import torch
import transformers as tf
import nltk
from tqdm import tqdm
import time

from src.interp.activations_nnsight import extract_batch_prompt_activations
from src.reasoning_model_utils import has_thinking, THINK_END_TAG, GEMMA4_THINK_END_TAG, MINISTRAL_THINK_END_TAG


def future_probe_controlled_generation_fast(
    model,
    wrapped_model,
    tokenizer,
    device,
    input_prompt,
    total_max_new_tokens,
    temperature,
    num_candidates,
    probe,
    probing_layer,
    negative=False,
    verbose: bool = False,
    candidate_tokens: int = 32,
    early_exit: bool = False,
    sampling: bool = False,
    think_end_tag: str = THINK_END_TAG,
) -> str:



    inputs = tokenizer([input_prompt] * num_candidates, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    generated_ids = input_ids.new_zeros((num_candidates, 0))
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64, device=device)

    past_key_values = tf.DynamicCache()
    current_input_ids = torch.cat([input_ids, generated_ids], dim=1)
    current_attention_mask = inputs.attention_mask

    progress_bar = tqdm(total=total_max_new_tokens, desc="Generating response")
    while generated_ids.shape[1] < total_max_new_tokens:
       
        start_ts = time.time()
        # Step 1: Generate K candidate continuations
        step_1_start_ts = time.time()


        outputs = model.generate(
            current_input_ids,
            attention_mask=current_attention_mask,
            max_new_tokens=candidate_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )


        step_1_end_ts = time.time()
        if verbose:
            print(f"Time taken for step 1: {step_1_end_ts - step_1_start_ts} seconds")

        continuations = outputs.sequences[:, current_input_ids.shape[1]:]
        past_key_values = outputs.past_key_values
        num_continuation_tokens = continuations.shape[1]
        del outputs

        # Extract first sentence from each candidate
        tokenized_candidates = []
        hit_eos = []
        hit_end_think_tag = []
        for i in range(num_candidates):
            raw_ids = continuations[i]
            text = tokenizer.decode(raw_ids, skip_special_tokens=False)
            sentences = nltk.sent_tokenize(text)

            candidate = sentences[0] if sentences else ""
            if len(sentences) > 1 and has_thinking(sentences[1]):
                candidate += think_end_tag
            
            if early_exit and generated_ids.shape[1] > 2048:
                candidate += think_end_tag

            hit_eos.append(
                len(sentences) <= 1
                and (raw_ids == tokenizer.eos_token_id).any()
            )
            tokenized_candidate = tokenizer(candidate, return_tensors="pt", add_special_tokens=False).to(device)
            tokenized_candidates.append(tokenized_candidate.input_ids)
            hit_end_think_tag.append(has_thinking(candidate))

        candidate_lengths = [tc.shape[1] for tc in tokenized_candidates]
        del continuations

        # Crop generation cache: discard KV beyond the longest first sentence.
        max_cand_len = max(candidate_lengths) if candidate_lengths else 0
        excess_tokens = num_continuation_tokens - max_cand_len
        if excess_tokens > 0:
            past_key_values.crop(past_key_values.get_seq_length() - excess_tokens)

        # Step 2: Score each candidate's activation at probing_layer.
        # Compute the shared prefix KV cache once, then run only each
        # candidate's short suffix against it (crop back after each).
        step_2_start_ts = time.time()

        full_prefix = torch.cat([input_ids[0:1], generated_ids[0:1]], dim=1)
        prefix_len = full_prefix.shape[1]

        base_model = model.model if hasattr(model, 'model') else model.transformer
        decoder_layers = base_model.layers if hasattr(base_model, 'layers') else base_model.h
        target_layer = decoder_layers[probing_layer]

        captured_activation = {}

        def _capture_hook(_module, _input, output):
            captured_activation['value'] = output[0] if isinstance(output, tuple) else output

        hook_handle = target_layer.register_forward_hook(_capture_hook)

        try:
            scoring_cache = tf.DynamicCache()
            model(full_prefix, past_key_values=scoring_cache, use_cache=True, return_dict=True)
            prefix_last_activation = captured_activation['value'][0, -1, :].clone()
            del captured_activation['value']

            candidate_activations = []
            for i in range(num_candidates):
                cand_len = candidate_lengths[i]
                if cand_len > 0:
                    model(
                        tokenized_candidates[i],
                        past_key_values=scoring_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    activation = captured_activation['value'][0, -1, :].clone()
                    del captured_activation['value']
                    scoring_cache.crop(prefix_len)
                else:
                    activation = prefix_last_activation

                candidate_activations.append(activation)

            batch_activations = torch.stack(candidate_activations)
        finally:
            hook_handle.remove()
            del scoring_cache

        step_2_end_ts = time.time()
        if verbose:
            print(f"Time taken for step 2: {step_2_end_ts - step_2_start_ts} seconds")

        token_scores = probe.predict_proba_tensor(
            batch_activations.to(probe.mean.device)
        )  # (num_candidates,)

        del batch_activations

        # if early_exit:
        # # We changed the logic of early exit to just enforce ending thinking.
        #     # We reward the candidate if it has the end think tag.
        #     for i in range(num_candidates):
        #         if hit_end_think_tag[i]:
        #             if verbose:
        #                 print(f"Candidate {i} has end think tag. Score: {token_scores[i].item()}")
        #             if negative:
        #                 token_scores[i] -= 1
        #             else:
        #                 token_scores[i] += 1

        # Step 3: Pick best and append
        if negative: 
            if sampling:
                reverse_scores = 1 - token_scores 
                reverse_probs = reverse_scores / reverse_scores.sum(dim=-1, keepdim=True)
                chosen_idx = torch.multinomial(reverse_probs, num_samples=1).item()
                if verbose:
                    print(f"Token scores: {token_scores}")
                    print(f"Reverse probs: {reverse_probs}")
                    print(f"Choosing random sample with score: {token_scores[chosen_idx].item()}")
            else:
                chosen_idx = token_scores.argmin().item()
                if verbose:
                    print(f"Choosing minimum score: {token_scores[chosen_idx].item()}")
        else:
            if sampling:
                token_probs = token_scores / token_scores.sum(dim=-1, keepdim=True)
                chosen_idx = torch.multinomial(token_probs, num_samples=1).item()
                if verbose:
                    print(f"Token scores: {token_scores}")
                    print(f"Token probs: {token_probs}")
                    print(f"Choosing random sample with score: {token_scores[chosen_idx].item()}")
            else:
                chosen_idx = token_scores.argmax().item()
                if verbose:
                    print(f"Choosing maximum score: {token_scores[chosen_idx].item()}")

        chosen_len = candidate_lengths[chosen_idx]
        if chosen_len == 0:
            break
        chosen_ids = tokenized_candidates[chosen_idx][0, :chosen_len].repeat(num_candidates, 1)
        generated_ids = torch.cat([generated_ids, chosen_ids], dim=1)

        # Update the cache. Each batch element contains a copy of the chosen tokens. They are also cropped to the chosen length.
        chosen_idx_tensor = torch.tensor([chosen_idx], device="cpu")
        past_key_values.crop(input_ids.shape[1] + generated_ids.shape[1] - 1)
        past_key_values.batch_select_indices(chosen_idx_tensor)
        past_key_values.batch_repeat_interleave(num_candidates)

        # Update the current input ids. We only need the last token for the next sentence generation, since everything else is already in the cache.
        current_input_ids = generated_ids[:, -1:]
        # Update the attention mask.  Following https://huggingface.co/docs/transformers/v5.2.0/en/cache_explanation,  we extend it by the amount of new tokens.
        current_attention_mask = torch.cat([current_attention_mask, current_attention_mask.new_ones((current_attention_mask.shape[0], chosen_ids.shape[1]))], dim=-1)

        # Cache position contains the position of the last generated token (it was not cached in the previous step), in other words the first token that needs to be cached.
        # It is equal to the amount (length) of tokens already cached. (see how we crop past_key_values above)
        # correct version
        cache_position = torch.tensor([input_ids.shape[1] + generated_ids.shape[1] - 1], device=device, dtype=torch.int64)

        if verbose:
            print(
                f"Added {chosen_ids.shape[1]} tokens to the generated sequence with score {token_scores[chosen_idx].item()}"
            )
            print(
                f"Added sentence: {tokenizer.decode(chosen_ids[0], skip_special_tokens=False)}\n"
            )

        if hit_eos[chosen_idx]:
            break
        
        progress_bar.update(chosen_ids.shape[1])

        end_ts = time.time()
        if verbose:
            print(f"Time taken for one sentence: {end_ts - start_ts} seconds")

    progress_bar.close()

    return tokenizer.decode(generated_ids[0], skip_special_tokens=False)

