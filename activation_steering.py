import argparse
import json
import os
import torch
import gc
from tqdm import tqdm
from nnsight import LanguageModel

from src.custom_datasets import get_dataset
from src.model_utils import get_model_and_tokenizer
from src.reasoning_model_utils import extract_response_part, extract_thinking_content_from_response, has_thinking, THINK_END_TAG, GPT_OSS_THINK_END_TAG, GEMMA4_THINK_END_TAG, MINISTRAL_THINK_END_TAG


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def get_nnsight_layer(nn_model, hf_model, layer_idx):
    """Return the nnsight Envoy for decoder layer `layer_idx`."""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return nn_model.model.layers[layer_idx]
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return nn_model.transformer.h[layer_idx]
    raise ValueError(f"Unsupported model architecture: {type(hf_model)}")


def get_layer_device_dtype(hf_model, layer_idx):
    """Return (device, dtype) of decoder layer `layer_idx` in the raw HF model."""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        p = next(hf_model.model.layers[layer_idx].parameters())
    elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        p = next(hf_model.transformer.h[layer_idx].parameters())
    else:
        raise ValueError(f"Unsupported model architecture: {type(hf_model)}")
    return p.device, p.dtype


def _steered_generate(nn_model, tokenizer, input_prompt, input_len, max_new_tokens, temperature, target_layer, scaled_sv):
    """Run a single steered generation and return the raw generated token ids."""
    with torch.no_grad():
        with nn_model.generate(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        ) as tracer:
            with tracer.invoke(input_prompt):
                with tracer.iter[:]:
                    original_output = target_layer.output.save()
                    target_layer.output = original_output + scaled_sv
            with tracer.invoke():
                final_output = nn_model.generator.output.save()
    return final_output[0][input_len:]


def generate_steered_responses(
    nn_model, hf_model, tokenizer, dataset, example, input_prompt,
    max_new_tokens, temperature, num_responses,
    sv_tensor, layer, effective_multiplier,
    defer_behavior_detection=False, verbose=False,
    early_exit=False, think_end_tag=THINK_END_TAG,
) -> tuple[list[dict], list[bool | None]]:

    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    input_len = input_ids.shape[1]

    target_layer = get_nnsight_layer(nn_model, hf_model, layer)
    scaled_sv = sv_tensor * effective_multiplier

    thinking_budget = 2048

    response_dicts = []
    behaviors = []

    for _ in tqdm(range(num_responses), total=num_responses, leave=False):
        if early_exit:
            phase1_tokens = _steered_generate(
                nn_model, tokenizer, input_prompt, input_len,
                min(thinking_budget, max_new_tokens), temperature,
                target_layer, scaled_sv,
            )
            phase1_text = tokenizer.decode(phase1_tokens, skip_special_tokens=False)
            hit_eos = (phase1_tokens == tokenizer.eos_token_id).any().item()
            thinking_finished = think_end_tag in phase1_text

            if hit_eos or thinking_finished:
                full_response = phase1_text
            else:
                continuation_prompt = input_prompt + phase1_text + think_end_tag
                continuation_ids = tokenizer(continuation_prompt, return_tensors="pt").input_ids
                continuation_len = continuation_ids.shape[1]
                remaining_budget = max(max_new_tokens - thinking_budget, 256)

                phase2_tokens = _steered_generate(
                    nn_model, tokenizer, continuation_prompt, continuation_len,
                    remaining_budget, temperature,
                    target_layer, scaled_sv,
                )
                phase2_text = tokenizer.decode(phase2_tokens, skip_special_tokens=False)
                full_response = phase1_text + think_end_tag + phase2_text
        else:
            generated_tokens = _steered_generate(
                nn_model, tokenizer, input_prompt, input_len,
                max_new_tokens, temperature, target_layer, scaled_sv,
            )
            full_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        response_has_thinking = has_thinking(full_response) or (think_end_tag in full_response)
        answer_content = extract_response_part(full_response)
        thinking_content = extract_thinking_content_from_response(full_response)

        if defer_behavior_detection:
            behavior = None
        else:
            behavior = dataset.detect_behavior(example, answer_content)
            if not response_has_thinking:
                behavior = None
                if verbose:
                    print("Response does not have thinking, so behavior is None")
            if verbose:
                print(f"Reasoning content: {thinking_content}")
                print(f"Answer content: {answer_content}")
            print(f"Behavior: {behavior}")

        response_dict = {
            "response": full_response,
            "behavior": behavior,
            "thinking_content": thinking_content,
            "answer_content": answer_content,
            "response_has_thinking": response_has_thinking,
        }
        response_dicts.append(response_dict)
        behaviors.append(behavior)

    return response_dicts, behaviors


def steer_with_vector(
    steering_vector_path, results_file, multiplier, negative, responses, normalize, mean_act_norm=False, verbose=False, early_exit=False,
    output_dir_name="activation_steering",
):
    if mean_act_norm and normalize:
        raise ValueError("Cannot use both mean activation norm and normalization.")

    with open(results_file, "r") as f:
        results = json.load(f)

    outputs_file = results_file.replace("results.json", "outputs.json")
    with open(outputs_file, "r") as f:
        outputs = json.load(f)

    sv_data = torch.load(steering_vector_path, map_location="cpu")
    sv_tensor = sv_data["steering_vector"]
    sv_config = sv_data["config"]
    layer = sv_config["layer"]
    overall_mean_norm = sv_config["overall_mean_norm"]

    if normalize:
        sv_tensor = sv_tensor / sv_tensor.norm()
    
    if mean_act_norm:
        sv_tensor = sv_tensor / sv_tensor.norm()
        sv_tensor = sv_tensor * overall_mean_norm


    output_dir = results_file.replace("behavioral_stability", output_dir_name)
    output_dir = output_dir.replace("_results.json", "")
    os.makedirs(output_dir, exist_ok=True)

    model_name = results["model_name"]
    dataset_name = results["dataset_name"]
    subset = results["subset"]
    seed = results["seed"]
    temperature = results["temperature"]
    max_new_tokens = results["max_new_tokens"]
    num_samples = results["num_samples"]
    disable_reasoning = results.get("disable_reasoning", False)

    print(f"Loading model and tokenizer: {model_name}")
    hf_model, tokenizer = get_model_and_tokenizer(
        model_name=model_name, device=DEVICE, half_precision=True, multi_gpu=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    layer_device, layer_dtype = get_layer_device_dtype(hf_model, layer)
    sv_tensor = sv_tensor.to(device=layer_device, dtype=layer_dtype)

    nn_model = LanguageModel(hf_model, tokenizer=tokenizer)

    dataset = get_dataset(dataset_name, subset, seed=seed, disable_reasoning=disable_reasoning)
    defer_behavior = dataset.requires_model_for_detection
    if defer_behavior:
        print("Behavior detection will be deferred until after all generations are complete.")

    if responses is None:
        responses = num_samples

    if "gpt-oss" in model_name:
        think_end_tag = GPT_OSS_THINK_END_TAG
    elif "gemma-4" in model_name:
        think_end_tag = GEMMA4_THINK_END_TAG
    elif "ministral" in model_name.lower():
        think_end_tag = MINISTRAL_THINK_END_TAG
    else:
        think_end_tag = THINK_END_TAG
    effective_multiplier = -multiplier if negative else multiplier

    augmented_outputs = []
    total_none_count = 0
    total_response_count = 0

    def save_partial_results(augmented_outputs_list, total_none, total_responses, output_dir, pos_vs_neg_suffix, layer_suffix, mult_suffix, norm_suffix, mean_act_norm_suffix, early_exit_suffix):
        """Save partial results every 10 samples."""
        average_steered_behaviors = [output.get("average_steered_behavior", 0.0) for output in augmented_outputs_list]
        total_avg_behavior = sum(average_steered_behaviors) / len(average_steered_behaviors) if average_steered_behaviors else 0.0
        
        partial_results = {
            "steering_vector_path": steering_vector_path,
            "steering_vector_config": {k: v for k, v in sv_config.items() if not isinstance(v, torch.Tensor)},
            "multiplier": multiplier,
            "negative": negative,
            "normalize": normalize,
            "early_exit": early_exit,
            "responses": responses,
            "layer": layer,
            "original_results": results,
            "average_steered_behaviors": average_steered_behaviors,
            "total_avg_behavior": total_avg_behavior,
            "none_behavior_count": total_none,
            "total_response_count": total_responses,
            "none_behavior_fraction": total_none / total_responses if total_responses > 0 else 0.0,
            "samples_processed": len(augmented_outputs_list),
        }
        
        partial_results_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}_{mult_suffix}{norm_suffix}{mean_act_norm_suffix}{early_exit_suffix}_partial_results.json")
        with open(partial_results_path, "w") as f:
            json.dump(partial_results, f, indent=4)
        
        partial_outputs_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}_{mult_suffix}{norm_suffix}{mean_act_norm_suffix}{early_exit_suffix}_partial_outputs.json")
        with open(partial_outputs_path, "w") as f:
            json.dump(augmented_outputs_list, f, indent=4)
        
        print(f"Saved partial results after {len(augmented_outputs_list)} samples")

    for idx, (example, output) in enumerate(tqdm(zip(dataset, outputs), total=len(dataset))):
        input_prompt = output["input_prompt"]

        steered_responses, steered_behaviors = generate_steered_responses(
            nn_model=nn_model,
            hf_model=hf_model,
            tokenizer=tokenizer,
            dataset=dataset,
            example=example,
            input_prompt=input_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_responses=responses,
            sv_tensor=sv_tensor,
            layer=layer,
            effective_multiplier=effective_multiplier,
            defer_behavior_detection=defer_behavior,
            verbose=verbose,
            early_exit=early_exit,
            think_end_tag=think_end_tag,
        )

        output["steered_responses"] = steered_responses

        if not defer_behavior:
            not_none_behaviors = [b for b in steered_behaviors if b is not None]
            total_none_count += len(steered_behaviors) - len(not_none_behaviors)
            total_response_count += len(steered_behaviors)
            average_steered_behavior = sum(not_none_behaviors) / len(not_none_behaviors) if not_none_behaviors else 0.0
            output["average_steered_behavior"] = average_steered_behavior

        augmented_outputs.append(output)
        
        # Save partial results every 10 samples
        if (idx + 1) % 10 == 0:
            pos_vs_neg_suffix = "negative" if negative else "positive"
            layer_suffix = f"layer{layer}"
            mult_suffix = f"mult{multiplier}"
            norm_suffix = "_normed" if normalize else ""
            mean_act_norm_suffix = "_mean_act_normed" if mean_act_norm else ""
            early_exit_suffix = "_early_exit" if early_exit else ""
            save_partial_results(augmented_outputs, total_none_count, total_response_count, output_dir, 
                                pos_vs_neg_suffix, layer_suffix, mult_suffix, norm_suffix, mean_act_norm_suffix, early_exit_suffix)

    if defer_behavior:
        print("Freeing generation model memory before behavior detection...")
        del nn_model
        del hf_model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        print("Computing deferred behavior detection...")
        for example, output in tqdm(
            zip(dataset, augmented_outputs),
            total=len(dataset),
            desc="Detecting behaviors",
        ):
            for response_dict in output["steered_responses"]:
                if not response_dict["response_has_thinking"]:
                    # Do not detect behavior for responses that did not finish thinking.
                    behavior = None
                else:
                    behavior = dataset.detect_behavior(example, response_dict["answer_content"])

                response_dict["behavior"] = behavior
                print(f"Behavior: {behavior}")

            behaviors = [rd["behavior"] for rd in output["steered_responses"]]
            not_none_behaviors = [b for b in behaviors if b is not None]
            total_none_count += len(behaviors) - len(not_none_behaviors)
            total_response_count += len(behaviors)
            average_steered_behavior = sum(not_none_behaviors) / len(not_none_behaviors) if not_none_behaviors else 0.0
            output["average_steered_behavior"] = average_steered_behavior

    average_steered_behaviors = [output["average_steered_behavior"] for output in augmented_outputs]
    total_avg_behavior = sum(average_steered_behaviors) / len(average_steered_behaviors)

    augmented_results = {
        "steering_vector_path": steering_vector_path,
        "steering_vector_config": {k: v for k, v in sv_config.items() if not isinstance(v, torch.Tensor)},
        "multiplier": multiplier,
        "negative": negative,
        "normalize": normalize,
        "early_exit": early_exit,
        "responses": responses,
        "layer": layer,
        "original_results": results,
        "average_steered_behaviors": average_steered_behaviors,
        "total_avg_behavior": total_avg_behavior,
        "none_behavior_count": total_none_count,
        "total_response_count": total_response_count,
        "none_behavior_fraction": total_none_count / total_response_count if total_response_count > 0 else 0.0,
    }

    pos_vs_neg_suffix = "negative" if negative else "positive"
    layer_suffix = f"layer{layer}"
    mult_suffix = f"mult{multiplier}"
    norm_suffix = "_normed" if normalize else ""
    mean_act_norm_suffix = "_mean_act_normed" if mean_act_norm else ""
    early_exit_suffix = "_early_exit" if early_exit else ""
    augmented_results_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}_{mult_suffix}{norm_suffix}{mean_act_norm_suffix}{early_exit_suffix}_results.json")
    print(f"Saving final results to {augmented_results_path}")
    with open(augmented_results_path, "w") as f:
        json.dump(augmented_results, f, indent=4)

    augmented_outputs_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}_{mult_suffix}{norm_suffix}{mean_act_norm_suffix}{early_exit_suffix}_outputs.json")
    print(f"Saving final outputs to {augmented_outputs_path}")
    with open(augmented_outputs_path, "w") as f:
        json.dump(augmented_outputs, f, indent=4)

    partial_results_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}_{mult_suffix}{norm_suffix}{mean_act_norm_suffix}{early_exit_suffix}_partial_results.json")
    partial_outputs_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}_{mult_suffix}{norm_suffix}{mean_act_norm_suffix}{early_exit_suffix}_partial_outputs.json")
    for partial_path in (partial_results_path, partial_outputs_path):
        if os.path.exists(partial_path):
            os.remove(partial_path)
            print(f"Deleted partial checkpoint {partial_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steer the model using a precomputed steering vector (from 26_compute_steering_vector.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--steering_vector", type=str, required=True,
        help="Path to the steering vector .pt file from 26_compute_steering_vector.py.",
    )
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to the results of 6_behavioral_stability.py file.",
    )
    parser.add_argument(
        "--multiplier", type=float, default=1.0,
        help="Multiplier for the steering vector.",
    )
    parser.add_argument(
        "--negative", type=lambda x: x.lower() == "true", default=False,
        help="Steer towards negative behavior (negate the vector). Use --negative=True or --negative=False",
    )
    parser.add_argument(
        "--responses", type=int, default=None,
        help="Number of responses to generate per prompt. Defaults to num_samples from results.",
    )
    parser.add_argument(
        "--normalize", type=lambda x: x.lower() == "true", default=False,
        help="Normalize the steering vector to unit norm before applying. Use --normalize=True or --normalize=False",
    )
    parser.add_argument(
        "--mean_act_norm", type=lambda x: x.lower() == "true", default=False,
        help="Normalize the steering vector to the norm of the mean activation vector (following https://arxiv.org/pdf/2506.18167)",
    )
    parser.add_argument(
        "--verbose", type=lambda x: x.lower() == "true", default=False,
        help="Verbose mode. Print more information.",
    )
    parser.add_argument(
        "--early_exit", type=lambda x: x.lower() == "true", default=False,
        help="Finish reasoning after 2048 tokens if not already finished. Use --early_exit=True or --early_exit=False",
    )
    parser.add_argument(
        "--output_dir_name", type=str, default="activation_steering",
        help="Name to replace 'behavioral_stability' with in the output directory path. "
             "Use e.g. 'probe_steering' to separate probe-based steering results.",
    )

    args = parser.parse_args()

    steer_with_vector(
        steering_vector_path=args.steering_vector,
        results_file=args.results,
        multiplier=args.multiplier,
        negative=args.negative,
        responses=args.responses,
        normalize=args.normalize,
        mean_act_norm=args.mean_act_norm,
        verbose=args.verbose,
        early_exit=args.early_exit,
        output_dir_name=args.output_dir_name,
    )
