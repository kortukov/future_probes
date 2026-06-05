import argparse
import json
import os
import nnsight
import torch
from tqdm import tqdm


from src.custom_datasets import get_dataset
from src.model_utils import get_model_and_tokenizer
from src.interp import activations_nnsight

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def get_activations_and_labels_for_one_response(
    response_dict, input_prompt, wrapped_model, layer, before_only, example, dataset,
    batch_size: int = 1,
    response_only: bool = False,
    reasoning_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    partial_prompts = []
    labels = []

    response_start_index = response_dict.get("response_start_index", 0)

    for idx, partial_prompt_dict in enumerate(response_dict["partial_prompts"]):
        if response_only and idx <= response_start_index:
            continue
        if reasoning_only and idx > response_start_index:
            continue

        partial_response = partial_prompt_dict["partial_response"]

        if before_only:
            behavior_already_in_partial = dataset.detect_behavior(example, partial_response)
            if behavior_already_in_partial:
                continue

        labels.append(partial_prompt_dict["probability"])
        partial_prompts.append(input_prompt + partial_response)

    if not partial_prompts:
        return torch.empty(0), torch.tensor([], dtype=torch.float16)

    all_activations = []
    for batch_start in range(0, len(partial_prompts), batch_size):
        batch_prompts = partial_prompts[batch_start:batch_start + batch_size]

        tokenized = wrapped_model.tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
        )
        input_ids = tokenized.input_ids        # (B, seq_len)
        attention_mask = tokenized.attention_mask  # (B, seq_len)
        seq_len = input_ids.shape[1]
        # With left-padding, the last real token is always at position seq_len - 1
        prompt_lengths = [seq_len] * len(batch_prompts)

        batch_activations = activations_nnsight.extract_batch_prompt_activations(
            wrapped_model, input_ids, layer, prompt_lengths,
            attention_mask=attention_mask,
        )  # (B, hidden_size)
        all_activations.append(batch_activations)

    activations = torch.cat(all_activations, dim=0)  # (num_sentences, hidden_size)
    labels = torch.tensor(labels, dtype=torch.float16)  # (num_sentences,)

    return activations, labels


def get_sample_activation_data_for_response_sentences(example, sample_idx, data_sample, wrapped_model, layer, before_only, dataset, batch_size: int = 1, response_only: bool = False, reasoning_only: bool = False) -> dict:
    """For each response in the data sample, save activations at the end of each of the response sentences."""
    sample_activation_data = {"sample_idx": sample_idx, "responses": []}
    input_prompt = data_sample["input_prompt"]
    for response_dict in tqdm(data_sample["responses"], desc="Processing responses"):
        # (num_sentences, hidden_size), (num_sentences,)
        response_activations, response_labels = (
            get_activations_and_labels_for_one_response(
                response_dict, input_prompt, wrapped_model, layer, before_only, example, dataset,
                batch_size=batch_size, response_only=response_only, reasoning_only=reasoning_only,
            )
        )
        num_sentences = response_activations.shape[0]

        response_activation_data = {
            "activations": response_activations,  # (num_sentences, hidden_size)
            "labels": response_labels,  # (num_sentences,)
        }
        sample_activation_data["responses"].append(response_activation_data)
    return sample_activation_data


def get_mean_response_activations_for_one_response(
    response_dict, input_prompt, wrapped_model, layer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get mean activation per response sentence (averaged over the sentence's tokens).

    Runs the model once on the full prompt (input + complete response) with
    full_sequence=True, then uses tokenized boundary lengths of consecutive
    partial prompts to identify which token positions belong to each response
    sentence.  Each sentence's hidden states are averaged into a single vector.

    Returns:
        activations: (num_response_sentences, hidden_size) or empty tensor.
        labels: (num_response_sentences,) probability labels.
    """
    partial_prompts_list = response_dict["partial_prompts"]
    response_start_index = response_dict.get("response_start_index", 0)

    num_sentences = len(partial_prompts_list) - 1  # index 0 is the base prompt
    num_response_sentences = num_sentences - response_start_index

    if num_response_sentences <= 0:
        return torch.empty(0), torch.tensor([], dtype=torch.float16)

    full_prompt = input_prompt + partial_prompts_list[-1]["partial_response"]
    full_tokens = wrapped_model.tokenizer(full_prompt, return_tensors="pt").input_ids

    # For each response sentence i (0-indexed in the sentences list), its
    # tokens span from len(tok(prompt_before_i)) to len(tok(prompt_after_i))-1.
    # prompt_before_i = input_prompt + partial_prompts_list[i]["partial_response"]
    # prompt_after_i  = input_prompt + partial_prompts_list[i+1]["partial_response"]
    boundaries = []
    for i in range(response_start_index, num_sentences):
        before_text = input_prompt + partial_prompts_list[i]["partial_response"]
        after_text = input_prompt + partial_prompts_list[i + 1]["partial_response"]
        start_tok = len(wrapped_model.tokenizer(before_text, return_tensors="pt").input_ids[0])
        end_tok = len(wrapped_model.tokenizer(after_text, return_tensors="pt").input_ids[0])
        label = partial_prompts_list[i + 1]["probability"]
        if start_tok < end_tok:
            boundaries.append((start_tok, end_tok, label))

    if not boundaries:
        return torch.empty(0), torch.tensor([], dtype=torch.float16)

    full_seq_activations = activations_nnsight.extract_prompt_activations(
        wrapped_model, full_tokens, full_sequence=True, extraction_layer=layer,
    )  # (seq_len, hidden_size)

    mean_activations = []
    labels = []
    for start, end, label in boundaries:
        sentence_acts = full_seq_activations[start:end]  # (num_tokens, hidden_size)
        mean_activations.append(sentence_acts.mean(dim=0))
        labels.append(label)

    activations = torch.stack(mean_activations)  # (num_response_sentences, hidden_size)
    labels = torch.tensor(labels, dtype=torch.float16)  # (num_response_sentences,)
    return activations, labels


def get_sample_mean_response_activation_data(
    example, sample_idx, data_sample, wrapped_model, layer, dataset,
) -> dict:
    """For each response, save per-sentence mean activations over response tokens."""
    sample_activation_data = {"sample_idx": sample_idx, "responses": []}
    input_prompt = data_sample["input_prompt"]

    for response_dict in tqdm(data_sample["responses"], desc="Processing responses"):
        response_activations, response_labels = get_mean_response_activations_for_one_response(
            response_dict, input_prompt, wrapped_model, layer,
        )

        response_activation_data = {
            "activations": response_activations,  # (num_response_sentences, hidden_size)
            "labels": response_labels,  # (num_response_sentences,)
        }
        sample_activation_data["responses"].append(response_activation_data)

    return sample_activation_data


def gather_per_sentence_activations(
    results_file: str, layer: int, before_only: bool, batch_size: int = 1,
    response_only: bool = False, reasoning_only: bool = False,
    response_mean_sentence: bool = False, use_kernels: bool = False,
):
    outputs_file = results_file.replace("_results.json", "_outputs.json")

    with open(results_file, "r") as f:
        results = json.load(f)

    with open(outputs_file, "r") as f:
        outputs = json.load(f)

    # Prepare output file structure
    output_dir = results_file.replace("_results.json", f"_activations/layer{layer}")
    if before_only:
        output_dir += "_before_only"
    if response_only:
        output_dir += "_response_only"
    if reasoning_only:
        output_dir += "_reasoning_only"
    if response_mean_sentence:
        output_dir += "_response_mean_sentence"
    os.makedirs(output_dir, exist_ok=True)

    model_name = results["model_name"]
    dataset_name = results["dataset_name"]
    subset = results["subset"]
    num_base_responses = results["num_base_responses"]
    num_samples = results["num_samples"]
    seed = results["seed"]
    disable_reasoning = results.get("disable_reasoning", False)

    assert len(outputs) == subset, (
        f"Lengths of outputs and subset do not match: {len(outputs)}, {subset}"
    )

    print(f"Loading model {model_name}")
    model, tokenizer = get_model_and_tokenizer(
        model_name, device=DEVICE, half_precision=True, multi_gpu=True, use_kernels=use_kernels
    )

    print(f"Loading dataset: {dataset_name}")
    dataset = get_dataset(dataset_name, subset, seed=seed, disable_reasoning=disable_reasoning)

    wrapped_model = nnsight.LanguageModel(model, tokenizer=tokenizer, dispatch=True)

    activation_data = []
    for sample_idx, (example, data_sample) in enumerate(
        tqdm(zip(dataset, outputs), desc="Processing data samples", total=len(outputs))
    ):
        if response_mean_sentence:
            sample_activation_data = get_sample_mean_response_activation_data(
                example, sample_idx, data_sample, wrapped_model, layer, dataset,
            )
        else:
            sample_activation_data = get_sample_activation_data_for_response_sentences(
                example, sample_idx, data_sample, wrapped_model, layer, before_only, dataset,
                batch_size=batch_size, response_only=response_only, reasoning_only=reasoning_only,
            )

        activation_data.append(sample_activation_data)

    resulting_activation_data = {
        "activation_data": activation_data,
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "subset": subset,
            "num_base_responses": num_base_responses,
            "num_samples": num_samples,
            "seed": seed,
            "disable_reasoning": disable_reasoning,
            "layer": layer,
            "response_only": response_only,
            "reasoning_only": reasoning_only,
            "response_mean_sentence": response_mean_sentence,
        },
    }

    output_file = os.path.join(output_dir, f"activations.pt")
    torch.save(resulting_activation_data, output_file)
    print(f"Saved activations and labels to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gather per-sentence activations from results of 19_get_per_sentence_probabilities.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_file", type=str, required=True, help="Path to the results file."
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to use for gathering activations.",
    )
    parser.add_argument(
        "--before_only",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Only gather activations before the behavior happens.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for activation extraction.",
    )
    parser.add_argument(
        "--response_only",
        action="store_true",
        default=False,
        help="Only gather activations from response sentences (skip thinking sentences).",
    )
    parser.add_argument(
        "--reasoning_only",
        action="store_true",
        default=False,
        help="Only gather activations from reasoning sentences, before the final response starts.",
    )
    parser.add_argument(
        "--response_mean_sentence",
        action="store_true",
        default=False,
        help="Instead of last-token activations per sentence, compute the mean of all "
             "token activations within each response sentence. Yields one vector per "
             "response sentence (same shape as --response_only).",
    )
    parser.add_argument(
        "--use_kernels",
        action="store_true",
        default=False,
        help="Use kernels for activation extraction.",
    )
    args = parser.parse_args()

    if args.response_mean_sentence and args.response_only:
        parser.error("--response_mean_sentence already considers only response tokens; "
                      "--response_only is redundant.")
    if args.response_only and args.reasoning_only:
        parser.error("--response_only and --reasoning_only are mutually exclusive.")

    gather_per_sentence_activations(
        args.results_file, args.layer, args.before_only, args.batch_size,
        response_only=args.response_only,
        reasoning_only=args.reasoning_only,
        response_mean_sentence=args.response_mean_sentence,
        use_kernels=args.use_kernels,
    )
