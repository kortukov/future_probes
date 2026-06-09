import argparse
import json
import os
import nnsight
import torch
from tqdm import tqdm

from src.custom_datasets import get_dataset
from src.model_utils import get_model_and_tokenizer
import src.reasoning_model_utils as rmu
from src.interp import activations_nnsight

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BINARY_THRESHOLD_LOW = 0.05
BINARY_THRESHOLD_HIGH = 0.95


def get_response_token_activations(
    response_dict, input_prompt, wrapped_model, layer,
) -> tuple[torch.Tensor, bool]:
    """Extract response token activations and a binary label for a single response.

    Returns:
        activations: (num_response_tokens, hidden_size)
        is_positive: True if behavior is present (probability >= 0.95)
    """
    final_probability = response_dict["per_sentence_probabilities"][-1]
    if BINARY_THRESHOLD_LOW < final_probability < BINARY_THRESHOLD_HIGH:
        raise ValueError(f"Final probability is not binary: {final_probability}")

    is_positive = final_probability >= BINARY_THRESHOLD_HIGH

    full_response = response_dict["response"]
    if not rmu.has_thinking(full_response):
        raise ValueError(f"Response does not contain thinking")

    thinking_content = rmu.extract_thinking_content_from_response(full_response)
    prompt_with_thinking = input_prompt + thinking_content
    prompt_with_thinking_ids = wrapped_model.tokenizer(prompt_with_thinking, return_tensors="pt").input_ids
    prompt_with_thinking_len = prompt_with_thinking_ids.shape[1]

    prompt_with_response = input_prompt + full_response
    input_ids = wrapped_model.tokenizer(prompt_with_response, return_tensors="pt").input_ids

    full_sequence_activations = activations_nnsight.extract_prompt_activations(
        wrapped_model, input_ids, full_sequence=True, extraction_layer=layer
    )  # (seq_len, hidden_size)
    response_activations = full_sequence_activations[prompt_with_thinking_len:, :]  # (num_response_tokens, hidden_size)

    return response_activations, is_positive


def compute_steering_vector(results_file: str, layer: int, num_responses: int = None):
    outputs_file = results_file.replace("_results.json", "_outputs.json")

    with open(results_file, "r") as f:
        results = json.load(f)

    with open(outputs_file, "r") as f:
        outputs = json.load(f)

    output_dir = results_file.replace("_results.json", f"_steering_vectors/layer{layer}")
    os.makedirs(output_dir, exist_ok=True)

    model_name = results["model_name"]
    dataset_name = results["dataset_name"]
    subset = results["subset"]
    seed = results["seed"]
    disable_reasoning = results.get("disable_reasoning", False)

    assert len(outputs) == subset, (
        f"Lengths of outputs and subset do not match: {len(outputs)}, {subset}"
    )

    print(f"Loading model {model_name}")
    model, tokenizer = get_model_and_tokenizer(
        model_name, device=DEVICE, half_precision=True, multi_gpu=True
    )

    print(f"Loading dataset: {dataset_name}")
    dataset = get_dataset(dataset_name, subset, seed=seed, disable_reasoning=disable_reasoning)

    wrapped_model = nnsight.LanguageModel(model, tokenizer=tokenizer, dispatch=True)

    positive_sum = None
    negative_sum = None
    positive_count = 0
    negative_count = 0
    skipped_count = 0

    for sample_idx, (example, data_sample) in enumerate(
        tqdm(zip(dataset, outputs), desc="Computing steering vector", total=len(outputs))
    ):
        input_prompt = data_sample["input_prompt"]
        for response_idx, response_dict in enumerate(data_sample["responses"]):
            try:
                response_activations, is_positive = get_response_token_activations(
                    response_dict, input_prompt, wrapped_model, layer,
                )
            except ValueError:
                skipped_count += 1
                continue

            if num_responses is not None and response_idx >= num_responses:
                break

            # (num_response_tokens, hidden_size) -> (hidden_size,)
            token_sum = response_activations.sum(dim=0).float()
            num_tokens = response_activations.shape[0]

            per_token_norm = response_activations.norm(dim=1)
            mean_vector = response_activations.mean(dim=0)
            # print(f"Num response tokens: {num_tokens}")
            # print(f"Each token has norm: {per_token_norm.tolist()}")
            # print(f"Mean has norm: {mean_vector.norm().item():.4f}")

            if is_positive:
                if positive_sum is None:
                    positive_sum = torch.zeros_like(token_sum)
                positive_sum += token_sum
                positive_count += num_tokens
            else:
                if negative_sum is None:
                    negative_sum = torch.zeros_like(token_sum)
                negative_sum += token_sum
                negative_count += num_tokens

    print(f"Positive tokens: {positive_count}, Negative tokens: {negative_count}, Skipped responses: {skipped_count}")

    if positive_count == 0 or negative_count == 0:
        raise RuntimeError(
            f"Cannot compute steering vector: {positive_count} positive tokens, {negative_count} negative tokens"
        )

    mean_positive = positive_sum / positive_count
    mean_negative = negative_sum / negative_count
    steering_vector = mean_positive - mean_negative  # (hidden_size,)
    sv_norm = steering_vector.norm().item()
    mean_positive_norm = mean_positive.norm().item()
    mean_negative_norm = mean_negative.norm().item()
    overall_mean_vector = (positive_sum + negative_sum) / (positive_count + negative_count)
    overall_mean_norm = overall_mean_vector.norm().item()

    output_data = {
        "steering_vector": steering_vector,
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "subset": subset,
            "seed": seed,
            "disable_reasoning": disable_reasoning,
            "layer": layer,
            "positive_token_count": positive_count,
            "negative_token_count": negative_count,
            "skipped_response_count": skipped_count,
            "sv_norm": sv_norm,
            "mean_positive_norm": mean_positive_norm,
            "mean_negative_norm": mean_negative_norm,
            "overall_mean_norm": overall_mean_norm,
        },
    }

    responses_suffix = f"_nresp{num_responses}" if num_responses is not None else ""
    output_file = os.path.join(output_dir, f"diff_of_means_steering_vector{responses_suffix}.pt")
    torch.save(output_data, output_file)
    print(f"Saved steering vector to {output_file}")
    print(f"Steering vector norm: {sv_norm:.4f}")
    print(f"Mean positive norm: {mean_positive_norm:.4f}")
    print(f"Mean negative norm: {mean_negative_norm:.4f}")
    print(f"Overall mean norm: {overall_mean_norm:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute a difference-of-means steering vector from per-sentence probability results. "
                    "Streams through response token activations without saving them to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_file", type=str, required=True, help="Path to the results file from 19_get_per_sentence_probabilities.py."
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to extract activations from.",
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=None,
        help="Number of responses to each prompt to use for computing the steering vector. Defaults to all responses.",
    )
    args = parser.parse_args()

    compute_steering_vector(args.results_file, args.layer, args.num_responses)
