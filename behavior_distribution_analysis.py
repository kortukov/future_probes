"""
Get per-sentence behavior probabilities for model responses.

For each example in the dataset:
1. Generate `num_base_responses` base completions
2. For each base response, split into sentences and prepare partial prompts
3. For each sentence prefix, generate `num_samples` completions
4. Free the vllm model, then compute base behavior probability from base completions
   (deferred so datasets that load their own detection models have enough GPU memory)
5. Compute per-sentence behavior probabilities and save results

Output format per example:
{
    "input_prompt": "...",
    "base_behavior_probability": 0.6,
    "responses": [
        {
            "response": "Full response text...",
            "sentences": ["Sentence 1.", "Sentence 2.", ...],
            "per_sentence_probabilities": [0.6, 0.7, 0.8, ...],  # Starts with base prob
            "response_with_probabilities": "(0.6) Sentence 1. (0.7) Sentence 2. ...",
            "partial_prompts": [
                {"partial_response": "", "probability": 0.6},
                {"partial_response": "Sentence 1.", "probability": 0.7},
                ...
            ]
        },
        ...  # num_base_responses entries
    ]
}
"""

import argparse
import json
import os
import torch
import vllm
import numpy as np
from tqdm import tqdm
import itertools

from src.model_utils import (
    get_vllm_model_and_tokenizer,
    delete_vllm_model_and_free_memory,
)
from src.custom_datasets import get_dataset
from src.interp.probing import format_response_with_probabilities
from src.reasoning_model_utils import (
    extract_response_part,
    split_response_with_reasoning,
    reconstruct_partial_response,
)

# Fix for VLLM CUDA multiprocessing issue
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def variable_length_batched(iterable, lengths):
    # variable_length_batched([1, 2, 3, 4, 5], [2, 3]) → [[1, 2], [3, 4, 5]]
    sum_lengths = sum(lengths)
    if sum_lengths != len(iterable):
        raise ValueError(
            "The sum of the lengths must be equal to the length of the iterable."
        )

    iterator = iter(iterable)
    for length in lengths:
        batch = tuple(itertools.islice(iterator, length))
        yield batch



def create_partial_prompts_from_sentences(
    input_prompt: str, sentences: list[str], response_start_index: int = 0,
) -> list[str]:
    """Create prompts with progressively more sentences appended.

    For reasoning models (response_start_index > 0), thinking sentences are
    wrapped in <think>...</think> tags so the model is prompted to generate
    the actual response.

    Returns a list of prompts (does NOT include the bare base prompt,
    since base probability is computed separately):
    - input_prompt + sentence[0]
    - input_prompt + sentence[0] + sentence[1]
    - etc.
    """
    partial_prompts = []
    for i in range(len(sentences)):
        partial_response = reconstruct_partial_response(
            sentences, i, response_start_index
        )
        partial_prompts.append(input_prompt + partial_response)
    return partial_prompts


def compute_behavior_probability(
    behaviors: list[int | None],
) -> float:
    """Compute the probability of behavior being True from a list of behaviors."""
    int_behaviors = [b for b in behaviors if b is not None]
    if not int_behaviors:
        return 0.0
    return sum(int_behaviors) / len(int_behaviors)


def run_per_sentence_probabilities(
    model_name: str,
    dtype: str,
    dataset_name: str,
    subset: int | None,
    results_file: str,
    num_base_responses: int,
    num_samples: int,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    disable_reasoning: bool,
    quantize: bool,
):
    """Main function to compute per-sentence behavior probabilities."""

    dataset = get_dataset(
        dataset_name,
        subset,
        icl_examples=None,
        seed=seed,
        disable_reasoning=disable_reasoning,
    )

    model, tokenizer, lora_request = get_vllm_model_and_tokenizer(
        model_name, dtype, multi_gpu=True, max_model_len=max_new_tokens + 2000, quantize=quantize
    )

    is_gpt_oss = "gpt-oss" in model_name.lower()
    is_gemma4 = "gemma-4" in model_name.lower()
    is_ministral = "ministral" in model_name.lower()
    keep_special_tokens = is_gpt_oss or is_gemma4

    # Step 1: Generate base responses for all prompts
    print("Step 1: Generating base responses...")
    base_sampling_params = vllm.SamplingParams(
        n=num_base_responses,
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
        skip_special_tokens=not keep_special_tokens,
    )

    input_prompts = [
        dataset.get_model_input_from_an_example(example, tokenizer)[1]
        for example in dataset
    ]

    base_outputs = model.generate(
        input_prompts, base_sampling_params, lora_request=lora_request
    )

    # Step 2: Prepare partial prompts
    print("Step 2: Preparing partial prompts...")

    all_partial_prompts = []  # Flattened list of all partial prompts
    partial_prompt_counts = []  # Number of partial prompts per (example, response) pair
    example_response_counts = []  # Number of responses per example (should be num_base_responses)

    base_responses_per_example = []  # Store responses for each example

    for example, base_output, input_prompt in zip(dataset, base_outputs, input_prompts):
        # Extract all base responses
        responses = [output.text for output in base_output.outputs]

        # For each response, create partial prompts
        example_responses_info = []
        response_count = 0

        for response in responses:
            sentences, response_start_index = split_response_with_reasoning(response, is_gpt_oss=is_gpt_oss, is_gemma4=is_gemma4, is_ministral=is_ministral)

            # Create partial prompts: after each sentence (not including bare base prompt)
            partial_prompts = create_partial_prompts_from_sentences(
                input_prompt, sentences, response_start_index
            )

            # Store info for later processing
            num_partials = len(partial_prompts)  # = len(sentences)
            example_responses_info.append(
                {
                    "response": response,
                    "sentences": sentences,
                    "num_partials": num_partials,
                    "response_start_index": response_start_index,
                }
            )

            if num_partials > 0:
                all_partial_prompts.extend(partial_prompts)
                partial_prompt_counts.append(num_partials)
            else:
                # Empty response - no partial prompts to generate
                partial_prompt_counts.append(0)
            response_count += 1

        example_response_counts.append(response_count)
        base_responses_per_example.append(example_responses_info)

    print(f"Total partial prompts to process: {len(all_partial_prompts)}")
    avg_sentences = np.mean(
        [info["num_partials"] for infos in base_responses_per_example for info in infos]
    )
    print(f"Average sentences per response: {avg_sentences:.2f}")

    # Step 3: Generate completions for all partial prompts
    print("Step 3: Generating completions for all partial prompts...")

    tokens_to_sample = max_new_tokens 
    if num_samples == 1:
        tokens_to_sample = 1
        print(f"--------------------------------")
        print()
        print(f"WARNING: If num_samples is 1, we set max_new_tokens to 1 to make it fast.")
        print(f"The assumption is that you only need base behavior probabilities after resampling from the prompt.")
        print(f"--------------------------------")


    partial_sampling_params = vllm.SamplingParams(
        n=num_samples,
        temperature=temperature,
        max_tokens=tokens_to_sample,
        seed=seed,
    )

    partial_outputs = model.generate(
        all_partial_prompts, partial_sampling_params, lora_request=lora_request
    )

    # Free model memory
    delete_vllm_model_and_free_memory(model)

    # Step 4: Compute base behavior probabilities (after freeing vllm memory,
    # so datasets that load their own models for detection have enough GPU memory)
    print("Step 4: Computing base behavior probabilities...")
    base_behavior_probabilities = []
    for example, responses_info in zip(dataset, base_responses_per_example):
        response_parts = [extract_response_part(info["response"]) for info in responses_info]
        behaviors = dataset.detect_behavior_batched(
            [example] * len(response_parts), response_parts
        )
        behaviors = [int(b) if b is not None else b for b in behaviors]
        base_probability = compute_behavior_probability(behaviors)
        base_behavior_probabilities.append(base_probability)

    # Step 5: Globally batch behavior detection, then assemble results
    print("Step 5: Processing results...")

    # Split outputs back into per-(example, response) groups
    per_response_outputs = list(
        variable_length_batched(partial_outputs, partial_prompt_counts)
    )

    # Further split into per-example groups
    per_example_outputs = list(
        variable_length_batched(per_response_outputs, example_response_counts)
    )

    # --- Collect ALL items that need behavior detection into flat lists ---
    # Instead of calling detect_behavior_batched once per sentence (thousands of
    # small calls), we collect everything and make a single batched call.
    all_detection_examples = []
    all_detection_responses = []
    detection_counts = []  # num_samples per (example, response, sentence)

    for example, example_outputs, responses_info in tqdm(
        zip(dataset, per_example_outputs, base_responses_per_example),
        total=len(dataset),
        desc="Collecting detection items",
    ):
        for response_outputs, response_info in zip(example_outputs, responses_info):
            sentences = response_info["sentences"]
            response_start_index = response_info["response_start_index"]
            for i, partial_output in enumerate(response_outputs):
                partial_response = reconstruct_partial_response(
                    sentences, i, response_start_index
                )
                completions = [output.text for output in partial_output.outputs]
                full_responses = [partial_response + c for c in completions]
                response_parts = [extract_response_part(fr) for fr in full_responses]

                all_detection_examples.extend([example] * len(response_parts))
                all_detection_responses.extend(response_parts)
                detection_counts.append(len(response_parts))

    print(f"Running behavior detection on {len(all_detection_responses)} items in a single batch...")
    all_behaviors = dataset.detect_behavior_batched(
        all_detection_examples, all_detection_responses
    )

    # Split flat behavior results back into per-sentence groups
    behaviors_per_sentence = list(
        variable_length_batched(all_behaviors, detection_counts)
    )

    # --- Assemble results using pre-computed behaviors ---
    outputs_to_save = []
    sentence_idx = 0
    for example, base_probability, responses_info, input_prompt in tqdm(
        zip(
            dataset,
            base_behavior_probabilities,
            base_responses_per_example,
            input_prompts,
        ),
        total=len(dataset),
        desc="Assembling results",
    ):
        example_output = {
            "input_prompt": input_prompt,
            "base_behavior_probability": base_probability,
            "responses": [],
        }

        for response_info in responses_info:
            sentences = response_info["sentences"]
            response_start_index = response_info["response_start_index"]

            probabilities = [base_probability]
            partial_prompts_info = [{"partial_response": "", "probability": base_probability}]

            for i in range(len(sentences)):
                behaviors = behaviors_per_sentence[sentence_idx]
                sentence_idx += 1

                probability = compute_behavior_probability(list(behaviors))
                probabilities.append(probability)

                partial_response = reconstruct_partial_response(
                    sentences, i, response_start_index
                )
                partial_prompts_info.append(
                    {"partial_response": partial_response, "probability": probability}
                )

            formatted_response = format_response_with_probabilities(sentences, probabilities)

            response_data = {
                "response": response_info["response"],
                "sentences": sentences,
                "per_sentence_probabilities": probabilities,
                "response_with_probabilities": formatted_response,
                "partial_prompts": partial_prompts_info,
                "response_start_index": response_start_index,
            }
            example_output["responses"].append(response_data)

        outputs_to_save.append(example_output)
    
    certain_examples = [(prob == 1.0) or (prob == 0.0) for prob in base_behavior_probabilities]
    uncertain_examples = [not certain for certain in certain_examples]
    num_certain_examples = sum(certain_examples)
    num_uncertain_examples = sum(uncertain_examples)


    # Save results
    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "subset": subset,
        "num_base_responses": num_base_responses,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "temperature": temperature,
        "disable_reasoning": disable_reasoning,
        "base_behavior_probabilities": base_behavior_probabilities,
        "fraction_certain_examples": num_certain_examples / len(base_behavior_probabilities),
        "fraction_uncertain_examples": num_uncertain_examples / len(base_behavior_probabilities),
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {results_file}")

    outputs_file = results_file.replace("results.json", "outputs.json")
    with open(outputs_file, "w") as f:
        json.dump(outputs_to_save, f, indent=4)
    print(f"Saved outputs to {outputs_file}")

    # Print summary statistics
    all_base_probs = base_behavior_probabilities
    print(f"\nSummary:")
    print(f"  Number of examples: {len(outputs_to_save)}")
    print(
        f"  Base behavior probability: mean={np.mean(all_base_probs):.3f}, std={np.std(all_base_probs):.3f}"
    )

    total_responses = sum(len(ex["responses"]) for ex in outputs_to_save)
    total_sentences = sum(
        len(r["sentences"]) for ex in outputs_to_save for r in ex["responses"]
    )
    print(f"  Total responses processed: {total_responses}")
    print(f"  Total sentences analyzed: {total_sentences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-sentence behavior probabilities for model responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="Subset of the dataset to use."
    )
    parser.add_argument(
        "--seed", type=int, default=43, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--num_base_responses",
        type=int,
        default=10,
        help="Number of base responses to generate for each prompt.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate for each partial prompt.",
    )
    parser.add_argument(
        "--disable_reasoning",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Disable reasoning by the model.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type to use for model inference.",
    )
    parser.add_argument(
        "--quantize",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use quantization for the model.",
    )

    args = parser.parse_args()

    short_model_name = args.model_name.split("/")[-1]
    output_dir = f"results/per_sentence_probabilities/{short_model_name}/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"n{args.subset if args.subset else 'full'}_nbase{args.num_base_responses}_nsamp{args.num_samples}_l{args.max_new_tokens}_s{args.seed}_t{args.temperature}{'_no_reasoning' if args.disable_reasoning else ''}_results.json"
    output_file = os.path.join(output_dir, filename)

    run_per_sentence_probabilities(
        args.model_name,
        args.dtype,
        args.dataset,
        args.subset,
        output_file,
        args.num_base_responses,
        args.num_samples,
        args.max_new_tokens,
        args.seed,
        args.temperature,
        args.disable_reasoning,
        args.quantize,
    )
