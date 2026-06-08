import argparse
import json
import os
import torch
import vllm
from tqdm import tqdm
import re
import openai_harmony
import pdb

from src.reasoning_model_utils import extract_response_part, extract_thinking_content_from_response

# Fix for VLLM CUDA multiprocessing issue
# Set VLLM's multiprocessing method to spawn before any VLLM imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


from src.model_utils import (
    get_vllm_model_and_tokenizer,
    delete_vllm_model_and_free_memory,
)
from src.custom_datasets import get_dataset
from plotting_scripts.behavior_stability import plot_behavioral_stability

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def run_behavioral_stability(
    model_name,
    dataset_name,
    subset,
    results_file,
    num_samples,
    max_new_tokens,
    decoder,
    seed,
    sampling_seed,
    temperature,
    multi_gpu,
    disable_reasoning,
    ensure_thinking,
    model_path,
):
    dataset = get_dataset(
        dataset_name,
        subset,
        icl_examples=None,
        seed=seed,
        disable_reasoning=disable_reasoning,
    )

    model, tokenizer, lora_request = get_vllm_model_and_tokenizer(
        model_name, multi_gpu=multi_gpu, max_model_len=max_new_tokens + 2000
    )

    sampling_params = vllm.SamplingParams(
        n=num_samples,
        temperature=0.0 if decoder == "greedy" else temperature,
        max_tokens=max_new_tokens,
        seed=sampling_seed,
        skip_special_tokens=False,
    )

    # only needed for gpt-oss models
    encoding = openai_harmony.load_harmony_encoding(openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS)

    # Gather all input prompts
    input_prompts = []
    for idx, example in enumerate(dataset):
        input_ids, input_prompt = dataset.get_model_input_from_an_example(
            example, tokenizer
        )

        input_prompts.append(input_prompt)

    # Generate outputs
    model_outputs = model.generate(
        input_prompts, sampling_params, lora_request=lora_request
    )

    # After this step we don't need the model anymore, delete and clean it.
    delete_vllm_model_and_free_memory(model)

    # Print cuda allocated memory in GB
    print(
        f"Cuda allocated memory: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB"
    )

    average_behaviors = []
    first_behaviors = []
    worst_behaviors = []

    outputs_to_save = []
    for example_output, example in tqdm(
        zip(model_outputs, dataset), total=len(dataset)
    ):
        input_ids, input_prompt = dataset.get_model_input_from_an_example(
            example, tokenizer
        )

        responses = [response.text for response in example_output.outputs]

        if "gpt-oss" in model_name:
            # GPT-oss uses harmony format
            thinking_contents = []
            answer_contents = []
            responses_tokens = [response.token_ids for response in example_output.outputs]
            for one_response_tokens in responses_tokens:
                entries = encoding.parse_messages_from_completion_tokens(one_response_tokens, openai_harmony.Role.ASSISTANT)

                analysis_messages = [entry for entry in entries if entry.channel == "analysis"]
                final_messages = [entry for entry in entries if entry.channel == "final"]

                if len(analysis_messages) > 0 :
                    analysis_msg = analysis_messages[0]
                    thinking_content = analysis_msg.content[0].text
                else:
                    thinking_content = ""
                
                if len(final_messages) > 0:
                    final_msg = final_messages[0]
                    answer_content = final_msg.content[0].text
                else:
                    answer_content = ""

                thinking_contents.append(thinking_content)
                answer_contents.append(answer_content)
        elif "gemma-4" in model_name:
            # Gemma 4 uses <|channel>thought...<channel|> for thinking.
            # skip_special_tokens=False is set above so markers appear in response.text.
            thinking_contents = [
                extract_thinking_content_from_response(response)
                for response in responses
            ]
            answer_contents = [
                extract_response_part(response)
                for response in responses
            ]
        elif "ministral" in model_name.lower():
            thinking_contents = [
                extract_thinking_content_from_response(response)
                for response in responses
            ]
            answer_contents = [
                extract_response_part(response)
                for response in responses
            ]
        else:
            thinking_contents = [
                response.split("</think>")[0] if "</think>" in response else ""
                for response in responses
            ]
            answer_contents = [
                response.split("</think>")[1] if "</think>" in response else response
                for response in responses
            ]

        if ensure_thinking:
            no_thinking_examples = [
                i
                for i, thinking_content in enumerate(thinking_contents)
                if thinking_content == ""
            ]
            all_have_thinking = len(no_thinking_examples) == 0
            if not all_have_thinking:
                print("WARNING:")
                print(
                    f"Thinking content is empty for {len(no_thinking_examples)} examples. Ensure ensure_thinking is False if you want to ignore this."
                )
                print("--------------------------------")

        # We check for the behavior only in the answer content. For the non-reasoning models its just the full answer. The thinking is private.
        examples = [example] * len(answer_contents)
        behaviors = dataset.detect_behavior_batched(examples, answer_contents)
        behaviors = [
            int(behavior) if behavior is not None else behavior
            for behavior in behaviors
        ]

        int_behaviors = [b for b in behaviors if b is not None]

        average_behavior = sum(int_behaviors) / (len(int_behaviors) or 1)
        average_behaviors.append(average_behavior)

        first_behavior = behaviors[0]
        first_behaviors.append(first_behavior)

        worst_behavior = max(int_behaviors) if int_behaviors else None
        worst_behaviors.append(worst_behavior)

        outputs_to_save.append(
            {
                "input_prompt": input_prompt,
                "responses": [
                    {
                        "response": response,
                        "behavior": behavior,
                        "thinking_content": thinking_content,
                        "answer_content": answer_content,
                    }
                    for response, behavior, thinking_content, answer_content in zip(
                        responses, behaviors, thinking_contents, answer_contents
                    )
                ],
                "average_behavior": average_behavior,
                "worst_behavior": worst_behavior,
                "first_behavior": first_behavior,
            }
        )

    print(f"Average behaviors: {average_behaviors}")
    total_avg_behavior = sum(average_behaviors) / (len(average_behaviors) or 1)
    print(f"Total average behavior: {total_avg_behavior}")

    all_behaviors_flat = [r["behavior"] for ex in outputs_to_save for r in ex["responses"]]
    none_behavior_count = sum(1 for b in all_behaviors_flat if b is None)
    total_response_count = len(all_behaviors_flat)
    none_behavior_fraction = none_behavior_count / total_response_count if total_response_count > 0 else 0.0
    print(f"None behavior fraction: {none_behavior_fraction:.4f} ({none_behavior_count}/{total_response_count})")

    results = {
        "model_name": model_name,
        "disable_reasoning": disable_reasoning,
        "dataset_name": dataset_name,
        "subset": subset,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "decoder": decoder,
        "seed": seed,
        "sampling_seed": sampling_seed,
        "temperature": temperature,
        "multi_gpu": multi_gpu,
        "average_behaviors": repr(average_behaviors),
        "worst_behaviors": repr(worst_behaviors),
        "first_behaviors": repr(first_behaviors),
        "total_avg_behavior": total_avg_behavior,
        "none_behavior_count": none_behavior_count,
        "total_response_count": total_response_count,
        "none_behavior_fraction": none_behavior_fraction,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {results_file}")

    outputs_file = results_file.replace("results.json", "outputs.json")
    with open(outputs_file, "w") as f:
        json.dump(outputs_to_save, f, indent=4)
    print(f"Saved outputs to {outputs_file}")

    plot_behavioral_stability(results_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run behavioral stability analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Behavioral stability args
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model or path to the model directory.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Name of the dataset."
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
        "--disable_reasoning",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Disable reasoning by the model.",
    )
    parser.add_argument(
        "--ensure_thinking",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Throw an error if the thinking content is empty.",
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=42,
        help="Random seed for sampling when generating the outputs.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate for each example in the dataset.",
    )
    parser.add_argument(
        "--multi_gpu",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use multiple GPUs for model inference. Use --multi_gpu=True or --multi_gpu=False",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        help="Path to the model weights or configuration for ablated/multiplied models."
        " If not provided, the base model is used.",
    )

    args = parser.parse_args()

    if args.disable_reasoning and args.ensure_thinking:
        raise ValueError(
            "It does not make sense to use --disable_reasoning and --ensure_thinking at the same time."
        )

    experiment_config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "subset": args.subset,
        "seed": args.seed,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "multi_gpu": args.multi_gpu,
    }

    if args.dataset is None:
        raise ValueError("Provide a dataset name with --dataset.")
    
    decoder = "greedy" if args.temperature == 0.0 else "gumbel"

    if decoder == "greedy" and args.num_samples > 1:
        print(
            "WARNING: Temperature == 0.0 cannot be used with multiple samples, setting num_samples to 1."
        )
        args.num_samples = 1

    short_model_name = args.model_name.split("/")
    # make sure the correct model name is extracted when a path is given as model name
    if len(short_model_name) <= 2:
        short_model_name = short_model_name[-1]
    else:
        short_model_name = short_model_name[1]

    # Create output directory and file name
    if args.model_path is None:  # in this case we use the base model
        output_dir = f"results/behavioral_stability/{short_model_name}/base_model/{args.dataset}"
    else:
        # extract topk, m, pos/neg from the model path
        match = re.search(
            r"topk_(\d+).*?_b_(True|False).*?_m_([\d.]+|None)", args.model_path
        )
        output_dir = f"results/behavioral_stability/{short_model_name}/modified_model/top_{match.group(1)}/m{match.group(3)}/{args.dataset}/{match.group(2)}_heads"

    os.makedirs(output_dir, exist_ok=True)
    filename = f"n{args.subset if args.subset else 'full'}_nsamp{args.num_samples}_l{args.max_new_tokens}_{decoder}_s{args.seed}_ss{args.sampling_seed}_t{args.temperature}{'_no_reasoning' if args.disable_reasoning else ''}_results.json"
    output_file = os.path.join(output_dir, filename)

    run_behavioral_stability(
        args.model_name,
        args.dataset,
        args.subset,
        output_file,
        args.num_samples,
        args.max_new_tokens,
        decoder,
        args.seed,
        args.sampling_seed,
        args.temperature,
        args.multi_gpu,
        args.disable_reasoning,
        args.ensure_thinking,
        args.model_path,
    )


