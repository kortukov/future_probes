import argparse
import json
import os
import nnsight
import torch
from tqdm import tqdm
import gc

from src.interp.probing import ProbabilisticProbe
from src.custom_datasets import get_dataset
from src.model_utils import get_model_and_tokenizer
from src.decoding_strategies import future_probe_controlled_generation_fast
from src.reasoning_model_utils import extract_response_part, extract_thinking_content_from_response, THINK_END_TAG, GPT_OSS_THINK_END_TAG, GEMMA4_THINK_END_TAG, MINISTRAL_THINK_END_TAG


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def generate_responses_with_future_probe_controlled_generation(
    model, wrapped_model, tokenizer, dataset, example, input_prompt, total_max_new_tokens, temperature, num_samples, probe, probing_layer, negative, responses, verbose, candidate_tokens, num_candidates, early_exit, sampling,
    defer_behavior_detection=False, think_end_tag=THINK_END_TAG,
) -> tuple[list[dict], list[bool | None]]:

    if responses is None:
        print(f"WARNING: responses is None, setting it to num_samples: {num_samples}")
        responses = num_samples
    
    if num_candidates is None:
        print(f"WARNING: num_candidates is None, setting it to num_samples: {num_samples}")
        num_candidates = num_samples


    response_dicts = []
    behaviors = []
    for _ in tqdm(range(responses), total=responses):

        with torch.no_grad():
            steered_response = future_probe_controlled_generation_fast(
                model=model,
                wrapped_model=wrapped_model,
                tokenizer=tokenizer,
                device=DEVICE,
                input_prompt=input_prompt,
                total_max_new_tokens=total_max_new_tokens,
                temperature=temperature,
                num_candidates=num_candidates,
                probe=probe,
                probing_layer=probing_layer,
                negative=negative,
                verbose=verbose,
                candidate_tokens=candidate_tokens,
                early_exit=early_exit,
                sampling=sampling,
                think_end_tag=think_end_tag,
            )

        answer_content = extract_response_part(steered_response)
        thinking_content = extract_thinking_content_from_response(steered_response)

        if defer_behavior_detection:
            behavior = None
        else:
            behavior = dataset.detect_behavior(example, answer_content)
            print(f"Behavior: {behavior}")

        response_dict = {
            "response": steered_response,
            "behavior": behavior,
            "thinking_content": thinking_content,
            "answer_content": answer_content,
        }
        response_dicts.append(response_dict)
        behaviors.append(behavior)

    return response_dicts, behaviors


def get_future_probe_controlled_generation_results(probe_path, results_file, probing_layer, negative, responses, verbose, candidate_tokens, num_candidates, early_exit, sampling, dataset_subset):
    with open(results_file, "r") as f:
        results = json.load(f)

    outputs_file = results_file.replace("results.json", "outputs.json")
    with open(outputs_file, "r") as f:
        outputs = json.load(f)

    probe = ProbabilisticProbe.load(probe_path)
    probe = probe.to(DEVICE)
    probe.eval()


    output_dir = results_file.replace("behavioral_stability", "future_probe_controlled_generation")
    output_dir = output_dir.replace("_results.json", "")
    os.makedirs(output_dir, exist_ok=True)

    model_name = results["model_name"]
    dataset_name = results["dataset_name"]
    subset = results["subset"]
    seed = results["seed"]
    temperature = results["temperature"]
    max_new_tokens = results["max_new_tokens"]
    num_samples = results["num_samples"]

    if "gpt-oss" in model_name:
        think_end_tag = GPT_OSS_THINK_END_TAG
    elif "gemma-4" in model_name:
        think_end_tag = GEMMA4_THINK_END_TAG
    elif "ministral" in model_name.lower():
        think_end_tag = MINISTRAL_THINK_END_TAG
    else:
        think_end_tag = THINK_END_TAG

    print(f"Loading model and tokenizer: {model_name}")
    model, tokenizer = get_model_and_tokenizer(
        model_name=model_name, device=DEVICE, half_precision=True, multi_gpu=True
    )
    tokenizer.pad_token = tokenizer.eos_token


    print(f"Wrapping model with nnsight: {model_name}")
    wrapped_model = nnsight.LanguageModel(model, tokenizer=tokenizer, dispatch=True)

    dataset = get_dataset(dataset_name, subset, seed=seed)
    defer_behavior = dataset.requires_model_for_detection
    if defer_behavior:
        print("Behavior detection will be deferred until after all generations are complete.")

    augmented_outputs = []
    total_none_count = 0
    total_response_count = 0

    def save_partial_results(augmented_outputs_list, total_none, total_responses, output_dir, pos_vs_neg_suffix, layer_suffix, num_candidates_suffix, early_exit_suffix, sampling_suffix, dataset_subset_suffix):
        """Save partial results every 10 samples."""
        average_steered_behaviors = [output.get("average_steered_behavior", 0.0) for output in augmented_outputs_list]
        total_avg_behavior = sum(average_steered_behaviors) / len(average_steered_behaviors) if average_steered_behaviors else 0.0

        partial_results = {
            "probe_path": probe_path,
            "early_exit": early_exit,
            "num_candidates": num_candidates,
            "responses": responses,
            "negative": negative,
            "candidate_tokens": candidate_tokens,
            "probing_layer": probing_layer,
            "original_results": results,
            "average_steered_behaviors": average_steered_behaviors,
            "total_avg_behavior": total_avg_behavior,
            "none_behavior_count": total_none,
            "total_response_count": total_responses,
            "none_behavior_fraction": total_none / total_responses if total_responses > 0 else 0.0,
            "samples_processed": len(augmented_outputs_list),
        }

        partial_results_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}{num_candidates_suffix}{early_exit_suffix}{sampling_suffix}{dataset_subset_suffix}_partial_results.json")
        with open(partial_results_path, "w") as f:
            json.dump(partial_results, f, indent=4)

        partial_outputs_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}{num_candidates_suffix}{early_exit_suffix}{sampling_suffix}{dataset_subset_suffix}_partial_outputs.json")
        with open(partial_outputs_path, "w") as f:
            json.dump(augmented_outputs_list, f, indent=4)

        print(f"Saved partial results after {len(augmented_outputs_list)} samples")

    for idx, (example, output) in enumerate(tqdm(zip(dataset, outputs), total=len(dataset))):

        if dataset_subset is not None and idx > dataset_subset:
            print(f"Reached dataset subset limit of {dataset_subset} examples. Stopping generation.")
            break

        input_prompt = output["input_prompt"]
        
        steered_responses, steered_behaviors = generate_responses_with_future_probe_controlled_generation(
            model=model,
            wrapped_model=wrapped_model,
            tokenizer=tokenizer,
            dataset=dataset,
            example=example,
            input_prompt=input_prompt,
            total_max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_samples=num_samples,
            probe=probe,
            probing_layer=probing_layer,
            negative=negative,
            responses=responses,
            verbose=verbose,
            candidate_tokens=candidate_tokens,
            num_candidates=num_candidates,
            early_exit=early_exit,
            sampling=sampling,
            defer_behavior_detection=defer_behavior,
            think_end_tag=think_end_tag,
            )

        output["steered_responses"] = steered_responses

        if not defer_behavior:
            not_none_behaviors = [b for b in steered_behaviors if b is not None]
            total_none_count += len(steered_behaviors) - len(not_none_behaviors)
            total_response_count += len(steered_behaviors)
            average_steered_behavior = sum(not_none_behaviors) / (len(not_none_behaviors) or 1)
            output["average_steered_behavior"] = average_steered_behavior

        augmented_outputs.append(output)

        # Save partial results every 2 samples
        if (idx + 1) % 2 == 0:
            pos_vs_neg_suffix = "negative" if negative else "positive"
            num_candidates_suffix = f"_nc{num_candidates}" if num_candidates is not None else ""
            early_exit_suffix = "_early_exit" if early_exit else ""
            sampling_suffix = "_sampling" if sampling else ""
            layer_suffix = f"layer{probing_layer}"
            dataset_subset_suffix = f"_subset{dataset_subset}" if dataset_subset is not None else ""
            save_partial_results(
                augmented_outputs,
                total_none_count,
                total_response_count,
                output_dir,
                pos_vs_neg_suffix,
                layer_suffix,
                num_candidates_suffix,
                early_exit_suffix,
                sampling_suffix,
                dataset_subset_suffix,
            )

    if defer_behavior:
        print("Freeing generation model memory before behavior detection...")
        del wrapped_model
        del model
        del tokenizer
        del probe
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
        "probe_path": probe_path,
        "early_exit": early_exit,
        "num_candidates": num_candidates,
        "responses": responses,
        "negative": negative,
        "candidate_tokens": candidate_tokens,
        "probing_layer": probing_layer,
        "original_results": results,
        "average_steered_behaviors": average_steered_behaviors,
        "total_avg_behavior": total_avg_behavior,
        "none_behavior_count": total_none_count,
        "total_response_count": total_response_count,
        "none_behavior_fraction": total_none_count / total_response_count if total_response_count > 0 else 0.0,
    }
    pos_vs_neg_suffix = "negative" if negative else "positive"
    num_candidates_suffix = f"_nc{num_candidates}" if num_candidates is not None else ""
    early_exit_suffix = "_early_exit" if early_exit else ""
    sampling_suffix = "_sampling" if sampling else ""
    layer_suffix = f"layer{probing_layer}"
    dataset_subset_suffix = f"_subset{dataset_subset}" if dataset_subset is not None else ""
    augmented_results_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}{num_candidates_suffix}{early_exit_suffix}{sampling_suffix}{dataset_subset_suffix}_results.json")
    print(f"Saving final results to {augmented_results_path}")
    with open(augmented_results_path, "w") as f:
        json.dump(augmented_results, f, indent=4)

    augmented_outputs_path = augmented_results_path.replace("results.json", "outputs.json")
    print(f"Saving final outputs to {augmented_outputs_path}")
    with open(augmented_outputs_path, "w") as f:
        json.dump(augmented_outputs, f, indent=4)

    partial_results_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}{num_candidates_suffix}{early_exit_suffix}{sampling_suffix}{dataset_subset_suffix}_partial_results.json")
    partial_outputs_path = os.path.join(output_dir, f"{pos_vs_neg_suffix}_{layer_suffix}{num_candidates_suffix}{early_exit_suffix}{sampling_suffix}{dataset_subset_suffix}_partial_outputs.json")
    for partial_path in (partial_results_path, partial_outputs_path):
        if os.path.exists(partial_path):
            os.remove(partial_path)
            print(f"Deleted partial checkpoint {partial_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steer the model using the probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--probe", type=str, required=True, help="Path to the probe."
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the results of 6_behavioral_stability.py file.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=25,
        help="Layer to use for probing.",
    )
    parser.add_argument(
        "--negative",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Steer towards negative behavior (behavior False)",
    )
    parser.add_argument(
        "--responses",
        type=int,
        default=None,
        help="Number of responses to generate. If not provided, will generate num_samples responses per prompt.",
    )
    parser.add_argument(
        "--verbose",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Print verbose output. Use --verbose=True or --verbose=False",
    )
    parser.add_argument(
        "--candidate_tokens",
        type=int,
        default=32,
        help="Number of candidate tokens to generate. If not provided, will generate 32 tokens per candidate.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=10,
        help="Number of candidates to generate. If not provided, will generate num_samples candidates.",
    )
    parser.add_argument(
        "--early_exit",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Early exit if the desired behavior is achieved. Use --early_exit=True or --early_exit=False",
    )
    parser.add_argument(
        "--sampling",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Sample the candidate sentences instead of choosing the best one.",
    )
    parser.add_argument(
        "--dataset_subset",
        type=int,
        default=None,
        help="Only use a subset of the dataset.",
    )
    args = parser.parse_args()

    print(f"Arguments: {args.__dict__}")

    get_future_probe_controlled_generation_results(
        probe_path=args.probe,
        results_file=args.results,
        probing_layer=args.layer,
        negative=args.negative,
        responses=args.responses,
        verbose=args.verbose,
        candidate_tokens=args.candidate_tokens,
        num_candidates=args.num_candidates,
        early_exit=args.early_exit,
        sampling=args.sampling,
        dataset_subset=args.dataset_subset,
    )
