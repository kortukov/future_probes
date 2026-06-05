"""
Evaluate a probabilistic probe or blackbox monitor on per-sentence data.

This script evaluates how well a trained predictor (either a probabilistic probe
operating on activations, or a blackbox text classifier) predicts behavior
probabilities at different positions in a response.

Inputs (probe mode: --probe + --activations):
- Probe from 21_train_probabilistic_probe.py
- Activations from 20_gather_per_sentence_activations.py
- Results/outputs from 19_get_per_sentence_probabilities.py

Inputs (blackbox mode: --blackbox_monitor):
- Trained blackbox monitor model directory from 28_train_blackbox_monitor.py
- Results/outputs from 19_get_per_sentence_probabilities.py

Outputs:
- Enriched outputs file in the same format as 19_get_per_sentence_probabilities.py
  with additional prediction information (for use with visualization)
- Evaluation metrics summary

Evaluation metrics (always computed):
1. Overall MAE - mean absolute error across all positions
2. Overall binarized accuracy - fraction where (pred > 0.5) matches (gt > 0.5)
3. MAE at position 0 - error when predicting from just the prompt (empty partial response)
4. Direction accuracy - for each consecutive pair of positions, whether the
   predicted change direction (up/down/equal) matches the ground truth direction

Per-position metrics (--per_position flag):
5. MAE by sentence position (0, 1, 2, ...) with support counts
6. Direction accuracy broken down by sentence position

Behavior-relative metrics (--relative_to_behavior flag):
7. MAE after behavior happened (first position where detect_behavior returns True)
8. MAE at relative offsets from behavior position (-1, -2, -3, ...)
   where -1 is the position right before behavior, -2 is two positions before, etc.

Reasoning parts metrics (--reasoning_parts flag):
9. MAE and binarized accuracy for the 1st, 2nd, and last thirds of the
   reasoning chain (positions corresponding to thinking sentences)
10. MAE and binarized accuracy for the final response (positions after reasoning)
    For non-reasoning models, all non-position-0 positions are "final response".

Behavior position is determined by loading the corresponding dataset and using
its detect_behavior function on progressive partial responses.
"""

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from src.custom_datasets import get_dataset
from src.interp.probing import ProbabilisticProbe, format_response_with_probabilities


# ============================================================================ #
# Data Structures
# ============================================================================ #


@dataclass
class ResponsePrediction:
    """Predictions and ground truth for one response."""

    sample_idx: int
    response_idx: int
    predictions: np.ndarray  # (num_positions,) predicted probabilities
    ground_truth: np.ndarray  # (num_positions,) actual probabilities
    num_sentences: int  # number of sentences in response
    behavior_position: int | None = None  # set externally via dataset.detect_behavior
    response_start_index: int = 0  # index of first non-thinking sentence (0 for non-reasoning)

    @property
    def num_positions(self) -> int:
        """Number of positions (including position 0 = empty partial response)."""
        return len(self.predictions)


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""

    # Overall metrics
    overall_mae: float
    overall_bce: float
    overall_binarized_accuracy: float
    n_total: int

    # Standard errors for overall metrics (std / sqrt(n))
    overall_mae_se: float = 0.0
    overall_bce_se: float = 0.0
    overall_binarized_accuracy_se: float = 0.0

    # Position 0 metrics (empty partial response)
    position_0_mae: float = 0.0
    n_position_0: int = 0
    position_0_mae_se: float = 0.0

    # Per-position metrics (only populated when --per_position is set)
    per_position_mae: dict[int, float] = field(default_factory=dict)
    per_position_support: dict[int, int] = field(default_factory=dict)

    # Direction accuracy: fraction of responses where the predicted direction of
    # change (up/down/equal) matches the ground truth direction.
    # overall_direction_accuracy is always computed; per-position breakdown only
    # when --per_position is set. Only defined for positions >= 1.
    per_position_direction_accuracy: dict[int, float] = field(default_factory=dict)
    per_position_direction_support: dict[int, int] = field(default_factory=dict)
    overall_direction_accuracy: float | None = None
    overall_direction_accuracy_se: float | None = None
    n_direction_total: int = 0

    # Before behavior metrics (all positions strictly before behavior_position)
    # Only populated when --relative_to_behavior is set
    before_behavior_mae: float | None = None
    before_behavior_mae_se: float | None = None
    n_before_behavior: int = 0

    # After behavior metrics (positions from behavior_position onwards)
    after_behavior_mae: float | None = None
    after_behavior_mae_se: float | None = None
    n_after_behavior: int = 0

    # Relative-to-behavior metrics (offsets: -1, -2, -3, ...)
    # Offset -1 = one position before behavior, -2 = two positions before, etc.
    relative_to_behavior_mae: dict[int, float] = field(default_factory=dict)
    relative_to_behavior_support: dict[int, int] = field(default_factory=dict)

    # Reasoning parts metrics (only populated when --reasoning_parts is set)
    # Dict keys: 0 = first third, 1 = second third, 2 = last third
    reasoning_third_mae: dict[int, float] = field(default_factory=dict)
    reasoning_third_mae_se: dict[int, float] = field(default_factory=dict)
    reasoning_third_binarized_accuracy: dict[int, float] = field(default_factory=dict)
    reasoning_third_binarized_accuracy_se: dict[int, float] = field(default_factory=dict)
    reasoning_third_support: dict[int, int] = field(default_factory=dict)

    # Final response metrics (positions after reasoning chain)
    final_response_mae: float | None = None
    final_response_mae_se: float | None = None
    final_response_binarized_accuracy: float | None = None
    final_response_binarized_accuracy_se: float | None = None
    n_final_response: int = 0


# ============================================================================ #
# Evaluation Functions
# ============================================================================ #


def load_data(
    activations_path: Path, results_path: Path
) -> tuple[list[dict], list[dict], dict, dict]:
    """
    Load activations, outputs, and results data.

    Returns:
        activation_data: list of sample dicts with activations
        outputs: list of output dicts with ground truth
        results: dict with metadata from script 19
        config: config dict from activations file
    """
    # Load activations
    act_data = torch.load(activations_path, map_location="cpu", weights_only=False)
    activation_data = act_data["activation_data"]
    config = act_data["config"]

    # Load results metadata
    with open(results_path, "r") as f:
        results = json.load(f)

    # Load outputs (ground truth probabilities)
    outputs_path = results_path.parent / results_path.name.replace(
        "_results.json", "_outputs.json"
    )
    with open(outputs_path, "r") as f:
        outputs = json.load(f)

    return activation_data, outputs, results, config


def load_outputs_and_results(results_path: Path) -> tuple[list[dict], dict]:
    """Load outputs and results data (no activations needed).

    Used by blackbox monitor evaluation which doesn't require activation files.
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    outputs_path = results_path.parent / results_path.name.replace(
        "_results.json", "_outputs.json"
    )
    with open(outputs_path, "r") as f:
        outputs = json.load(f)

    return outputs, results


class BlackboxPredictor:
    """Wraps a HuggingFace sequence classification model for probability prediction.

    Loads a model trained by 28_train_blackbox_monitor.py and predicts behavior
    probabilities from (prompt + partial_response) text.
    """

    def __init__(self, model_path: str | Path, max_length: int = 8192, device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_path = str(model_path)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=1
        )
        if self.tokenizer.pad_token_id is not None and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict behavior probabilities for a list of texts."""
        probs = []
        for text in texts:
            inputs = self.tokenizer(
                text, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
                prob = torch.sigmoid(logits).cpu().item()
            probs.append(prob)
        return np.array(probs)


def make_probe_predict_fn(probe, activation_data: list[dict]):
    """Create a predict_fn that uses a probe on activations."""

    def predict_fn(sample_idx, response_idx, input_prompt, out_response):
        act_sample = activation_data[sample_idx]
        assert act_sample["sample_idx"] == sample_idx
        act_response = act_sample["responses"][response_idx]
        activations = act_response["activations"]
        if isinstance(activations, torch.Tensor):
            activations = activations.float()
        else:
            activations = torch.tensor(activations, dtype=torch.float32)
        return probe.predict_proba(activations)

    return predict_fn


def make_blackbox_predict_fn(predictor: BlackboxPredictor):
    """Create a predict_fn that uses a blackbox text classifier."""

    def predict_fn(sample_idx, response_idx, input_prompt, out_response):
        texts = [
            input_prompt + pp["partial_response"]
            for pp in out_response["partial_prompts"]
        ]
        return predictor.predict_proba(texts)

    return predict_fn


def compute_behavior_positions(
    dataset,
    outputs: list[dict],
) -> list[list[int | None]]:
    """
    Compute behavior_position for each response using the dataset's detect_behavior.

    For each response, builds progressive partial responses from sentences and
    calls detect_behavior to find the first position where behavior is detected.

    Position 0 = empty response, Position k = after sentences[0:k].

    Args:
        dataset: Dataset object with detect_behavior method
        outputs: List of output dicts from script 19

    Returns:
        Nested list [sample_idx][response_idx] -> behavior_position or None
    """
    all_behavior_positions = []

    for example, output_sample in zip(dataset, outputs):
        sample_positions = []
        for response in output_sample["responses"]:
            sentences = response["sentences"]

            # Build partial responses for each position
            # Position 0: empty string
            # Position k (k >= 1): " ".join(sentences[:k])
            partial_responses = [""]
            for k in range(1, len(sentences) + 1):
                partial_responses.append(" ".join(sentences[:k]))

            # Check behavior for each partial response
            behavior_position = None
            behaviors = dataset.detect_behavior_batched(
                [example] * len(partial_responses), partial_responses
            )

            for pos, behavior_detected in enumerate(behaviors):
                if behavior_detected:
                    behavior_position = pos
                    break

            sample_positions.append(behavior_position)

        all_behavior_positions.append(sample_positions)

    return all_behavior_positions


def compute_predictions_and_enrich_outputs(
    predict_fn,
    outputs: list[dict],
    behavior_positions: list[list[int | None]],
) -> tuple[list[ResponsePrediction], list[dict]]:
    """
    Compute predictions for all responses and create enriched outputs.

    Args:
        predict_fn: callable(sample_idx, response_idx, input_prompt, out_response) -> np.ndarray
            Returns predicted probabilities for each position in the response.
        outputs: list of output dicts with ground truth
        behavior_positions: nested list [sample_idx][response_idx] -> behavior_position

    Returns:
        predictions: List of ResponsePrediction objects
        enriched_outputs: Deep copy of outputs with prediction info added
    """
    predictions = []
    enriched_outputs = copy.deepcopy(outputs)

    for sample_idx, (out_sample, enriched_sample) in tqdm(enumerate(
        zip(outputs, enriched_outputs)
    ), total=len(outputs), desc="Computing predictions"):
        input_prompt = out_sample["input_prompt"]
        sample_predictions = []
        sample_errors = []

        for response_idx, (out_response, enriched_response) in enumerate(
            zip(out_sample["responses"], enriched_sample["responses"])
        ):
            pred_probs = predict_fn(
                sample_idx=sample_idx,
                response_idx=response_idx,
                input_prompt=input_prompt,
                out_response=out_response,
            )

            gt_probs = np.array(
                [pp["probability"] for pp in out_response["partial_prompts"]]
            )

            assert len(pred_probs) == len(gt_probs), (
                f"Length mismatch: pred={len(pred_probs)}, gt={len(gt_probs)}"
            )

            errors = np.abs(pred_probs - gt_probs)

            enriched_response["predicted_probabilities"] = pred_probs.tolist()
            enriched_response["response_with_predicted_probabilities"] = (
                format_response_with_probabilities(
                    out_response["sentences"], pred_probs.tolist()
                )
            )
            enriched_response["per_position_error"] = errors.tolist()
            enriched_response["response_mae"] = float(errors.mean())

            for partial_prompt, pred_prob, error in zip(
                enriched_response["partial_prompts"], pred_probs, errors
            ):
                partial_prompt["predicted_probability"] = float(pred_prob)
                partial_prompt["prediction_error"] = float(error)

            sample_predictions.extend(pred_probs.tolist())
            sample_errors.extend(errors.tolist())

            bpos = behavior_positions[sample_idx][response_idx]
            enriched_response["behavior_position"] = bpos

            predictions.append(
                ResponsePrediction(
                    sample_idx=sample_idx,
                    response_idx=response_idx,
                    predictions=pred_probs,
                    ground_truth=gt_probs,
                    num_sentences=len(out_response["sentences"]),
                    behavior_position=bpos,
                    response_start_index=out_response.get("response_start_index", 0),
                )
            )

        enriched_sample["predicted_base_behavior_probability"] = float(
            np.mean(
                [r["predicted_probabilities"][0] for r in enriched_sample["responses"]]
            )
        )
        enriched_sample["sample_mae"] = float(np.mean(sample_errors))

    return predictions, enriched_outputs


def compute_metrics(
    predictions: list[ResponsePrediction],
    per_position: bool = False,
    relative_to_behavior: bool = False,
    reasoning_parts: bool = False,
) -> EvaluationResults:
    """
    Compute all evaluation metrics from predictions.

    Args:
        predictions: list of ResponsePrediction objects
        per_position: whether to compute per-position MAE and direction accuracy
        relative_to_behavior: whether to compute behavior-relative metrics

    Returns:
        EvaluationResults with all metrics
    """
    # Collect all predictions and ground truth
    all_preds = []
    all_gt = []

    # Position 0 (empty partial response)
    pos0_preds = []
    pos0_gt = []

    # Per-position
    per_pos_preds = defaultdict(list)
    per_pos_gt = defaultdict(list)

    # Direction accuracy: per-position, does predicted delta match GT direction?
    # Keyed by position (>= 1), value is list of bools (correct or not)
    per_pos_direction_correct = defaultdict(list)

    # Before behavior (all positions strictly before behavior_position)
    before_behavior_preds = []
    before_behavior_gt = []

    # After behavior
    after_behavior_preds = []
    after_behavior_gt = []

    # Relative to behavior (offsets: -1, -2, -3, ...)
    relative_preds = defaultdict(list)
    relative_gt = defaultdict(list)

    # Reasoning parts (thirds of reasoning chain + final response)
    reasoning_third_preds: dict[int, list] = {0: [], 1: [], 2: []}
    reasoning_third_gt: dict[int, list] = {0: [], 1: [], 2: []}
    final_response_preds_list: list = []
    final_response_gt_list: list = []

    for pred in predictions:
        # Overall
        all_preds.extend(pred.predictions)
        all_gt.extend(pred.ground_truth)

        # Position 0
        pos0_preds.append(pred.predictions[0])
        pos0_gt.append(pred.ground_truth[0])

        # Direction accuracy for positions >= 1 (always computed)
        for pos in range(1, pred.num_positions):
            gt_diff = pred.ground_truth[pos] - pred.ground_truth[pos - 1]
            pred_diff = pred.predictions[pos] - pred.predictions[pos - 1]
            gt_dir = np.sign(gt_diff)
            pred_dir = np.sign(pred_diff)
            per_pos_direction_correct[pos].append(gt_dir == pred_dir)

        if per_position:
            for pos in range(pred.num_positions):
                per_pos_preds[pos].append(pred.predictions[pos])
                per_pos_gt[pos].append(pred.ground_truth[pos])

        if relative_to_behavior:
            bpos = pred.behavior_position
            if bpos is not None:
                if bpos > 0:
                    before_behavior_preds.extend(pred.predictions[:bpos])
                    before_behavior_gt.extend(pred.ground_truth[:bpos])

                after_behavior_preds.extend(pred.predictions[bpos:])
                after_behavior_gt.extend(pred.ground_truth[bpos:])

                for offset in range(1, bpos + 1):
                    pos = bpos - offset
                    relative_preds[-offset].append(pred.predictions[pos])
                    relative_gt[-offset].append(pred.ground_truth[pos])

        if reasoning_parts:
            R = pred.response_start_index
            if R > 0:
                reasoning_positions = list(range(1, R + 1))
                thirds = np.array_split(reasoning_positions, 3)
                for third_idx, third_positions in enumerate(thirds):
                    for pos in third_positions:
                        reasoning_third_preds[third_idx].append(pred.predictions[pos])
                        reasoning_third_gt[third_idx].append(pred.ground_truth[pos])
            for pos in range(R + 1, pred.num_positions):
                final_response_preds_list.append(pred.predictions[pos])
                final_response_gt_list.append(pred.ground_truth[pos])

    all_preds = np.array(all_preds)
    all_gt = np.array(all_gt)
    pos0_preds = np.array(pos0_preds)
    pos0_gt = np.array(pos0_gt)

    def _se(arr: np.ndarray) -> float:
        """Standard error of the mean (ddof=1)."""
        n = len(arr)
        return float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    # Compute overall metrics
    overall_abs_errors = np.abs(all_preds - all_gt)
    overall_mae = float(overall_abs_errors.mean())
    overall_mae_se = _se(overall_abs_errors)

    # Binarized accuracy: threshold at 0.5 for both predictions and labels
    thresholded_preds = (all_preds > 0.5).astype(float)
    binarized_labels = (all_gt > 0.5).astype(float)
    binarized_correct = (thresholded_preds == binarized_labels).astype(float)
    overall_binarized_accuracy = float(binarized_correct.mean())
    overall_binarized_accuracy_se = _se(binarized_correct)

    eps = 1e-7
    all_preds_clipped = np.clip(all_preds, eps, 1 - eps)
    per_element_bce = -(
        all_gt * np.log(all_preds_clipped)
        + (1 - all_gt) * np.log(1 - all_preds_clipped)
    )
    overall_bce = float(per_element_bce.mean())
    overall_bce_se = _se(per_element_bce)

    # Position 0 metrics
    pos0_abs_errors = np.abs(pos0_preds - pos0_gt)
    pos0_mae = float(pos0_abs_errors.mean())
    pos0_mae_se = _se(pos0_abs_errors)

    # Per-position metrics
    per_position_mae = {}
    per_position_support = {}
    per_position_direction_accuracy = {}
    per_position_direction_support = {}
    overall_direction_accuracy = None
    n_direction_total = 0

    if per_position:
        for pos in sorted(per_pos_preds.keys()):
            preds_pos = np.array(per_pos_preds[pos])
            gt_pos = np.array(per_pos_gt[pos])
            per_position_mae[pos] = float(np.abs(preds_pos - gt_pos).mean())
            per_position_support[pos] = len(preds_pos)

    # Direction accuracy (always computed)
    all_direction_correct = []
    for pos in sorted(per_pos_direction_correct.keys()):
        correct = per_pos_direction_correct[pos]
        if per_position:
            per_position_direction_accuracy[pos] = float(np.mean(correct))
            per_position_direction_support[pos] = len(correct)
        all_direction_correct.extend(correct)

    overall_direction_accuracy_se = None
    if all_direction_correct:
        dir_arr = np.array(all_direction_correct, dtype=float)
        overall_direction_accuracy = float(dir_arr.mean())
        overall_direction_accuracy_se = _se(dir_arr)
        n_direction_total = len(all_direction_correct)

    # Before behavior metrics
    before_behavior_mae = None
    before_behavior_mae_se = None
    n_before_behavior = len(before_behavior_preds)
    if relative_to_behavior and n_before_behavior > 0:
        before_behavior_preds = np.array(before_behavior_preds)
        before_behavior_gt = np.array(before_behavior_gt)
        bb_errs = np.abs(before_behavior_preds - before_behavior_gt)
        before_behavior_mae = float(bb_errs.mean())
        before_behavior_mae_se = _se(bb_errs)

    # After behavior metrics
    after_behavior_mae = None
    after_behavior_mae_se = None
    n_after_behavior = len(after_behavior_preds)
    if relative_to_behavior and n_after_behavior > 0:
        after_behavior_preds = np.array(after_behavior_preds)
        after_behavior_gt = np.array(after_behavior_gt)
        ab_errs = np.abs(after_behavior_preds - after_behavior_gt)
        after_behavior_mae = float(ab_errs.mean())
        after_behavior_mae_se = _se(ab_errs)

    # Relative-to-behavior metrics
    relative_to_behavior_mae = {}
    relative_to_behavior_support = {}

    if relative_to_behavior:
        for offset in sorted(relative_preds.keys()):
            preds_off = np.array(relative_preds[offset])
            gt_off = np.array(relative_gt[offset])
            relative_to_behavior_mae[offset] = float(np.abs(preds_off - gt_off).mean())
            relative_to_behavior_support[offset] = len(preds_off)

    # Reasoning parts metrics
    reasoning_third_mae_result: dict[int, float] = {}
    reasoning_third_mae_se_result: dict[int, float] = {}
    reasoning_third_binarized_accuracy_result: dict[int, float] = {}
    reasoning_third_binarized_accuracy_se_result: dict[int, float] = {}
    reasoning_third_support_result: dict[int, int] = {}
    final_response_mae_val = None
    final_response_mae_se_val = None
    final_response_binarized_accuracy_val = None
    final_response_binarized_accuracy_se_val = None
    n_final_response = 0

    if reasoning_parts:
        for third_idx in range(3):
            preds_t = reasoning_third_preds[third_idx]
            gt_t = reasoning_third_gt[third_idx]
            if preds_t:
                preds_arr = np.array(preds_t)
                gt_arr = np.array(gt_t)
                t_errs = np.abs(preds_arr - gt_arr)
                t_correct = ((preds_arr > 0.5) == (gt_arr > 0.5)).astype(float)
                reasoning_third_mae_result[third_idx] = float(t_errs.mean())
                reasoning_third_mae_se_result[third_idx] = _se(t_errs)
                reasoning_third_binarized_accuracy_result[third_idx] = float(t_correct.mean())
                reasoning_third_binarized_accuracy_se_result[third_idx] = _se(t_correct)
                reasoning_third_support_result[third_idx] = len(preds_t)

        n_final_response = len(final_response_preds_list)
        if n_final_response > 0:
            fr_preds = np.array(final_response_preds_list)
            fr_gt = np.array(final_response_gt_list)
            fr_errs = np.abs(fr_preds - fr_gt)
            fr_correct = ((fr_preds > 0.5) == (fr_gt > 0.5)).astype(float)
            final_response_mae_val = float(fr_errs.mean())
            final_response_mae_se_val = _se(fr_errs)
            final_response_binarized_accuracy_val = float(fr_correct.mean())
            final_response_binarized_accuracy_se_val = _se(fr_correct)

    return EvaluationResults(
        overall_mae=overall_mae,
        overall_mae_se=overall_mae_se,
        overall_bce=overall_bce,
        overall_bce_se=overall_bce_se,
        overall_binarized_accuracy=overall_binarized_accuracy,
        overall_binarized_accuracy_se=overall_binarized_accuracy_se,
        n_total=len(all_preds),
        position_0_mae=pos0_mae,
        position_0_mae_se=pos0_mae_se,
        n_position_0=len(pos0_preds),
        per_position_mae=per_position_mae,
        per_position_support=per_position_support,
        per_position_direction_accuracy=per_position_direction_accuracy,
        per_position_direction_support=per_position_direction_support,
        overall_direction_accuracy=overall_direction_accuracy,
        overall_direction_accuracy_se=overall_direction_accuracy_se,
        n_direction_total=n_direction_total,
        before_behavior_mae=before_behavior_mae,
        before_behavior_mae_se=before_behavior_mae_se,
        n_before_behavior=n_before_behavior,
        after_behavior_mae=after_behavior_mae,
        after_behavior_mae_se=after_behavior_mae_se,
        n_after_behavior=n_after_behavior,
        relative_to_behavior_mae=relative_to_behavior_mae,
        relative_to_behavior_support=relative_to_behavior_support,
        reasoning_third_mae=reasoning_third_mae_result,
        reasoning_third_mae_se=reasoning_third_mae_se_result,
        reasoning_third_binarized_accuracy=reasoning_third_binarized_accuracy_result,
        reasoning_third_binarized_accuracy_se=reasoning_third_binarized_accuracy_se_result,
        reasoning_third_support=reasoning_third_support_result,
        final_response_mae=final_response_mae_val,
        final_response_mae_se=final_response_mae_se_val,
        final_response_binarized_accuracy=final_response_binarized_accuracy_val,
        final_response_binarized_accuracy_se=final_response_binarized_accuracy_se_val,
        n_final_response=n_final_response,
    )


def results_to_dict(results: EvaluationResults) -> dict:
    """Convert EvaluationResults to a JSON-serializable dict.

    Only includes per-position and behavior sections when they were computed.
    """
    d: dict = {
        "overall": {
            "mae": results.overall_mae,
            "mae_se": results.overall_mae_se,
            "bce": results.overall_bce,
            "bce_se": results.overall_bce_se,
            "binarized_accuracy": results.overall_binarized_accuracy,
            "binarized_accuracy_se": results.overall_binarized_accuracy_se,
            "n": results.n_total,
        },
        "position_0": {
            "mae": results.position_0_mae,
            "mae_se": results.position_0_mae_se,
            "n": results.n_position_0,
        },
    }

    if results.per_position_mae:
        d["per_position"] = {
            "mae": {str(k): v for k, v in results.per_position_mae.items()},
            "support": {str(k): v for k, v in results.per_position_support.items()},
        }
    direction_accuracy_entry: dict = {
        "overall": results.overall_direction_accuracy,
        "overall_se": results.overall_direction_accuracy_se,
        "n": results.n_direction_total,
    }
    if results.per_position_direction_accuracy:
        direction_accuracy_entry["per_position"] = {
            str(k): v
            for k, v in results.per_position_direction_accuracy.items()
        }
        direction_accuracy_entry["support"] = {
            str(k): v
            for k, v in results.per_position_direction_support.items()
        }
    d["direction_accuracy"] = direction_accuracy_entry
    if results.before_behavior_mae is not None or results.after_behavior_mae is not None:
        d["before_behavior"] = {
            "mae": results.before_behavior_mae,
            "mae_se": results.before_behavior_mae_se,
            "n": results.n_before_behavior,
        }
        d["after_behavior"] = {
            "mae": results.after_behavior_mae,
            "mae_se": results.after_behavior_mae_se,
            "n": results.n_after_behavior,
        }
    if results.relative_to_behavior_mae:
        d["relative_to_behavior"] = {
            "mae": {str(k): v for k, v in results.relative_to_behavior_mae.items()},
            "support": {
                str(k): v for k, v in results.relative_to_behavior_support.items()
            },
        }

    third_names = {0: "first_third", 1: "second_third", 2: "last_third"}
    if results.reasoning_third_mae or results.final_response_mae is not None:
        reasoning_parts: dict = {}
        for tidx, name in third_names.items():
            if tidx in results.reasoning_third_mae:
                reasoning_parts[name] = {
                    "mae": results.reasoning_third_mae[tidx],
                    "mae_se": results.reasoning_third_mae_se.get(tidx),
                    "binarized_accuracy": results.reasoning_third_binarized_accuracy[tidx],
                    "binarized_accuracy_se": results.reasoning_third_binarized_accuracy_se.get(tidx),
                    "n": results.reasoning_third_support[tidx],
                }
        reasoning_parts["final_response"] = {
            "mae": results.final_response_mae,
            "mae_se": results.final_response_mae_se,
            "binarized_accuracy": results.final_response_binarized_accuracy,
            "binarized_accuracy_se": results.final_response_binarized_accuracy_se,
            "n": results.n_final_response,
        }
        d["reasoning_parts"] = reasoning_parts

    return d


def print_results(
    results: EvaluationResults,
    certain_results: EvaluationResults | None = None,
    uncertain_results: EvaluationResults | None = None,
) -> None:
    """Print evaluation results as consolidated tables with All/Certain/Uncertain columns."""
    splits: list[tuple[str, EvaluationResults]] = [("All", results)]
    if certain_results is not None:
        splits.append(("Certain", certain_results))
    if uncertain_results is not None:
        splits.append(("Uncertain", uncertain_results))

    LBL = 20  # label column width
    COL = 12  # data column width

    def _header():
        return f"  {'':>{LBL}}" + "".join(f"  {n:>{COL}}" for n, _ in splits)

    def _sep():
        return f"  {'-' * LBL}" + "".join(f"  {'-' * COL}" for _ in splits)

    def _row(label: str, getter, fmt: str = ".4f"):
        cells = []
        for _, r in splits:
            v = getter(r)
            if v is None:
                cells.append(f"  {'n/a':>{COL}}")
            elif isinstance(v, int):
                cells.append(f"  {v:>{COL}}")
            else:
                cells.append(f"  {v:>{COL}{fmt}}")
        print(f"  {label:>{LBL}}" + "".join(cells))

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # ---- Summary table ----
    print(f"\n{'=' * 30} SUMMARY {'=' * 31}")
    print(_header())
    print(_sep())
    _row("N (positions)", lambda r: r.n_total)
    _row("N (responses)", lambda r: r.n_position_0)
    _row("MAE", lambda r: r.overall_mae)
    _row("BCE", lambda r: r.overall_bce)
    _row("Binarized Acc", lambda r: r.overall_binarized_accuracy)
    _row("Pos 0 MAE", lambda r: r.position_0_mae)
    _row("Direction Acc", lambda r: r.overall_direction_accuracy)
    _row("N (dir)", lambda r: r.n_direction_total)

    has_behavior = results.before_behavior_mae is not None or results.after_behavior_mae is not None
    if has_behavior:
        _row("Before Beh MAE", lambda r: r.before_behavior_mae)
        _row("N (before beh)", lambda r: r.n_before_behavior if r.n_before_behavior > 0 else None)
        _row("After Beh MAE", lambda r: r.after_behavior_mae)
        _row("N (after beh)", lambda r: r.n_after_behavior if r.n_after_behavior > 0 else None)

    # ---- Reasoning parts table ----
    has_reasoning = bool(results.reasoning_third_mae) or results.final_response_mae is not None
    if has_reasoning:
        third_names = {0: "1st Third", 1: "2nd Third", 2: "Last Third"}
        print(f"\n{'=' * 22} REASONING PARTS METRICS {'=' * 23}")
        print(_header())
        print(_sep())
        for tidx in range(3):
            _row(f"Reas {third_names[tidx]} MAE", lambda r, t=tidx: r.reasoning_third_mae.get(t))
            _row(f"Reas {third_names[tidx]} Acc", lambda r, t=tidx: r.reasoning_third_binarized_accuracy.get(t))
            _row(f"N (reas {third_names[tidx].lower()})", lambda r, t=tidx: r.reasoning_third_support.get(t))
        _row("Final Resp MAE", lambda r: r.final_response_mae)
        _row("Final Resp Acc", lambda r: r.final_response_binarized_accuracy)
        _row("N (final resp)", lambda r: r.n_final_response if r.n_final_response > 0 else None)

    # ---- Per-position table ----
    if results.per_position_mae:
        print(f"\n{'=' * 24} PER-POSITION METRICS {'=' * 25}")
        # Sub-header per split: MAE  DirAcc  N
        sub_hdr = f"  {'Pos':>4}"
        sub_sep = f"  {'-' * 4}"
        for name, _ in splits:
            sub_hdr += f"  |  {'MAE':>8}{'DirAcc':>8}{'N':>8}"
            sub_sep += f"  |  {'-' * 8}{'-' * 8}{'-' * 8}"
        # Print split names above sub-headers
        name_line = f"  {'':>4}"
        for name, _ in splits:
            name_line += f"  |  {name:^24}"
        print(name_line)
        print(sub_hdr)
        print(sub_sep)

        all_positions = sorted(results.per_position_mae.keys())
        for pos in all_positions:
            line = f"  {pos:>4}"
            for _, r in splits:
                mae = r.per_position_mae.get(pos)
                dir_acc = r.per_position_direction_accuracy.get(pos)
                support = r.per_position_support.get(pos)
                mae_s = f"{mae:>8.4f}" if mae is not None else f"{'n/a':>8}"
                dir_s = f"{dir_acc:>8.4f}" if dir_acc is not None else f"{'n/a':>8}"
                n_s = f"{support:>8}" if support is not None else f"{'n/a':>8}"
                line += f"  |  {mae_s}{dir_s}{n_s}"
            print(line)

    # ---- Relative-to-behavior offsets table ----
    if results.relative_to_behavior_mae:
        print(f"\n{'=' * 19} RELATIVE TO BEHAVIOR (OFFSETS) {'=' * 20}")
        off_hdr = f"  {'Offset':>7}"
        off_sep = f"  {'-' * 7}"
        name_line = f"  {'':>7}"
        for name, _ in splits:
            name_line += f"  |  {name:^16}"
            off_hdr += f"  |  {'MAE':>8}{'N':>8}"
            off_sep += f"  |  {'-' * 8}{'-' * 8}"
        print(name_line)
        print(off_hdr)
        print(off_sep)
        all_offsets = sorted(results.relative_to_behavior_mae.keys(), reverse=True)
        for offset in all_offsets:
            line = f"  {offset:>7}"
            for _, r in splits:
                mae = r.relative_to_behavior_mae.get(offset)
                support = r.relative_to_behavior_support.get(offset)
                mae_s = f"{mae:>8.4f}" if mae is not None else f"{'n/a':>8}"
                n_s = f"{support:>8}" if support is not None else f"{'n/a':>8}"
                line += f"  |  {mae_s}{n_s}"
            print(line)

    print("\n" + "=" * 70)


def compute_baseline_metrics(
    predictions: list[ResponsePrediction],
    per_position: bool = False,
    relative_to_behavior: bool = False,
    reasoning_parts: bool = False,
) -> tuple[dict[str, EvaluationResults], float]:
    """Compute metrics for random (constant 0.5) and dataset-mean baselines."""
    all_gt = np.concatenate([p.ground_truth for p in predictions])
    dataset_mean_value = float(all_gt.mean())

    baseline_results = {}
    for name, value in [("random", 0.5), ("dataset_mean", dataset_mean_value)]:
        baseline_preds = [
            ResponsePrediction(
                sample_idx=p.sample_idx,
                response_idx=p.response_idx,
                predictions=np.full_like(p.predictions, value),
                ground_truth=p.ground_truth,
                num_sentences=p.num_sentences,
                behavior_position=p.behavior_position,
                response_start_index=p.response_start_index,
            )
            for p in predictions
        ]
        baseline_results[name] = compute_metrics(
            baseline_preds,
            per_position=per_position,
            relative_to_behavior=relative_to_behavior,
            reasoning_parts=reasoning_parts,
        )

    return baseline_results, dataset_mean_value


def print_baseline_comparison(
    predictor_name: str,
    eval_results: EvaluationResults,
    baseline_results: dict[str, EvaluationResults],
    dataset_mean_value: float,
) -> None:
    """Print a compact comparison table: predictor vs random and dataset-mean baselines."""
    print(f"\n{'=' * 26} BASELINE COMPARISON {'=' * 24}")

    LBL = 20
    COL = 16

    columns = [
        (predictor_name, eval_results),
        ("Random (0.5)", baseline_results["random"]),
        (f"Mean ({dataset_mean_value:.3f})", baseline_results["dataset_mean"]),
    ]

    header = f"  {'':>{LBL}}" + "".join(f"  {n:>{COL}}" for n, _ in columns)
    sep = f"  {'-' * LBL}" + "".join(f"  {'-' * COL}" for _ in columns)

    def _row(label, getter, fmt=".4f"):
        cells = []
        for _, r in columns:
            v = getter(r)
            if v is None:
                cells.append(f"  {'n/a':>{COL}}")
            elif isinstance(v, int):
                cells.append(f"  {v:>{COL}}")
            else:
                cells.append(f"  {v:>{COL}{fmt}}")
        print(f"  {label:>{LBL}}" + "".join(cells))

    print(header)
    print(sep)
    _row("MAE", lambda r: r.overall_mae)
    _row("BCE", lambda r: r.overall_bce)
    _row("Binarized Acc", lambda r: r.overall_binarized_accuracy)
    _row("Pos 0 MAE", lambda r: r.position_0_mae)
    _row("Direction Acc", lambda r: r.overall_direction_accuracy)
    print("=" * 70)


# ============================================================================ #
# Main
# ============================================================================ #


def get_probe_name(probe_path: Path) -> str:
    """Extract probe name from probe path.

    Examples:
        linear_probe.pt -> linear
        mlp_256_128_probe.pt -> mlp_256_128
        my_custom_probe.pt -> my_custom
    """
    stem = probe_path.stem
    # Remove common suffixes
    for suffix in ["_probe", "_model"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a probabilistic probe or blackbox monitor on per-sentence data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Predictor type (mutually exclusive)
    pred_group = parser.add_mutually_exclusive_group(required=True)
    pred_group.add_argument(
        "--probe",
        type=Path,
        help="Path to the probe file from 21_train_probabilistic_probe.py",
    )
    pred_group.add_argument(
        "--blackbox_monitor",
        type=Path,
        help="Path to a trained blackbox monitor model directory from 28_train_blackbox_monitor.py",
    )

    # Probe-specific args
    parser.add_argument(
        "--activations",
        type=Path,
        default=None,
        help="Path to activations file from 20_gather_per_sentence_activations.py "
        "(required with --probe)",
    )

    # Blackbox-specific args
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Max tokenized sequence length for blackbox monitor (only used with --blackbox_monitor)",
    )

    # Common args
    parser.add_argument(
        "--results_file",
        type=Path,
        required=True,
        help="Path to results file from 19_get_per_sentence_probabilities.py",
    )
    parser.add_argument(
        "--predictor_name",
        type=str,
        default=None,
        help="Name for the predictor (defaults to probe/model filename). "
        "Use this to distinguish predictors trained on different data.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--per_position",
        action="store_true",
        help="Compute and show per-position metrics (MAE and direction accuracy by sentence position)",
    )
    parser.add_argument(
        "--relative_to_behavior",
        action="store_true",
        help="Compute and show metrics relative to behavior position "
        "(requires dataset with detect_behavior)",
    )
    parser.add_argument(
        "--reasoning_parts",
        action="store_true",
        help="Compute MAE and binarized accuracy for the 1st, 2nd, and last thirds "
        "of the reasoning chain, and for the final response after reasoning. "
        "Only meaningful for reasoning models (response_start_index > 0).",
    )
    args = parser.parse_args()

    is_probe_mode = args.probe is not None

    # Validate probe-mode requirements
    if is_probe_mode and args.activations is None:
        parser.error("--activations is required when using --probe")

    # Determine predictor name
    if is_probe_mode:
        predictor_name = args.predictor_name or get_probe_name(args.probe)
    else:
        if args.predictor_name:
            predictor_name = args.predictor_name
        elif args.blackbox_monitor.name == "model":
            predictor_name = f"blackbox_{args.blackbox_monitor.parent.name}"
        else:
            predictor_name = f"blackbox_{args.blackbox_monitor.name}"

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif is_probe_mode:
        output_dir = args.probe.parent / f"{args.probe.stem}_predictions"
    else:
        monitor_dir = (
            args.blackbox_monitor.parent
            if args.blackbox_monitor.name == "model"
            else args.blackbox_monitor
        )
        output_dir = monitor_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_stem = args.results_file.stem
    if results_stem.endswith("_results"):
        results_stem = results_stem[: -len("_results")]

    print(f"Output directory: {output_dir}")
    print(f"Predictor name: {predictor_name}")
    print(f"Results stem: {results_stem}")

    # Load predictor and data
    if is_probe_mode:
        print(f"\nLoading probe from {args.probe}")
        probe = ProbabilisticProbe.load(args.probe)
        print(f"  Input size: {probe.input_size}")
        print(f"  Hidden dims: {probe.hidden_dims}")

        print(f"\nLoading activations from {args.activations}")
        print(f"Loading results from {args.results_file}")
        activation_data, outputs, original_results, config = load_data(
            args.activations, args.results_file
        )
        print(f"  Loaded {len(activation_data)} samples")
        print(f"  Config: {config}")

        predict_fn = make_probe_predict_fn(probe, activation_data)
    else:
        print(f"\nLoading blackbox monitor from {args.blackbox_monitor}")
        blackbox = BlackboxPredictor(
            args.blackbox_monitor, max_length=args.max_length
        )
        print(f"  Model: {args.blackbox_monitor}")
        print(f"  Max length: {args.max_length}")
        print(f"  Device: {blackbox.device}")

        print(f"\nLoading results from {args.results_file}")
        outputs, original_results = load_outputs_and_results(args.results_file)
        print(f"  Loaded {len(outputs)} samples")
        config = None

        predict_fn = make_blackbox_predict_fn(blackbox)

    # Load dataset for behavior detection
    print("\nLoading dataset for behavior detection...")
    dataset_name = original_results["dataset_name"]
    dataset_subset = original_results.get("subset")
    dataset_seed = original_results.get("seed")
    dataset = get_dataset(dataset_name, subset=dataset_subset, seed=dataset_seed)
    print(f"  Dataset: {dataset_name}, subset={dataset_subset}, seed={dataset_seed}")
    print(f"  Loaded {len(dataset)} examples")

    assert len(dataset) == len(outputs), (
        f"Dataset has {len(dataset)} examples but outputs has {len(outputs)} samples. "
        f"Check that dataset_name={dataset_name}, subset={dataset_subset}, seed={dataset_seed} "
        f"match the parameters used to generate the results file."
    )

    # Compute behavior positions (only if needed)
    if args.relative_to_behavior:
        print("\nComputing behavior positions using detect_behavior...")
        behavior_positions = compute_behavior_positions(dataset, outputs)
        n_with_behavior = sum(
            1 for sample in behavior_positions for pos in sample if pos is not None
        )
        total_responses = sum(len(sample) for sample in behavior_positions)
        print(f"  Behavior detected in {n_with_behavior}/{total_responses} responses")
    else:
        behavior_positions = [
            [None] * len(sample["responses"]) for sample in outputs
        ]

    # Compute predictions and enrich outputs
    print("\nComputing predictions...")
    predictions, enriched_outputs = compute_predictions_and_enrich_outputs(
        predict_fn, outputs, behavior_positions
    )
    print(f"  Computed predictions for {len(predictions)} responses")

    # Split predictions into certain (base prob == 0 or 1) and uncertain
    certain_preds = [p for p in predictions if p.ground_truth[0] in (0.0, 1.0)]
    uncertain_preds = [p for p in predictions if p.ground_truth[0] not in (0.0, 1.0)]

    # Compute metrics
    print("\nComputing metrics...")
    metric_kwargs = dict(
        per_position=args.per_position,
        relative_to_behavior=args.relative_to_behavior,
        reasoning_parts=args.reasoning_parts,
    )
    eval_results = compute_metrics(predictions, **metric_kwargs)
    certain_results = compute_metrics(certain_preds, **metric_kwargs) if certain_preds else None
    uncertain_results = compute_metrics(uncertain_preds, **metric_kwargs) if uncertain_preds else None

    # Compute baseline metrics (random=0.5, dataset mean)
    baseline_results, dataset_mean_value = compute_baseline_metrics(predictions, **metric_kwargs)

    # Print results
    print_results(eval_results, certain_results, uncertain_results)
    print_baseline_comparison(predictor_name, eval_results, baseline_results, dataset_mean_value)

    # ========================================================================
    # Save enriched outputs (same format as script 19, with predictions added)
    # ========================================================================
    enriched_outputs_path = output_dir / f"{results_stem}_outputs.json"
    with open(enriched_outputs_path, "w") as f:
        json.dump(enriched_outputs, f, indent=2)
    print(f"\nEnriched outputs saved to {enriched_outputs_path}")
    print("  (Compatible with per_sentence_probabilities.html visualization)")

    # ========================================================================
    # Save enriched results (same format as script 19 results, with metrics)
    # ========================================================================
    enriched_results = copy.deepcopy(original_results)
    evaluation_dict = results_to_dict(eval_results)
    if certain_results is not None:
        evaluation_dict["certain"] = results_to_dict(certain_results)
    if uncertain_results is not None:
        evaluation_dict["uncertain"] = results_to_dict(uncertain_results)
    enriched_results["evaluation"] = evaluation_dict
    enriched_results["evaluation"]["baselines"] = {
        name: results_to_dict(br) for name, br in baseline_results.items()
    }
    enriched_results["evaluation"]["baselines"]["dataset_mean_value"] = dataset_mean_value

    if is_probe_mode:
        enriched_results["predictor_config"] = {
            "type": "probe",
            "probe_path": str(args.probe),
            "predictor_name": predictor_name,
            "activations_path": str(args.activations),
            "input_size": probe.input_size,
            "hidden_dims": probe.hidden_dims,
        }
    else:
        enriched_results["predictor_config"] = {
            "type": "blackbox_monitor",
            "model_path": str(args.blackbox_monitor),
            "predictor_name": predictor_name,
            "max_length": args.max_length,
        }

    enriched_results["predicted_base_behavior_probabilities"] = [
        sample["predicted_base_behavior_probability"] for sample in enriched_outputs
    ]

    enriched_results_path = output_dir / f"{results_stem}_results.json"
    with open(enriched_results_path, "w") as f:
        json.dump(enriched_results, f, indent=2)
    print(f"Enriched results saved to {enriched_results_path}")

    # ========================================================================
    # Save evaluation metrics summary (for quick reference)
    # ========================================================================
    metrics_summary = results_to_dict(eval_results)
    if certain_results is not None:
        metrics_summary["certain"] = results_to_dict(certain_results)
    if uncertain_results is not None:
        metrics_summary["uncertain"] = results_to_dict(uncertain_results)

    metrics_summary["baselines"] = {
        name: results_to_dict(br) for name, br in baseline_results.items()
    }
    metrics_summary["baselines"]["dataset_mean_value"] = dataset_mean_value

    if is_probe_mode:
        metrics_summary["config"] = {
            "type": "probe",
            "probe_path": str(args.probe),
            "predictor_name": predictor_name,
            "activations_path": str(args.activations),
            "results_file": str(args.results_file),
            "activation_config": config,
        }
    else:
        metrics_summary["config"] = {
            "type": "blackbox_monitor",
            "model_path": str(args.blackbox_monitor),
            "predictor_name": predictor_name,
            "max_length": args.max_length,
            "results_file": str(args.results_file),
        }

    metrics_path = output_dir / f"{results_stem}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics summary saved to {metrics_path}")

    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("=" * 70)
    print(f"Directory: {output_dir}")
    print(f"  {enriched_outputs_path.name}")
    print("    -> Enriched outputs compatible with visualization")
    print(f"  {enriched_results_path.name}")
    print("    -> Results with evaluation metrics and original metadata")
    print(f"  {metrics_path.name}")
    print("    -> Evaluation metrics summary")


if __name__ == "__main__":
    main()
