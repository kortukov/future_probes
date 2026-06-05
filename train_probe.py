"""
Train and evaluate a probabilistic probing classifier on per-sentence activations.

This script trains a logistic regression probe (linear + sigmoid) to predict
probability labels from hidden state activations. Uses BCE loss with soft targets.

Input: activations from 20_gather_per_sentence_activations.py
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.interp.probing import ProbabilisticProbe, AttentionProbe, GDMAttentionProbe


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================ #
# Data Loading and Splitting
# ============================================================================ #


@dataclass
class DataSplit:
    """Train/eval split with metadata."""

    X_train: torch.Tensor  # (N_train, hidden_size)
    y_train: torch.Tensor  # (N_train,)
    mask_train: torch.Tensor  # (N_train, max_seq_len) or None
    X_eval: torch.Tensor  # (N_eval, hidden_size)
    y_eval: torch.Tensor  # (N_eval,)
    mask_eval: torch.Tensor  # (N_eval, max_seq_len) or None
    config: dict
    n_train_samples: int  # number of data samples (prompts) in train
    n_eval_samples: int  # number of data samples (prompts) in eval
    hidden_size: int # dimension of the activations


def pad_and_stack_activations(activations_list: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Pad and stack a list of activations and masks.
    """
    max_seq_len = max([activation.shape[0] for activation in activations_list])

    padded_activations_list = []
    padded_masks_list = []
    for activation in activations_list:
        # activation has shape (seq_len, hidden_size)
        seq_len = activation.shape[0]
        pad_len = max_seq_len - seq_len
        act_pad = (0, 0, 0, pad_len) # Padding the second to last dimension on the right, which is seq len
        padded_activation = torch.nn.functional.pad(activation, act_pad, mode="constant", value=0) # (max_seq_len, hidden_size)

        padded_mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)], dim=0) # (max_seq_len,)
        padded_mask = padded_mask.bool()

        padded_activation = padded_activation.unsqueeze(0) # (1, max_seq_len, hidden_size)
        padded_mask = padded_mask.unsqueeze(0) # (1, max_seq_len,)

        padded_activations_list.append(padded_activation)
        padded_masks_list.append(padded_mask)
    
    return padded_activations_list, padded_masks_list


def load_and_split_data(
    path: Path,
    eval_fraction: float = 0.2,
    seed: int = 42,
    first_only: bool = False,
    last_only: bool = False,
    act_type: str = "per_sentence",
) -> DataSplit:
    """
    Load activations and split by data sample (prompt).

    The split is done at the sample level: all responses and sentences for a given
    prompt stay together in either train or eval. This prevents data leakage.
    """
    if first_only and last_only:
        raise ValueError("Cannot use both first_only and last_only at the same time.")

    data = torch.load(path, map_location="cpu", weights_only=False)
    activation_data = data["activation_data"]
    n_samples = len(activation_data)

    # Determine train/eval sample indices
    n_eval_samples = max(1, int(n_samples * eval_fraction))
    n_train_samples = n_samples - n_eval_samples

    rng = np.random.default_rng(seed)
    sample_indices = rng.permutation(n_samples)
    eval_sample_indices = set(sample_indices[:n_eval_samples])

    # Collect activations and labels by split
    train_acts, train_labels = [], []
    eval_acts, eval_labels = [], []

    for sample in activation_data:
        sample_idx = sample["sample_idx"]
        is_eval = sample_idx in eval_sample_indices

        if act_type == "last_prompt_token":
            acts = sample["activations"].unsqueeze(0)  # (1, hidden_size)
            hidden_size = acts.shape[1]
            labels = torch.tensor(sample["label"]).unsqueeze(0) # (1,)
            if is_eval:
                eval_acts.append(acts)
                eval_labels.append(labels)
            else:
                train_acts.append(acts)
                train_labels.append(labels)
        elif act_type == "full_prompt_sequence":
            acts = sample["activations"] # (seq_len, hidden_size)
            labels = torch.tensor(sample["label"]).unsqueeze(0) # (1,)
            hidden_size = acts.shape[1]
            if is_eval:
                eval_acts.append(acts)
                eval_labels.append(labels)
            else:
                train_acts.append(acts)
                train_labels.append(labels)
        else:
            for response_idx, response in enumerate(sample["responses"]):
                if first_only and response_idx > 0:
                    continue

                acts = response["activations"]  # (num_sentences, hidden_size)
                labels = response["labels"]  # (num_sentences,)

                hidden_size = acts.shape[1]

                if first_only:
                    acts = acts[0].unsqueeze(0) # (1, hidden_size)
                    labels = labels[0].unsqueeze(0) # (1,)
                
                if last_only:
                    acts = acts[-1].unsqueeze(0) # (1, hidden_size)
                    labels = labels[-1].unsqueeze(0) # (1,)

                if is_eval:
                    eval_acts.append(acts)
                    eval_labels.append(labels)
                else:
                    train_acts.append(acts)
                    train_labels.append(labels)
    
    if act_type == "full_prompt_sequence":
        train_acts, train_masks = pad_and_stack_activations(train_acts)
        eval_acts, eval_masks = pad_and_stack_activations(eval_acts)
        mask_train = torch.cat(train_masks, dim=0)
        mask_eval = torch.cat(eval_masks, dim=0)
    else:
        mask_train = None
        mask_eval = None


    return DataSplit(
        X_train=torch.cat(train_acts, dim=0).float(),
        y_train=torch.cat(train_labels, dim=0).float(),
        mask_train=mask_train,
        X_eval=torch.cat(eval_acts, dim=0).float(),
        y_eval=torch.cat(eval_labels, dim=0).float(),
        mask_eval=mask_eval,
        config=data["config"],
        n_train_samples=n_train_samples,
        n_eval_samples=n_eval_samples,
        hidden_size=hidden_size,
    )


# ============================================================================ #
# Training and Evaluation
# ============================================================================ #


def train_probe(
    probe_type: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    mask_train: torch.Tensor,
    hidden_size: int,
    hidden_dims: list[int] | None = None,
    attn_hidden_dims: list[int] | None = None,
    num_heads: int = 1,
    dropout: float = 0.0,
    position_bias: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 256,
    verbose: bool = True,
    device: torch.device = DEVICE,
) -> ProbabilisticProbe:
    """
    Train a probe using BCE loss with soft targets.

    Args:
        X_train: (N, hidden_size) activations
        y_train: (N,) probability labels in [0, 1]
        hidden_dims: list of hidden layer sizes for MLP, or None for linear probe
        lr: learning rate
        weight_decay: L2 regularization
        epochs: number of training epochs
        batch_size: mini-batch size
        verbose: print training progress

    Returns:
        Trained ProbabilisticProbe
    """
    if probe_type == "probabilistic":
        probe = ProbabilisticProbe(input_size=hidden_size, hidden_dims=hidden_dims).to(device)
    elif probe_type == "attention":
        probe = AttentionProbe(input_size=hidden_size, num_heads=num_heads, hidden_dims=hidden_dims, attn_hidden_dims=attn_hidden_dims, dropout=dropout, use_position_bias=position_bias).to(device)
    elif probe_type == "gdm_attention":
        probe = GDMAttentionProbe(input_size=hidden_size, hidden_dims=hidden_dims).to(device)
    else:
        raise ValueError(f"Invalid probe type: {probe_type}")

    if probe_type == "probabilistic":
        # Compute and set normalization parameters
        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0).clamp(min=1e-8)
        probe.set_normalization(mean, std, device)
    elif probe_type == "attention":
        # Compute per-feature mean/std across all valid (non-padded) tokens
        # X_train: (N, max_seq_len, hidden_size), mask_train: (N, max_seq_len)
        valid_mask = mask_train.unsqueeze(-1).expand_as(X_train)  # (N, max_seq_len, hidden_size)
        valid_tokens = X_train[valid_mask].view(-1, X_train.shape[-1])  # (total_valid_tokens, hidden_size)
        mean = valid_tokens.mean(dim=0)
        std = valid_tokens.std(dim=0).clamp(min=1e-8)
        probe.set_normalization(mean, std, device)

    # Setup training
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()  # more numerically stable

    if probe_type == "probabilistic":
        dataset = TensorDataset(X_train, y_train)
    elif probe_type == "attention" or probe_type == "gdm_attention":
        dataset = TensorDataset(X_train, y_train, mask_train)
    else:
        raise ValueError(f"Invalid probe type: {probe_type}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    probe.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()

            if probe_type == "probabilistic":
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = probe(X_batch)
            elif probe_type in {"attention", "gdm_attention"}:
                X_batch, y_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                logits = probe(X_batch, mask_batch)
            else:
                raise ValueError(f"Invalid probe type: {probe_type}")

            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(X_train)
            print(f"  Epoch {epoch + 1}/{epochs}: BCE loss = {avg_loss:.4f}")

    probe.eval()
    return probe


def evaluate_probe(
    probe: ProbabilisticProbe,
    probe_type: str,
    X: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    """
    Evaluate probe predictions.

    Returns dict with: bce, mse, mae
    """
    probe.eval()
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device) if mask is not None else None
        if probe_type == "probabilistic":
            logits = probe(X)
        elif probe_type in {"attention", "gdm_attention"}:
            logits = probe(X, mask)
        else:
            raise ValueError(f"Invalid probe type: {probe_type}")
        probs = torch.sigmoid(logits)

    bce = nn.functional.binary_cross_entropy_with_logits(logits, y).item()
    mse = ((probs - y) ** 2).mean().item()
    mae = (probs - y).abs().mean().item()
    thresholded_predictions = (probs > 0.5).float()
    binarized_labels = (y > 0.5).float()
    binarized_accuracy = (thresholded_predictions == binarized_labels).float().mean().item()

    return {
        "bce": bce,
        "mse": mse,
        "mae": mae,
        "binarized_accuracy": binarized_accuracy,
    }


# ============================================================================ #
# Main
# ============================================================================ #


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a probabilistic probing classifier on per-sentence activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to activations file from 20_gather_per_sentence_activations.py",
    )
    parser.add_argument(
        "--act_type",
        type=str,
        default="per_sentence",
        choices=["per_sentence","last_prompt_token", "full_prompt_sequence"],
        help="Type of probe to train.",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="probabilistic",
        choices=["probabilistic", "attention", "gdm_attention"],
        help="Type of probe to train.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to same as activations file)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization (weight decay)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default=None,
        help="MLP hidden layer dimensions (comma-separated, e.g., '256,128'). If not set, uses linear probe.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--attn_hidden_dims",
        type=str,
        default=None,
        help="Attention hidden layer dimensions (comma-separated, e.g., '256,128'). If not set, uses a linear query projection..",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability after each ReLU in attention probe MLPs.",
    )
    parser.add_argument(
        "--position_bias",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Add a learnable linear position bias to attention logits (attention probe only).",
    )
    parser.add_argument(
        "--first_only",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Only train the probe on the first activation and label of each sample only. Since they are same for each response, we also take only the first response.",
    )
    parser.add_argument(
        "--last_only",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Only train the probe on the last activation and label of each sample only. Unlike first_only, we take all responses for each sample.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for output files. Defaults to 'linear' or 'mlp_256_128' based on architecture.",
    )
    parser.add_argument(
        "--eval_fraction",
        type=float,
        default=0.2,
        help="Fraction of data samples for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if args.probe_type == "attention" and args.act_type != "full_prompt_sequence":
        raise ValueError("Attention probe can only be trained on full prompt sequence activations.")
    
    if args.first_only and args.last_only:
        raise ValueError("Cannot use both first_only and last_only at the same time.")

    set_seed(args.seed)

    # Determine output directory
    output_dir = args.output_dir or args.activations.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and split data (split by sample/prompt to prevent leakage)
    print(f"Loading activations from {args.activations}")
    data = load_and_split_data(args.activations, args.eval_fraction, args.seed, args.first_only, args.last_only, args.act_type)

    hidden_size = data.hidden_size
    print(f"  Hidden size: {hidden_size}")
    print(f"  Label range: [{data.y_train.min():.3f}, {data.y_train.max():.3f}]")
    print(f"\nData split (by sample/prompt):")
    print(f"  Train: {data.n_train_samples} samples -> {len(data.X_train)} activations")
    print(f"  Eval:  {data.n_eval_samples} samples -> {len(data.X_eval)} activations")

    # Parse hidden dims and determine name
    hidden_dims = None
    if args.hidden_dims:
        hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",")]
    
    attn_hidden_dims = None
    if args.attn_hidden_dims:
        attn_hidden_dims = [int(x.strip()) for x in args.attn_hidden_dims.split(",")]
    
    if args.name:
        probe_name = args.name
    elif hidden_dims or attn_hidden_dims:
        value_suffix, query_suffix = "", ""
        if hidden_dims:
            value_suffix = "_mlp_" + "_".join(str(d) for d in hidden_dims)
        if attn_hidden_dims:
            query_suffix = "_attn_mlp_" + "_".join(str(d) for d in attn_hidden_dims)
        
        if args.probe_type == "attention":
            heads_suffix = f"_h{args.num_heads}"
        else:
            heads_suffix = ""
        probe_name = f"{args.probe_type}" + heads_suffix + value_suffix + query_suffix
    else:
        probe_name = f"{args.probe_type}_linear"

    
    if args.position_bias:
        probe_name += "_posbias"

    if args.first_only:
        probe_name += "_first_only"
    
    if args.last_only:
        probe_name += "_last_only"

    # Train probe
    probe_architecture = f"MLP {hidden_dims}" if hidden_dims else "linear"
    print(
        f"\nTraining {probe_architecture} probe '{probe_name}' (lr={args.lr}, weight_decay={args.weight_decay}, epochs={args.epochs})..."
    )
    probe = train_probe(
        args.probe_type,
        data.X_train,
        data.y_train,
        data.mask_train,
        hidden_size=hidden_size,
        hidden_dims=hidden_dims,
        attn_hidden_dims=attn_hidden_dims,
        num_heads=args.num_heads,
        dropout=args.dropout,
        position_bias=args.position_bias,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=DEVICE,
    )

    # Evaluate
    train_metrics = evaluate_probe(probe, args.probe_type, data.X_train, data.y_train, data.mask_train, device=DEVICE)
    eval_metrics = evaluate_probe(probe, args.probe_type, data.X_eval, data.y_eval, data.mask_eval, device=DEVICE)

    print("\nResults:")
    print(
        f"  Train: BCE={train_metrics['bce']:.4f}, MSE={train_metrics['mse']:.4f}, MAE={train_metrics['mae']:.4f}, Binarized Accuracy={train_metrics['binarized_accuracy']:.4f}"
    )
    print(
        f"  Eval:  BCE={eval_metrics['bce']:.4f}, MSE={eval_metrics['mse']:.4f}, MAE={eval_metrics['mae']:.4f}, Binarized Accuracy={eval_metrics['binarized_accuracy']:.4f}"
    )

    # Save probe
    probe_path = output_dir / f"{probe_name}_probe.pt"
    probe.save(probe_path)
    print(f"\nProbe saved to {probe_path}")

    # Save results
    results = {
        "config": data.config,
        "args": {
            "activations": str(args.activations),
            "name": probe_name,
            "hidden_dims": hidden_dims,
            "position_bias": args.position_bias,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_fraction": args.eval_fraction,
            "seed": args.seed,
        },
        "data": {
            "n_train_samples": data.n_train_samples,
            "n_eval_samples": data.n_eval_samples,
            "n_train_activations": len(data.X_train),
            "n_eval_activations": len(data.X_eval),
            "hidden_size": hidden_size,
        },
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }

    results_path = output_dir / f"{probe_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
