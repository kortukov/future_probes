#!/usr/bin/env python3
"""Build the trimmed, curated dataset that powers the demo site.

For each (model, dataset) pair we read the held-out probabilistic-MLP probe
prediction file, trim every example down to just the fields the visualization
needs, curate a handful of examples, and write one small JSON per pair plus a
manifest the front-end uses to populate its dropdowns.

Usage:
    python build_demo.py                      # diverse curation, 15 examples
    python build_demo.py --curation best_fit  # lowest-MAE examples first
    python build_demo.py --curation random    # reproducible random sample
    python build_demo.py --examples 30
    python build_demo.py --standalone         # also emit index_standalone.html
"""

import argparse
import glob
import json
import os
import random

# Repo root = parent of this demo/ directory.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results", "per_sentence_probabilities")

# Model -> residual-stream layer for the probabilistic_mlp_1024 probe (paper config).
MODELS = {
    "Qwen3-14B": 32,
    "QwQ-32B": 50,
    "gpt-oss-20b": 20,
    "DeepSeek-R1-Distill-Llama-8B": 25,
}

DATASETS = [
    "elephant_aita",
    "myopic_reward",
    "sep",
    "sorrybench",
    "survival_instinct",
    "wealth_seeking",
]

# Pretty labels for the UI.
MODEL_LABELS = {
    "Qwen3-14B": "Qwen3-14B",
    "QwQ-32B": "QwQ-32B",
    "gpt-oss-20b": "gpt-oss-20b",
    "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
}
DATASET_LABELS = {
    "elephant_aita": "Elephant (AITA)",
    "myopic_reward": "Myopic reward",
    "sep": "SEP",
    "sorrybench": "SorryBench",
    "survival_instinct": "Survival instinct",
    "wealth_seeking": "Wealth seeking",
}


def find_prediction_file(model, layer, dataset):
    """Return the non-empty probe-prediction outputs file for a pair, or None."""
    pattern = os.path.join(
        RESULTS,
        model,
        dataset,
        "*_activations",
        f"layer{layer}",
        "probabilistic_mlp_1024_probe_predictions",
        "*_outputs.json",
    )
    candidates = [p for p in glob.glob(pattern) if os.path.getsize(p) > 0]
    if not candidates:
        return None
    # Prefer the longer-budget (l8192) / nsamp30 file when several exist.
    candidates.sort(key=os.path.getsize, reverse=True)
    return candidates[0]


def trim_example(ex):
    """Keep only what the visualization renders."""
    return {
        "input_prompt": ex["input_prompt"],
        "base_behavior_probability": ex["base_behavior_probability"],
        "predicted_base_behavior_probability": ex.get("predicted_base_behavior_probability"),
        "sample_mae": ex.get("sample_mae"),
        "responses": [
            {
                "sentences": r["sentences"],
                "per_sentence_probabilities": r["per_sentence_probabilities"],
                "predicted_probabilities": r.get("predicted_probabilities"),
                "response_mae": r.get("response_mae"),
            }
            for r in ex["responses"]
        ],
    }


def trajectory_swing(ex):
    """Mean |last - first| behavior-probability change across responses."""
    swings = []
    for r in ex["responses"]:
        p = r["per_sentence_probabilities"]
        if len(p) >= 2:
            swings.append(abs(p[-1] - p[0]))
    return sum(swings) / len(swings) if swings else 0.0


def curate(examples, mode, n):
    """Return up to n example indices, deterministically, per curation mode."""
    idx = list(range(len(examples)))
    if mode == "random":
        rng = random.Random(42)
        rng.shuffle(idx)
        return sorted(idx[:n])
    if mode == "best_fit":
        idx.sort(key=lambda i: (examples[i].get("sample_mae") if examples[i].get("sample_mae") is not None else 1e9))
        return idx[:n]
    # diverse: interleave big-swing, flat, and best-probe-fit examples across the
    # base-probability range, deterministically.
    by_swing = sorted(idx, key=lambda i: trajectory_swing(examples[i]), reverse=True)
    by_flat = sorted(idx, key=lambda i: trajectory_swing(examples[i]))
    by_fit = sorted(idx, key=lambda i: (examples[i].get("sample_mae") if examples[i].get("sample_mae") is not None else 1e9))
    by_base = sorted(idx, key=lambda i: examples[i]["base_behavior_probability"])
    # Round-robin across the orderings, skipping duplicates, until we have n.
    picked = []
    seen = set()
    pools = [by_swing, by_fit, by_base, by_flat]
    cursors = [0, 0, 0, 0]
    while len(picked) < n and any(cursors[k] < len(pools[k]) for k in range(len(pools))):
        for k in range(len(pools)):
            while cursors[k] < len(pools[k]) and pools[k][cursors[k]] in seen:
                cursors[k] += 1
            if cursors[k] < len(pools[k]) and len(picked) < n:
                i = pools[k][cursors[k]]
                picked.append(i)
                seen.add(i)
                cursors[k] += 1
    return sorted(picked)


def mean_sample_mae(examples):
    vals = [e.get("sample_mae") for e in examples if e.get("sample_mae") is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curation", choices=["diverse", "best_fit", "random"], default="diverse")
    ap.add_argument("--examples", type=int, default=15)
    ap.add_argument("--standalone", action="store_true")
    args = ap.parse_args()

    data_dir = os.path.join(HERE, "data")
    os.makedirs(data_dir, exist_ok=True)

    manifest = []
    embedded = {}  # for the optional standalone build
    for model, layer in MODELS.items():
        for dataset in DATASETS:
            src = find_prediction_file(model, layer, dataset)
            if src is None:
                print(f"SKIP  {model}/{dataset}: no mlp_1024 prediction file found")
                continue
            with open(src) as f:
                examples = json.load(f)
            chosen = curate(examples, args.curation, args.examples)
            trimmed = [trim_example(examples[i]) for i in chosen]

            out_rel = f"data/{model}/{dataset}.json"
            out_abs = os.path.join(HERE, out_rel)
            os.makedirs(os.path.dirname(out_abs), exist_ok=True)
            with open(out_abs, "w") as f:
                json.dump(trimmed, f)

            size_mb = os.path.getsize(out_abs) / 1024 / 1024
            manifest.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "layer": layer,
                "probe": "probabilistic_mlp_1024",
                "file": out_rel,
                "num_examples": len(trimmed),
                "mean_mae": mean_sample_mae(trimmed),
            })
            embedded[f"{model}/{dataset}"] = trimmed
            print(f"OK    {model}/{dataset}: {len(trimmed)} examples, {size_mb:.2f} MB")

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "models": [{"id": m, "label": MODEL_LABELS[m]} for m in MODELS],
            "datasets": [{"id": d, "label": DATASET_LABELS[d]} for d in DATASETS],
            "pairs": manifest,
            "curation": args.curation,
        }, f, indent=2)

    total_mb = sum(os.path.getsize(os.path.join(HERE, p["file"])) for p in manifest) / 1024 / 1024
    print(f"\nWrote {len(manifest)} pairs, manifest.json, total data {total_mb:.1f} MB")

    if args.standalone:
        build_standalone(embedded, manifest)


def build_standalone(embedded, manifest):
    """Emit a single offline HTML file with all pair data embedded."""
    tpl_path = os.path.join(HERE, "index.html")
    with open(tpl_path) as f:
        html = f.read()
    blob = json.dumps({"manifest": {"pairs": manifest}, "data": embedded})
    inject = (
        "<script>window.__EMBEDDED__ = " + blob + ";</script>\n</body>"
    )
    html = html.replace("</body>", inject)
    out = os.path.join(HERE, "index_standalone.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote standalone {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
