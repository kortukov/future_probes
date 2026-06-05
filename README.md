# Predicting Future Behaviors in Reasoning Models Enables Better Steering


![Teaser figure](./data/fpcg_teaser_v2_compressed.png)



## Installation
We use `uv` to manage the environment. You can install it [as described here](https://docs.astral.sh/uv/getting-started/installation/).

To create the environment, run the following command in the project root:
```bash
uv sync
```

# Experiments

## Behavior Distribution Analysis

To gather the ground truth behavior distribution dynamics for a dataset, run:
```bash
uv run behavior_distribution_analysis.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset myopic_reward \
    --subset 100 \ 
    --max_new_tokens 8192 \
    --num_base_responses 10 \
    --num_samples 30 \
    --seed 42 \
```
We use negative seeds to get the non-overlapping test dataset, so to get test set results, use `--seed -42`.
In the paper, we use the datasets `myopic_reward`, `wealth_seeking`, `survival_instinct`, `sorrybench`, `sep`, and `elephant_aita`.

This stage of analysis generates the ground truth data for probing and FPCG.
This data is saved in the `results/per_sentence_probabilities` directory.


## Internal Representation of Output Behavior Distributions

### Predicting Future Behavior Distributions

This is done in three steps - gather activations, train the probe and evaluate it.

#### Gather Activations

We run activation gathering for the training and test datasets.
```bash
uv run gather_activations.py --results_file results/per_sentence_probabilities/DeepSeek-R1-Distill-Llama-8B/myopic_reward/n100_nbase10_nsamp30_l8192_s42_t1.0_results.json --layer 25
uv run gather_activations.py --results_file results/per_sentence_probabilities/DeepSeek-R1-Distill-Llama-8B/myopic_reward/n100_nbase10_nsamp30_l8192_s-42_t1.0_results.json --layer 25
```

#### Train the Linear Probe 
```bash
uv run train_probe.py --activations results/per_sentence_probabilities/DeepSeek-R1-Distill-Llama-8B/myopic_reward/n100_nbase10_nsamp30_l8192_s42_t1.0_activations/layer25/activations.pt
```

#### Evaluate the Probe
```bash
uv run evaluate_probe.py \
 --probe results/per_sentence_probabilities/DeepSeek-R1-Distill-Llama-8B/myopic_reward/n100_nbase10_nsamp30_l8192_s42_t1.0_activations/layer25/probabilistic_linear_probe.pt \
 --activations results/per_sentence_probabilities/DeepSeek-R1-Distill-Llama-8B/myopic_reward/n100_nbase10_nsamp30_l8192_s-42_t1.0_activations/layer25/activations.pt \
 --results_file results/per_sentence_probabilities/DeepSeek-R1-Distill-Llama-8B/myopic_reward/n100_nbase10_nsamp30_l8192_s-42_t1.0_results.json  --reasoning_parts
```

You can find the results in `results/per_sentence_probabilities/DeepSeek-R1-Distill-Llama-8B/myopic_reward/n100_nbase10_nsamp30_l8192_s42_t1.0_activations/layer25/linear_probe_predictions`.
