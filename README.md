# Predicting Future Behaviors in Reasoning Models Enables Better Steering


## Installation
We use `uv` to manage the environment. You can install it [as described here](https://docs.astral.sh/uv/getting-started/installation/).

To create the environment, run the following command in the project root:
```bash
uv sync
```

## Experiments

### Behavior Distribution Analysis

To gather the ground truth behavior distribution dynamics for a dataset, run:
```bash
uv run behavior_distribution_analysis \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset elephant_aita \
    --subset 100 \ 
    --max_new_tokens 8192 \
    --num_base_responses 10 \
    --num_samples 10 \
    --seed 42 \
```
We use negative seeds to get the non-overlapping test dataset, so to get test set results, use `--seed -42`.
In the paper, we use the datasets `myopic_reward`, `wealth_seeking`, `survival_instinct`, `sorrybench`, `sep`, and `elephant_aita`.

This stage of analysis generates the ground truth data for probing and FPCG.
This data is saved in the `results/per_sentence_probabilities` directory.

