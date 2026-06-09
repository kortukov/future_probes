# Behavior-Distribution Demo

Interactive visualization of **per-sentence behavior probability** over a reasoning
model's chain of thought, with the **probe's predicted trajectory** overlaid — the demo
companion to the paper *Predicting Behavior Distributions in LRMs Enables Better Steering*.

Pick a **model** and **dataset**; browse curated examples. Each line is one sampled
response: the solid line is the measured behavior probability after each sentence, the
dashed line is the probe's prediction. A ★ marks the end of the model's thinking.

- **Models:** Qwen3-14B (L32), QwQ-32B (L50), gpt-oss-20b (L20), DeepSeek-R1-Distill-Llama-8B (L25)
- **Datasets:** Elephant (AITA), Myopic reward, SEP, SorryBench, Survival instinct, Wealth seeking
- **Probe:** `probabilistic_mlp_1024`

## Run it locally

The page loads data with `fetch()`, so it must be served over HTTP (opening `index.html`
via `file://` will not work — browsers block `fetch` from the filesystem).

```bash
cd demo
python -m http.server 8000
# open http://localhost:8000
```

For an offline single file instead, build the standalone (see below) and open
`index_standalone.html` directly — it has all data embedded, no server needed.

## Load full data (beyond the curated examples)

The bundled `data/` only holds a handful of curated examples per pair. To browse the
**full** set, use **Load full data…** in the controls: it reads a raw
`results/per_sentence_probabilities/<model>/<dataset>/..._outputs.json` you downloaded
locally and renders all of its examples in memory (nothing is uploaded; reload to revert
to the bundled data).

1. Pick the matching **model** and **dataset** in the dropdowns first — the helper line
   under the button ("loads into: …") shows which pair the file will be attached to. The
   **dataset** drives the behavior-meaning callout, so as long as you pick a real dataset
   the callout stays even when the model is **Other**. Choose **Other** for the model when
   it isn't bundled (probe/layer info is then hidden, since there's no matching pair); choose
   **Other** for the dataset only when it isn't one of the known behaviors (the callout is
   then hidden).
2. Click **Load full data…** and pick the `..._outputs.json`.

Both file flavors work: raw outputs (ground truth only — the prediction overlay is hidden)
and the downstream `probabilistic_mlp_1024_probe_predictions/..._outputs.json` (which adds
the dashed probe predictions). Large files (hundreds of MB) load but may take a few seconds.

## Rebuild the data

`build_demo.py` reads the probe-prediction outputs under
`../results/per_sentence_probabilities/<model>/<dataset>/.../probabilistic_mlp_1024_probe_predictions/`,
trims each example to just the fields the viz needs, curates a handful, and writes
`data/<model>/<dataset>.json` plus `data/manifest.json` (which drives the dropdowns).

```bash
python build_demo.py                      # 15 examples/pair, "diverse" curation (default)
python build_demo.py --curation best_fit  # lowest-MAE examples first
python build_demo.py --curation random    # reproducible random sample (seed 42)
python build_demo.py --examples 30        # more examples per pair
python build_demo.py --standalone         # also write index_standalone.html (embeds all data)
```

Curation modes:
- `diverse` — a deterministic spread: big behavior swings, flat trajectories, a range of
  base probabilities, and good probe fits.
- `best_fit` — sorted by lowest sample MAE (probe at its most accurate).
- `random` — a reproducible random sample.

Model→layer mapping and the dataset list are at the top of `build_demo.py`.

## Publish to GitHub Pages

The whole `demo/` folder is a self-contained static site (`index.html` + `data/`).

1. Create a new **public** GitHub repo.
2. Copy the contents of this `demo/` folder into it (you do **not** need the multi-GB
   `results/` tree — only `data/` is required at runtime).
3. Push, then in the repo: **Settings → Pages → Build and deployment → Deploy from a
   branch**, branch `main`, folder `/ (root)`.
4. Your demo will be live at `https://<user>.github.io/<repo>/`.

Edit `PAPER_URL` and `REPO_URL` near the top of the `<script>` in `index.html` to point at
your paper and code.

## Files

- `index.html` — the site (model/dataset dropdowns + Plotly viz).
- `build_demo.py` — regenerates `data/` from the research results.
- `data/manifest.json` — list of available pairs + probe metadata (built).
- `data/<model>/<dataset>.json` — trimmed, curated per-pair data (built, committed).
- `index_standalone.html` — optional offline single file (built with `--standalone`).
