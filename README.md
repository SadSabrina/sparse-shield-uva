# SparseShield: Towards Understanding the Robustness of Sparse Autoencoders

[![Paper](https://img.shields.io/badge/ACL%20Findings-2026-b31b1b.svg)](https://aclanthology.org/2026.findings-acl.1298/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.18756-b31b1b.svg)](https://arxiv.org/abs/2604.18756)
[![Official code](https://img.shields.io/badge/code-AikyamLab%2Fsparse--jailbreak-blue.svg)](https://github.com/AikyamLab/sparse-jailbreak)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Companion repository for the paper

> **Towards Understanding the Robustness of Sparse Autoencoders**
> Ahson Saiyed¹, Sabrina Sadiekh², Chirag Agarwal¹
> ¹University of Virginia ([Aikyam Lab](https://github.com/AikyamLab)), ²Independent Researcher
> *Findings of the Association for Computational Linguistics: ACL 2026*, pp. 26066–26086

The paper is my collaboration with the [Aikyam Lab](https://github.com/AikyamLab) at the University of Virginia. This repository contains the black-box evaluation and the statistical and feature-level analyses drafts as a part of the work.

It contains the **black-box evaluation pipeline**, the **raw judge verdicts and ASR tables**, the **statistical analysis of white-box transfer**, the **sparse-feature (Jaccard) analysis** and the notebooks that produce the paper figures.
GCG/BEAST suffix generation, SAE feature extraction and the gradient spectral analysis live in the official repository: [AikyamLab/sparse-jailbreak](https://github.com/AikyamLab/sparse-jailbreak).

---

## TL;DR

We insert a pretrained Sparse Autoencoder (SAE) into one residual-stream layer of an LLM at inference time:

$$\hat h_\ell = W_{dec}\,\mathrm{ReLU}(W_{enc} h_\ell + b_{enc}) + b_{dec}$$

The model weights stay unchanged and gradients still flow through the SAE, so white-box attackers can optimise straight through it. Even so, the sparse encode–decode bottleneck:

- reduces jailbreak success rates by **up to 5×** under GCG and BEAST,
- **reduces cross-model transfer** of adversarial suffixes and makes it asymmetric (BASE→SAE < SAE→BASE),
- shows a **dose–response** effect (sparser SAE → more robust) and a **layer-dependent** defense–utility tradeoff,
- concentrates GCG gradients into fewer directions (higher inter-step cosine similarity, lower effective rank, higher final loss).

| Attack | Median ASR, baseline | Median ASR, SAE | Mann–Whitney *p* |
|---|---:|---:|---:|
| GCG | 55.0% | 19.05% | 0.0087 |
| BEAST | 19.35% | 9.7% | 0.0411 |

Black-box prompts (Salad-Data, prompt-injections, SafeEval; 1,500 prompts per model): the SAE lowers ASR under the HarmBench classifier (mean 23.2% → 16.0%). The effect is small under WildGuard and absent under the refusal heuristic, so black-box conclusions depend on the judge.

## Models and SAEs

| Model | SAE | Layer | Width |
|---|---|---:|---:|
| Gemma-2-2B | `gemma-scope-2b-pt-res` | 12 | 16K |
| Gemma-2-9B | `gemma-scope-9b-pt-res` | 19 | 16K |
| Gemma-2-27B | `gemma-scope-27b-pt-res` | 34 | 131K |
| Llama-3.1-8B-Instruct | `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` | 19 | 65K |
| Llama-3.3-70B-Instruct | `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` | 50 | 65K |
| Mistral-7B | `JoshEngels/Mistral-7B-Residual-Stream-SAEs` | 16 | 65K |

## Repository structure

```
.
├── scripts/blackbox/
│   ├── generate_gemma.py       # generate responses: Gemma-2 2B/9B/27B (+ Gemma Scope SAE via sae_lens)
│   ├── generate_llama.py       # generate responses: Llama-3.1-8B / Llama-3.3-70B (+ Goodfire SAE)
│   ├── generate_mistral.py     # generate responses: Mistral-7B (+ Engels et al. SAE)
│   └── aggregate_asr.py        # per-prompt verdicts -> asr_by_file.csv, asr_summary.csv
├── notebooks/
│   ├── 01_blackbox_asr_analysis.ipynb        # black-box ASR, CIs, box plots (App. B.3, Table 13)
│   ├── 02_whitebox_transfer_statistics.ipynb # GCG/BEAST transfer matrices, MWU tests, bootstrap CIs (Tables 1-2, Fig. 1)
│   ├── 03_jaccard_feature_overlap.ipynb      # top-K SAE feature overlap between suffix sources (App. D)
│   └── 04_paper_figures.ipynb                # transfer and Jaccard heatmaps (Figs. 2-3)
├── results/
│   ├── blackbox_asr/           # <model>/<judge>/<baseline|sae>_<dataset>[.json|_detailed.jsonl] + summary CSVs
│   └── jaccard/                # <gcg|beast>/<model>/ Jaccard matrices, top-K features, overlap stats, heatmaps
├── figures/                    # paper and exploratory figures
└── data/                       # random-text baseline suffixes (large artefacts not tracked, see data/README.md)
```

## Setup

```bash
git clone https://github.com/<your-username>/SparseShield.git
cd SparseShield
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Generation and judging need a CUDA GPU (the 70B model and the 13B HarmBench judge need ≥ 80 GB, or several GPUs). Gemma and Llama checkpoints are gated on Hugging Face: accept their licenses and pass `--hf-token` or run `huggingface-cli login`.

The analysis notebooks only need the files in `results/` and run on CPU.

## Reproducing the black-box evaluation

**1. Generate responses** (500 prompts per benchmark, sampled with `shuffle(seed=42)`, greedy decoding). The Llama and Mistral scripts append a single space to every prompt: without it these models often emit EOS right away on jailbreak-style prompts and return an empty response. Gemma-2 is trained to emit EOS differently and does not show this early termination, so the Gemma script uses the prompts as is (this is how the Gemma results were produced). Each run writes `<model>_<SAE|baseline>_<benchmark>_<timestamp>.jsonl` to `--output-dir`.

```bash
# Gemma-2 (SAE from Gemma Scope via sae_lens)
python scripts/blackbox/generate_gemma.py --model-name google/gemma-2-9b \
    --sae-release gemma-scope-9b-pt-res-canonical --sae-id layer_19/width_16k/canonical \
    --max-samples 500 --output-dir data/responses/gemma-9b
python scripts/blackbox/generate_gemma.py --model-name google/gemma-2-9b --no-sae \
    --max-samples 500 --output-dir data/responses/gemma-9b

# Llama-3.1-8B-Instruct (Goodfire SAE, layer 19, 16x expansion)
python scripts/blackbox/generate_llama.py --max-samples 500 --output-dir data/responses/llama-8b
python scripts/blackbox/generate_llama.py --no-sae --max-samples 500 --output-dir data/responses/llama-8b

# Llama-3.3-70B-Instruct (Goodfire SAE, layer 50, 8x expansion)
python scripts/blackbox/generate_llama.py --model-name meta-llama/Llama-3.3-70B-Instruct \
    --sae-name Llama-3.3-70B-Instruct-SAE-l50 --sae-layer model.layers.50 --expansion-factor 8 \
    --max-samples 500 --output-dir data/responses/llama-70b

# Mistral-7B (layer 16)
python scripts/blackbox/generate_mistral.py --sae-layer 16 --max-samples 500 --output-dir data/responses/mistral-7b
python scripts/blackbox/generate_mistral.py --no-sae --max-samples 500 --output-dir data/responses/mistral-7b
```

Use `--benchmarks safe_eval injections salad` to run a subset. Gemma-2-2B uses layer 12 (`gemma-scope-2b-pt-res-canonical`, `layer_12/width_16k/canonical`), and Gemma-2-27B uses layer 34 (`gemma-scope-27b-pt-res-canonical`, `layer_34/width_131k/canonical`).

**2. Judge the responses** with three judges: [WildGuard](https://huggingface.co/allenai/wildguard), the [HarmBench classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) (each with its standard prompt template) and a keyword-based refusal detector (pattern categories in the paper, Table 10). Save the per-prompt verdicts as JSON lines (`{"prompt_id": ..., "is_harmful": true|false, "judge": ...}`) to `results/blackbox_asr/<model>/<judge>/<baseline|sae>_<dataset>_detailed.jsonl`.

**3. Aggregate** all verdicts into ASR tables:

```bash
python scripts/blackbox/aggregate_asr.py results/blackbox_asr
```

Then open `notebooks/01_blackbox_asr_analysis.ipynb` for confidence intervals and plots.

## Results format

`results/blackbox_asr/<model>/<judge>/`

| File | Content |
|---|---|
| `<exp>_<dataset>.json` | summary: `asr` (%), `total`, `harmful`, `judge` (most files also record the judged `input_file`) |
| `<exp>_<dataset>_detailed.jsonl` | one line per prompt: `prompt_id`, `is_harmful`, `judge` |

`<exp>` ∈ {`baseline`, `sae`}, `<dataset>` ∈ {`salad`, `injections`, `safe_eval`}, `<judge>` ∈ {`refusal`, `wildguard`, `harmbench`}.
`asr_summary.csv` has one row per (model, exp, dataset, judge), and `asr_by_file.csv` adds the source file.
Both CSVs match the numbers reported in the paper. The per-prompt file for `gemma-9b/refusal/baseline_salad` is not included, so re-running `aggregate_asr.py` gives 107 rows instead of 108.

`results/jaccard/<attack>/<model>/` (the model is the one whose SAE was used to encode the suffixes)

| File | Content |
|---|---|
| `jaccard_matrix_top{K}_mean.csv` | pairwise Jaccard similarity of top-K features between suffix sources |
| `top_features_top{K}_mean.json` | top-K feature indices and mean activations per source |
| `overlap_statistics_top{K}_mean.json` | pairwise overlap, union and features common to all sources |
| `active_feature_stats_thr0.0_top{K}_mean.csv` | mean and median number of active features per sample |
| `jaccard_similarity_top{K}_mean.png` | heatmap |

`gcg/` and `beast/` include random-text baselines (`random_*` sources). `gcg_without_random_baseline/` is an earlier run without them.

## Data availability

Some artefacts are **not** part of this repository (see [`data/README.md`](data/README.md)):

- **SAE feature activations** (`data/sae_features/`, ~6 GB of pickles): too large for git. They can be regenerated with the feature-extraction scripts in the [official repository](https://github.com/AikyamLab/sparse-jailbreak).
- **Optimised adversarial suffixes and raw model responses**: withheld in line with the paper's ethics statement.

## Responsible use

This work studies adversarial attacks in order to build defenses. All prompts come from established red-teaming benchmarks (HarmBench, Salad-Data, prompt-injections-benchmark, SafeEval). Use the code only for safety research.

## Citation

```bibtex
@inproceedings{saiyed-etal-2026-towards,
    title     = {Towards Understanding the Robustness of Sparse Autoencoders},
    author    = {Saiyed, Ahson and Sadiekh, Sabrina and Agarwal, Chirag},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
    year      = {2026},
    pages     = {26066--26086},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2026.findings-acl.1298/}
}
```

## Acknowledgements

SAEs: [Gemma Scope](https://huggingface.co/google/gemma-scope), [Goodfire](https://huggingface.co/Goodfire), [Engels et al.](https://huggingface.co/JoshEngels/Mistral-7B-Residual-Stream-SAEs). Judges: [WildGuard](https://huggingface.co/allenai/wildguard), [HarmBench classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls). Benchmarks: [Salad-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data), [prompt-injections-benchmark](https://huggingface.co/datasets/qualifire/prompt-injections-benchmark), [SafeEval](https://huggingface.co/datasets/dynamoai/safe_eval).

## License

[MIT](LICENSE)
