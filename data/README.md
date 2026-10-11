# Data

| Path | Tracked in git | Content |
|---|---|---|
| `random_suffixes/` | yes | Random-text baseline suffixes (5 types: `full_alphanumeric_symbols`, `mixed_case`, `text_lowercase`, `text_numbers`, `unicode_text`) as `.txt` (one per line) and `.pkl` |
| `sae_features/` | no (~6 GB) | SAE latent activations of HarmBench prompts with each suffix source appended, used by `notebooks/03_jaccard_feature_overlap.ipynb` |
| `suffixes/` | no | Optimised GCG/BEAST adversarial suffixes (withheld, see the paper's ethics statement) |
| `responses/` | no | Raw model generations from `scripts/blackbox/generate_*.py` |

## `sae_features/` layout

```
sae_features/
├── gcg/<sae-model>/<source>_sae_features.pkl     # GCG suffixes from every source model + random baselines
├── beast/<sae-model>/<source>_sae_features.pkl   # BEAST suffixes from every source model + random baselines
└── random/<sae-model>/random_<type>_sae_features.pkl
```

`<sae-model>` ∈ {`gemma-9b`, `llama-8b`, `mistral-7b`} is the model whose SAE encodes the inputs. `<source>` is the model the suffix was optimised on, e.g. `llama_8b_base` (baseline model) or `llama_8b_sae_l19` (SAE-augmented model, SAE at layer 19).
Each pickle is a dict with a `features` tensor of shape `[n_prompts, d_sae]`.

These files can be regenerated with the feature-extraction scripts in [AikyamLab/sparse-jailbreak](https://github.com/AikyamLab/sparse-jailbreak).
