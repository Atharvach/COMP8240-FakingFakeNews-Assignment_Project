# Reproducing and Extending *Faking Fake News for Real Fake News Detection*

COMP8240 (Applications of Data Science), Macquarie University. Final project by Atharva Chitgopekar.

This repository reproduces the RoBERTa-large detector of
[Huang et al. (ACL 2023)](https://aclanthology.org/2023.acl-long.815/), which is trained on the
propaganda-loaded PropaNews corpus. It then tests the detector on new data: **LIAR**, **WELFake**, and
**WikiPropaNews**, a 594-article test set built for this project from openly licensed Wikinews articles (2020–2026).

## Key results

| Test set | n | AUC [95% CI] |
|---|---|---|
| PolitiFact (original) | 886 | 0.578 [0.537, 0.615] (paper: 0.653 ± 0.045) |
| Snopes (original) | 710 | 0.433 [0.389, 0.479] (paper: 0.530 ± 0.037) |
| LIAR / LIAR-strict | 1,283 / 1,016 | 0.564 / 0.581 |
| WELFake (sample) | 1,000 | 0.573 [0.538, 0.608] |
| **WikiPropaNews (ours)** | 594 | 0.517 [0.470, 0.565] |

Within matched real/fake pairs, the detector lowers P(real) when **propaganda** is inserted
(sign test p < 0.001 for appeal to authority, p = 0.002 for loaded language), but not when only a **false claim** is inserted (p = 0.66).
The full report is in `report/`.

## Repository layout

```
COMP8240_pipeline.ipynb      # one Colab notebook that runs everything in order (start here)
replication/
  setup_and_patch.sh         # clones the authors' code at a pinned commit and applies the AdamW fix
  train_log_20261007_142926.txt
  README.md                  # replication commands, results and code-vs-paper notes
new_data/
  01_collect_wikinews.py     # real articles from the Wikinews API (CC BY 2.5)
  02_generate_fakes.py       # PropaNews-style fakes with Qwen2.5-7B-Instruct (3 fake types)
  03_build_dataset.py        # roberta-large-mnli contradiction filter; builds the final dataset
  04_llm_judge.py            # LLM-as-judge validation (Phi-3.5-mini-instruct)
  05_annotation_agreement.py # blind annotation sheet; Cohen's kappa (human vs. judge)
  06_analyse.py              # result tables, bootstrap CIs and figures
  prep_existing.py           # LIAR (binary and strict) and WELFake (label meaning checked from the data)
  eval_any.py                # evaluates the checkpoint on any JSONL file and saves per-example scores
  common.py, README.md, DATASET_CARD.md
analysis/
  nli_by_type.py             # NLI pass rate per fake type (report Table 5)
  paired_sign_test.py        # within-pair sign test (report Table 12)
data/                        # final dataset, judge labels, annotation, predictions, results (see data/README.md)
report/                      # LaTeX source (ACL template) and compiled PDF
```

## How to run

1. Open `COMP8240_pipeline.ipynb` in Google Colab with a **T4 GPU**, and set `REPO_URL` in the first cell.
2. Run the cells top to bottom.
   - **Part A** trains the detector (about 75 min) and evaluates it on Snopes and PolitiFact.
   - **Part B** builds the new data and evaluates the detector on all six test sets.
3. All outputs go to `MyDrive/fakenewsproject/` on Google Drive.

Each long-running step appends to its output file and skips finished items, so a disconnect only loses the current batch.
All sampling uses seed 42.

To use the trained checkpoint instead of retraining, set `CKPT` to its path and start Part B at Step 6.

## For the marker

### Option A: check the results in 2 minutes (no GPU needed)

The per-example predictions, judge labels and annotations are in `data/`. Every table in the report
can be recomputed from them on any machine with Python 3.10+:

```bash
pip install pandas numpy scikit-learn scipy matplotlib tabulate
cd new_data
python 06_analyse.py --out_dir ../data/results_check --full ../data/wikipropanews/wikipropanews_full.jsonl --pred \
  Snopes=../data/preds/preds_snopes.csv PolitiFact=../data/preds/preds_politifact.csv LIAR=../data/preds/preds_liar.csv \
  LIAR-strict=../data/preds/preds_liar_strict.csv WELFake=../data/preds/preds_welfake.csv WikiPropaNews=../data/preds/preds_wikipropanews.csv
python ../analysis/paired_sign_test.py --pred ../data/preds/preds_wikipropanews.csv --full ../data/wikipropanews/wikipropanews_full.jsonl
python 05_annotation_agreement.py score --judge ../data/judge.jsonl --sheet ../data/annotation/annotation_sheet_FILLED.csv --key ../data/annotation/annotation_key.csv
```

These reproduce report Table 10 (overall results), Table 11 (by topic), Table 12 (by fake type and paired sign test) and Table 9 (human–judge agreement).

### Option B: run the detector (Colab T4, about 10 min)

Download our trained checkpoint (1.4 GB): https://drive.google.com/file/d/1ktPquPDnze38ohhXelLIQbRh0IGp8ukk/view?usp=sharing

Open `COMP8240_pipeline.ipynb` in Colab and run the first two code cells. In Part A, run the
"use our checkpoint" cell instead of training, then go to Part B, Step 6, to evaluate the detector on any test set.

### Option C: everything from scratch

Run the whole notebook, about 3–4 GPU-hours in total.

## Important conventions

- `label: true` means **real** news, as in the original repository. The classifier outputs **P(real)**.
- The original code has no licence file, so it is **not copied** here. `replication/setup_and_patch.sh` fetches it.
- LIAR and WELFake are not redistributed; `prep_existing.py` downloads and converts them.
- WikiPropaNews contains deliberately false statements and is for detection research only. Read `new_data/DATASET_CARD.md`.

## Acknowledgements

Original work: Huang, McKeown, Nakov, Choi and Ji (2023), https://github.com/khuangaf/FakingFakeNews.
Wikinews content: Wikinews contributors, CC BY 2.5.
Generative AI (Claude) assisted with drafting code and text. All experiments were run and checked by the author.
