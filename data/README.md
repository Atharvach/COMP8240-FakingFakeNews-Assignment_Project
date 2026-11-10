# Data

The pipeline writes everything to `new_data_out/` on Google Drive. Copy these folders and files here:

| Copy from `new_data_out/` | What it is |
|---|---|
| `wikipropanews/` | `wikipropanews_test.jsonl` (model-ready), `wikipropanews_full.jsonl` (with metadata), `stats.json`, `stats.md` |
| `judge.jsonl` | LLM-judge labels (Phi-3.5-mini) for all 594 items |
| `annotation/` | Blind annotation sheet, its key, the filled sheet and `agreement_results.csv` |
| `preds/` | Per-example detector scores for all six test sets |
| `results/` | Result tables (CSV/MD) and figures |
| `wikinews_raw.jsonl`, `generations.jsonl` | Collected articles and all raw generations, including rejected ones (optional) |

`annotation/annotation_sheet_FILLED.csv` (the author's labels) is already included.

**Do not add** `new_data_out/existing/`, which holds LIAR and WELFake. They are third-party datasets; `new_data/prep_existing.py`
regenerates them.

Licences: Wikinews text is CC BY 2.5, with attribution through each record's `url`. See `new_data/DATASET_CARD.md`.
