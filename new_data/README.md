# WikiPropaNews — new test data for the FakingFakeNews reproduction (COMP8240)

This folder builds and evaluates the new data for the reproduction of Huang et al. (ACL 2023),
*Faking Fake News for Real Fake News Detection*. The notebook `COMP8240_pipeline.ipynb` at the repository root copies this folder into the cloned FakingFakeNews
code, next to `roberta/` and `data/`, and runs every step in order.

## What gets built

**WikiPropaNews (newly constructed).** Real articles are collected from English Wikinews
(CC BY 2.5, published 2020–May 2026, after PropaNews' source news and RoBERTa's
pre-training data). For each real article a paired fake is made the way PropaNews makes them: one
salient sentence is replaced with a plausible but false claim, in one of three styles matching the
three PropaNews fake types (`sent-fake`, `loaded-language`, `appeal-to-authority`). Fakes are kept
only if an NLI model labels the false sentence as contradicting the original. Final size:
594 articles (297 real + 297 fake: 100 sent-fake, 97 loaded-language, 100 appeal-to-authority), drawn from Wikinews
articles published 2020 to May 2026. See `DATASET_CARD.md`.

**Existing datasets (new to this method).** LIAR test split, as a 6→2 binary mapping and a
"strict" version without half-true; and a balanced 1,000-article sample of WELFake.

## Label convention

As in the original repository, `label: true` = REAL and `label: false` = FAKE; the classifier outputs
P(real), and F1 uses REAL as the positive class.

## Pipeline

| Step | Script | Needs GPU | Output |
|---|---|---|---|
| 1 | `01_collect_wikinews.py` | no | `wikinews_raw.jsonl` |
| 2 | `02_generate_fakes.py` (Qwen2.5-7B-Instruct, 4-bit) | yes | `generations.jsonl` |
| 3 | `03_build_dataset.py` (roberta-large-mnli filter) | yes (faster) | `wikipropanews_full.jsonl`, `wikipropanews_test.jsonl`, `stats.json` |
| 4 | `04_llm_judge.py` (Phi-3.5-mini-instruct) | yes | `judge.jsonl` |
| 5 | `05_annotation_agreement.py` | no | blind annotation sheet; Cohen's κ human vs judge |
| – | `prep_existing.py` | no | `liar_test_binary.jsonl`, `liar_test_strict.jsonl`, `welfake_sample.jsonl` |
| 6 | `eval_any.py` | yes | per-example predictions (`preds_*.csv`) |
| 7 | `06_analyse.py` | no | result tables + figures |

Every long step appends to its output file and skips finished items, so a Colab disconnect only
costs the current batch. Seeds are fixed (42) throughout.

## Ethics / licence

Real article text: Wikinews contributors, CC BY 2.5 — source URL and title are kept for every
example. Fake articles are research artefacts containing deliberately false statements; they are
labelled `false` and must not be re-published as news. Generated quotes are attributed only to
unnamed authorities, never to real named people.

## Third-party data

LIAR and WELFake are not redistributed here. Run `prep_existing.py` to download and convert them.

## Top-up generation

`02_generate_fakes.py --pool_size 700 --force_type loaded-language` adds extra loaded-language fakes,
for the articles not already generated. We used it because only 29% of loaded-language generations passed the NLI filter.
