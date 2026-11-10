# Replication of Huang et al. (ACL 2023), RoBERTa-large on PropaNews

## Setup

`bash replication/setup_and_patch.sh` clones https://github.com/khuangaf/FakingFakeNews at commit `9164941`
and applies the only code change we needed. Both RoBERTa scripts import `AdamW` from `torch.optim`,
because `transformers` no longer exports it. The authors' Google Drive checkpoint is no longer reachable,
so we train from scratch.

## Train (about 73 min on a Colab T4)

```bash
cd FakingFakeNews/roberta
python train_roberta.py --output_dir /content/drive/MyDrive/fakenewsproject/outputs --data_dir ../data/ --max_epoch 10
```

Every other hyperparameter stays at the repository default: batch 2, accumulation 8, LR 5e-5 / 1e-3,
5 warm-up epochs, maximum length 512, seed 42. The script prints Snopes results at the end.
Our full log is in `train_log_20261007_142926.txt`.

## PolitiFact

```bash
python test_roberta.py --checkpoint_path <output_dir>/<timestamp>/best.pt --data_dir ../data/
```

`test_roberta.py` always reads `politifact_test.jsonl` from `--data_dir`. For any other file, use `new_data/eval_any.py`.

## Results

| Test set | Paper AUC (mean ± s.d., 4 runs) | Ours (1 run, 95% bootstrap CI) |
|---|---|---|
| PolitiFact | 65.34 ± 4.5 | 57.76 [53.7, 61.5] |
| Snopes | 53.03 ± 3.7 | 43.32 [38.9, 47.9] |

**Label convention.** In the original data, `label: true` means **real**. The sigmoid output is P(real).

**Code vs. paper.**
- The paper trains for at most 20 epochs; the code defaults to 30.
- The paper says it selects by dev accuracy; the code selects by dev **AUC**, because the accuracy variable is overwritten.
