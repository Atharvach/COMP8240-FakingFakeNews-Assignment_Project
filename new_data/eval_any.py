"""Evaluate a trained FakingFakeNews RoBERTa checkpoint on ANY jsonl test file and save per-example scores.

Same model definition, tokenisation (max 512 tokens) and checkpoint format as roberta/test_roberta.py,
but (a) the test file is an argument instead of the hard-coded politifact_test.jsonl, and (b) every
example's P(real) is written to CSV with its id, so results can be broken down by subgroup.

Usage:
  python eval_any.py --checkpoint_path .../best.pt --test_file .../wikipropanews_test.jsonl --out_csv .../preds_wikipropanews.csv
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

class BERTModelForClassification(nn.Module):          # identical to the original repo
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return torch.sigmoid(self.linear(hidden_states[:, 0, :]))

def to_bool(x):
    return bool(int(x)) if isinstance(x, (int, float, bool)) else str(x).lower() in {"true", "1", "real"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_name", default="roberta-large")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test_file, encoding="utf-8") if l.strip()]
    ids = [r.get("id", i) for i, r in enumerate(rows)]
    labels = np.array([to_bool(r["label"]) for r in rows], dtype=int)    # 1 = real (repo convention)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = BERTModelForClassification(args.model_name).cuda()
    model.load_state_dict(torch.load(args.checkpoint_path, map_location="cuda")["model"], strict=True)
    model.eval()

    probs = []
    with torch.no_grad():
        for b in range(0, len(rows), args.batch_size):
            enc = tok([r["txt"] for r in rows[b:b + args.batch_size]], max_length=args.max_len,
                      padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(enc["input_ids"], enc["attention_mask"]).view(-1)
            probs.extend(out.float().cpu().tolist())
    probs = np.array(probs)
    pred = (probs > 0.5).astype(int)
    pd.DataFrame({"id": ids, "label_real": labels, "prob_real": probs}).to_csv(args.out_csv, index=False)
    print(f"file: {args.test_file}  n={len(rows)}  real={labels.sum()}  fake={len(labels)-labels.sum()}")
    print(f"AUC={roc_auc_score(labels, probs):.4f}  F1(real=positive, as repo)={f1_score(labels, pred):.4f}  "
          f"macro-F1={f1_score(labels, pred, average='macro'):.4f}  acc={accuracy_score(labels, pred):.4f}  "
          f"predicted-real rate={pred.mean():.3f}")

if __name__ == "__main__":
    main()
