"""Within-pair analysis for WikiPropaNews (report Table 10).

For every real article and its own fake, delta = P(real | real) - P(real | fake).
Reports correct (+), wrong (-) and tied pairs per fake type and a two-sided sign test that
excludes ties. Ties occur because half-precision inference can give two nearly identical
articles exactly the same score.

Usage:
  python analysis/paired_sign_test.py --pred $OUT/preds/preds_wikipropanews.csv \
      --full $OUT/wikipropanews/wikipropanews_full.jsonl [--out_csv results_paired_sign.csv]
"""
import argparse, json
import pandas as pd
from scipy.stats import binomtest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    p = pd.read_csv(args.pred)
    m = pd.DataFrame([json.loads(l) for l in open(args.full, encoding="utf-8")])[["id", "pair_id", "fake_type"]]
    d = p.merge(m, on="id")
    piv = d.pivot_table(index="pair_id", columns="label_real", values="prob_real").dropna()
    delta = piv[1] - piv[0]
    ftype = d[d.label_real == 0].set_index("pair_id")["fake_type"].reindex(delta.index)
    rows = []
    for t in ["sent-fake", "loaded-language", "appeal-to-authority", "ALL"]:
        x = delta if t == "ALL" else delta[ftype == t]
        pos, neg, tie = int((x > 0).sum()), int((x < 0).sum()), int((x == 0).sum())
        pval = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
        rows.append({"fake_type": t, "pairs": len(x), "correct": pos, "wrong": neg, "tie": tie,
                     "mean_delta": round(float(x.mean()), 4), "sign_test_p": pval})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    if args.out_csv:
        res.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
