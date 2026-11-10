"""Step 6 - Results tables and figures for the report.

Reads the per-example prediction CSVs written by eval_any.py and produces:
  results_overall.csv/.md   AUC (bootstrap 95% CI), F1 (real = positive, as in repo), macro-F1, accuracy,
                            predicted-real rate, for every test set
  results_wikipropanews_by_type.csv / _by_topic.csv
  paired analysis           mean P(real|real article) - P(real|its fake), and % of pairs ranked correctly
  fig_auc_by_dataset.png, fig_auc_by_fake_type.png, fig_prob_hist_wikipropanews.png

Usage:
  python 06_analyse.py --pred Snopes=preds_snopes.csv PolitiFact=preds_politifact.csv LIAR=preds_liar.csv \
      LIAR-strict=preds_liar_strict.csv WELFake=preds_welfake.csv WikiPropaNews=preds_wikipropanews.csv \
      --full wikipropanews_full.jsonl --out_dir results/
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import read_jsonl

def boot_auc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])

def metrics(y, p):
    pred = (p > 0.5).astype(int)
    lo, hi = boot_auc(y, p)
    return {"n": len(y), "n_real": int(y.sum()), "n_fake": int(len(y) - y.sum()),
            "AUC": roc_auc_score(y, p), "AUC_CI_low": lo, "AUC_CI_high": hi,
            "F1_real_pos": f1_score(y, pred, zero_division=0), "macro_F1": f1_score(y, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, pred), "pred_real_rate": pred.mean()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="Name=path.csv pairs")
    ap.add_argument("--full", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    preds = {kv.split("=", 1)[0]: pd.read_csv(kv.split("=", 1)[1]) for kv in args.pred}
    overall = pd.DataFrame([{"dataset": k, **metrics(v["label_real"].values, v["prob_real"].values)}
                            for k, v in preds.items()]).round(4)
    overall.to_csv(os.path.join(args.out_dir, "results_overall.csv"), index=False)
    with open(os.path.join(args.out_dir, "results_overall.md"), "w") as f:
        try:
            f.write(overall.to_markdown(index=False))
        except ImportError:                       # tabulate not installed
            f.write(overall.to_string(index=False))
    print(overall.to_string(index=False))

    wk = next(k for k in preds if "wiki" in k.lower())
    meta = pd.DataFrame(read_jsonl(args.full))[["id", "pair_id", "label", "fake_type", "topic", "n_words"]]
    df = preds[wk].merge(meta, on="id")
    reals = df[df["label_real"] == 1]

    by_type = []
    for t in ["sent-fake", "loaded-language", "appeal-to-authority"]:
        sub = pd.concat([reals, df[df["fake_type"] == t]])
        m = metrics(sub["label_real"].values, sub["prob_real"].values)
        by_type.append({"fake_type": t, **m, "mean_P_real_of_fakes": df[df["fake_type"] == t]["prob_real"].mean()})
    by_type = pd.DataFrame(by_type).round(4)
    by_type.to_csv(os.path.join(args.out_dir, "results_wikipropanews_by_type.csv"), index=False)
    print("\nBy fake type (each type's fakes vs ALL reals):\n", by_type[["fake_type", "n", "AUC", "AUC_CI_low", "AUC_CI_high", "mean_P_real_of_fakes"]].to_string(index=False))

    by_topic = []
    for t, sub in df.groupby("topic"):
        if sub["label_real"].nunique() == 2 and len(sub) >= 20:
            by_topic.append({"topic": t, "n": len(sub), "AUC": roc_auc_score(sub["label_real"], sub["prob_real"])})
    by_topic = pd.DataFrame(by_topic, columns=["topic", "n", "AUC"]).sort_values("n", ascending=False).round(4)
    by_topic.to_csv(os.path.join(args.out_dir, "results_wikipropanews_by_topic.csv"), index=False)
    print("\nBy topic (topics with >= 20 examples):\n", by_topic.to_string(index=False))

    piv = df.pivot_table(index="pair_id", columns="label_real", values="prob_real")
    piv = piv.dropna()
    delta = piv[1] - piv[0]
    ft = df[df["label_real"] == 0].set_index("pair_id")["fake_type"]
    paired = pd.DataFrame({"delta": delta, "fake_type": ft.reindex(delta.index)})
    summary = paired.groupby("fake_type")["delta"].agg(["count", "mean", lambda d: (d > 0).mean()]).round(4)
    summary.columns = ["pairs", "mean_delta_P_real", "pct_pairs_ranked_correctly"]
    summary.loc["ALL"] = [len(delta), round(delta.mean(), 4), round((delta > 0).mean(), 4)]
    summary.to_csv(os.path.join(args.out_dir, "results_wikipropanews_paired.csv"))
    print("\nPaired analysis (same article, real vs fake):\n", summary.to_string())

    # Figures
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(overall))
    ax.bar(x, overall["AUC"], yerr=[overall["AUC"] - overall["AUC_CI_low"], overall["AUC_CI_high"] - overall["AUC"]],
           capsize=4, color="#4C72B0")
    ax.axhline(0.5, ls="--", c="grey", lw=1); ax.set_xticks(x); ax.set_xticklabels(overall["dataset"], rotation=20)
    ax.set_ylabel("AUC (95% bootstrap CI)"); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_auc_by_dataset.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(by_type["fake_type"], by_type["AUC"], yerr=[by_type["AUC"] - by_type["AUC_CI_low"], by_type["AUC_CI_high"] - by_type["AUC"]],
           capsize=4, color=["#55A868", "#C44E52", "#8172B2"])
    ax.axhline(0.5, ls="--", c="grey", lw=1); ax.set_ylabel("AUC vs all real"); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_auc_by_fake_type.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(df[df.label_real == 1]["prob_real"], bins=20, alpha=.6, label="real")
    ax.hist(df[df.label_real == 0]["prob_real"], bins=20, alpha=.6, label="fake")
    ax.set_xlabel("classifier P(real)"); ax.set_ylabel("count"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_prob_hist_wikipropanews.png"), dpi=200)
    print("\nFigures written to", args.out_dir)

if __name__ == "__main__":
    main()
