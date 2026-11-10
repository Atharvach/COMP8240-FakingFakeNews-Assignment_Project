"""Step 5 - Human annotation sample + agreement statistics.

  make_sheet : writes a BLIND, shuffled annotation sheet (default 20 items per fake type + 20 controls = 80).
               Open it in Google Sheets/Excel and fill the four answer columns yourself, without looking
               at the judge output or the hidden key.
  score      : Cohen's kappa (human vs LLM judge) for each yes/no question, quadratic-weighted kappa for
               plausibility, raw % agreement, and how often human / judge recover the DESIGN label
               (what the generator was asked to do).
Usage:
  python 05_annotation_agreement.py make_sheet --judge .../judge.jsonl --out_dir .../annotation
  python 05_annotation_agreement.py score --judge .../judge.jsonl --sheet .../annotation/annotation_sheet_FILLED.csv --key .../annotation/annotation_key.csv
"""
import argparse, os, random
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from common import read_jsonl

QS = ["contradicts", "loaded_language", "appeal_to_authority"]

def norm_yn(v):
    v = str(v).strip().lower()
    if v in {"y", "yes", "1", "true"}:
        return "yes"
    if v in {"n", "no", "0", "false"}:
        return "no"
    return None          # blank / unreadable -> row dropped

def design_label(kind, ftype, q):
    if q == "contradicts":
        return "yes" if kind == "fake" else "no"
    if q == "loaded_language":
        return "yes" if ftype == "loaded-language" else "no"
    if q == "appeal_to_authority":
        return "yes" if ftype == "appeal-to-authority" else "no"

def make_sheet(args):
    rows = [r for r in read_jsonl(args.judge) if r["judge"]]
    rng = random.Random(args.seed)
    sample = []
    for group in ["sent-fake", "loaded-language", "appeal-to-authority", "control"]:
        g = [r for r in rows if (r["kind"] == "control" if group == "control" else r["fake_type"] == group)]
        rng.shuffle(g)
        sample.extend(g[:args.per_group])
    rng.shuffle(sample)
    os.makedirs(args.out_dir, exist_ok=True)
    sheet = pd.DataFrame([{"item_id": r["item_id"], "original_sentence": r["orig"], "edited_passage": r["edited"],
                           "contradicts": "", "loaded_language": "", "appeal_to_authority": "", "plausible": ""}
                          for r in sample])
    key = pd.DataFrame([{"item_id": r["item_id"], "kind": r["kind"], "fake_type": r["fake_type"]} for r in sample])
    sheet.to_csv(os.path.join(args.out_dir, "annotation_sheet.csv"), index=False)
    key.to_csv(os.path.join(args.out_dir, "annotation_key.csv"), index=False)
    print(f"wrote {len(sheet)} items. Fill contradicts/loaded_language/appeal_to_authority with yes/no and plausible with 1/2/3.")

def score(args):
    sheet = pd.read_csv(args.sheet, dtype=str)
    key = pd.read_csv(args.key)
    judge = {r["item_id"]: r["judge"] for r in read_jsonl(args.judge) if r["judge"]}
    df = sheet.merge(key, on="item_id")
    df = df[df["item_id"].isin(judge)].copy()
    for q in QS + ["plausible"]:
        df["judge_" + q] = df["item_id"].map(lambda i: judge[i][q])
    for q in QS:
        df[q] = df[q].map(norm_yn)
        df["design_" + q] = [design_label(k, t, q) for k, t in zip(df["kind"], df["fake_type"])]
    df = df.dropna(subset=QS)
    out = []
    for q in QS:
        out.append({"question": q, "n": len(df),
                    "kappa_human_vs_judge": round(cohen_kappa_score(df[q], df["judge_" + q]), 3),
                    "agreement_human_vs_judge_%": round(100 * (df[q] == df["judge_" + q]).mean(), 1),
                    "human_matches_design_%": round(100 * (df[q] == df["design_" + q]).mean(), 1),
                    "judge_matches_design_%": round(100 * (df["judge_" + q] == df["design_" + q]).mean(), 1),
                    "kappa_human_vs_design": round(cohen_kappa_score(df[q], df["design_" + q]), 3)})
    df["plausible"] = pd.to_numeric(df["plausible"], errors="coerce")
    p = df.dropna(subset=["plausible"])
    out.append({"question": "plausible (1-3)", "n": len(p),
                "kappa_human_vs_judge": round(cohen_kappa_score(p["plausible"].astype(int), p["judge_plausible"].astype(int),
                                                                weights="quadratic"), 3),
                "agreement_human_vs_judge_%": round(100 * (p["plausible"].astype(int) == p["judge_plausible"].astype(int)).mean(), 1),
                "human_matches_design_%": None, "judge_matches_design_%": None, "kappa_human_vs_design": None})
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    print("\nMean human plausibility by item type:\n",
          p.assign(pl=p["plausible"].astype(int)).groupby("fake_type")["pl"].mean().round(2).to_string())
    res.to_csv(os.path.join(os.path.dirname(args.sheet), "agreement_results.csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make_sheet"); a.add_argument("--judge", required=True); a.add_argument("--out_dir", required=True)
    a.add_argument("--per_group", type=int, default=20); a.add_argument("--seed", type=int, default=7)
    b = sub.add_parser("score"); b.add_argument("--judge", required=True); b.add_argument("--sheet", required=True)
    b.add_argument("--key", required=True)
    args = ap.parse_args()
    make_sheet(args) if args.cmd == "make_sheet" else score(args)

if __name__ == "__main__":
    main()
