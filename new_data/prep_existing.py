"""Prepare the EXISTING new-to-the-method datasets in the repo's format (label True = REAL).

LIAR (Wang, 2017), test split, 6-way -> binary:
  liar_test_binary.jsonl : true / mostly-true / half-true -> REAL ; barely-true / false / pants-fire -> FAKE
  liar_test_strict.jsonl : drops the ambiguous half-true class (sensitivity analysis)
WELFake (Verma et al., 2021): balanced random sample of full articles (default 1,000).
  The label meaning is checked from the data itself (Reuters wire copy should be REAL), not assumed.

NOTE: if your earlier LIAR run (update presentation) mapped real -> 0, its labels were INVERTED
relative to the repo, where True/1 = real. AUC under inverted labels equals 1 - true AUC.
"""
import argparse, io, os, random, re
import pandas as pd
import requests
from common import write_jsonl, n_words

LIAR_URL = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv"
LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
             "barely_true_c", "false_c", "half_true_c", "mostly_true_c", "pants_fire_c", "context"]
REAL6, FAKE6 = {"true", "mostly-true", "half-true"}, {"barely-true", "false", "pants-fire"}

def prep_liar(out_dir):
    df = pd.read_csv(io.StringIO(requests.get(LIAR_URL, timeout=60).text), sep="\t", header=None,
                     names=LIAR_COLS, quoting=3)
    print("LIAR test rows:", len(df), "\n", df["label"].value_counts().to_string())
    binary = [{"id": f"liar-{r.id}", "txt": str(r.statement), "label": r.label in REAL6, "label6": r.label}
              for r in df.itertuples() if r.label in REAL6 | FAKE6]
    strict = [r for r in binary if r["label6"] != "half-true"]
    write_jsonl(os.path.join(out_dir, "liar_test_binary.jsonl"), binary)
    write_jsonl(os.path.join(out_dir, "liar_test_strict.jsonl"), strict)
    for name, rows in [("binary", binary), ("strict", strict)]:
        n_real = sum(r["label"] for r in rows)
        print(f"LIAR {name}: n={len(rows)} real={n_real} fake={len(rows)-n_real} "
              f"mean words={sum(n_words(r['txt']) for r in rows)/len(rows):.1f}")

def prep_welfake(out_dir, n, seed):
    from datasets import load_dataset
    df = load_dataset("davanstrien/WELFake", split="train").to_pandas()
    df = df.dropna(subset=["text"]).copy()
    df["title"] = df["title"].fillna("")
    df["n_words"] = df["text"].str.split().str.len()
    df = df[(df["n_words"] >= 100) & (df["n_words"] <= 1500)].drop_duplicates(subset=["text"])
    # Data-driven check of label meaning: Reuters wire articles are real news
    reuters = df[df["text"].str.contains(r"\(Reuters\)", regex=True)]
    real_value = int(reuters["label"].mode()[0])
    print(f"WELFake: {len(reuters)} '(Reuters)' articles, {100*(reuters['label']==real_value).mean():.1f}% have "
          f"label={real_value} -> treating label {real_value} as REAL")
    rng = random.Random(seed)
    rows = []
    for lab_value in sorted(df["label"].unique()):
        sub = df[df["label"] == lab_value]
        idx = rng.sample(list(sub.index), n // 2)
        for i in idx:
            r = df.loc[i]
            text = r["text"]
            # remove the "(Reuters)" dateline so the source tag cannot act as a shortcut
            text = re.sub(r"^.{0,80}?\(Reuters\)\s*-\s*", "", text)
            rows.append({"id": f"welfake-{i}", "txt": f"{r['title']}\n\n{text}".strip(), "label": int(lab_value) == real_value})
    rng.shuffle(rows)
    write_jsonl(os.path.join(out_dir, "welfake_sample.jsonl"), rows)
    print(f"WELFake sample: n={len(rows)} real={sum(r['label'] for r in rows)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--welfake_n", type=int, default=1000)
    ap.add_argument("--skip_welfake", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prep_liar(args.out_dir)
    if not args.skip_welfake:
        prep_welfake(args.out_dir, args.welfake_n, args.seed)

if __name__ == "__main__":
    main()
