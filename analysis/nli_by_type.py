"""NLI label distribution per fake type for ALL format-valid generations (report Table 5).

Re-scores every generation with status 'ok' (step 3 only keeps the ones it uses), so the pass
rates show how often each prompt type actually produced a contradiction.

Usage (from new_data/, so that common.py is importable):
  python ../analysis/nli_by_type.py --gen $OUT/generations.jsonl
"""
import argparse, collections, importlib, json, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--nli_model", default="FacebookAI/roberta-large-mnli")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    build = importlib.import_module("03_build_dataset")
    gens = [json.loads(l) for l in open(args.gen, encoding="utf-8") if l.strip()]
    ok = [g for g in gens if g["gen_status"] == "ok"]
    labels = build.nli_scores([(g["target_sentence"], g["fake_sentence"]) for g in ok], args.nli_model)
    tab = collections.defaultdict(collections.Counter)
    for g, (lab, _) in zip(ok, labels):
        tab[g["fake_type"]][lab] += 1
    rows = []
    for t, c in tab.items():
        n = sum(c.values())
        rows.append((t, n, c["CONTRADICTION"] / n, c["NEUTRAL"] / n, c["ENTAILMENT"] / n))
        print(f"{t:22s} n={n:3d}  contradiction={rows[-1][2]:.1%}  neutral={rows[-1][3]:.1%}  entailment={rows[-1][4]:.1%}")
    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("fake_type,n,contradiction,neutral,entailment\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")

if __name__ == "__main__":
    main()
