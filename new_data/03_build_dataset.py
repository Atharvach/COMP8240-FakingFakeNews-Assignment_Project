"""Step 3 - NLI validity filter + assemble the final WikiPropaNews test set.

NLI check (inspired by the paper's use of NLI to make sure generated content is genuinely
inconsistent with the original): premise = original sentence, hypothesis = generated false sentence.
--nli_mode strict keeps only CONTRADICTION; lenient keeps anything that is not ENTAILMENT.

Outputs (in --out_dir):
  wikipropanews_full.jsonl   every example with full provenance/metadata (for analysis + annotation)
  wikipropanews_test.jsonl   model-ready {"id","txt","label"} (label True = real, as in the original repo)
  politifact_test.jsonl      identical copy, so the ORIGINAL test_roberta.py can be pointed at this folder
  stats.json / stats.md      dataset statistics for the report
"""
import argparse, collections, json, os, random, statistics
from common import read_jsonl, write_jsonl, n_words, FAKE_TYPES, REAL, FAKE

def nli_scores(pairs, model_name, batch_size=16):
    from transformers import pipeline
    import torch
    pipe = pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
    res = pipe([{"text": p, "text_pair": h} for p, h in pairs], top_k=None, batch_size=batch_size, truncation=True)
    out = []
    for r in res:
        d = {x["label"].upper(): x["score"] for x in r}
        out.append((max(d, key=d.get), d))
    return out

def build_text(title, body):
    return f"{title}\n\n{body}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_type", type=int, default=100, help="fake articles kept per fake type")
    ap.add_argument("--nli_model", default="FacebookAI/roberta-large-mnli")
    ap.add_argument("--nli_mode", choices=["strict", "lenient"], default="strict")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gens = [g for g in read_jsonl(args.gen) if g["gen_status"] == "ok"]
    print(f"generations with status ok: {len(gens)}")
    scores = nli_scores([(g["target_sentence"], g["fake_sentence"]) for g in gens], args.nli_model)
    for g, (lab, d) in zip(gens, scores):
        g["nli_label"] = lab
        g["nli_contradiction"] = round(d.get("CONTRADICTION", 0.0), 4)
        g["nli_entailment"] = round(d.get("ENTAILMENT", 0.0), 4)
    print("NLI label distribution:", collections.Counter(g["nli_label"] for g in gens))
    passed = [g for g in gens if (g["nli_label"] == "CONTRADICTION" if args.nli_mode == "strict"
                                  else g["nli_label"] != "ENTAILMENT")]

    rng = random.Random(args.seed)
    kept = []
    for t in FAKE_TYPES:
        cand = [g for g in passed if g["fake_type"] == t]
        rng.shuffle(cand)
        if len(cand) < args.per_type:
            print(f"WARNING: only {len(cand)} valid '{t}' fakes (wanted {args.per_type})")
        kept.extend(cand[:args.per_type])

    full, model_ready = [], []
    for g in kept:
        body_real = g["body"]
        insert = g["fake_sentence"] + (" " + g["authority_sentence"] if g["authority_sentence"] else "")
        assert body_real.count(g["target_sentence"]) == 1
        body_fake = body_real.replace(g["target_sentence"], insert, 1)
        common_meta = {k: g[k] for k in ["pair_id", "pageid", "title", "url", "pub_date", "topic", "licence"]}
        real = {**common_meta, "id": g["pair_id"] + "-real", "label": REAL, "txt": build_text(g["title"], body_real),
                "fake_type": "none", "target_sentence": g["target_sentence"], "target_index": g["target_index"]}
        fake = {**common_meta, "id": g["pair_id"] + "-fake", "label": FAKE, "txt": build_text(g["title"], body_fake),
                "fake_type": g["fake_type"], "target_sentence": g["target_sentence"], "target_index": g["target_index"],
                "fake_sentence": g["fake_sentence"], "authority_sentence": g["authority_sentence"],
                "nli_label": g["nli_label"], "nli_contradiction": g["nli_contradiction"],
                "generator": g["generator"], "words_changed_pct": None}
        # share of the fake article's words that were newly generated (PropaNews-style minimal edit)
        fake["words_changed_pct"] = round(100 * n_words(insert) / n_words(body_fake), 2)
        for r in (real, fake):
            r["n_words"] = n_words(r["txt"])
            full.append(r)
            model_ready.append({"id": r["id"], "txt": r["txt"], "label": r["label"]})

    rng.shuffle(model_ready)
    write_jsonl(os.path.join(args.out_dir, "wikipropanews_full.jsonl"), full)
    write_jsonl(os.path.join(args.out_dir, "wikipropanews_test.jsonl"), model_ready)
    write_jsonl(os.path.join(args.out_dir, "politifact_test.jsonl"), model_ready)

    reals = [r for r in full if r["label"] == REAL]
    fakes = [r for r in full if r["label"] == FAKE]
    years = collections.Counter((r["pub_date"] or "")[-4:] or "unknown" for r in reals)
    stats = {
        "n_examples": len(full), "n_real": len(reals), "n_fake": len(fakes),
        "fake_types": dict(collections.Counter(r["fake_type"] for r in fakes)),
        "topics": dict(collections.Counter(r["topic"] for r in reals).most_common()),
        "pub_years": dict(sorted(years.items())),
        "words_mean": round(statistics.mean(r["n_words"] for r in full), 1),
        "words_median": statistics.median(r["n_words"] for r in full),
        "words_min": min(r["n_words"] for r in full), "words_max": max(r["n_words"] for r in full),
        "fake_words_changed_pct_mean": round(statistics.mean(r["words_changed_pct"] for r in fakes), 2),
        "generation_attempted": len(read_jsonl(args.gen)), "generation_ok": len(gens),
        "nli_mode": args.nli_mode, "nli_passed": len(passed),
        "nli_distribution": dict(collections.Counter(g["nli_label"] for g in gens)),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out_dir, "stats.md"), "w") as f:
        f.write("| Statistic | Value |\n|---|---|\n")
        for k, v in stats.items():
            f.write(f"| {k} | {v} |\n")
    print(json.dumps(stats, indent=2))

    ex = next((r for r in fakes if r["fake_type"] == "appeal-to-authority"), fakes[0] if fakes else None)
    if ex:
        print("\nExample for the report figure:\nORIGINAL :", ex["target_sentence"],
              "\nFAKE     :", ex["fake_sentence"], "\nAUTHORITY:", ex["authority_sentence"], "\nURL:", ex["url"])

if __name__ == "__main__":
    main()
