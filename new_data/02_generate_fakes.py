"""Step 2 - Generate PropaNews-style fake versions of real Wikinews articles.

Mirrors the three fake-article types in PropaNews (train/dev/test path suffixes):
  sent-fake            -> one salient sentence replaced by a plausible but FALSE claim
  loaded-language      -> the false sentence is also written with emotionally loaded wording
  appeal-to-authority  -> the false sentence plus an inserted quote from an (unnamed) authority supporting it
Difference from the paper (deliberate, discussed in the report): the paper used a fine-tuned BART
generator with NLI-guided self-critical training; we prompt an off-the-shelf open-weights instruction
model (default Qwen2.5-7B-Instruct, 4-bit) and apply an NLI filter afterwards (step 3).

Only ONE sentence (plus one inserted sentence for appeal-to-authority) is changed per article, as in
PropaNews, so real/fake differ minimally and the detector cannot rely on topic or length.

Usage (Colab GPU):
  python 02_generate_fakes.py --raw .../wikinews_raw.jsonl --out .../generations.jsonl --pool_size 540
Resumable: rows already in --out are skipped.
"""
import argparse, random, re, collections
from common import read_jsonl, append_jsonl, choose_target_sentence, FAKE_TYPES

SYSTEM = ("You are assisting a university research project that builds a labelled test set for "
          "evaluating fake-news DETECTORS. The edited text is never published; it is stored with a FAKE label.")

INSTR = {
"sent-fake": (
"Rewrite the TARGET SENTENCE so that it states a specific, plausible but FALSE claim that contradicts "
"the original (for example change a number, an outcome, who did what, where, or why). Keep a neutral, "
"factual news style and roughly the same length. Do not add opinions or emotional words. "
"Output ONLY the rewritten sentence."),
"loaded-language": (
"Rewrite the TARGET SENTENCE so that it states a specific, plausible but FALSE claim that contradicts "
"the original (for example change a number, an outcome, who did what, where, or why), and write it "
"using LOADED LANGUAGE: emotionally charged words meant to provoke a reaction (e.g. 'shocking', "
"'reckless', 'devastating', 'slammed'). Keep roughly the same length. Output ONLY the rewritten sentence."),
"appeal-to-authority": (
"Line 1: rewrite the TARGET SENTENCE so that it states a specific, plausible but FALSE claim that "
"contradicts the original, in a neutral factual news style.\n"
"Line 2: write ONE new sentence containing a direct quotation that supports the false claim, attributed "
"to a credible-sounding but UNNAMED authority relevant to the story (e.g. 'a senior official at the "
"transport ministry', 'an independent economist'). Never name a real person as the source of the quote.\n"
"Output exactly two lines and nothing else."),
}

BAD_PREFIX = re.compile(r"^(here('s| is)[^:]*:|rewritten( target)? sentence:|line \d:|sentence:|output:)\s*", re.I)
REFUSAL = re.compile(r"\b(I can(no|')t|I am unable|as an AI|I won't|I will not)\b", re.I)

def build_prompt(tok, art, ftype, context_words=250):
    context = " ".join(art["body"].split()[:context_words])
    user = (f"ARTICLE TITLE: {art['title']}\n\nARTICLE CONTEXT (beginning):\n{context}\n\n"
            f"TARGET SENTENCE:\n{art['target_sentence']}\n\nTASK:\n{INSTR[ftype]}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def clean_line(s):
    s = BAD_PREFIX.sub("", s.strip()).strip()
    if len(s) > 1 and s[0] in "\"\u201c'" and s[-1] in "\"\u201d'" and s.count('"') <= 2:
        s = s[1:-1].strip()
    return s

def postprocess(raw, art, ftype):
    lines = [clean_line(l) for l in raw.strip().split("\n") if clean_line(l)]
    if not lines:
        return None, None, "empty"
    if REFUSAL.search(raw):
        return None, None, "refusal"
    fake = lines[0]
    auth = None
    if ftype == "appeal-to-authority":
        if len(lines) < 2:
            return None, None, "missing_authority_line"
        auth = lines[1]
        if not any(q in auth for q in ['"', "\u201c", "\u201d"]):
            return None, None, "authority_without_quote"
    orig = art["target_sentence"]
    if fake.strip().lower() == orig.strip().lower():
        return None, None, "unchanged"
    ratio = len(fake.split()) / max(1, len(orig.split()))
    if ratio < 0.5 or ratio > 2.0:
        return None, None, f"length_ratio_{ratio:.2f}"
    if fake[-1] not in ".!\"\u201d'":
        fake = fake + "."
    return fake, auth, "ok"

def select_pool(raw, pool_size, seed):
    """Topic-stratified round-robin sample so no single topic dominates; fake types assigned in rotation."""
    rng = random.Random(seed)
    by_topic = collections.defaultdict(list)
    for a in raw:
        sent, idx = choose_target_sentence(a["body"], rng)
        if sent:
            by_topic[a["topic"]].append({**a, "target_sentence": sent, "target_index": idx})
    for t in by_topic:
        rng.shuffle(by_topic[t])
    topics = sorted(by_topic, key=lambda t: -len(by_topic[t]))
    pool = []
    while len(pool) < pool_size and any(by_topic.values()):
        for t in topics:
            if by_topic[t] and len(pool) < pool_size:
                pool.append(by_topic[t].pop())
    rng.shuffle(pool)
    for i, a in enumerate(pool):
        a["fake_type"] = FAKE_TYPES[i % 3]
        a["pair_id"] = f"wpn-{a['pageid']}"
    return pool

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool_size", type=int, default=540, help="articles to attempt (over-generate; step 3 filters)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_type", choices=FAKE_TYPES, default=None,
                    help="top-up mode: give every NOT-yet-generated article this fake type")
    ap.add_argument("--dry_run", action="store_true", help="only build the pool and print 3 prompts")
    args = ap.parse_args()

    raw = read_jsonl(args.raw)
    pool = select_pool(raw, args.pool_size, args.seed)
    print(f"raw articles: {len(raw)} | pool: {len(pool)} | topics: {collections.Counter(a['topic'] for a in pool)}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.dry_run:
        for a in pool[:3]:
            print("=" * 80, a["fake_type"], "\n", build_prompt(tok, a, a["fake_type"]))
        return

    qcfg = None if args.no_4bit else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qcfg,
                                                 torch_dtype=torch.float16, device_map="auto")
    model.eval()
    torch.manual_seed(args.seed)

    done = {r["pair_id"] for r in read_jsonl(args.out)}
    todo = [a for a in pool if a["pair_id"] not in done]
    if args.force_type:
        for a in todo:
            a["fake_type"] = args.force_type
    print(f"already done: {len(done)} | to generate: {len(todo)}")

    def generate(batch, temperature):
        prompts = [build_prompt(tok, a, a["fake_type"]) for a in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=140, do_sample=True, temperature=temperature,
                                 top_p=0.9, pad_token_id=tok.pad_token_id)
        return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

    for b in range(0, len(todo), args.batch_size):
        batch = todo[b:b + args.batch_size]
        outs = generate(batch, 0.7)
        for a, raw_out in zip(batch, outs):
            fake, auth, status = postprocess(raw_out, a, a["fake_type"])
            attempts = 1
            if status != "ok":                      # one retry at a different temperature
                raw_out = generate([a], 0.9)[0]
                fake, auth, status = postprocess(raw_out, a, a["fake_type"])
                attempts = 2
            rec = {k: a[k] for k in ["pair_id", "pageid", "title", "url", "pub_date", "topic", "body",
                                     "n_words", "licence", "target_sentence", "target_index", "fake_type"]}
            rec.update({"fake_sentence": fake, "authority_sentence": auth, "raw_output": raw_out,
                        "gen_status": status, "attempts": attempts, "generator": args.model})
            append_jsonl(args.out, rec)
        print(f"  {min(b + args.batch_size, len(todo))}/{len(todo)} generated")
    rows = read_jsonl(args.out)
    print("status counts:", collections.Counter(r["gen_status"] for r in rows))

if __name__ == "__main__":
    main()
