"""Step 4 - LLM-as-judge validation of the constructed data.

A DIFFERENT model family from the generator (default judge: microsoft/Phi-3.5-mini-instruct; generator
was Qwen) to reduce self-preference bias. For every pair the judge sees the ORIGINAL sentence and an
EDITED passage and answers four questions. Two kinds of item are judged:
  fake    : edited passage = generated false sentence (+ authority quote)
  control : edited passage = the original sentence unchanged (taken from the paired REAL article)
Controls give every question a 'no' class, which the agreement statistics in step 5 need.

Usage: python 04_llm_judge.py --full .../wikipropanews_full.jsonl --out .../judge.jsonl
"""
import argparse, json, re, collections
from common import read_jsonl, append_jsonl, REAL

PROMPT = """You are checking items in a research dataset of news sentences.

ORIGINAL SENTENCE (from a real news article):
{orig}

EDITED PASSAGE (replaces the original sentence in the article):
{edited}

Answer these questions about the EDITED PASSAGE:
1. contradicts: Does it make a factual claim that contradicts or is clearly inconsistent with the ORIGINAL SENTENCE? (yes/no)
2. loaded_language: Does it use emotionally loaded or charged wording intended to provoke a reaction, beyond neutral news style? (yes/no)
3. appeal_to_authority: Does it cite an authority figure, expert or official (e.g. via a quotation) to lend credibility to its claim, where the original does not? (yes/no)
4. plausible: How plausible would it look to an ordinary reader as part of a real news article? (1 = obviously fake, 2 = somewhat plausible, 3 = fully plausible)

Respond with ONLY a JSON object, e.g. {{"contradicts": "yes", "loaded_language": "no", "appeal_to_authority": "no", "plausible": 3}}"""

def items_from_full(full):
    items = []
    for r in full:
        if r["label"] == REAL:
            items.append({"item_id": r["pair_id"] + "-control", "pair_id": r["pair_id"], "kind": "control",
                          "fake_type": "none", "orig": r["target_sentence"], "edited": r["target_sentence"]})
        else:
            edited = r["fake_sentence"] + (" " + r["authority_sentence"] if r.get("authority_sentence") else "")
            items.append({"item_id": r["pair_id"] + "-fake", "pair_id": r["pair_id"], "kind": "fake",
                          "fake_type": r["fake_type"], "orig": r["target_sentence"], "edited": edited})
    return items

def parse_json(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ["contradicts", "loaded_language", "appeal_to_authority"]:
        v = str(d.get(k, "")).strip().lower()
        out[k] = "yes" if v.startswith("y") else ("no" if v.startswith("n") else None)
    try:
        out["plausible"] = int(d.get("plausible"))
    except (TypeError, ValueError):
        out["plausible"] = None
    return out if None not in out.values() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto",
                                                 attn_implementation="eager")
    model.eval()

    done = {r["item_id"] for r in read_jsonl(args.out)}
    items = [it for it in items_from_full(read_jsonl(args.full)) if it["item_id"] not in done]
    print(f"items to judge: {len(items)} (already done {len(done)})")

    def run(batch):
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(orig=i["orig"], edited=i["edited"])}],
                                           tokenize=False, add_generation_prompt=True) for i in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

    for b in range(0, len(items), args.batch_size):
        batch = items[b:b + args.batch_size]
        for it, raw in zip(batch, run(batch)):
            parsed = parse_json(raw)
            if parsed is None:                      # one retry, alone
                raw = run([it])[0]
                parsed = parse_json(raw)
            append_jsonl(args.out, {**it, "judge_raw": raw, "judge": parsed, "judge_model": args.model})
        print(f"  {min(b + args.batch_size, len(items))}/{len(items)} judged")

    rows = read_jsonl(args.out)
    ok = [r for r in rows if r["judge"]]
    print(f"parsed {len(ok)}/{len(rows)}")
    for kind in ["control"] + ["sent-fake", "loaded-language", "appeal-to-authority"]:
        sub = [r for r in ok if (r["kind"] == "control" if kind == "control" else r["fake_type"] == kind)]
        if not sub:
            continue
        rate = lambda k: sum(r["judge"][k] == "yes" for r in sub) / len(sub)
        print(f"{kind:22s} n={len(sub):3d} contradicts={rate('contradicts'):.2f} loaded={rate('loaded_language'):.2f} "
              f"authority={rate('appeal_to_authority'):.2f} plausible_mean={sum(r['judge']['plausible'] for r in sub)/len(sub):.2f}")

if __name__ == "__main__":
    main()
