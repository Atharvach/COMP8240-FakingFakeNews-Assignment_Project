"""Shared helpers for the WikiPropaNews construction pipeline (COMP8240 final project).

Label convention (IMPORTANT): this follows the FakingFakeNews repository exactly.
In the original data files, label == True means REAL news and label == False means FAKE.
The RoBERTa classifier's sigmoid output is therefore P(real), and the repo's F1 is computed
with REAL as the positive class. Every file produced here uses the same convention.
"""
import json, os, re, random

REAL, FAKE = True, False
FAKE_TYPES = ["sent-fake", "loaded-language", "appeal-to-authority"]
# Mirrors PropaNews' three kinds of fake article (file suffixes .sent-fake, .ll_filled, .app-auth-final)

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def to_bool_label(x):
    """Accept True/False, 1/0, 'true'/'false' and return a Python bool (True = real)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    return str(x).strip().lower() in {"true", "1", "real"}

def n_words(text):
    return len(text.split())

_QUOTE_CHARS = ['"', "\u201c", "\u201d", "''"]

def split_sentences(paragraph):
    """Sentence splitter: NLTK punkt if available, regex fallback otherwise."""
    try:
        import nltk
        try:
            return [s.strip() for s in nltk.sent_tokenize(paragraph) if s.strip()]
        except LookupError:
            for res in ("punkt_tab", "punkt"):
                try:
                    nltk.download(res, quiet=True)
                except Exception:
                    pass
            return [s.strip() for s in nltk.sent_tokenize(paragraph) if s.strip()]
    except Exception:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\u201c"])', paragraph)
        return [p.strip() for p in parts if p.strip()]

def salience(sentence):
    """Simple salience proxy: numbers and named-entity-like capitalised tokens carry checkable facts."""
    toks = sentence.split()
    nums = sum(1 for t in toks if re.search(r"\d", t))
    caps = sum(1 for t in toks[1:] if t[:1].isupper())
    return 2 * nums + caps

def choose_target_sentence(body, rng, max_index=10, min_words=12, max_words=50):
    """Pick the most salient checkable sentence (not the first, no direct quotes).
    Returns (sentence, global_index) or (None, None)."""
    sents = []
    for para in body.split("\n\n"):
        sents.extend(split_sentences(para))
    cands = []
    for i, s in enumerate(sents):
        if i == 0 or i > max_index:
            continue
        w = n_words(s)
        if w < min_words or w > max_words:
            continue
        if any(q in s for q in _QUOTE_CHARS) or s.endswith("?") or not s[:1].isupper():
            continue
        if body.count(s) != 1:          # must be uniquely replaceable
            continue
        cands.append((salience(s), rng.random(), i, s))
    if not cands:
        return None, None
    cands.sort(reverse=True)
    _, _, i, s = cands[0]
    return s, i
