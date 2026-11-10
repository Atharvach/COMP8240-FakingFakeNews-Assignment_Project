"""Step 1 - Collect REAL news articles from English Wikinews (CC BY 2.5).

Why Wikinews: openly licensed (CC BY 2.5, so the dataset can be shared publicly with attribution),
reviewed before publication, dated, topically diverse, and the archive is frozen (read-only since
4 May 2026), so the collection is reproducible. Default window 2022..2026 post-dates
both PropaNews' source news (TIMELINE17, ~2009-2013) and RoBERTa's pre-training corpus (2019).

Usage (Colab):
  python 01_collect_wikinews.py --out /content/drive/MyDrive/fakenewsproject/new_data_out/wikinews_raw.jsonl
Resumable: articles already in --out are skipped.
"""
import argparse, re, time, sys
import requests
from bs4 import BeautifulSoup
from common import read_jsonl, append_jsonl, n_words

API = "https://en.wikinews.org/w/api.php"
TOPICS = ["Politics and conflicts", "Crime and law", "Disasters and accidents", "Economy and business",
          "Health", "Science and technology", "Environment", "Culture and entertainment", "Sports",
          "Weather", "Education", "Transport"]
STOP_HEADINGS = {"sources", "related news", "related articles", "see also", "external links",
                 "references", "sister links", "interview"}
SKIP_TITLE = re.compile(r"^(Wikinews Shorts|Wikinews interviews|Wikinews Interviews|Today in History)", re.I)
SKIP_CATS = {"Interviews", "Wikinews Shorts", "Opinion", "Satire"}
DATE_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
                     r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                     r"\s+\d{1,2},\s+\d{4}")

def api_get(session, params, tries=4):
    params = {**params, "format": "json", "formatversion": 2}
    for k in range(tries):
        try:
            r = session.get(API, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** k
            print(f"  request failed ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"API failed for {params}")

def list_published(session):
    """All mainspace pages in Category:Published. Category-link timestamps on Wikinews do not reflect
    publication dates, so dates are taken from each article itself (see article_year)."""
    params = {"action": "query", "list": "categorymembers", "cmtitle": "Category:Published",
              "cmnamespace": 0, "cmtype": "page", "cmlimit": 500, "cmprop": "ids|title|timestamp"}
    out = []
    while True:
        data = api_get(session, params)
        out.extend(data.get("query", {}).get("categorymembers", []))
        if "continue" not in data:
            break
        params.update(data["continue"])
    return out

def clean_html(html):
    """Return (pub_date_string_or_None, body_text) from rendered article HTML."""
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(".mw-parser-output") or soup
    for bad in root.select("table, figure, .thumb, .infobox, sup.reference, .reference, style, script, "
                           ".mw-editsection, .noprint, .metadata, .ambox, .mbox-small, .navbox"):
        bad.decompose()
    pub_date, paras = None, []
    for el in root.find_all(["p", "h2", "h3"]):
        text = re.sub(r"\s+", " ", el.get_text("")).strip()
        if el.name in ("h2", "h3"):
            if text.lower().strip(" :") in STOP_HEADINGS:
                break
            continue
        text = re.sub(r"\[\d+\]", "", text).strip()
        if not text:
            continue
        m = DATE_RE.match(text)
        if m:
            pub_date = pub_date or m.group(0)
            text = text[m.end():].strip(" -\u2013\u2014")
            if not text:
                continue
        paras.append(text)
    return pub_date, "\n\n".join(paras)

def first_revision_year(session, pageid):
    data = api_get(session, {"action": "query", "prop": "revisions", "pageids": pageid,
                             "rvlimit": 1, "rvdir": "newer", "rvprop": "timestamp"})
    try:
        return int(data["query"]["pages"][0]["revisions"][0]["timestamp"][:4])
    except (KeyError, IndexError, ValueError):
        return None

def article_year(session, pageid, pub_date):
    """Publication year from the article's own dateline, else from its first revision."""
    if pub_date:
        return int(pub_date[-4:]), "dateline"
    return first_revision_year(session, pageid), "first_revision"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-05-04")
    ap.add_argument("--min_words", type=int, default=150)
    ap.add_argument("--max_words", type=int, default=1500)
    ap.add_argument("--max_articles", type=int, default=700)
    ap.add_argument("--stop_after_old", type=int, default=300,
                    help="stop after this many consecutive articles older than --start (pages are scanned newest first)")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--contact", default="student project",
                    help="put your email here; Wikimedia asks API clients to identify themselves")
    args = ap.parse_args()

    s = requests.Session()
    s.headers["User-Agent"] = f"COMP8240-WikiPropaNews/1.0 (Macquarie University research project; {args.contact})"

    start_y, end_y = int(args.start[:4]), int(args.end[:4])
    members = list_published(s)
    # higher page ids were created later, so scanning in descending id order goes newest -> oldest
    members.sort(key=lambda m: m["pageid"], reverse=True)
    print(f"Category:Published pages: {len(members)} (scanning newest first, keeping {start_y}-{end_y})")
    if not members:
        sys.exit("No articles returned - check the network.")

    done = {r["pageid"] for r in read_jsonl(args.out)}
    skipped = {"title": 0, "category": 0, "length": 0, "error": 0, "out_of_window": 0}
    kept, old_streak = len(done), 0
    for i, m in enumerate(members):
        if kept >= args.max_articles:
            print("Reached --max_articles.")
            break
        if old_streak >= args.stop_after_old:
            print(f"{old_streak} consecutive articles older than {start_y}: stopping.")
            break
        if m["pageid"] in done:
            continue
        if SKIP_TITLE.search(m["title"]):
            skipped["title"] += 1
            continue
        try:
            data = api_get(s, {"action": "parse", "pageid": m["pageid"], "prop": "text|categories",
                               "redirects": 1, "disablelimitreport": 1, "disableeditsection": 1})
            p = data["parse"]
        except Exception as e:
            skipped["error"] += 1
            continue
        cats = [c["category"].replace("_", " ") for c in p.get("categories", []) if not c.get("hidden")]
        if any(c in SKIP_CATS for c in cats):
            skipped["category"] += 1
            continue
        pub_date, body = clean_html(p["text"])
        year, year_source = article_year(s, m["pageid"], pub_date)
        if year is None or year < start_y or year > end_y:
            skipped["out_of_window"] += 1
            old_streak = old_streak + 1 if (year is not None and year < start_y) else old_streak
            continue
        old_streak = 0
        w = n_words(body)
        if w < args.min_words or w > args.max_words:
            skipped["length"] += 1
            continue
        topic = next((t for t in TOPICS if t in cats), "Other")
        append_jsonl(args.out, {
            "pageid": m["pageid"], "title": p["title"],
            "url": "https://en.wikinews.org/wiki/" + p["title"].replace(" ", "_"),
            "pub_date": pub_date, "pub_year": year, "year_source": year_source,
            "topic": topic, "categories": cats, "body": body, "n_words": w,
            "licence": "CC BY 2.5 (Wikinews contributors)"})
        kept += 1
        if kept % 25 == 0:
            print(f"  kept {kept} (scanned {i+1}/{len(members)}) skipped={skipped}")
        time.sleep(args.sleep)
    print(f"Done. Articles kept: {kept}. Skipped: {skipped}")

if __name__ == "__main__":
    main()
