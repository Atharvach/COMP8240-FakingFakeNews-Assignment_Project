# Dataset card: WikiPropaNews (COMP8240 research artefact)

**What it is.** 594 English news articles: 297 real articles from English Wikinews (published 2020 to May 2026)
and 297 paired fakes. Each fake replaces one salient sentence of its real article with an NLI-verified false claim,
optionally written with loaded language (97) or followed by a quotation from an unnamed authority (100);
the other 100 contain a plain false claim. Built for evaluating fake-news detectors.

**Files.** `wikipropanews_test.jsonl` (`id`, `txt`, `label`; `label: true` = real, following FakingFakeNews);
`wikipropanews_full.jsonl` (adds source URL, publication date, topic, fake type, original and edited sentence,
NLI label and generator); `judge.jsonl` (LLM-judge labels); `annotation/` (human annotation sample).

**Licence and attribution.** Real-article text: Wikinews contributors, CC BY 2.5 (https://en.wikinews.org).
Each record links to its source article. Edited text is released under the same licence.

**Intended use.** Research on detecting disinformation only.

**Do not** republish individual fake articles as news, or strip their labels. Fake articles contain deliberately
false statements, some involving real public figures. Generated quotations are attributed only to unnamed authorities.

**Known limitations.** Single generator (Qwen2.5-7B-Instruct) and single NLI filter (roberta-large-mnli);
loaded-language fakes skew towards politics; human validation by one annotator, with low agreement on loaded language.
