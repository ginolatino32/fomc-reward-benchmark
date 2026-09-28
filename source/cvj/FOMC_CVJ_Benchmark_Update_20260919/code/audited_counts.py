"""Audited, independent reconstruction of Appendix B; NOT original-author code.

The uploaded reconstruction is retained in sources/. Corrections are fixed from
Appendix B and synthetic code tests, not selected using market outcomes.
Remaining departures: explicit local stop-word/descriptive lists, whole-document
scope, and stated ambiguity resolution. No semantic accuracy labels are claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uploaded_reconstruction", ROOT / "sources/replicate_author_method_counts.py")
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
STOCK_PHRASES = list(legacy.STOCK_PHRASES)
NEGATIVE = ["adjust* downward" if x == "adjust*" else x for x in legacy.NEGATIVE_DIRECTIONS]
NEGATIVE.insert(NEGATIVE.index("slow* down"), "slow*")
POSITIVE = list(legacy.POSITIVE_DIRECTIONS)
POSITIVE.insert(POSITIVE.index("strength*"), "stop decline")
PREFIXES = ["rate* of", "growth of", "level* of", "index* of", "indices of"]
NOUNS = STOCK_PHRASES + [f"{p} {n}" for p in PREFIXES for n in STOCK_PHRASES]
STOP_WORDS = set(legacy.STOP_WORDS)
BOUNDARIES = set(legacy.SUBSENTENCE_BOUNDARIES)
DESC = [("somewhat", False), ("unusual", True), ("remarkabl", True), ("much", False), ("rapid", True), ("experience", True), ("show", False), ("register", True)]
EXCLUDED_AFTER = {"reduced", "reduce", "reducing", "boosted", "boost", "boosting", "fostered", "foster", "fostering", "encouraged", "encourage"}
LEX = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?|[,;.!?\n]")

@dataclass(frozen=True)
class Tok:
    text: str
    start: int
    end: int

@dataclass(frozen=True)
class Hit:
    start: int
    end: int
    label: str
    sign: str = ""
    words: int = 0
    negated: bool = False

def clean(text: str) -> str:
    # Appendix's printed pattern is "s p 500 index". Offsets refer to cleaned text.
    text = re.sub(r"\bs\s*&\s*p\b", "s p", text, flags=re.I)
    text = legacy.clean_text(text)
    text = re.sub(r"\b([A-Z])\.(?=\s+[A-Z][a-z])", r"\1", text)
    return text

def pattern(phrase: str) -> tuple[str, ...]:
    return legacy.normalize_pattern(phrase)

def matches(p: str, word: str) -> bool:
    return word.startswith(p[:-1]) if p.endswith("*") else p == word

def is_barrier(t: Tok) -> bool:
    return t.text in BOUNDARIES or t.text in {",", ";", ".", "!", "?", "\n"}

def removable(t: Tok) -> bool:
    if t.text == "not" or is_barrier(t):
        return False
    return t.text in STOP_WORDS or any(t.text.startswith(p) if wild else t.text == p for p, wild in DESC)

def match_from(tokens: list[Tok], i: int, pats: tuple[str, ...]) -> int | None:
    """Match explicit internal connectors; otherwise never cross a boundary.
    Local stop/descriptive words may be removed within a multiword expression.
    The first token is not skipped, so source spans remain interpretable.
    """
    pos = i
    for j, p in enumerate(pats):
        if j:
            while pos < len(tokens) and not matches(p, tokens[pos].text) and removable(tokens[pos]):
                pos += 1
        if pos >= len(tokens) or not matches(p, tokens[pos].text):
            return None
        pos += 1
    return pos

def nonoverlap(candidates: list[Hit]) -> list[Hit]:
    chosen: list[Hit] = []
    for h in sorted(candidates, key=lambda z: (-z.words, -(z.end-z.start), z.start, z.label)):
        if not any(h.start < q.end and h.end > q.start for q in chosen):
            chosen.append(h)
    return sorted(chosen, key=lambda z: z.start)

NP = [(pattern(x), x) for x in NOUNS]
DP = [(pattern(x), x, "negative") for x in NEGATIVE] + [(pattern(x), x, "positive") for x in POSITIVE]

def extract(text: str) -> list[dict]:
    """Return every noun hit, including unclassified/ambiguous audit rows."""
    toks = [Tok(m.group(0).lower(), m.start(), m.end()) for m in LEX.finditer(text)]
    nc: list[Hit] = []
    dc: list[Hit] = []
    for i, t in enumerate(toks):
        for pats, label in NP:
            if matches(pats[0], t.text):
                end = match_from(toks, i, pats)
                if end is not None:
                    nc.append(Hit(i, end, label, words=len(pats)))
        for pats, label, sign in DP:
            if matches(pats[0], t.text):
                end = match_from(toks, i, pats)
                if end is not None:
                    dc.append(Hit(i, end, label, sign, len(pats)))
    nouns, directions = nonoverlap(nc), nonoverlap(dc)
    # Attach "not" to its following direction, before checking noun adjacency.
    expanded: list[Hit] = []
    for h in directions:
        j = h.start - 1
        while j >= 0 and removable(toks[j]):
            j -= 1
        if j >= 0 and toks[j].text == "not":
            expanded.append(Hit(j, h.end, "not " + h.label, "positive" if h.sign == "negative" else "negative", h.words + 1, True))
        else:
            expanded.append(h)
    out = []
    for noun in nouns:
        ds = []
        for h in expanded:
            if h.end <= noun.start:
                gap = toks[h.end:noun.start]
            elif noun.end <= h.start:
                gap = toks[noun.end:h.start]
                # Apply the exact finite list, not the whole stem family.
                verb_start = h.start + 1 if h.negated else h.start
                if toks[verb_start].text in EXCLUDED_AFTER:
                    continue
            else:
                continue
            if all(removable(t) for t in gap):
                ds.append(h)
        signs = {h.sign for h in ds}
        sign = next(iter(signs)) if len(signs) == 1 else ("ambiguous" if signs else "unclassified")
        a, b = toks[noun.start].start, toks[noun.end-1].end
        out.append({"phrase": noun.label, "matched_text": text[a:b], "start": a, "end": b,
                    "sign": sign, "direction": " | ".join(h.label for h in ds),
                    "direction_spans": json.dumps([[toks[h.start].start,toks[h.end-1].end] for h in ds]),
                    "context": re.sub(r"\s+", " ", text[max(0,a-110):min(len(text),b+150)])})
    return out

def run(minutes_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = legacy.load_minutes(minutes_root)
    if not docs:
        raise ValueError(f"No minutes found under {minutes_root}/Minutes")
    meetings, rows = [], []
    for idx, doc in enumerate(docs):
        # Read original text to normalize S&P before the uploaded & substitution.
        text = clean(Path(doc.path).read_text(encoding="utf-8"))
        hits = extract(text)
        counts = Counter(h["sign"] for h in hits)
        meetings.append({"meeting_date": str(doc.date), "negative_count": counts["negative"], "positive_count": counts["positive"],
                         "classified_total": counts["negative"]+counts["positive"], "noun_hits_total": len(hits),
                         "ambiguous_count": counts["ambiguous"], "unclassified_count": counts["unclassified"],
                         "scope": "whole_document", "cleaned_sha256": hashlib.sha256(text.encode()).hexdigest()})
        for j, h in enumerate(hits):
            rows.append({"meeting_date": str(doc.date), "hit_id": f"{doc.date}_{j:04}", **h})
        if (idx+1)%25 == 0:
            print("audited documents", idx+1, flush=True)
    pd.DataFrame(meetings).sort_values("meeting_date").to_csv(out_dir/'meeting_counts.csv',index=False)
    pd.DataFrame(rows).sort_values(['meeting_date','start']).to_csv(out_dir/'all_phrase_audit.csv',index=False)
    dictionary = {"noun_patterns": STOCK_PHRASES, "prefixes": PREFIXES, "negative_patterns": NEGATIVE,
                  "positive_patterns": POSITIVE, "stop_words": sorted(STOP_WORDS), "descriptive_patterns": DESC,
                  "postposed_exclusions": sorted(EXCLUDED_AFTER),
                  "provenance": "Independent reconstruction of Appendix B. Explicit local stop list retained; historical package version unavailable."}
    (out_dir/'dictionary.json').write_text(json.dumps(dictionary,indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--minutes-root',type=Path,default=ROOT/'data/minutes')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/audited_reconstruction')
    args=p.parse_args();run(args.minutes_root,args.output_dir)
