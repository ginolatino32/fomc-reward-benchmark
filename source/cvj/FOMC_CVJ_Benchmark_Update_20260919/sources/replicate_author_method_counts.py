"""Reconstruct the Cieslak-Vissing-Jorgensen appendix stock-market algorithm.

This is a transparent reconstruction from the published online appendix. It is
not author-supplied code. The output counts only positive and negative matches
under the appendix algorithm; neutral and hypothetical mentions are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd


APPENDIX_URL = "https://back.nber.org/appendix/w26894/w26894appendix.pdf"


STOCK_PHRASES = [
    "asset index*", "asset indic*", "asset market*", "asset price index*",
    "asset price indic*", "asset price*", "asset valu*", "equities",
    "equity and home price*", "equity and home valu*", "equity and house price*",
    "equity and housing price*", "equity index*", "equity indic*",
    "equity market index*", "equity market indic*", "equity market price*",
    "equity market valu*", "equity market*", "equity price index*",
    "equity price indic*", "equity price measure*", "equity price*",
    "equity valu*", "financial wealth", "home and equity price*",
    "house and equity price*", "household wealth", "household* net worth",
    "housing and equity price*", "price* of risk* asset*", "ratio of wealth to income",
    "risk* asset price*", "s p 500 index", "stock index*", "stock indic*",
    "stock market index*", "stock market price*", "stock market wealth",
    "stock market*", "stock price indic*", "stock price*", "stock prices index*",
    "stock val*", "us stock market price*", "wealth effect*", "wealth to income ratio",
]

# The appendix describes additional word combinations formed by prefixing the
# stock-market noun phrases with these expressions.
WORD_COMBINATION_PREFIXES = ["rate of", "growth of", "level of", "index of", "indices of"]
STOCK_PHRASE_VARIANTS = [(phrase, phrase) for phrase in STOCK_PHRASES]
STOCK_PHRASE_VARIANTS += [
    (f"{prefix} {phrase}", f"{prefix} {phrase}")
    for prefix in WORD_COMBINATION_PREFIXES
    for phrase in STOCK_PHRASES
]

# Group 1 in Appendix Table 8 is negative; group 2 is positive.
NEGATIVE_DIRECTIONS = [
    "adjust*", "adverse", "burst*", "contract*", "cool*", "deceler*",
    "declin*", "decreas*", "deteriorat*", "down", "downturn", "downward",
    "downward adjust*", "downward movement", "downward revision", "drop*",
    "eas*", "edge* down", "fall*", "fell", "go* down", "limit*", "low*",
    "moderate*", "moderati*", "mov* down", "mov* downward", "mov* lower",
    "plummet*", "pressure*", "pull* back", "pullback", "reduc*",
    "revis* down*", "slow* down", "soft*", "stagnate*", "stall*", "strain*",
    "stress*", "subdu*", "take* toll on", "tension*", "tick* down", "tight*",
    "took toll on", "tumbl*", "weak*", "weigh* on", "went down", "worse*",
]

POSITIVE_DIRECTIONS = [
    "acceler*", "adjust* upward", "advanc*", "bolster*", "boost*", "edge* up",
    "elevat*", "encourag*", "expand*", "fast*", "favor*", "gain*", "go* up",
    "high*", "improv*", "increas*", "mov* high*", "mov* up", "mov* upward",
    "pick* up", "rais*", "rallied", "rally*", "rebound*", "recoup*",
    "revis* up*", "rise*", "rising", "rose", "run up", "runup", "strength*",
    "strong*", "tick* up", "up", "upward", "upward adjust*", "upward movement",
    "upward revision", "went up",
]

SUBSENTENCE_BOUNDARIES = {
    "and", "as", "or", "to", "of", "after", "because", "but", "from", "if",
    "so", "when", "where", "while", "although", "however", "though", "whereas",
    "despite",
}

# The appendix removes ordinary stop words and selected descriptive words before
# testing distance. Keep direction and phrase vocabulary even when a word is a
# general stop word, such as "up".
STOP_WORDS = {
    "a", "an", "the", "are", "am", "is", "was", "were", "be", "been", "being",
    "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "that",
    "to", "with", "it", "its", "this", "these", "those", "their", "there", "then",
    "than", "which", "who", "whom", "what", "where", "when", "how", "had", "has",
    "have", "having", "do", "does", "did", "but", "if", "so", "because", "about",
    "after", "before", "during", "through", "over", "under", "between", "very",
    "some", "such", "can", "could", "may", "might", "must", "should", "would",
}

DESCRIPTIVE_BASES = {
    "somewhat", "unusual", "remarkabl", "much", "rapid", "experience", "show", "register",
}

ORDER_EXCLUDED_AFTER_BASES = {"reduc", "boost", "foster", "encourag"}

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass
class MinuteDoc:
    date: date
    path: str
    text: str
    word_count: int
    sentence_count: int
    sha256: str


def clean_text(text: str) -> str:
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
        "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\ufeff": "",
        "&": " and ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # The appendix removes periods in common abbreviations before sub-sentence
    # parsing. These substitutions cover the forms common in the minutes.
    text = re.sub(r"\bU\.S\.", "US", text, flags=re.I)
    text = re.sub(r"\b([AaPp])\.m\.", r"\1m", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    return text


def wildcard_match(pattern: str, token: str) -> bool:
    if pattern.endswith("*"):
        return token.startswith(pattern[:-1])
    return token == pattern


def normalize_pattern(phrase: str) -> tuple[str, ...]:
    words = re.findall(r"[a-z0-9]+\*?", phrase.lower().replace("&", " and "))
    # Keep all words in dictionary phrases. Phrase-internal conjunctions and
    # prepositions must remain part of the noun phrase.
    return tuple(words)


def vocabulary_bases(patterns: list[str]) -> set[str]:
    bases: set[str] = set()
    for pattern in patterns:
        for word in normalize_pattern(pattern):
            bases.add(word[:-1] if word.endswith("*") else word)
    return bases


PHRASE_BASES = vocabulary_bases([pattern for pattern, _ in STOCK_PHRASE_VARIANTS])
DIRECTION_BASES = vocabulary_bases(NEGATIVE_DIRECTIONS + POSITIVE_DIRECTIONS)


def tokenize(text: str, offset: int = 0) -> list[Token]:
    return [Token(m.group(0).lower(), offset + m.start(), offset + m.end()) for m in TOKEN_RE.finditer(text)]


def is_dictionary_word(token: str, bases: set[str]) -> bool:
    return any(token == base or token.startswith(base) for base in bases)


def is_descriptive(token: str) -> bool:
    return any(token == base or token.startswith(base) for base in DESCRIPTIVE_BASES)


def filtered_tokens(tokens: list[Token]) -> list[Token]:
    kept: list[Token] = []
    for token in tokens:
        word = token.text
        keep_for_dictionary = is_dictionary_word(word, PHRASE_BASES | DIRECTION_BASES)
        if word == "not" or keep_for_dictionary or (word not in STOP_WORDS and not is_descriptive(word)):
            kept.append(token)
    return kept


def pattern_at(pattern: tuple[str, ...], tokens: list[Token], index: int) -> bool:
    if index + len(pattern) > len(tokens):
        return False
    return all(wildcard_match(pat, tok.text) for pat, tok in zip(pattern, tokens[index:index + len(pattern)]))


def raw_boundary(tokens: list[Token], left_end: int, right_start: int) -> bool:
    """Reject matches crossing a punctuation or appendix sub-sentence boundary."""
    if left_end >= right_start:
        return False
    between = tokens[left_end:right_start]
    return any(tok.text in SUBSENTENCE_BOUNDARIES for tok in between)


def group_from_direction(pattern: tuple[str, ...], side: str) -> str:
    del side
    rendered = " ".join(pattern)
    if rendered in {normalize_pattern(x) for x in NEGATIVE_DIRECTIONS}:
        return "negative"
    return "positive"


def direction_candidates(tokens: list[Token], phrase_start: int, phrase_end: int) -> list[tuple[str, str, int, int]]:
    candidates: list[tuple[str, str, int, int]] = []
    direction_patterns = [(normalize_pattern(x), "negative", x) for x in NEGATIVE_DIRECTIONS]
    direction_patterns += [(normalize_pattern(x), "positive", x) for x in POSITIVE_DIRECTIONS]

    for pattern, sign, label in direction_patterns:
        plen = len(pattern)
        # Direct adjacency after stop-word/descriptive removal.
        for start, end, side in [
            (phrase_start - plen, phrase_start, "before"),
            (phrase_end, phrase_end + plen, "after"),
        ]:
            if start < 0 or end > len(tokens) or not pattern_at(pattern, tokens, start):
                continue
            if raw_boundary(tokens, end, phrase_start) or raw_boundary(tokens, phrase_end, start):
                continue
            if side == "after" and any(
                token.text.startswith(base) for token in tokens[start:end] for base in ORDER_EXCLUDED_AFTER_BASES
            ):
                continue
            candidates.append((sign, label, start, end))

        # Invert the direction for constructions such as "not encouraging".
        for start, end, side in [
            (phrase_start - plen - 1, phrase_start, "before"),
            (phrase_end, phrase_end + plen + 1, "after"),
        ]:
            if start < 0 or end > len(tokens):
                continue
            if side == "before":
                direction_slice = tokens[start:start + plen]
                not_index = start + plen
            else:
                not_index = phrase_end
                direction_slice = tokens[phrase_end + 1:end]
            if tokens[not_index].text != "not" or not pattern_at(pattern, direction_slice, 0):
                continue
            if raw_boundary(tokens, not_index, phrase_start) or raw_boundary(tokens, phrase_end, not_index):
                continue
            candidates.append(("positive" if sign == "negative" else "negative", f"not {label}", start, end))
    return candidates


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?\n]+", text):
        end = match.end()
        if text[start:end].strip():
            ranges.append((start, end))
        start = end
    if text[start:].strip():
        ranges.append((start, len(text)))
    return ranges


def phrase_hits_in_sentence(text: str, start: int, end: int) -> list[dict[str, object]]:
    raw = tokenize(text[start:end], offset=start)
    tokens = filtered_tokens(raw)
    candidates: list[tuple[int, int, str, tuple[str, ...]]] = []
    # Longest-first prevents "stock market" from double-counting "stock market price".
    patterns = sorted(
        ((normalize_pattern(pattern), label) for pattern, label in STOCK_PHRASE_VARIANTS),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    for pattern, label in patterns:
        for i in range(len(tokens)):
            if pattern_at(pattern, tokens, i):
                candidates.append((i, i + len(pattern), label, pattern))

    chosen: list[tuple[int, int, str, tuple[str, ...]]] = []
    occupied: list[tuple[int, int]] = []
    for i, j, label, pattern in sorted(candidates, key=lambda row: (-len(row[3]), row[0])):
        span = (tokens[i].start, tokens[j - 1].end)
        if any(span[0] < right and span[1] > left for left, right in occupied):
            continue
        occupied.append(span)
        chosen.append((i, j, label, pattern))

    hits: list[dict[str, object]] = []
    for i, j, label, pattern in sorted(chosen, key=lambda row: row[0]):
        dirs = direction_candidates(tokens, i, j)
        signs = {row[0] for row in dirs}
        if len(signs) != 1:
            if not signs:
                continue
            sign = "ambiguous"
        else:
            sign = next(iter(signs))
        if sign == "ambiguous":
            direction = "ambiguous"
        else:
            direction = sorted(dirs, key=lambda row: abs(row[2] - j) if row[2] >= j else abs(i - row[3]))[0][1]
        phrase_start = tokens[i].start
        phrase_end = tokens[j - 1].end
        hits.append({
            "phrase": label,
            "matched_text": text[phrase_start:phrase_end],
            "direction": direction,
            "sign": sign,
            "start": phrase_start,
            "end": phrase_end,
            "sentence_text": re.sub(r"\s+", " ", text[start:end]).strip(),
        })
    return hits


def section_info(text: str, meeting_date: date) -> tuple[list[tuple[int, str]], str]:
    headings = [
        ("1. Staff Review of Economic Situation", re.compile(r"\bstaff review of (?:the )?economic situation\b", re.I)),
        ("2. Staff Review of Financial Situation", re.compile(r"\bstaff review of (?:the )?financial situation\b", re.I)),
        ("3. Staff Economic Outlook", re.compile(r"\bstaff economic outlook\b", re.I)),
        ("4. Participants' Views", re.compile(r"\bparticipants[']?\s+views\b", re.I)),
        ("5. Committee Policy Action", re.compile(r"\bcommittee policy action\b", re.I)),
    ]
    found: list[tuple[int, str]] = [(0, "Unparsed/Other")]
    for label, pattern in headings:
        found.extend((m.start(), label) for m in pattern.finditer(text))
    found = sorted(set(found), key=lambda row: row[0])
    if len(found) >= 3:
        return found, "explicit_heading"
    return [(0, "Unavailable: pre-2009 manual assignment not supplied")], "unavailable_original_manual_assignment"


def section_for_offset(sections: list[tuple[int, str]], offset: int) -> str:
    current = sections[0][1]
    for position, label in sections:
        if position <= offset:
            current = label
        else:
            break
    return current


def section_group(label: str) -> str:
    if label.startswith(("1.", "2.", "3.")):
        return "staff"
    if label.startswith(("4.", "5.")):
        return "participants"
    return "unavailable"


def load_minutes(minutes_root: Path) -> list[MinuteDoc]:
    docs: list[MinuteDoc] = []
    paths = sorted((minutes_root / "Minutes").rglob("*.txt"))
    for path in paths:
        match = re.search(r"fomcminutes(\d{4})(\d{2})(\d{2})", path.name, re.I)
        if not match:
            continue
        y, m, d = map(int, match.groups())
        meeting_date = date(y, m, d)
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        docs.append(MinuteDoc(
            date=meeting_date,
            path=str(path),
            text=text,
            word_count=len(re.findall(r"\b\w+\b", text)),
            sentence_count=len(sentence_ranges(text)),
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ))
    return docs


def run(minutes_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = load_minutes(minutes_root)
    mention_rows: list[dict[str, object]] = []
    meeting_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for doc in docs:
        sections, section_method = section_info(doc.text, doc.date)
        hits: list[dict[str, object]] = []
        for start, end in sentence_ranges(doc.text):
            hits.extend(phrase_hits_in_sentence(doc.text, start, end))
        # The appendix's automated series uses positive/negative matches only.
        classified = [hit for hit in hits if hit["sign"] in {"positive", "negative"}]
        counts = Counter(str(hit["sign"]) for hit in classified)
        section_counts = Counter()
        for idx, hit in enumerate(classified, start=1):
            label = section_for_offset(sections, int(hit["start"]))
            group = section_group(label)
            section_counts[f"{group}_{hit['sign']}"] += 1
            mention_rows.append({
                "mention_id": f"minutes_{doc.date.isoformat()}_author_method_{idx:04d}",
                "meeting_date": doc.date.isoformat(),
                "source_path": doc.path,
                "phrase": hit["phrase"],
                "matched_text": hit["matched_text"],
                "direction": hit["direction"],
                "sign": hit["sign"],
                "sentence_text": hit["sentence_text"],
                "start": hit["start"],
                "section": label,
                "section_group": group,
                "section_assignment_method": section_method,
            })
        meeting_rows.append({
            "meeting_date": doc.date.isoformat(),
            "source_path": doc.path,
            "word_count": doc.word_count,
            "sentence_count": doc.sentence_count,
            "author_method_total_classified": len(classified),
            "author_method_negative_count": counts.get("negative", 0),
            "author_method_positive_count": counts.get("positive", 0),
            "author_method_ambiguous_or_unclassified_phrase_count": len(hits) - len(classified),
            "staff_negative_count": section_counts.get("staff_negative", 0),
            "staff_positive_count": section_counts.get("staff_positive", 0),
            "participants_negative_count": section_counts.get("participants_negative", 0),
            "participants_positive_count": section_counts.get("participants_positive", 0),
            "section_assignment_method": section_method,
        })
        audit_rows.append({
            "meeting_date": doc.date.isoformat(),
            "source_path": doc.path,
            "sha256": doc.sha256,
            "total_phrase_hits_with_direction_or_ambiguous": len(hits),
            "classified_positive_negative": len(classified),
            "unclassified_or_ambiguous": len(hits) - len(classified),
            "section_assignment_method": section_method,
        })

    mentions = pd.DataFrame(mention_rows).sort_values(["meeting_date", "start"]) if mention_rows else pd.DataFrame()
    meetings = pd.DataFrame(meeting_rows).sort_values("meeting_date")
    audit = pd.DataFrame(audit_rows).sort_values("meeting_date")
    mentions.to_csv(output_dir / "author_method_mentions.csv", index=False)
    meetings.to_csv(output_dir / "author_method_minutes_meeting_counts.csv", index=False)
    audit.to_csv(output_dir / "author_method_document_audit.csv", index=False)

    dictionary = {
        "source": APPENDIX_URL,
        "stock_phrase_count": len(STOCK_PHRASES),
        "stock_phrase_variant_count_with_word_combinations": len(STOCK_PHRASE_VARIANTS),
        "word_combination_prefixes": WORD_COMBINATION_PREFIXES,
        "negative_direction_count": len(NEGATIVE_DIRECTIONS),
        "positive_direction_count": len(POSITIVE_DIRECTIONS),
        "appendix_prose_direction_counts": {"negative": 52, "positive": 41},
        "direction_count_note": "The printed Appendix Table 8 transcription contains 51 negative and 40 positive pattern rows; the appendix prose describes the lists as 52 and 41 words.",
        "stock_phrases": STOCK_PHRASES,
        "negative_directions_group_1": NEGATIVE_DIRECTIONS,
        "positive_directions_group_2": POSITIVE_DIRECTIONS,
        "distance_after_stopword_and_descriptive_removal": 0,
        "neutral_or_hypothetical_matches": "excluded from the automated series",
        "pre_2009_section_assignments": "not available; output marked unavailable",
    }
    (output_dir / "author_method_dictionary.json").write_text(json.dumps(dictionary, indent=2) + "\n", encoding="utf-8")

    section_summary = (
        meetings.groupby("section_assignment_method", dropna=False)[
            ["author_method_total_classified", "staff_negative_count", "participants_negative_count"]
        ].sum().reset_index().to_string(index=False)
    )
    report = f"""# Appendix-Based Author-Method Reconstruction

Source method: [Cieslak and Vissing-Jorgensen online appendix]({APPENDIX_URL}).
This is an independent reconstruction from the published description and
Appendix Table 8. It is not author-provided code or an original author data file.

## Scope

- Minutes: {len(docs)} official text files covering {meetings['meeting_date'].min()} through {meetings['meeting_date'].max()}.
- Stock-market noun phrases: {len(STOCK_PHRASES)}.
- Word-combination prefixes: {', '.join(WORD_COMBINATION_PREFIXES)}.
- Phrase variants searched, including word combinations: {len(STOCK_PHRASE_VARIANTS)}.
- Negative direction words, group 1: {len(NEGATIVE_DIRECTIONS)}.
- Positive direction words, group 2: {len(POSITIVE_DIRECTIONS)}.
- The appendix prose describes these lists as 52 negative and 41 positive words; the printed table contains 51 negative and 40 positive pattern rows, which are the patterns used here.
- Distance: zero words after stop-word and selected descriptive-word removal.
- Neutral and hypothetical phrases: excluded, consistent with the appendix's automated approach.
- Sub-sentence boundaries and negation handling: implemented from the appendix description.

## Output totals

- Classified negative matches: {int((meetings['author_method_negative_count']).sum())}.
- Classified positive matches: {int((meetings['author_method_positive_count']).sum())}.
- Classified total: {int((meetings['author_method_total_classified']).sum())}.
- Ambiguous or otherwise unclassified phrase hits: {int((meetings['author_method_ambiguous_or_unclassified_phrase_count']).sum())}.

## Section assignment limitation

The appendix states that minutes sections were manually assigned before April
2009. Those author assignments are not in the available files. The output uses
explicit headings where they exist and marks earlier meetings as
`unavailable_original_manual_assignment`; it does not invent a pre-2009 staff or
participant assignment.

Section-method totals:

```text
{section_summary}
```

## Files

- `author_method_mentions.csv`: classified phrase-level matches.
- `author_method_minutes_meeting_counts.csv`: meeting-level automated counts.
- `author_method_document_audit.csv`: document hashes and classification audit.
- `author_method_dictionary.json`: exact dictionary used by this reconstruction.

This series is suitable for a closer comparison with the paper's automated
robustness approach. It should not be described as an exact reproduction of the
authors' original series because the original code and pre-2009 manual section
assignments are unavailable.
"""
    (output_dir / "AUTHOR_METHOD_REPLICATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.minutes_root.expanduser().resolve(), args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
