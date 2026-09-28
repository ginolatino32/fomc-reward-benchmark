# Appendix-Based Author-Method Reconstruction

Source method: [Cieslak and Vissing-Jorgensen online appendix](https://back.nber.org/appendix/w26894/w26894appendix.pdf).
This is an independent reconstruction from the published description and
Appendix Table 8. It is not author-provided code or an original author data file.

## Scope

- Minutes: 200 official text files covering 2000-02-02 through 2024-12-18.
- Stock-market noun phrases: 47.
- Word-combination prefixes: rate of, growth of, level of, index of, indices of.
- Phrase variants searched, including word combinations: 282.
- Negative direction words, group 1: 51.
- Positive direction words, group 2: 40.
- The appendix prose describes these lists as 52 negative and 41 positive words; the printed table contains 51 negative and 40 positive pattern rows, which are the patterns used here.
- Distance: zero words after stop-word and selected descriptive-word removal.
- Neutral and hypothetical phrases: excluded, consistent with the appendix's automated approach.
- Sub-sentence boundaries and negation handling: implemented from the appendix description.

## Output totals

- Classified negative matches: 339.
- Classified positive matches: 397.
- Classified total: 736.
- Ambiguous or otherwise unclassified phrase hits: 14.

## Section assignment limitation

The appendix states that minutes sections were manually assigned before April
2009. Those author assignments are not in the available files. The output uses
explicit headings where they exist and marks earlier meetings as
`unavailable_original_manual_assignment`; it does not invent a pre-2009 staff or
participant assignment.

Section-method totals:

```text
             section_assignment_method  author_method_total_classified  staff_negative_count  participants_negative_count
                      explicit_heading                             514                   150                           32
unavailable_original_manual_assignment                             222                     0                            0
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
