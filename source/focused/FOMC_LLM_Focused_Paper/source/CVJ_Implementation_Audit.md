# Audit of the uploaded CVJ-method reconstruction

Date: September 19, 2026.

## Provenance and scope

The uploaded `replicate_author_method_counts.py` describes itself as an independent reconstruction, not author-provided code. It has been preserved byte-for-byte in `sources/`. The supplied `w26894.pdf` is the March 2020 NBER working paper, not the online appendix. The appendix named inside the script was separately consulted through its original NBER link: https://back.nber.org/appendix/w26894/w26894appendix.pdf. Appendix B, pages 17–20, and the rendered dictionary in Appendix Table 8 were checked. The web reader accessed this appendix, but a direct container download failed; no local appendix PDF or invented download hash is claimed.

The paper explicitly distinguishes human-coded minutes from its algorithmic checks. We do not reproduce the human-coded series or its original policy regressions. The current experiment evaluates an independently reconstructed algorithm on the manuscript's own state-recovery task. This is not an exact reproduction of the original authors' results.

## Findings and changes

| Item | Uploaded code | Audited reconstruction |
|---|---|---|
| Direction dictionary | 51 negative and 40 positive patterns; asserts a prose/table inconsistency | Rendered Table 8 contains 52 and 41. Replace `adjust*` by `adjust* downward`; restore `slow*` and `stop decline`. The stated discrepancy is a transcription problem, not a verified inconsistency in Table 8. |
| Noun-prefix forms | `rate of`, `level of`, `index of` are literal | Retain the printed wildcards in `rate* of`, `level* of`, and `index* of`. |
| Clause boundaries | Commas and semicolons disappear during tokenization; the adjacency boundary calls examine empty/reversed intervals | Preserve punctuation and named connector boundaries before matching. Explicitly named connectors inside a dictionary phrase remain part of that phrase. |
| Preposed negation | Before a noun, the code looks for direction + `not`, rather than `not` + direction; unnegated direction can be counted | Attach `not` to its following direction before assessing proximity to a noun, on either side of the noun. |
| Stop-word protection | Exempts all tokens starting with any dictionary token; short tokens such as `s` and `p` protect unrelated vocabulary | Protect complete matched expressions, not all words sharing a prefix. Remove only the explicit local stop/descriptive terms in allowed gaps. |
| Postposed order exclusions | Suppresses every word beginning with excluded stems, broader than the printed finite list | Use the printed finite forms. For example, an `encouraging` construction is not automatically suppressed as if it were `encourage`. |
| Direction overlap | Competing short and long direction matches can create spurious conflict | Prefer the longest direction expression before assigning polarity; conflicting distinct directions abstain. |
| S&P normalization | `&` becomes `and`, which no longer matches `s p 500 index` | Normalize the abbreviation before generic ampersand processing. Offsets and hashes refer to cleaned text. |
| Unclassified audit rows | Nouns without a direction are discarded before the reported unclassified count | Preserve all noun matches and report unclassified and ambiguous cases separately. |
| Hypothetical language | Report claims exclusion, but `would` can be dropped and a hypothetical fall counted | No general semantic hypothetical exclusion is claimed. The code reports directional pattern matches, not a validated taxonomy of factual/hypothetical statements. |
| Sections | Heading detector misses curly apostrophes and searches prose; early manual labels are unavailable | Primary count comparison aggregates whole documents and makes no staff/participant claim. This is a documented scope departure. |

These changes were specified before inspecting the new financial prediction errors. They were not tuned to enlarge an embedding advantage. All three versions—uploaded, dictionary-only, and audited—are retained.

## What remains a reconstruction assumption

The historical version of the authors' `stop_words` package and their complete descriptive-removal list are not supplied. To avoid silently substituting an unrelated dictionary, the audited implementation retains the uploaded explicit stop-word list and the printed descriptive examples, with wildcard versus exact forms documented. The whole-document corpus includes material that the authors may have excluded through section selection. Remaining match-overlap choices are documented in code. Original author-output equivalence therefore remains unverified.

Twenty-four constructed matching tests check deterministic behavior (polarity, boundaries, longest matches, abbreviations and negation). They are code tests, not an independently labeled sample of real minutes. Passing them does not estimate natural-corpus accuracy. No human annotation or fresh model judging was performed.

## Comparability of the downstream experiment

The manuscript target, all 199 available return labels, the existing sentence-count control, 3,072-dimensional Gemini vectors, and all 99 evaluation meetings are unchanged. The primary reconstructed-count model uses negative/positive counts in one feature block and length in another, under the same linear ridge implementation and inner tuning grid as Gemini. Joint numeric-block ridge and ordinary least squares are retained as sensitivities. Original signed counts, mention counts, historical means and TF–IDF are rerun rather than redefined.

Counts are computed over full document text. Gemini retains the existing selected-passage representation. This compares complete measurement procedures; the count-versus-Gemini comparison is not an identical-input ablation. TF–IDF and Gemini do retain identical passage inputs. Whole-document counts are not restricted to the old extraction dictionary to manufacture a matched sample.

The source paper's full 1994–2016 text sample is not present: this corpus overlaps during 2000–2016. Consequently, aggregate counts cannot be forced to match the published 1994–2016 totals. Original meeting-level outputs have not been supplied. Period-overlap summaries are not numerical agreement tests against the authors' series. Later-period results are also not new holdouts: they were part of the already explored manuscript sample.

## Interpretation

Any favorable error comparison supports performance against this explicitly documented reconstruction under the common task. It does not show superiority over the original human coding, establish exact original-algorithm replication, identify an economic mechanism, or create a real-time return forecast.
