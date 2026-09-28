# New-model extension: independent review

Review of `FOMC_New_Model_Extension_Final_20260920.zip`.

## Assessment

The extension strengthens the observed advantage of learned language representations over the reconstructed rule-based benchmark, but it does not establish that newer encoders improve performance or that full embedding vectors are necessary. The existing Gemini representation remains the strongest original-text full-vector specification. Flash-derived five-category tone counts perform almost as well. Voyage's within-meeting contextualization increases error rather than reducing it under the supplied grouping, pooling and regression design.

This is an analysis of the supplied caches and code. No external model generation, human annotation, new financial data collection, or independent holdout evaluation was performed in this review. Model identities for embedding caches are as recorded in their supplied manifests; offline reproducibility does not authenticate an external provider invocation.

## 1. What was independently checked

The submitted offline pipeline was rerun in a separate output directory. Its 2,079 saved predictions (21 configurations times 99 evaluation meetings) reproduce with maximum absolute difference below 2.4e-9. The source package remains unchanged.

A separate implementation using scikit-learn Ridge and explicit feature-block transformations, rather than the submitted eigensolver, independently refits the 16 non-PCA estimated configurations that appear in the returned prediction table. Its maximum prediction discrepancy is below 2.1e-9, with no selected-penalty differences. The historical-mean baseline is recomputed directly. All 137 recorded structural, identity, alignment, error-reconstruction and deterministic checks pass. These are computational checks, not estimates of semantic accuracy.

The input contains 200 minutes documents and 1,197 unique selected passages. The common target has 199 nonmissing observations. Initial training uses 100 observations; the five test blocks contain 20, 20, 20, 20 and 19 observations. Later training sets expand to 120, 140, 160 and 180. The returned README's phrase '199 observations before the common evaluation sample' is incorrect: 199 is the entire labeled panel, not a pre-evaluation training sample.

Sources: `results/submitted_pipeline_rerun_comparison.csv`, `results/refit_vs_submitted.csv`, `results/integrity_checks.csv`, `source_tables/folds.csv`, and the supplied `README.md`.

## 2. Main economic-state recovery result

All rows use the same 99 evaluation meetings and already-realized intermeeting excess price-return proxy. Length is the unchanged `document_sentence_count` variable. Lower RMSE is better; the MSE gain uses the audited split-block count reconstruction as denominator.

| Representation | RMSE, percentage points | OOS R2 | MSE gain versus audited counts |
|---|---:|---:|---:|
| Audited phrase-and-direction counts + length | 4.100 | 0.324 | Reference |
| TF-IDF + length | 4.613 | 0.144 | -26.6% |
| Legacy Gemini Embedding 001 + length | 3.522 | 0.501 | 26.2% |
| Refreshed Gemini Embedding 001 + length | 3.522 | 0.501 | 26.2% |
| Gemini Embedding 2 + length | 3.626 | 0.471 | 21.8% |
| Voyage Context 4, isolated + length | 3.731 | 0.440 | 17.2% |
| Voyage Context 4, contextualized + length | 4.164 | 0.303 | -3.2% |
| Flash five-category counts + length | 3.560 | 0.490 | 24.6% |

The refreshed Gemini-001 result is numerically indistinguishable from the legacy benchmark at reported precision. Fresh passage vectors are pooled in float64 in the submitted extension, whereas the handoff requested saving the meeting mean in float32 before fitting. The largest pooling-rounding difference is very small; this does not materially explain model rankings. Preserve the legacy cache as the reference rather than overwrite it.

The generalization here is across representations on the same studied corpus. It is not a new test period. The contextualized Voyage row is a counterexample to a blanket claim that every newer embedding method improves on the rule-based comparator.

Sources: `source_tables/table_03_performance.csv`, `results/recomputed_reported_metrics.csv`, `results/independent_performance.csv`, and `results/pooling_precision.json`.

## 3. The important conceptual result: modern tone counts nearly match full vectors

Legacy Gemini has 2.15% lower MSE than Flash's five-category representation. This is a small observed difference, not a demonstrated full-vector advantage over a strong LLM-based classifier. The H5 two-sided conditional loss-test p-value is 0.8537; its corrected percentile gain interval is -30.2% to 22.9%. Non-rejection is not statistical equivalence, and the interval is too wide to conclude that the methods are interchangeable in a new population.

Flash's 24.6% MSE improvement over the audited rule-based comparator provides a more specific interpretation of 'beyond word counts': an LLM can improve the labels being counted. It would now be misleading to write that counting as an aggregation operation necessarily loses the useful information that only a 3,072-coordinate representation retains. The comparison supports learned language interpretation in this task; it does not by itself isolate linguistic reasoning as the causal source of the improvement.

Keep the current Gemini result as the reference and retain Flash as a serious, lower-dimensional comparator. Do not replace the reference simply because a masked or post-discovery variant obtains a slightly better point estimate.

Source: corrected H5 and H4 in `results/corrected_paired_contrasts.csv` and the main performance table.

## 4. More context did not help in this experiment

Within-meeting Voyage has 24.6% higher MSE than isolated Voyage on identical selected-passage inputs. Its registered H2 comparison has conditional p = 0.0013 and Holm-adjusted p = 0.0104 across the eight original-text contrasts. Of those eight contrasts, this is the only one below 0.05 after the specified Holm adjustment.

The conclusion is configuration-specific: contextualizing the selected overlapping windows and then averaging their vectors did not improve the measured return recovery. It is not evidence that all contextual embedding systems fail or that full-document context is universally harmful. The contextual run did not embed the entire minutes; it grouped the same selected passages by meeting. Possible explanations such as dilution of locally informative direction or mismatch with mean pooling remain hypotheses, not findings established by these data.

Sources: H2 in `source_tables/table_05_paired_contrasts.csv`; corrected ratio intervals in `results/corrected_paired_contrasts.csv`; grouping code in supplied `source/extension_pipeline.py`.

## 5. Numerical masking largely preserves the embedding signal

| Representation | Original RMSE | Masked RMSE | MSE change after masking |
|---|---:|---:|---:|
| Refreshed Gemini 001 | 3.522 | 3.517 | -0.25% |
| Gemini Embedding 2 | 3.626 | 3.606 | -1.15% |
| Voyage isolated | 3.731 | 3.736 | +0.28% |
| Voyage contextualized | 4.164 | 4.110 | -2.60% |
| Flash five-category counts | 3.560 | 3.631 | +4.04% |
| TF-IDF | 4.613 | 4.619 | +0.29% |

The actual supplied mask changes 542 of 1,197 passages, or 45.3%, affecting 171 of 200 meetings and 96 of the 99 evaluation meetings. For the three isolated encoders, all 655 unchanged texts have identical original/masked vectors, and none of the changed texts has identical vectors. This supports that the transformation was actually reflected in the cached inputs and outputs rather than only relabeled.

Performance retention is evidence against the narrow explanation that the advantage depends mainly on literal numerical expressions removed by this mask. It does not eliminate pretrained exposure, qualitative magnitude language, dates not covered by the transformation, or other routes to recognizing an economic episode. The masking procedure is not a perfect intervention on 'all numerical information'.

Flash changes 14 passage labels across original and masked variants. Nine changes occur on altered text; five occur on identical text. Accordingly, its 4.0% MSE increase cannot be attributed entirely to number removal: repeated generative classification also introduces variability. No statistical equivalence between original and masked performance is claimed.

Sources: `source_tables/table_06_number_masking.csv`, `results/mask_coverage.json`, `results/mask_vector_checks.csv`, and `results/flash_and_provenance_audit.json`.

## 6. Period stability and the meaning of the loss improvement

Gemini-001, Gemini-2 and Flash each outperform the audited count model in all five outer evaluation blocks. Isolated Voyage wins in four of five; contextualized Voyage wins in two of five. Gemini-001's gains versus the audited comparator range from 2.9% to 56.4%, Gemini-2's from 7.7% to 44.4%, and Flash's from 5.1% to 31.2%.

The MSE advantage does not mean better predictions at every meeting. Gemini-001 has lower squared error at 52 of 99 meetings; Gemini-2 at 55; isolated Voyage at 51; Flash at 57. For Gemini-001, its five largest positive paired-loss contributions total 85.4% of the net MSE advantage. This is not evidence that those events should be discarded. It shows why MSE, MAE, period results and eventual external validation should all be reported.

As an additional descriptive check, restricting saved predictions to exclude 2020 leaves gains of 21.0% for legacy Gemini, 17.4% for Gemini-2, 13.0% for isolated Voyage and 25.5% for Flash. These restrictions neither remove the omitted years from training nor create an independent validation sample. They were calculated during this review and must be labeled post-review descriptive checks if added to the manuscript.

Sources: `results/by_fold_with_gains.csv` and `results/descriptive_stability_checks.csv`.

## 7. Corrections required before incorporating supporting tables

### A. PCA must be fitted within inner training folds

In supplied `fit_pca_readout`, PCA is fitted once on the full outer-training observations and reused inside inner validation. Thus the inner validation covariates help determine the PCA basis used to select the ridge penalty. The outer test observations are not used, so this is not direct leakage of outer evaluation labels. It nevertheless differs from the declared fully nested design.

I refitted PCA separately in every inner-training split and on each outer training set. The primary full-vector results do not change.

| Encoder | Supplied three-PC R2 | Correctly nested three-PC R2 | Correctly nested RMSE |
|---|---:|---:|---:|
| Legacy Gemini 001 | 0.472 | 0.477 | 3.605 |
| Gemini Embedding 2 | 0.459 | 0.459 | 3.669 |
| Voyage isolated | 0.447 | 0.450 | 3.700 |
| Voyage contextualized | 0.042 | 0.008 | 4.965 |

The low-dimensional Gemini result remains informative: three components preserve much of the observed full-vector performance. Equal numbers of components do not imply identical effective complexity or economic interpretability.

Use `results/corrected_capacity_comparison.csv`; its predictions and full nested tuning are in `results/independent_predictions.csv` and `results/independent_tuning.csv`.

### B. Bootstrap the reported ratio, not a fixed-denominator surrogate

The delivered contrast code bootstraps the mean loss difference and divides every draw by the original comparator MSE. The requested statistic was the paired ratio, recomputing both MSEs in each resample. These have the same sample point estimate but generally different percentile intervals.

I corrected all eight contrasts at block lengths 4, 2 and 8. For H5, the interval changes from -21.7% to 25.0% to -30.2% to 22.9%. For H2, the corrected interval for contextual minus isolated improvement is -41.9% to -9.3%. Negative gains indicate worse contextual performance.

The centered-loss p-values and their Holm adjustment test the loss differential rather than the ratio and are unchanged by this interval correction. Percentile intervals and centered-bootstrap p-values are not inversion pairs; do not infer a p-value from whether a percentile ratio interval excludes zero. All remain conditional on saved predictions, not on refitting the full selection process and not adjusted for the earlier research search.

Use `results/corrected_paired_contrasts.csv`.

### C. Repair execution and provenance statements

The following findings affect reporting rather than the independently reproduced headline metrics:

- The Flash prompt differs from the original handoff: it adds an instruction to avoid malformed extraction characters by quoting a shorter exact span. This is a plausible formatting repair, but it must be recorded as a prompt revision; the returned files do not identify the prompt version used for every cached response.
- The protocol file is hash-consistent, but its recorded lock time is after all 2,551 timestamped retained Flash responses. `build_generation` overwrites this file on each resumed run. The earlier handoff provides a prespecified plan, but the returned lock file cannot establish the original run's immutable timing. Preserve the first lock and append amendments instead.
- The final cache contains 2,394 valid labels. The retained histories also contain 157 invalid JSON attempts and one API failure; the request log includes 323 failed record-level processing attempts across resumed runs. Final success is not zero execution failures. Failed histories replaced on resume are not fully retained.
- The repeat table uses stale primary-status fields. Joining to the final primary labels gives 43 valid repeats, all 43 agreeing, out of 50 requested repeats. The seven invalid repeats remain unresolved. This is repeat consistency, not classification accuracy.
- Embedding logs mostly contain locally assigned record IDs and hashes, not full provider-returned model/version identifiers or per-request timestamps. The inventory populates `resolved_model` from the requested name; that is not an independently resolved revision. Provider-call authentication is therefore beyond this offline review.
- Actual Flash usage is present in retained response objects even though the inventory fields are blank. The retained totals are 1,456,266 input, 152,038 output and 764,892 thought tokens, excluding missing historical attempts and repeat/smoke-test accounting. They are not an invoice. The nominal USD 5 ceiling is stored, but the code does not enforce a cumulative spending check.
- The 15-item 'baseline reproduction gate' reads pre-existing reference metrics rather than testing regenerated predictions. The main models do in fact refit correctly, as verified here, but the gate itself must be updated.
- The returned extension omits several requested comparator rows, pooled-vector files and a pinned environment lockfile. This review restores length-only, initial signed counts and cached E5-plus-length as auxiliary refits; these do not replace the audited comparator. The missing pooled caches can be derived without new provider calls.

Sources: supplied generation, PCA and contrast functions; `results/handoff_comparison.csv`; `results/handoff_diff_flash_classification_prompt.txt.txt`; `results/flash_and_provenance_audit.json`; and `results/flash_repeat_corrected_pairs.csv`.

## 8. Recommended manuscript interpretation

The existing title, 'Beyond Word Counts', remains suitable. The extension should be a comparison of rule-based counts, learned tone counts and embedding representations, not a ranking in which a newer model is presumed to win.

Suggested results paragraph:

> On the same 99 chronological evaluation meetings, the legacy Gemini representation retains the lowest error among original-text full-vector specifications (RMSE 3.522; OOS R2 0.501). Gemini Embedding 2 and isolated Voyage representations also have lower point-estimate error than the audited phrase-and-direction reconstruction, with MSE reductions of 21.8% and 17.2%, respectively. A five-category Flash tone-count representation achieves RMSE 3.560 and OOS R2 0.490, close to the legacy full-vector result; the paired comparison does not establish a full-vector advantage over these learned labels. Within-meeting Voyage contextualization instead raises MSE by 24.6% relative to isolated encoding. The embedding results change little under the supplied numerical mask. Together, the findings favor learned language representations over the reconstructed rules in this retrospective task, without establishing that high dimensionality, additional context or model recency is necessary for the observed improvement.

Among the eight registered extension contrasts, the improvements of Gemini-2, isolated Voyage and Flash over the audited counts do not individually pass the supplied eight-test Holm adjustment. Report their point estimates and uncertainty without calling them independent statistical confirmations. In particular, the Flash comparison should not be presented as evidence of formal equivalence.

The extension supplies cross-model and representation diagnostics, not a new holdout or a conventional return-construction sensitivity. Those two previously identified tests remain the most valuable next evidence. Do not launch another provider tournament in response to the mixed ranking.

## 9. Deliverables and reproduction

`Review_Findings.pdf` is the formatted version of this review. The source tables are retained under `source_tables/`. Corrected and independently recomputed outputs are under `results/`. The audit does not overwrite the manuscript or original user archive.

Run the supplied `reproduce_audit.py` against the original returned ZIP. No API credentials or new inference are required. A separate `Coder_Corrections.md` distinguishes offline fixes already implemented here from missing execution-history information that only the coder can supply.
