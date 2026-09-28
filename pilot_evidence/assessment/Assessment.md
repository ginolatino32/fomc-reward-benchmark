# Adaptive-reader pilot: independent assessment

**Decision:** Retain the source-verified economics manuscript as the main paper. The new archive is useful implementation progress, not evidence that adaptation improves reading. Keep it as a separate, bounded workshop experiment. No completed adaptive result warrants replacing the existing manuscript.

**Scope of this review:** The uploaded `FOMC_Adaptive_Reader_Pilot_GPT_Pro_Review_20260920.zip` was extracted without changing source files. The reviewer rebuilt meeting-level features from saved classifications, independently refitted ridge models, reran the 11 supplied tests with runtime paths relocated, and corrected the TF-IDF nesting diagnostic. No LLM was invoked and no new financial observations were obtained. Numbers below are calculations from the supplied archive, not authenticated new provider results.

## 1. What is actually complete

| Stage | Verified artifact coverage |
|---|---|
| Frozen-reader calibration classifications | 253 passages from 60 meetings in fold 0 |
| Frozen-reader feedback classifications | 117 passages from 20 later meetings in fold 0 |
| Readout fit | Ridge trained on calibration C; inner chronological alpha selection |
| Feedback for reflection | Six error-selected feedback F episodes |
| Proposed instructions | Five blind proposals; one primary 367-word reflected proposal |
| Candidate classifications | None saved |
| Candidate validation scores | None saved |
| Fresh-reader final evaluation | None saved |
| Gated versus ungated/blind/static comparisons | All blank and BLOCKED |

There are 420 saved response files: 370 production plus 50 smoke-test responses, representing 376 unique passages. Forty-four smoke passages overlap calibration; the other six are from the first, unlabeled meeting. Every production response passes the supplied JSON schema, nonempty-quote substring checks, and input-hash comparison. These checks do not establish semantic accuracy or authenticate the vendor invocation.

`agent_predictions.csv` has 396 predictions, all for four recalibrated reference controls. There is no adaptive-reader prediction in that file. `learning_curve.csv` records proposal creation, not a sequence of measured improvements. An untested proposal is neither an effective update nor an empirical negative result.

Sources: `source_documents/Pilot_Results_Report.md`, `source_documents/Pilot_Blockers.md`, `source_documents/agent_performance.csv`, `source_documents/learning_curve.csv`; computed audit in `results/audit_summary.json`.

## 2. The existing numerical claims reproduce

Independent primal ridge fitting reproduces all 396 reference-control predictions with maximum absolute discrepancy approximately 1.82e-14. The calibration-only design uses 60, 72, 84, 96 and 108 fitting observations across five outer folds, reserving the remaining outer-training observations for feedback and validation.

| Recalibrated control | Evaluation meetings | RMSE | OOS R-squared |
|---|---:|---:|---:|
| Existing cached embedding | 99 | 3.788297 | 0.475964 |
| Reconstructed linear rules | 99 | 4.085790 | 0.390427 |
| Reconstructed quadratic rules | 99 | 4.073765 | 0.394010 |
| Submitted TF-IDF implementation | 99 | 5.233138 | 0.000005 |
| TF-IDF with correctly nested vocabulary/IDF | 99 | 5.137792 | 0.036112 |

These are fresh downstream fits of old representations, not learning-agent results. They are not directly comparable to the earlier embedding RMSE of 3.522 as an adaptation effect: training sample sizes and the training-mean R-squared denominator differ.

The fresh-reader calibration selects alpha 1.0. Training MSE is 11.921789, selected inner-validation MSE is 14.691074, and feedback MSE is 40.108110. The feedback number is calculated on F, which supplies reflection input; it is not the final test of the adaptive procedure.

## 3. The feedback exposes an important credit-assignment problem

For the October 29, 2008 feedback episode, the supplied target is -30.985565 percentage points and the calibrated estimate is -5.767251. All five passages are already classified negative. This single episode accounts for 79.281% of feedback squared error. The calibration target range is approximately -18.273 to +9.898 percentage points, so this target is also outside the calibration range.

This is not an independently adjudicated statement that every label is right. It shows something more limited and important: the largest return error is not evidence that the reader failed to detect negative direction. It could arise from omitted magnitude, finite-sample calibration, shrinkage, or a mismatch between the scalar target and the language. The present evidence cannot isolate these explanations.

Outcome-guided reflection must not treat every numerical residual as a semantic labeling error. Forcing category changes merely to reproduce an unusually large return could make the measurement less faithful. The candidate's discussion of time horizons, net movement and relative sector performance is plausible, but no candidate output has yet tested it. Blind proposals cover overlapping ideas, making the blind-search control substantive rather than cosmetic.

Source: `results/feedback_error_concentration.csv`, `results/production_meeting_category_counts.csv`, and the archived feedback JSON.

## 4. Corrections needed before the next run

### 4.1 Refit each candidate's readout on its own calibration labels

The proposed handoff requires classifying C again under each candidate, fitting that candidate's downstream mapping only on C, and then scoring V and E. The abbreviated completion instructions in BLOCKERS.md suggest retaining the existing readout and only classifying V/E. That would test a different, frozen-decoder intervention. It must not be silently substituted for the declared calibrated-candidate experiment.

### 4.2 Refit TF-IDF within inner folds

`actual_harness.py` learns TF-IDF once on all C and then uses that matrix during inner alpha selection. Inner-validation texts influence vocabulary/IDF. This uses no outer-E labels, but it is not fully nested preprocessing. Correct nesting changes TF-IDF RMSE from 5.233 to 5.138. `results/independent_control_performance.csv` retains both values.

### 4.3 Separate fitting error from validation error

`alpha_selection.csv` labels 11.9218 as `cv_mse` for the fresh reader. It is training MSE, not cross-validation error. The selected inner-validation MSE is 14.691074. The independent audit writes all nine alpha scores to `results/correct_fresh_reader_alpha_selection.csv`. The originally selected penalty remains 1.0.

### 4.4 Repair proposal lineage and not-yet-evaluated fields

The blind-proposal hash in `learning_curve.csv` matches none of the five archived blind candidate hashes. There are also two distinct reflection text artifacts: `proposals/reflector_fold0_round1.json` and the primary `proposals/reflective_proposals/fold0_round1_actual.json`. Only the latter is designated as the scored candidate in the report, but neither has been evaluated. Explicitly supersede or identify the earlier artifact.

Unscored candidates currently have `accepted=False` and `valid_development_outputs=True`. Use NOT_EVALUATED/null for gate acceptance and development completeness, with a separate proposal-format-valid field. A rejection requires an executed gate; format-valid text is not complete development output.

### 4.5 Align the gate definition

The original handoff describes variance-normalized MSE minus token cost. The execution lock describes raw MSE and a zero token penalty when telemetry is unavailable. Declare one amended formula, its gate margin, and a resource-budget rule before computing new gate scores. Do not present unavailable inference usage as measured zero cost. Zero additional API billing, subscription cost, compute cost, and measured token use are distinct quantities.

### 4.6 Limit provenance claims to the evidence present

The archive contains response JSON, staging batches, reported native-agent task IDs and self-authored ledgers; it does not contain the underlying native-agent execution transcripts claimed in the report. All 11 supplied tests pass, but several check only literal code strings or expected metadata. That does not prove actual tool isolation, prove the provider was invoked, or establish that no other source of labels exists. No contrary provenance is asserted; the provided evidence is simply insufficient to authenticate these stronger claims.

Retain exact requests and tool-access transcripts for subsequent runs, with target-bearing controller files inaccessible to reader processes. The supplied batch files contain 37–53 passages from multiple meetings. Either document the actual batched-reader procedure and keep batch composition identical between arms, or use isolated requests. Changing that interface requires an appropriate fresh static control, rather than treating older batched outputs as identical.

## 5. A much smaller next experiment is possible

Do not launch the entire estimated 138,000-classification matrix merely to decide whether the branch is worth pursuing. Amend and freeze a one-fold, one-proposal pilot before gate/evaluation results exist.

Fold 0 contains C=253 passages, F=117, V=130 and E=129. Existing static C and F can be retained only if the reader invocation protocol remains the same.

| Additional work | Passage classifications |
|---|---:|
| Static reader on V and E | 259 |
| Existing reflected candidate on C, V and E | 512 |
| Total for static / ungated / gated single-proposal comparison | 771 |
| One outcome-blind, preselected blind proposal on C, V and E | +512 |
| Total including a matched one-proposal blind comparator | 1,283 |

These are passage-output counts, not API-request counts, token estimates, elapsed-time guarantees, or independent economic observations. They exclude repair attempts and extra replicates. A candidate readout is refitted on C; no extra F pass is needed for a single already-generated update. Ungated uses that candidate; gated uses it only if it passes V, otherwise it uses the static reader. Generate both static and candidate E outputs after the gate is locked so that ungated and gated evaluation are both observable without new prompt selection.

The 20 E meetings would complete one restricted historical pilot, not a full five-fold result. A single-fold result cannot establish robust transfer or a general benefit of gating. It supplies a bounded decision point: expand a scientifically interpretable, reproducible experiment, or retain the pilot as an implementation appendix without a learning-improvement claim.

## 6. Publication strategy

Keep the source-verified economics paper's completed contribution intact: learned automation improves on the tested lexical algorithm, with appropriately qualified comparisons against original human-informed counts. The new partial pilot neither contradicts nor strengthens that performance finding.

For RL4LLM-Agents, the official call welcomes harness-side reflection, validation gates, and short/system/reproducibility reports. It does not require weights to be updated. Thus a separate benchmark-and-partial-pilot report has a plausible scope fit, but the present archive supplies no completed learning effect and no gate-effect estimate. A negative-result framing would also be premature because adaptation has not been tested.

Source checked September 20, 2026: https://rl-for-llm-agents.github.io/ (official workshop call). Submission deadline listed as October 1, 2026; short and system reports are described separately from mature full papers. The site also disallows work under review elsewhere, so overlapping submissions require care.

**Bottom line:** useful engineering and a genuine research question remain, but the previous economics paper is the stronger empirical manuscript now. Keep the branches separate. Complete the bounded pilot only if pursuing the workshop remains a priority.
