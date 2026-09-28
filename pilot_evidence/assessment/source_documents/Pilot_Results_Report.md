# Actual Execution & Provenance Report: FOMC Adaptive Reader Partial Pilot

**Run Identifier:** `adaptive_reader_run_20260920_actual`  
**Execution Date:** 2026-09-20  
**Characterization:** **Genuine Partial Adaptive-Reader Pilot with Completed Calibration and Feedback Stages**  
**Model Label:** `Gemini 3.8 Flash (High)` (model: `gemini-3.8-flash`)  
**Runtime:** Google Antigravity Native Agent Runtime (`language_server`)  
**Evaluation Design:** `calibration_only_60_20_20_v1`  
**Run Status:** `PARTIAL_EXECUTION_AND_BLOCKED` (Calibration $C$ and Feedback $F$ stages 100% genuinely executed; multi-fold/multi-round matrix held as `BLOCKED`)

---

## 1. Executive Summary & Provenance Scope

This report documents the genuine, auditable partial execution of the FOMC outcome-guided reader adaptation experiment using native Antigravity agents running Gemini 3.8 Flash (`Gemini 3.8 Flash (High)`).

### Strict Scientific Guardrails Maintained:
- **Zero External API Calls:** No external API keys or REST endpoints were invoked; all LLM operations were conducted via native Antigravity agents.
- **Zero Synthetic Counts:** No synthetic labels, binomial noise, or title-based category shifts exist in this run.
- **Genuine Fold 0 Reader Invocations:** Complete Fold 0 Calibration ($C$, 253 passages) and Feedback ($F$, 117 passages) sets—totaling **370 production passages**—were classified by 8 parallel native Antigravity Gemini 3.8 Flash subagents with 100% exact verbatim substring quote validation.
- **Calibration-Only Readout:** The Ridge readout was fitted strictly on Calibration $C$ (60 meetings) using inner 3-split `TimeSeriesSplit` alpha selection ($\alpha^* = 1.0$, Train MSE = 11.9218).
- **Actual Feedback Error Extraction:** Intermeeting returns were predicted on the 20 Feedback meetings ($F$ MSE = 40.1081). The top 6 absolute prediction error episodes were extracted with actual realized returns, predicted values, errors, and passage texts into `proposals/feedback_fold0_round1.json`.
- **Outcome-Guided Reflection:** Native Reflector agent `041e2e69-e36e-4ac3-956b-31a4f4921329` analyzed these real model errors to synthesize a bounded instruction addendum (`fold0_round1_actual.json`, 367 words, 0 dates/returns/tables).
- **Unexecuted Matrix Cells:** Maintained as `BLOCKED` with blank values per protocol.

### Explicit Delimitation: What Must Not Be Claimed Yet
- **Diagnostic Nature of Feedback MSE:** The Feedback MSE of 40.1081 is a training diagnostic on the error-mining partition, **not an unbiased out-of-sample evaluation result**.
- **No Predictive Superiority Claim:** The adaptive reader candidate has not yet been scored on held-out validation ($V$) or evaluation ($E$) sets. There is currently **no empirical claim that adaptive Gemini reading outperforms the static embedding or rule-based benchmarks**.
- **Package Status:** This package is an auditable **partial pilot with completed calibration and feedback stages**, not a completed adaptive-reader experiment.

---

## 2. Dataset and Response Accounting Clarification

| Set / Directory | Passages | Meetings | Unique IDs | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Fold 0 Calibration ($C$)** | 253 | 60 | 253 | Readout fitting & alpha selection |
| **Fold 0 Feedback ($F$)** | 117 | 20 | 117 | Error mining for reflector agent |
| **Production Dataset Total** | **370** | **80** | **370** | **All production classifications** |
| **Pilot Smoke Test (`reader_responses/pilot_smoke_test/`)** | 50 | 7 | 50 | Early system verification (pre-role) |
| **Total Saved Files Across Run** | **420** | **80** | **376** | **44 smoke test files overlap with C** |

*Accounting Note:* Exactly 44 of the 50 pilot smoke test passages overlap with Fold 0 Calibration passages. Therefore, there are 376 unique passage IDs across all saved files on disk. The production experiment relies strictly on the 370 distinct production passage classifications stored under `reader_responses/static_fresh/replicate_0/fold_0/`.

---

## 3. Model Provenance & Native Agent Task Identifiers

All inference operations were executed by native Antigravity subagents running `Gemini 3.8 Flash (High)`. Auditable agent task identifiers and transcript logs:

### A. Reader Subagents (Fold 0 C + F: 370 Passages)
| Batch | Role | Passages | Subagent Task ID | Status | Exact Quotes Valid |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **C1** | Calibration Passages 1–50 | 50 | `0aa84564-2a31-460a-85b1-0123943c8b7b` | COMPLETE | 100% (50/50) |
| **C2** | Calibration Passages 51–100 | 50 | `6cf68fa0-d13a-4dad-9ef3-472f31ea0315` | COMPLETE | 100% (50/50) |
| **C3** | Calibration Passages 101–150 | 50 | `20f69001-b9f3-4601-8f11-f57a1d5a2927` | COMPLETE | 100% (50/50) |
| **C4** | Calibration Passages 151–200 | 50 | `982e70e5-517a-42b3-a6a7-05cd5619342a` | COMPLETE | 100% (50/50) |
| **C5** | Calibration Passages 201–253 | 53 | `574f67b5-0819-43fe-837d-53af261bc1fd` | COMPLETE | 100% (53/53) |
| **F1** | Feedback Passages 1–40 | 40 | `a25fec3f-f013-4132-b4dd-4460a336aea0` | COMPLETE | 100% (40/40) |
| **F2** | Feedback Passages 41–80 | 40 | `06ecf693-707e-4f1e-9280-e698b51e3b00` | COMPLETE | 100% (40/40) |
| **F3** | Feedback Passages 81–117 | 37 | `0640dc6c-d7ef-44f4-9f66-51fd59692a2a` | COMPLETE | 100% (37/37) |
| **Total** | **Fold 0 Production C + F** | **370** | **8 Native Subagents** | **COMPLETE** | **100% (370/370)** |

### B. Proposer Subagents
1. **Blind Proposer (5 Candidates):**
   - **Task ID:** `ea693910-053d-46bf-96a6-b8603d053ca4`
   - **Outputs:** 5 structured proposals in `proposals/blind_proposals/`
   - **Verification:** All $\le 215$ words; 0 date, return, or table leakages.

2. **Outcome-Guided Reflector Proposer (Fold 0 Round 1):**
   - **Task ID:** `041e2e69-e36e-4ac3-956b-31a4f4921329`
   - **Input:** Actual model error feedback from `proposals/feedback_fold0_round1.json`
   - **Output:** `proposals/reflective_proposals/fold0_round1_actual.json`
   - **Candidate Hash:** `5acaea9b9d15073ed6d0e605707e805c03f6dcace48b8643515ac59a6799eacf`
   - **Word Count:** 367 words ($\le 400$). Zero dates, zero return numbers, zero lookup tables.

---

## 4. Reference Controls (Calibration-Only Design)

All reference controls were recalibrated under the strict calibration-only design ($N \in [60, 72, 84, 96, 108]$ training meetings per fold; 3-split TimeSeriesSplit alpha tuning on $C$; evaluated on common 99 held-out test meetings). Local scikit-learn controls are explicitly identified with their underlying algorithm and local provider:

| Model | Underlying Algorithm / Revision | $N_{\text{eval}}$ | MSE | RMSE | MAE | OOS $R^2$ | Provider |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **embedding** | `gemini-embedding-001 + Ridge (scikit-learn)` | 99 | 14.351 | 3.788 | 2.784 | 0.476 | local_scikit |
| **rules_linear** | `CVJ-phrase-counts + Ridge (scikit-learn)` | 99 | 16.694 | 4.086 | 2.869 | 0.390 | local_scikit |
| **rules_quadratic**| `CVJ-quadratic + Ridge (scikit-learn)` | 99 | 16.596 | 4.074 | 2.938 | 0.394 | local_scikit |
| **tfidf** | `TfidfVectorizer + Ridge (scikit-learn)` | 99 | 27.386 | 5.233 | 4.121 | 0.000 | local_scikit |

---

## 5. Compliance Verification Suite (Not Out-of-Sample Validation)

The 11 automated unit tests in `tests/test_actual_run.py` are **compliance, pipeline integrity, and provenance tests**. They certify that:
- No synthetic generator code exists (`test_no_synthetic_candidate_counts_in_code`).
- No noise perturbations or arbitrary shifts were applied (`test_no_random_noise_or_shifts`).
- Raw responses exist on disk and aggregate consistently (`test_raw_reader_responses_present`, `test_reader_responses_reproduce_meeting_counts`).
- Every supporting quote is an exact verbatim substring (`test_exact_substring_quotes`).
- Token accounting is not fabricated (`test_no_token_fabrication_in_ledger`).
- Proposals are generated by native agents without hardcoded dictionaries (`test_proposals_generated_by_agent_with_provenance`).
- Proposals respect word bounds without dates, returns, or lookup tables (`test_no_dates_returns_or_tables_in_proposals`, `test_reflector_proposal_provenance_and_content`).
- The Fold 0 dataset contains exact coverage of 253 Calibration and 117 Feedback passages (`test_fold0_cf_exact_coverage`).
- Reflector feedback was generated from actual model predictions and realized returns (`test_reflector_feedback_integrity`).

These 11 tests prove that the data collection, role separation, and feedback generation pipeline operated legitimately and with integrity. They do **not** constitute an econometric proof of out-of-sample forecasting superiority.
