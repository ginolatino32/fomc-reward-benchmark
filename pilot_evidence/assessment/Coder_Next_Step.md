# Coder next step: finish one small, evaluable pilot

Do not generate another model comparison or launch the whole multi-fold matrix yet. Preserve all source files from the received pilot. The attached independent audit reproduces the existing reference controls and fresh-reader calibration/feedback results. It does not contain new model outputs.

## 1. Correct before generating new gate or test results

- Make all paths relative to the archive or configurable by command line; the supplied tests/harness currently hard-code a local workstation path.
- Refit TF-IDF vocabulary and IDF inside every inner training split. Use the corrected audit as a reference.
- Record alpha 1.0's inner-validation MSE as 14.691073562..., and training MSE separately as 11.9217892775.... Do not label the latter CV error.
- Join every learning-curve row to a real candidate hash. Identify or supersede the earlier reflection artifact. Store the final primary candidate as an immutable file; never replace its text after gate feedback.
- Replace unexecuted accepted=False / valid_development_outputs=True fields with null/NOT_EVALUATED and a distinct proposal-format-valid field.
- Write a NEW versioned execution amendment. Resolve the raw-MSE versus variance-normalized objective discrepancy and the unavailable-token rule. An existing timestamp is not a substitute for recording this amendment. Choose the resource rule and acceptance margin before scores are visible.
- Retain exact native requests and tool traces. Input hashes and task IDs alone do not prove role isolation. Readers must not access controller targets. Use the same reasoning setting, base prompt, request grouping, and retry limits across arms.
- If switching from multi-passage native-agent batches to isolated passage calls, regenerate the static C/F outputs as well; do not silently treat a changed invocation interface as the original control.

## 2. Freeze this reduced experiment

One outer fold: fold 0, same C/F/V/E dates. One generation replicate. One feedback-informed proposal (the already archived primary candidate). For a comparator, use one of the archived blind proposals selected by a deterministic outcome-free rule, such as the lowest nonce, not by inspecting predictive performance. This is a one-proposal-versus-one-proposal experiment, not completion of the five-proposal registered matrix.

The objective is to measure whether the generated instructions change outputs and generalize beyond the feedback episodes. Do not assume that the large negative-return residual establishes a classification error. The worst feedback meeting already has five negative labels. Do not redefine the ontology, insert return lookup information, or rewrite labels to force agreement with the financial outcome.

## 3. Run and calibrate

For each candidate, classify all 253 C passages under that candidate; aggregate five categories plus the unchanged document-length control. Refit its ridge readout on the 60 C meetings, selecting alpha through the existing three chronological inner splits with fully training-only transformations. Do not fit coefficients on F or V.

Generate the static V classifications (130) and candidate V classifications (130 per candidate). Score V with each candidate's C-fitted readout. Record gate decision without exposing V cases/outcomes to the proposer. Freeze artifacts and decisions.

Generate E classifications for static, reflected and the preselected blind candidate (129 each), using unchanged frozen artifacts. Ungated uses the reflected candidate; gated uses it only if V accepted it, otherwise static. A single candidate can supply both ungated and gated output when accepted; do not generate redundant identical branches merely to create separate table rows.

With reusable static C/F, static+reflected needs 771 additional passage outputs. Adding one blind comparator raises this to 1,283 before retries. Do not confuse passage outputs with API requests or independent meetings.

## 4. Return actual results

Return:
- A final 20-meeting table for static, blind, ungated, gated, embedding, linear rules, quadratic rules and corrected TF-IDF, all using the same C calibration and E sample.
- Per-meeting predictions, y, paired MSE and MAE differences, plus conditional uncertainty clearly marked as one historical fold. Do not compare the old 99-meeting RMSE directly with this 20-meeting result.
- Candidate C/F/V/E label coverage, prompt hashes, actually selected alpha, training versus inner-CV losses, V scores and acceptance decisions.
- A label-change table versus static by category and role. Retain exact evidence and mechanical quote/schema failures. This is not gold-standard accuracy.
- Exact costs where measured; null where unavailable; counts and elapsed times with clear units and boundaries.
- Raw requests/responses, all attempts, real generation transcripts where exportable, immutable amendment, source hashes and portable offline reproduction/tests.
- A comparison of reflected instructions with the blind control, not just with an empty prompt. Similar human-plausible wording is not evidence of superior feedback learning.

Stop at this amended budget and report the result, including no change or deterioration. Do not revise prompts after seeing E. The earlier broad design remains unexecuted. Any larger multi-fold study is a separately approved extension.
