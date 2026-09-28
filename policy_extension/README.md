# Policy-reaction extension (manuscript Section 7, Supplement S10)

1. `protocol_locked.json` was written and hashed (`protocol_locked.sha256`) before any estimate was computed.
2. From the archive root, run:

       python policy_extension/run_policy_extension.py --archive . --out policy_extension/results

   The script checks the protocol hash. It then reproduces the main-paper recovery RMSE (3.522 and 4.100) before estimating anything.
3. `posthoc_information_sets.py` is a POST HOC diagnostic outside the protocol. It is reported as such in Supplement S10.

`results/` contains the policy panel, out-of-sample predictions, performance, bootstrap contrasts with Holm adjustment, the asymmetry regression, and the run log. Two independent runs produce byte-identical CSVs.
