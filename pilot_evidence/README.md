# Pilot evidence: diagnostic only

The manuscript adds no adaptive-reader performance claim. `assessment/` preserves the independently audited pilot assessment and diagnostic outputs. `raw/` contains the 370 saved static calibration/feedback responses and the partition/target/feedback records needed to verify the cited example, plus the supplied execution lock and primary proposed update. This is a selected diagnostic evidence archive, not the complete adaptive-reader experiment.

Run `python code/verify_future_work.py` from the package root to verify record counts, meeting aggregates, the October 2008 diagnostic, and the unchanged economic-results files. No provider call or outcome-model fitting is performed by that check. The full supplied pilot and independent-assessment ZIP hashes appear in `revision/input_provenance.json`.

The assessment's script `assessment/code/audit_pilot.py` refers to original full source trees and is preserved as audit provenance, not presented as a standalone script on these selected inputs. Its corrected diagnostic CSVs are archived. The original incomplete candidate evaluations remain incomplete. A future continuation must fix the identified path, nesting, score-label, gate, and execution-provenance issues before candidate evaluation.
