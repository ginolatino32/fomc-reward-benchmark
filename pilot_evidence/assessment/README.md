# Independent assessment of the adaptive-reader pilot

Read `Assessment.md` for the publication decision and `Coder_Next_Step.md` for the bounded completion task. The previous manuscript was not rewritten or overwritten.

`results/` contains the independent numeric audit, correctly nested TF-IDF diagnostic, corrected alpha-selection table, response/role checks, and supplied-test results. No LLM inference was performed.

To reproduce, extract the user-supplied `FOMC_Adaptive_Reader_Pilot_GPT_Pro_Review_20260920.zip` to a directory containing `adaptive_reader_run_20260920_actual/` and `source_context/`. Then run:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/audit_pilot.py --source /path/to/extracted/pilot --output ./new_audit_results
```

The submitted pilot ZIP is not duplicated in this small audit package. `source_documents/` preserves the specific submitted reports/code referenced in the assessment. Their strong provenance claims are those of the source, not a certification by the auditor. Original artifacts are never modified by the audit script.

Absolute paths in the saved audit JSON describe this execution environment only; use `--source`/`--output` for your own directories. Independent numeric refits may differ by negligible floating-point precision on another platform.
