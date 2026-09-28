# FOMC reward benchmark

Code, data and results for:

> Manuel Bossi. *Which Outcome Should Teach an LLM Reader? A Verified FOMC Benchmark and Reward-Design Evidence for Outcome-Guided Agents in Finance.* Submitted to the Workshop on Reinforcement Learning for LLM-based Agents (RL4LLM-Agents) at ACM ICAIF 2026.

The paper is in [`paper/rl4llm_fomc_reward.pdf`](paper/rl4llm_fomc_reward.pdf).

The benchmark pairs the minutes of 200 Federal Open Market Committee (FOMC) meetings (February 2000 to December 2024) with the stock-market measures of Cieslak and Vissing-Jorgensen's study of the "Fed put" (*Review of Financial Studies*, 2021; NBER Working Paper 26894). It contains:

- 1,197 equity-market passages;
- fixed chronological splits (100 initial training meetings, 99 evaluation meetings in five expanding blocks);
- an audited reconstruction of the authors' phrase-and-direction algorithm, with 24 constructed tests;
- the authors' human-coded tone counts and return series, recovered from the vector graphics of their published figures;
- frozen text representations (3,072-dimensional Gemini embeddings and five-category tone labels from `gemini-3.8-flash`);
- a hashed, pre-specified protocol for the policy-reaction test;
- the saved outputs of an adaptive-reader pilot.

## Quick start

Requires Python 3.11 or later.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` refits every comparison offline from the released inputs in about 30 seconds and writes `results/final_verification.json` and `manuscript/tables/`. With the pinned versions in `requirements.txt`, the regenerated files in `results/` and `data/` are byte-identical to the released ones. No API calls are made.

The policy-reaction test (paper Section 4, Table 3, Figure 1b):

```bash
python policy_extension/run_policy_extension.py --archive . --out policy_extension/results
cd policy_extension && python posthoc_information_sets.py .. && cd ..   # post hoc, outside the protocol
```

The first script refuses to run if `policy_extension/protocol_locked.json` no longer matches `protocol_locked.sha256` (SHA-256 `1adb261e1d9cdef148f9d85d1ddc01a84f6aff7a00e47549bc86d019b39ec53e`). It also reproduces the recovery results before estimating anything.

## Where the paper's numbers come from

| Paper item | File |
| --- | --- |
| Table 2 (recovery reward, frozen readers) | `manuscript/tables/table1_full_period.csv`, created by `reproduce.py`; length-only row in `source/focused/FOMC_LLM_Focused_Paper/results/performance.csv` |
| Overlap with human coding (35 meetings) | `manuscript/tables/table2_original_source.csv`, `results/author_original_target_*.csv` |
| Matched-passage and return-definition checks | `manuscript/tables/table3_matched_inputs.csv`, `table4_target_sensitivity.csv` |
| Bootstrap intervals and Holm adjustment | `results/new_paired_contrasts.csv` |
| Table 3 and Figure 1b (policy reward) | `policy_extension/results/` |
| Table 4 and the 15.7% role-separation cost | `pilot_evidence/assessment/results/` |
| Figure 2 wording count (33 meetings) | `paper/scripts/intensity_language.py` (run from the repository root) |
| Figure 1 | `paper/scripts/make_figure1.py` (run from `paper/`) |

## Repository layout

| Path | Contents |
| --- | --- |
| `code/` | Feature construction, evaluation, source recovery and verification scripts called by `reproduce.py` |
| `data/` | Meeting-level targets and the source observations recovered from the published figures |
| `source/cvj/` | Minutes texts (200 files), passages, the audited rule reconstruction and its tests |
| `source/focused/` | Frozen embeddings (`representations.npz`), LLM tone labels, prompts, schemas, folds, predictions |
| `source/market/` | Federal funds target range and three-month Treasury bill series (FRED) |
| `results/` | Out-of-sample predictions, tuning records, paired contrasts and audits |
| `policy_extension/` | Hashed protocol, scripts and results for the policy-reaction test |
| `pilot_evidence/` | Adaptive-reader pilot: 370 saved reader responses, manifests, proposals and an independent audit |
| `provenance/` | Source registers with URLs and SHA-256 hashes, download scripts, request logs and the target-construction ledger |
| `paper/` | Workshop paper (PDF and LaTeX) and figure scripts |

## Data sources and what is not included

| Source | Status in this repository |
| --- | --- |
| FOMC minutes, Board of Governors of the Federal Reserve System | Included as extracted text (`source/cvj/.../data/minutes/`). The original HTML and PDF pages can be re-downloaded with `provenance/source_records/download_official_fomc_minutes.py`; their URLs and hashes are in `provenance/minutes_source_register.csv`. |
| Federal funds target and range (DFEDTAR, DFEDTARU, DFEDTARL) and three-month Treasury bill rate (DTB3), FRED | Included |
| S&P 500 daily closes (Yahoo Finance `^GSPC`; FRED `SP500`) | **Not included**; third-party terms restrict redistribution. Meeting-level return targets derived from them are included in `data/` and `provenance/primary_target_construction_ledger.csv` (endpoint price levels removed). `python scripts/fetch_market_data.py` downloads fresh copies; a fresh Yahoo download reproduced the released targets exactly on 28 September 2026. |
| Macro-financial controls (UNRATE, CPI, VIXCLS, BAA10Y) | **Not included**; used only in earlier exploratory work and not in any result reported in the paper. |
| Cieslak and Vissing-Jorgensen (2020, 2021) papers and online appendix | **Not included**. Obtain them from the publishers to rerun the figure recovery (`python reproduce.py --recover-sources` expects `source/w26894.pdf` and `external/FedPut_onlineappendix_rfs2.pdf`). The recovered observations are included in `data/`. |

With the price caches present, `reproduce.py` also rebuilds the alternative return targets. Without them it uses the released targets.

## Integrity and provenance

- `RELEASE_SHA256.txt` lists the SHA-256 hash of every released file. Verify it with `python scripts/verify_release.py`.
- `MANIFEST_SHA256.json` is the manifest of the full private research archive this release was cut from. It also lists files that are not released (manuscript drafts and the third-party files above).
- Five provenance files had local absolute paths replaced by placeholders (`<local_drive>`, `<local_desktop>`, `<home>`) before release, so their hashes differ from the archive manifest: `pilot_evidence/assessment/source_documents/Pilot_Results_Report.md`, `pilot_evidence/assessment/source_documents/submitted_actual_harness.py`, `provenance/source_records/minutes_download_log.txt`, `provenance/source_records/fred_target_metadata.json`, and `provenance/source_records/legacy_gemini_manifest.json`. The `start_price` and `end_price` columns were removed from `provenance/primary_target_construction_ledger.csv`.
- The adaptive-reader pilot responses pass JSON-schema and exact-quotation checks. Their semantic accuracy was not measured, and the archived records do not authenticate the calls to the model provider. No adaptive arm has been run.
- `pilot_evidence/assessment/code/audit_pilot.py` reruns the pilot audit. It expects the original pilot package layout described in `pilot_evidence/assessment/README.md`, and it additionally requires `jsonschema`.

## License

Code is released under the MIT License (`LICENSE`). Data and documentation created for this project are released under CC BY 4.0 (`DATA_LICENSE.md`). FOMC minutes and FRED series from the Federal Reserve are public-domain U.S. government works. Model outputs are subject to the provider's terms.

## Citation

See `CITATION.cff`. Please also cite Cieslak and Vissing-Jorgensen (2021), "The Economics of the Fed Put," *Review of Financial Studies* 34(9), 4045–4089.
