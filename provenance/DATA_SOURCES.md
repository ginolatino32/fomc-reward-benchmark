# Data sources, acquisition evidence, and analytic use

This register describes the inputs used by **Language Models and Count-Based Measures of FOMC Minutes**. It documents the archived files supplied for the study. No web download or model inference was performed in this editorial revision. Missing acquisition timestamps are reported as missing. File modification times are not acquisition evidence.

## How to use the inventory

Main Table 1 identifies the inputs used in the analysis. Supplement S9 gives stored financial-file coverage. The machine-readable `data_source_register.csv` contains one row per source component. `minutes_source_register.csv` provides the exact URL, raw file, extracted text, and hashes for each of the 200 minutes documents. `primary_target_construction_ledger.csv` records the actual endpoint rows and accrual calculation for all 199 primary targets.

Archived download files, graph-recovered observations, derived merges, and generated model outputs are distinct input types. The reconstructed algorithm is not the original authors’ raw output series. The original human counts were recovered from the source figures.

## Inputs

### D1. FOMC minutes and meeting dates

**Source:** Board of Governors of the Federal Reserve System

**Source address or identifier:** https://www.federalreserve.gov/monetarypolicy/fomchistorical2000.htm ; https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

**Archive file:** `provenance/minutes_source_register.csv`

**Use:** all text representations and meeting chronology.

**Acquisition record:** archived download manifest reports 200 successful extractions. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 2000-02-02 to 2024-12-18, 200 rows/records. Units: documents.

**Evidence:** `provenance/source_records/minutes_download_log.txt`

**SHA-256 of named archive file:** `d2d1d31d260820fccab49370452e2e199731c29ff98ee0084dd3cef85a5c264d`

199 HTML extractions and one PDF extraction (2008-06-25). Per-document URLs, raw bytes, and text hashes are in the register. No original download timestamp is present.

### D2. S&P 500 closing index

**Source:** Yahoo Finance, ticker ^GSPC

**Source address or identifier:** https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=946684800&period2=1735689600&interval=1d&events=history

**Archive file:** `source/market/yahoo_gspc_daily.csv`

**Use:** primary return proxy and two price-return sensitivities.

**Acquisition record:** archived data, not downloaded in this revision. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 2000-01-03 to 2024-12-31, 6289 rows/records. Units: index level.

**Evidence:** `provenance/source_records/daily_market_return_validation.py`

**SHA-256 of named archive file:** `d77806f49ee68e6f6cc91743fc0ddf95b44f0125ef0b654e7092daee6766a2d1`

Downloader permits chart endpoint or yfinance fallback. Successful retrieval route and exact date are not separately recorded. Nonmissing values in selected column: 6289.

### D3. Three-month Treasury-bill discount quote

**Source:** FRED, DTB3

**Source address or identifier:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3

**Archive file:** `source/market/DTB3.csv`

**Use:** approximate risk-free adjustment to primary target.

**Acquisition record:** archived data, not downloaded in this revision. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 1954-01-04 to 2026-05-19, 18882 rows/records. Units: annual discount quote, percent.

**Evidence:** `provenance/source_records/daily_market_return_validation.py`

**SHA-256 of named archive file:** `f3a09b7ee6aa4c5977c6e7559c4dcb2741ae8a2f7c7a74c61d6d2baa5772c9fb`

A discount quote is not a realized daily bill return. Only values within the 2000–2024 target windows enter the primary calculation. Nonmissing values in selected column: 18086.

### D4. Federal funds target and range

**Source:** FRED, DFEDTAR / DFEDTARL / DFEDTARU

**Source address or identifier:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTAR,DFEDTARU,DFEDTARL

**Archive file:** `source/market/fred_federal_funds_target_range.csv`

**Use:** effective-change-day return sensitivity; not a main regression control.

**Acquisition record:** archived data, not downloaded in this revision. Original acquisition timestamp: `2026-05-20T11:52:25`.

**Stored coverage:** 1982-09-27 to 2026-05-20, 15942 rows/records. Units: target rate, percent.

**Evidence:** `provenance/source_records/fred_target_metadata.json`

**SHA-256 of named archive file:** `69ab1c709bffb0612c2e97bba1e3831e84b08988628bff1638f679687348f412`

Metadata records this timestamp without timezone. Midpoint combines the single target and subsequent lower/upper range. Effective dates are not independently verified announcement dates. Nonmissing values in selected column: 15942.

### D5. S&P 500 validation cache

**Source:** FRED, SP500

**Source address or identifier:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500

**Archive file:** `source/market/SP500.csv`

**Use:** overlap quality check only.

**Acquisition record:** archived data, not downloaded in this revision. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 2016-05-20 to 2026-05-19, 2608 rows/records. Units: index level.

**Evidence:** `provenance/source_records/daily_market_return_validation.py`

**SHA-256 of named archive file:** `46cd98adebb7d23b815cab9b29d7f5735bd1c21027608d12f080813a5ce72a68`

Not the full-period price source. The saved Yahoo/FRED comparison uses 2,168 paired closes during 2016-05-20–2024-12-31. Nonmissing values in selected column: 2513.

### D6. Merged price/bill table

**Source:** Derived from D2 and D3

**Source address or identifier:** see D2 and D3

**Archive file:** `source/market/fred_sp500_dtb3_daily.csv`

**Use:** primary-target endpoint and accrual construction.

**Acquisition record:** archived data, not downloaded in this revision. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 2000-01-03 to 2026-05-19, 6882 rows/records. Units: index level and annual bill quote.

**Evidence:** `provenance/source_records/daily_market_return_validation.py`

**SHA-256 of named archive file:** `44a848819a80ac3123041cd0f80a9f247c75038daee87189686def976b4d1a7c`

Outer merge with forward filling. Price rows after 2024-12-31 repeat the last genuine Yahoo close and are not new price observations. Nonmissing values in selected column: 6882.

### D7. Published original attention and tone counts

**Source:** Cieslak and Vissing-Jorgensen, March 2020 NBER Working Paper 26894

**Source address or identifier:** https://www.nber.org/system/files/working_papers/w26894/w26894.pdf

**Archive file:** `source/w26894.pdf`

**Use:** recover Figures 4A and 5; verify Table 6 aggregates.

**Acquisition record:** source PDF supplied in earlier research archive. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 1994-02-04 to 2016-12-14, 184 rows/records. Units: meeting-level counts.

**Evidence:** `results/attention_recovery_audit.json; results/figure_recovery_audit.json`

**SHA-256 of named archive file:** `dbc3380b1463b0e72b6360a4e46cdc39658cfdf83606b9d010ebde0377d0bb88`

Numerical series recovered from PDF drawing objects. No original author raw dataset was received.

### D8. Original return criterion and count cross-check

**Source:** CVJ author-hosted June 2020 Internet Appendix

**Source address or identifier:** https://drive.google.com/file/d/1YA1jNNnWax_E8-QHpZwse7wOYjadf0JN/view

**Archive file:** `external/FedPut_onlineappendix_rfs2.pdf`

**Use:** recover Figure 2 returns and cross-check signed counts; algorithm from Appendix B.

**Acquisition record:** obtained through public author-hosted link during prior source-verification stage. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 1994-02-04 to 2016-12-14, 184 rows/records. Units: return percentage points and counts.

**Evidence:** `results/return_recovery_audit.json`

**SHA-256 of named archive file:** `025cb24a1fe23d36b3b618e1ced859596f5951324f63d5205bf8c763a126db46`

Exact download timestamp not retained here. Returns are coordinate approximations. Main article citation refers to the 2021 publication, but these exact source versions define the recovered series.

### D9. Gemini embedding-001 legacy metadata

**Source:** Google model name recorded in supplied generation manifest

**Source address or identifier:** https://arxiv.org/abs/2503.07891

**Archive file:** `provenance/source_records/legacy_gemini_manifest.json`

**Use:** lead pretrained text representation.

**Acquisition record:** supplied generation manifest and cached vectors. Original acquisition timestamp: `2026-06-10T11:32:08`.

**Stored coverage:** 2000-02-02 to 2024-12-18, 1314 rows/records. Units: 3072 vector coordinates per raw mention row.

**Evidence:** `provenance/source_records/legacy_gemini_manifest.json`

**SHA-256 of named archive file:** `8793f42059750b0199e14d6613bffa6a3d8a1b77cc6004fd7bfdbd873341faaa`

Generation time, not market-data download time. Deduplication yields 1,197 unique passages before pooling. Exact provider revision is not recorded.

### D10. Final original and masked Flash labels

**Source:** Supplied responses labeled gemini-3.8-flash

**Source address or identifier:** provider model designation from archived responses

**Archive file:** `source/focused/FOMC_LLM_Focused_Paper/data/flash_validated_labels.csv`

**Use:** five-category representation and masking diagnostic.

**Acquisition record:** archived model outputs. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 2000-02-02 to 2024-12-18, 2394 rows/records. Units: one categorical label per passage/variant.

**Evidence:** `provenance/extension_records/logs/flash_requests.jsonl; provenance/extension_records/inputs/handoff/prompts/flash_classification_prompt.txt`

**SHA-256 of named archive file:** `0cf3686f4bd29e48128609b298ab39dd5f948696995ae133b4473e76b931cf56`

1,197 original and 1,197 masked final labels. No immutable generation timestamp in the retained request log. Package label is dated September 20, 2026. Numerical re-estimation does not authenticate external inference.

### D11. Refreshed/masked embedding metadata

**Source:** Recorded Google gemini-embedding-001 configuration

**Source address or identifier:** https://arxiv.org/abs/2503.07891

**Archive file:** `provenance/extension_records/models/gemini001_refresh/original/embeddings.manifest.json`

**Use:** paired numerical-masking diagnostic.

**Acquisition record:** supplied generation records. Original acquisition timestamp: `not recorded`.

**Stored coverage:** 2000-02-02 to 2024-12-18, 2394 rows/records. Units: 3072 coordinates per original/masked record.

**Evidence:** `provenance/extension_records/models/gemini001_refresh/`

**SHA-256 of named archive file:** `992abc71d87f666472f02fbf803f2f6b9221f27d693fb51f79750515b81db8e4`

1,197 records per variant. No immutable request timestamp in retained logs. Original legacy pooled vectors remain the lead reference.

### D12. French daily factor ZIP and post-2024 minutes attempts

**Source:** Kenneth French Data Library and Federal Reserve Board

**Source address or identifier:** https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip

**Archive file:** `results/source_acquisition_manifest.json`

**Use:** planned validation, excluded from all empirical results.

**Acquisition record:** unsuccessful download attempts. Original acquisition timestamp: `2026-09-20T03:32:20–03:32:40+00:00`.

**Stored coverage:** not applicable to not applicable, 14 rows/records. Units: one factor-file and thirteen minutes-page attempts.

**Evidence:** `results/source_acquisition_manifest.json`

**SHA-256 of named archive file:** `0d51b8235b1fa864eab00ac4e11fc07a018e230d0b43391654e716ff9dd52f7a`

No fresh daily factor file or post-2024 same-model evaluation was added. The manifest header says data_cutoff 2026-09-19 but attempt timestamps are September 20 UTC; these fields describe different recorded dates.

## Source-version distinction

The economic article is CVJ (2021). Numerical recovery uses the exact March 2020 paper and June 2020 Internet Appendix. The March table reports 975 mentions. The appendix's algorithm-development prose refers to 983. These source-version totals have not been reconciled or overwritten. The recovered attention series and the primary aggregate cross-check use the March source. Human-count recovery does not establish exact algorithm-output replication.

## Newly included underlying documents

The raw HTML/PDF files in `raw_minutes/` and the extracted text files listed in the minutes register were recovered from the previously supplied full research archive. They were not downloaded anew. All 200 extracted-text hashes match the earlier CVJ benchmark input manifest. The archive extraction ledger records the original archive member and its hash. Raw source format comprises 199 HTML files and the June 25, 2008 PDF.

## Descriptive calculations

Run `python code/describe_data_sources.py` to recreate the source register, target-construction ledger, rule-box checks, and descriptive CSV tables. It performs no regression fitting, network requests, or model inference. Table 2A uses 184 source meetings in 1994–2016. Table 2B uses 135 common labeled meetings in March 2000–December 2016. Its correlations compare original human coding with a different reconstructed algorithm; they are not estimates of passage accuracy. Table 3 splits the existing 199-row panel into the initial 100 training meetings and 99 evaluation meetings.

## Rights and evidence limits

Original copyright and data-provider terms continue to apply. Inclusion in this research archive does not grant additional redistribution rights. Some provider versions and generation times are not recorded. Exact numerical reproduction does not authenticate undocumented external model calls. The archive contains no new post-2024 same-model validation sample.
