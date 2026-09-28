# Cost and Run Ledger

The provider run used the credentials supplied for this task and the locked
protocol recorded a nominal spending ceiling of USD 5.00. Provider responses
did not expose an account-level invoice, so the exact charged amount cannot be
verified from local outputs.

## Request coverage

| Provider/model family | Requests in the corpus run | Input basis | Status |
|---|---:|---|---|
| Gemini Embedding 001 | 2,394 records | 1,197 original plus 1,197 masked | Complete |
| Gemini Embedding 2 | 2,394 records | 1,197 original plus 1,197 masked | Complete |
| Voyage Context 4 isolated | 2,394 records | 1,197 original plus 1,197 masked | Complete |
| Voyage Context 4 contextualized | 2,394 records | 1,197 original plus 1,197 masked | Complete |
| Gemini Flash structured classifier | 2,394 target records, including retries | original plus masked | Complete |

The embedding arms reused completed cache records after the first interrupted
run. The final package therefore contains the vectors and request-status logs,
not a claim that every completed vector was generated in the final process.

The final primary run has 2,394 valid Flash labels and zero failed records.
The logs do not contain a provider invoice. Any billing statement should use
the provider account dashboard as the authority. Smoke tests are excluded from
the corpus request counts above.
