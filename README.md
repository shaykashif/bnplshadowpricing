# Shadow Pricing: Week 1 (Fundamental Credit & Research)

## Run order

```bash
pip install -r requirements.txt
python src/data/extract_anchor_deal.py      # -> data/processed/anchor_deal_structure.csv, anchor_deal_collateral.csv
python src/data/extract_affirm_kpis.py      # -> data/processed/bnpl_lender_kpis.csv
python src/models/synthetic_dti.py          # -> data/processed/default_multiplier_summary.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/02_synthetic_dti_calibration.ipynb
```

Filings are cached in `data/raw/` on first run. The EART asset-level XML is ~144 MB.

## Deliverables

| File | Contents |
|---|---|
| `data/processed/anchor_deal_structure.csv` | EART 2026-4 note classes: principal, coupon, accrual, final date, Moody's/S&P ratings, hard CE, subordination |
| `data/processed/anchor_deal_collateral.csv` | Pool balance, Gross WAC, WAM, WA FICO, WA PTI, OC initial/target/floor, reserve deposit, servicing fee |
| `data/processed/bnpl_lender_kpis.csv` | Affirm quarterly FY21-Q2 to FY26-Q4: GMV, active consumers, transactions/consumer, derived AOV, 30-59/60-89/90-119 DPD % of amortized cost, provision, 120-day charge-off policy |
| `src/models/synthetic_dti.py` | Synthetic DTI + OLS default multiplier (synthetic validation target and Affirm empirical target) |
| `notebooks/02_synthetic_dti_calibration.ipynb` | Loan stacking vs. income bracket, estimator validation, empirical M |

## Data decisions to review

- **Anchor deal is Exeter Automobile Receivables Trust 2026-4, not Upstart/LendingClub.** Their marketplace trusts are Rule 144A. They have no 424B2 or SF-3 on EDGAR, and a full-text search returns zero hits. EART 2026-4 is the closest publicly registered sub-prime consumer deal (424B5, WA FICO 566). It is **secured auto** credit, not unsecured.
- **No pool-level DTI is disclosed.** The collateral file reports balance-weighted **payment-to-income** (10.39%) from the Reg AB II loan-level data (ABS-EE Ex.102). That data covers the preliminary pool: 42,065 loans versus 45,996 at the final cutoff.
- **KBRA did not rate EART 2026-4** (Moody's and S&P only). Classes E and N are 144A and unrated.
- **Affirm does not report AOV in its SEC filings.** `aov_derived_usd` = TTM GMV / (active consumers × transactions per active consumer).
- Fiscal Q4 GMV and provision are calculated as the FY total minus the 9M year-to-date figure. From FY24 on, GMV is reported to $0.1B, so Q4 carries that rounding.
- The 2020-12-31 aging table uses unpaid principal balance including held-for-sale loans. All later tables use amortized cost.
- **Macro features are mock** (`tests/fixtures/mock_macro_features.csv`) until Track 1 writes `data/processed/macro_features.csv`, which the model picks up automatically.
"# bnplshadowpricing" 
