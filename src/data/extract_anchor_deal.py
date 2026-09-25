"""Builds the anchor consumer ABS capital structure from SEC EDGAR.

Anchor deal: Exeter Automobile Receivables Trust 2026-4 (sub-prime auto, 424B5 filed 2026-08-28).
Why not Upstart / LendingClub: their marketplace unsecured consumer trusts are Rule 144A deals, so no
424B2 / SF-3 prospectus exists on EDGAR (full-text search confirms zero hits). EART is the closest
publicly registered consumer credit shelf with a sub-prime borrower base (WA FICO 566).

Outputs:
  data/processed/anchor_deal_structure.csv   one row per note class
  data/processed/anchor_deal_collateral.csv  pool weighted averages + credit enhancement terms
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from edgar_client import download

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
BASE = "https://www.sec.gov/Archives/edgar/data/1654238"
PROSPECTUS = (f"{BASE}/000092963826003269/eart2026-4_424b5.htm", "eart2026-4_424b5.htm")
RATINGS_FWP = (f"{BASE}/000092963826003167/eart2026-4_ratingsfwp.htm", "eart2026-4_ratingsfwp.htm")
ASSET_DATA = (f"{BASE}/000092963826003150/eart2026-4_exhibit102.xml", "eart2026-4_absee_ex102.xml")
ABSEE_NS = "{http://www.sec.gov/edgar/document/absee/autoloan/assetdata}"


def flat_text(url_name: tuple) -> str:
    path = download(*url_name)
    t = BeautifulSoup(path.read_bytes(), "html.parser").get_text(" ")
    return re.sub(r"[\s\xa0]+", " ", t).replace("’", "'")


def _money(s: str) -> float:
    return float(s.replace(",", "").replace("$", ""))


def tranches(prospectus: str, fwp: str, pool_balance: float) -> pd.DataFrame:
    cover = prospectus[: prospectus.find("Price to Public")]
    rows = re.findall(r"Class ([A-Z](?:-\d)?) Notes ?(?:\(1\)(?:\(2\))?)? ?\$ ?([\d,]+) ?([\d.]+)% ?(Actual/360|30/360) ?(\w+ \d{1,2}, \d{4})", cover)
    df = pd.DataFrame(rows, columns=["note_class", "initial_principal_usd", "coupon_pct", "accrual", "final_scheduled_distribution"])
    df["initial_principal_usd"] = df["initial_principal_usd"].map(_money)
    df["coupon_pct"] = df["coupon_pct"].astype(float)

    ratings = dict(re.findall(r"Class ([A-Z](?:-\d)?) Notes ? ?([A-Za-z0-9+\-]+ \(sf\)) ?[A-Za-z0-9+\-]+ \(sf\)", fwp))
    sp = dict(re.findall(r"Class ([A-Z](?:-\d)?) Notes ? ?[A-Za-z0-9+\-]+ \(sf\) ?([A-Za-z0-9+\-]+ \(sf\))", fwp))
    df["rating_moodys"] = df["note_class"].map(ratings).fillna("NR (144A)")
    df["rating_sp"] = df["note_class"].map(sp).fillna("NR (144A)")
    df["rating_kbra"] = "not rated"
    df["offered"] = ~df["note_class"].isin(["E", "N"])

    # Class N is a residual-backed note paid below the E notes and outside the OC test, so it
    # carries no share of pool losses before E is written down; subordination is measured on A-E.
    pool_notes = df[df["note_class"] != "N"]
    seniority = {"A-1": "A", "A-2": "A", "A-3": "A"}
    grp = pool_notes["note_class"].map(lambda c: seniority.get(c, c))
    order = ["A", "B", "C", "D", "E"]
    cum = pool_notes.groupby(grp)["initial_principal_usd"].sum().reindex(order).cumsum()
    df["initial_hard_ce_pct"] = df["note_class"].map(lambda c: 1 - cum[seniority.get(c, c)] / pool_balance if c != "N" else np.nan)
    df["subordination_pct"] = df["note_class"].map(
        lambda c: (cum["E"] - cum[seniority.get(c, c)]) / pool_balance if c != "N" else np.nan
    )
    return df


def add_reserve(df: pd.DataFrame, reserve_pct: float) -> pd.DataFrame:
    df["initial_hard_ce_incl_reserve_pct"] = df["initial_hard_ce_pct"] + reserve_pct / 100
    return df


def collateral(prospectus: str) -> dict:
    grab = lambda pat: re.search(pat, prospectus).group(1)
    pool = _money(grab(r"aggregate Principal Balance of \$([\d,.]+);"))
    stats = {
        "deal": "Exeter Automobile Receivables Trust 2026-4",
        "form": "424B5",
        "accession": "0000929638-26-003269",
        "cutoff_date": pd.Timestamp(grab(r"cutoff date, which is (\w+ \d{1,2}, \d{4})")).strftime("%Y-%m-%d"),
        "original_collateral_balance_usd": pool,
        "number_of_contracts": int(grab(r"Number of Automobile Loan Contracts ?[\d,]+ ?[\d,]+ ?([\d,]+)").replace(",", "")),
        "gross_wac_pct": float(grab(r"weighted average annual percentage rate of approximately ([\d.]+)%")),
        "wa_original_term_months": int(grab(r"weighted average original term to maturity of approximately (\d+) months")),
        "wam_remaining_months": int(grab(r"weighted average remaining term to maturity of approximately (\d+) months")),
        "wa_fico": int(grab(r"Weighted Average FICO\W{0,3}Score \(4\) ?(\d+)")),
        "wa_credit_bureau_score": int(grab(r"credit bureau score is available\) of (\d+)")),
        "oc_initial_pct": float(grab(r"initial amount of overcollateralization with respect to the publicly offered notes and the Class E notes will be approximately ([\d.]+)%")),
        "oc_target_pct_current_pool": float(grab(r"Target Overcollateralization Amount .{0,120}?greater of \(i\) ([\d.]+)%")),
        "oc_floor_pct_cutoff_pool": float(grab(r"Target Overcollateralization Amount .{0,250}?\(ii\) ([\d.]+)%")),
        "reserve_initial_deposit_usd": _money(grab(r"At least \$([\d,]+) will be deposited into the reserve account")),
        "reserve_initial_pct_cutoff_pool": float(grab(r"deposited into the reserve account on the closing date, which is approximately ([\d.]+)%")),
        "reserve_specified_pct_cutoff_pool": float(grab(r"maintain the amount on deposit therein at ([\d.]+)% or more")),
        "servicing_fee_pct": float(grab(r"servicing fee, equal to ([\d.]+)%")),
    }
    return stats


def asset_level_pti() -> dict:
    """Balance-weighted payment-to-income from the Reg AB II asset-level file (the prospectus has no DTI)."""
    path = download(*ASSET_DATA)
    bal, pti = [], []
    for _, el in ET.iterparse(path):
        if el.tag == ABSEE_NS + "assets":
            bal.append(float(el.findtext(ABSEE_NS + "reportingPeriodActualEndBalanceAmount") or 0))
            pti.append(float(el.findtext(ABSEE_NS + "paymentToIncomePercentage") or "nan"))
            el.clear()
    bal, pti = np.array(bal), np.array(pti)
    ok = ~np.isnan(pti)
    return {
        "wa_reported_dti_pct": np.nan,  # not disclosed for auto ABS; PTI below is the closest reported burden metric
        "wa_payment_to_income_pct": round(100 * np.average(pti[ok], weights=bal[ok]), 2),
        "pti_loan_count": int(ok.sum()),
        "pti_source": "ABS-EE Ex.102 (0000929638-26-003150), preliminary pool",
    }


def build() -> tuple:
    prospectus, fwp = flat_text(PROSPECTUS), flat_text(RATINGS_FWP)
    pool = collateral(prospectus)
    pool.update(asset_level_pti())
    deal = add_reserve(tranches(prospectus, fwp, pool["original_collateral_balance_usd"]), pool["reserve_initial_pct_cutoff_pool"])
    deal.insert(0, "deal", pool["deal"])
    return deal, pd.DataFrame([pool])


if __name__ == "__main__":
    deal, pool = build()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    deal.round(6).to_csv(PROCESSED / "anchor_deal_structure.csv", index=False)
    pool.T.to_csv(PROCESSED / "anchor_deal_collateral.csv", header=["value"], index_label="field")
    print(deal.to_string())
    print(pool.T.to_string())
