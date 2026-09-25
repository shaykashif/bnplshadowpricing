"""Builds data/processed/bnpl_lender_kpis.csv from Affirm Holdings (CIK 1820953) 10-K / 10-Q filings.

Sources per filing:
  * Item 7 MD&A "Key Operating Metrics": GMV, active consumers, transactions per active consumer
  * Item 8 / Part I Item 1 loan footnote: aging of loans held for investment (amortized cost basis)
  * Consolidated statement of operations: provision for credit losses

Affirm does not disclose AOV in its SEC filings, so AOV is derived as
TTM GMV / (active consumers x transactions per active consumer), i.e. GMV per transaction.
Fiscal Q4 flow values (GMV, provision) are derived as FY total minus the 9M year-to-date figure.
"""
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from edgar_client import download, fetch

CIK = "1820953"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "affirm"
OUT = ROOT / "data" / "processed" / "bnpl_lender_kpis.csv"
CHARGE_OFF_DPD = 120  # "Loans are charged-off ... as the contractual principal becomes 120 days past due"


def filing_texts() -> list:
    subs = fetch(f"https://data.sec.gov/submissions/CIK{CIK.zfill(10)}.json").json()["filings"]["recent"]
    out = []
    for form, acc, doc, period in zip(subs["form"], subs["accessionNumber"], subs["primaryDocument"], subs["reportDate"]):
        if form not in ("10-K", "10-Q"):
            continue
        htm = download(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{acc.replace('-', '')}/{doc}", f"affirm/{period}_{form}.htm")
        txt = htm.with_suffix(".txt")
        if not txt.exists():
            t = BeautifulSoup(htm.read_bytes(), "html.parser").get_text("\n")
            txt.write_text("\n".join(l.strip() for l in t.splitlines() if l.strip()), encoding="utf8")
        flat = re.sub(r"[\s\xa0]+", " ", "|".join(txt.read_text(encoding="utf8").splitlines()))
        out.append({"period_end": period, "form": form, "accession": acc, "text": flat})
    return sorted(out, key=lambda f: f["period_end"])


def _num(tok: str) -> float:
    return float(tok.replace(",", "").replace("$", ""))


def _numbers(s: str, n: int) -> list:
    """First n numbers in a pipe-delimited run, treating '(|x|)' as negative."""
    s = re.sub(r"\(\|([\d,.]+)\|\)", r"-\1", s)
    toks = [t for t in s.split("|") if re.fullmatch(r"-?[\d,]+(\.\d+)?", t.strip())]
    return [_num(t) for t in toks[:n]]


def _prior_year_first(block: str) -> bool:
    """Early (FY2021) filings list the prior-year column first; detect via the header year order."""
    years = re.findall(r"\|(20\d\d)(?=\|)", block[:400])
    return len(years) >= 2 and int(years[0]) < int(years[1])


def _row(block: str, label: str) -> list:
    """Numeric cells of the table row starting with `label`, stopping at the next row label."""
    cells = []
    for tok in block[re.search(label, block).end():].split("|"):
        tok = tok.strip()
        if tok in ("", "$", "(1)"):
            continue
        if tok in ("—", "�", "-"):
            cells.append(0.0)
        elif re.fullmatch(r"[\d,]+(\.\d+)?", tok):
            cells.append(_num(tok))
        elif cells or re.search(r"[A-Za-z]", tok):
            break
    return cells


def parse_filing(f: dict) -> dict:
    s, is_k = f["text"], f["form"] == "10-K"
    month = int(f["period_end"][5:7])
    first_quarter = month == 9
    kpi = s[s.find("key operating metrics we use to evaluate"):][:6000]
    rev = _prior_year_first(kpi)

    m = re.search(r"GMV\)?\"?\)?\|\$\|([\d,.|$%A-Za-z ]+?)\|GMV", kpi) or re.search(r"(?:Gross Merchandise Volume|GMV)[^|]*\|\$\|(.+?)\|[A-Z]", kpi)
    vals = _numbers(re.sub(r"\|-?[\d,.]+\|%", "|", "|" + m.group(1)), 4)  # drop "% change" cells
    in_billions = "(in billions)" in kpi[:600]
    scale = 1000.0 if in_billions else 1 / 1000.0  # -> $ millions
    if is_k:
        gmv_period, gmv_ytd = vals[0] * scale, vals[0] * scale
    else:
        q, ytd = (vals[1], vals[3]) if rev else (vals[0], vals[0] if first_quarter else vals[2])
        gmv_period, gmv_ytd = q * scale, ytd * scale

    ac = _numbers(re.search(r"Active Consumers\|([\d,]+\|.{0,60})", s, re.I).group(1), 3)
    tp = _numbers(re.search(r"Transactions per Active Consumer(?: \(x\))?\|([\d.]+\|.{0,40})", s, re.I).group(1), 3)
    active = ac[-1] if rev else ac[0]
    tpac = tp[-1] if rev else tp[0]

    aging = s[re.search(r"aging analysis of the (?:unpaid principal balance|amortized cost basis)", s).start():][:2500]
    aging_basis = "unpaid_principal_balance" if "unpaid principal balance" in aging[:80] else "amortized_cost"

    vintage = "by fiscal year of origination" in aging[:300]
    dates = [pd.Timestamp(d) for d in re.findall(r"\|(\w+ \d{1,2}, 20\d\d)(?=\|)", aging[:600])]
    aging_rev = len(dates) >= 2 and dates[0] < dates[1]

    def bucket(label: str) -> float:
        vals = _row(aging, label)
        if vintage:
            return vals[-1]  # vintage table: Total is the last column
        return vals[1] if aging_rev else vals[0]

    total = bucket(r"Total (?:unpaid principal balance|amortized cost basis)\|")
    d30, d60, d90 = (bucket(fr"{a} ?[–\-�] ?{b} calendar days past due\|") for a, b in (("30", "59"), ("60", "89"), ("90", "119")))

    prov_vals = _numbers(re.search(r"Provision for credit losses\|(.{0,80})", s).group(1), 4)
    if is_k:
        prov_period = prov_ytd = prov_vals[0] / 1000
    else:
        q, ytd = (prov_vals[1], prov_vals[3]) if rev else (prov_vals[0], prov_vals[0] if first_quarter else prov_vals[2])
        prov_period, prov_ytd = q / 1000, ytd / 1000

    return {
        "period_end": f["period_end"], "form": f["form"], "accession": f["accession"],
        "gmv_period_musd": gmv_period, "gmv_ytd_musd": gmv_ytd,
        "active_consumers_k": active, "transactions_per_active_consumer": tpac,
        "loans_hfi_musd": total / 1000, "dpd_30_59_musd": d30 / 1000, "dpd_60_89_musd": d60 / 1000, "dpd_90_119_musd": d90 / 1000,
        "aging_basis": aging_basis, "provision_period_musd": prov_period, "provision_ytd_musd": prov_ytd,
    }


def build() -> pd.DataFrame:
    parsed = []
    for f in filing_texts():
        try:
            parsed.append(parse_filing(f))
        except Exception as e:
            raise RuntimeError(f"failed to parse {f['period_end']} {f['form']}") from e
    rows = pd.DataFrame(parsed)
    rows["period_end"] = pd.to_datetime(rows["period_end"])
    rows["fiscal_year"] = rows["period_end"].dt.year + (rows["period_end"].dt.month > 6)
    rows["fiscal_quarter"] = ((rows["period_end"].dt.month - 7) % 12) // 3 + 1

    # Fiscal Q4: 10-K reports the full year; subtract the Q3 10-Q year-to-date figure.
    q3_ytd = rows[rows["fiscal_quarter"] == 3].set_index("fiscal_year")[["gmv_ytd_musd", "provision_ytd_musd"]]
    annual = rows[rows["form"] == "10-K"].copy()
    for col in ("gmv", "provision"):
        rows.loc[rows["form"] == "10-K", f"{col}_period_musd"] = (
            annual[f"{col}_ytd_musd"].values - annual["fiscal_year"].map(q3_ytd[f"{col}_ytd_musd"]).values
        )

    q = rows.sort_values("period_end").reset_index(drop=True)
    q["gmv_ttm_musd"] = q["gmv_period_musd"].rolling(4).sum()
    q["aov_derived_usd"] = q["gmv_ttm_musd"] * 1e6 / (q["active_consumers_k"] * 1e3 * q["transactions_per_active_consumer"])
    for b in ("30_59", "60_89", "90_119"):
        q[f"dpd_{b}_pct"] = q[f"dpd_{b}_musd"] / q["loans_hfi_musd"]
    q["dpd_30plus_pct"] = q[["dpd_30_59_pct", "dpd_60_89_pct", "dpd_90_119_pct"]].sum(axis=1)
    q["dpd_60plus_pct"] = q[["dpd_60_89_pct", "dpd_90_119_pct"]].sum(axis=1)
    q["provision_to_loans_annualized"] = q["provision_period_musd"] * 4 / q["loans_hfi_musd"]
    q["chargeoff_policy_dpd"] = CHARGE_OFF_DPD
    q["period_end"] = q["period_end"].dt.strftime("%Y-%m-%d")

    cols = [
        "period_end", "fiscal_year", "fiscal_quarter", "form", "accession",
        "gmv_period_musd", "gmv_ttm_musd", "active_consumers_k", "transactions_per_active_consumer", "aov_derived_usd",
        "loans_hfi_musd", "aging_basis", "dpd_30_59_pct", "dpd_60_89_pct", "dpd_90_119_pct", "dpd_30plus_pct", "dpd_60plus_pct",
        "provision_period_musd", "provision_ytd_musd", "provision_to_loans_annualized", "chargeoff_policy_dpd",
    ]
    return q[cols]


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(OUT, index=False)
    print(df.to_string(max_cols=30))
    print(f"\n[SUCCESS] {len(df)} quarters written to {OUT}")
