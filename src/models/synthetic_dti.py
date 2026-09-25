"""Synthetic DTI: quantifies unrecorded BNPL (Pay-in-4) debt and calibrates the default multiplier M.

    DTI_true = DTI_reported + DTI_BNPL
    DTI_BNPL = ((AOV / 4) * 2 * N_active) / gross monthly income
    M        = b0 + b1 * (DTI_true - DTI_reported) / DTI_reported + b2 * savings + b3 * complaint velocity + e

The DTI surge varies over time because AOV and loan stacking are taken from Affirm's filed KPIs
(data/processed/bnpl_lender_kpis.csv): N_active scales with transactions per active consumer.

Two calibration targets are supported:
  * "synthetic": M generated from known coefficients plus seeded noise. Used to validate that the
    estimator recovers the planted betas before real data is plugged in.
  * "affirm":    M = Affirm 30+ DPD rate / its first-year average (quarterly, from 10-K/10-Q aging tables).

Macro features (savings rate, CFPB BNPL complaints) come from the Track 1 pipeline. Until it merges,
a seeded mock fixture is generated and every output row is tagged macro_source="mock".
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "mock_macro_features.csv"
TRACK1_PATH = ROOT / "data" / "processed" / "macro_features.csv"
KPI_PATH = ROOT / "data" / "processed" / "bnpl_lender_kpis.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "default_multiplier_summary.csv"

FEATURES = ["dti_surge", "savings_contraction", "complaint_velocity"]
# Coefficients planted in the synthetic target; the validation fit should recover these.
TRUE_BETAS = {"const": 1.0, "dti_surge": 0.40, "savings_contraction": 0.08, "complaint_velocity": 0.15}
SEED = 42
# Illustrative income brackets -> reported (bureau-visible) DTI. Lower incomes carry higher reported DTI.
INCOME_BRACKETS = {35000.0: 0.36, 50000.0: 0.34, 65000.0: 0.32, 85000.0: 0.30, 110000.0: 0.28}


def generate_mock_fixtures_if_needed(fixture_path: Path = FIXTURE_PATH) -> Path:
    """Writes seeded mock macro data covering Affirm's reporting history (Dec-2020 to Jun-2026)."""
    if fixture_path.exists():
        return fixture_path

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(start="2020-12-31", end="2026-06-30", freq="ME")
    n = len(dates)

    # Simulated squeeze: stimulus-era savings unwind, rising credit, surging complaints
    psavert = np.clip(np.linspace(13.0, 3.6, n) + rng.normal(0, 0.4, n), 2.0, 20.0)
    totalsl_yoy = np.linspace(0.03, 0.075, n) + rng.normal(0, 0.005, n)
    cpi_yoy = np.clip(np.linspace(0.05, 0.03, n) + rng.normal(0, 0.003, n), 0.01, 0.10)
    bnpl_complaints = np.clip(np.geomspace(120, 900, n) * rng.lognormal(0, 0.08, n), 50, 3000).astype(int)

    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "psavert": np.round(psavert, 2),
            "dspic96": np.round(np.linspace(16500, 17600, n), 1),
            "cpi_yoy": np.round(cpi_yoy, 4),
            "totalsl_yoy": np.round(totalsl_yoy, 4),
            "cfpb_bnpl_complaints": bnpl_complaints,
            "cfpb_narrative_count": (bnpl_complaints * 0.65).astype(int),
        }
    ).to_csv(fixture_path, index=False)
    print(f"[INIT] Mock macro fixtures generated at: {fixture_path}")
    return fixture_path


def load_macro() -> Tuple[pd.DataFrame, str]:
    """Track 1 output when available, otherwise the mock fixture."""
    if TRACK1_PATH.exists():
        return pd.read_csv(TRACK1_PATH, parse_dates=["date"]), "track1"
    return pd.read_csv(generate_mock_fixtures_if_needed(), parse_dates=["date"]), "mock"


class SyntheticDTIModel:
    """Quantifies unrecorded BNPL debt expansion and calibrates default multiplier M."""

    def __init__(self, aov: float = 320.0, active_loans: float = 3, reference_tpac: Optional[float] = None):
        self.aov = aov
        self.active_loans = active_loans
        # Transactions per active consumer at which the borrower carries `active_loans` loans.
        self.reference_tpac = reference_tpac

    @staticmethod
    def monthly_bnpl_payment(aov, active_loans):
        """Pay-in-4 installments are bi-weekly (AOV / 4), two cycles per month per stacked loan."""
        return (np.asarray(aov) / 4.0) * 2.0 * np.asarray(active_loans)

    def calculate_synthetic_dti(
        self, reported_dti, annual_gross_income, aov=None, active_loans=None
    ) -> Dict[str, np.ndarray]:
        """True DTI after adding unrecorded Pay-in-4 commitments. Inputs broadcast (scalars or arrays)."""
        aov = self.aov if aov is None else aov
        active_loans = self.active_loans if active_loans is None else active_loans
        reported_dti = np.asarray(reported_dti, dtype=float)
        monthly_income = np.asarray(annual_gross_income, dtype=float) / 12.0

        monthly_bnpl_debt = self.monthly_bnpl_payment(aov, active_loans)
        dti_bnpl = monthly_bnpl_debt / monthly_income
        true_dti = reported_dti + dti_bnpl

        return {
            "reported_dti": reported_dti,
            "monthly_income": monthly_income,
            "reported_debt_pmt": reported_dti * monthly_income,
            "monthly_bnpl_pmt": monthly_bnpl_debt,
            "dti_bnpl": dti_bnpl,
            "true_dti": true_dti,
            "dti_surge_pct": dti_bnpl / reported_dti,
        }

    def stacking_path(self, kpis: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
        """Monthly AOV and active-loan count, interpolated from Affirm's quarterly KPIs."""
        k = kpis.dropna(subset=["aov_derived_usd"])
        k = k.set_index(pd.to_datetime(k["period_end"]))
        ref_tpac = self.reference_tpac or k["transactions_per_active_consumer"].iloc[-1]
        path = pd.DataFrame(
            {
                "aov": k["aov_derived_usd"],
                "active_loans": self.active_loans * k["transactions_per_active_consumer"] / ref_tpac,
            }
        )
        idx = pd.DatetimeIndex(dates).union(path.index)
        # Interpolate between filings only; months outside the KPI history stay NaN (no extrapolation)
        return path.reindex(idx).interpolate(method="time", limit_area="inside").reindex(pd.DatetimeIndex(dates))

    def build_features(
        self, macro_df: pd.DataFrame, kpis: Optional[pd.DataFrame] = None,
        base_reported_dti: float = 0.32, benchmark_income: float = 65000.0,
        brackets: Optional[Dict[float, float]] = None,
    ) -> pd.DataFrame:
        """Monthly feature frame. With `brackets` ({income: reported_dti}) it is an income x month panel,
        which gives dti_surge cross-sectional variation that is not collinear with the macro trends."""
        df = macro_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if kpis is not None:
            path = self.stacking_path(kpis, df["date"])
            df["aov"], df["active_loans"] = path["aov"].values, path["active_loans"].values
        else:
            df["aov"], df["active_loans"] = self.aov, self.active_loans

        borrowers = pd.DataFrame(
            list((brackets or {benchmark_income: base_reported_dti}).items()), columns=["income", "reported_dti"]
        )
        df = df.merge(borrowers, how="cross").sort_values(["income", "date"]).reset_index(drop=True)
        stats = self.calculate_synthetic_dti(df["reported_dti"], df["income"], df["aov"], df["active_loans"])
        df["true_dti"] = stats["true_dti"]
        df["dti_surge"] = stats["dti_surge_pct"]
        df["savings_contraction"] = df["psavert"].max() - df["psavert"]
        # Complaint velocity: 3-month log change in BNPL complaints (smooths single-month noise)
        df["complaint_velocity"] = np.log(df["cfpb_bnpl_complaints"]).groupby(df["income"]).diff(3)
        return df.dropna(subset=FEATURES).sort_values(["income", "date"]).reset_index(drop=True)

    @staticmethod
    def synthetic_target(df: pd.DataFrame, noise_sd: float = 0.02, seed: int = SEED) -> pd.Series:
        rng = np.random.default_rng(seed)
        m = TRUE_BETAS["const"] + sum(TRUE_BETAS[f] * df[f] for f in FEATURES)
        return m + rng.normal(0, noise_sd, len(df))

    @staticmethod
    def affirm_target(df: pd.DataFrame, kpis: pd.DataFrame, baseline_quarters: int = 4) -> pd.DataFrame:
        """Collapse monthly features to fiscal quarter-ends and attach M = 30+ DPD / baseline 30+ DPD."""
        k = kpis.assign(date=pd.to_datetime(kpis["period_end"]))[["date", "dpd_30plus_pct"]]
        baseline = k["dpd_30plus_pct"].iloc[:baseline_quarters].mean()
        k["empirical_m"] = k["dpd_30plus_pct"] / baseline
        q = df.set_index("date").resample("QE").last().reset_index()  # single benchmark borrower
        return q.merge(k, on="date", how="inner")

    def calibrate_default_multiplier(
        self, macro_df: pd.DataFrame, kpis: Optional[pd.DataFrame] = None, target: str = "synthetic", **kwargs
    ) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
        """Fits an auditable OLS (Newey-West HAC errors, since the series are autocorrelated)."""
        if target == "synthetic":
            # Income x month panel; errors clustered by month since macro shocks hit every bracket at once
            df = self.build_features(macro_df, kpis, brackets=kwargs.pop("brackets", INCOME_BRACKETS), **kwargs)
            df["empirical_m"] = self.synthetic_target(df)
            fit_kw = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(df["date"])[0]}}
        elif target == "affirm":
            if kpis is None:
                raise ValueError("target='affirm' needs the Affirm KPI table")
            # Affirm M is a portfolio aggregate, so it is paired with the single benchmark borrower
            df = self.affirm_target(self.build_features(macro_df, kpis, **kwargs), kpis)
            fit_kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": 1}}
        else:
            raise ValueError(f"unknown target {target!r}")

        X = sm.add_constant(df[FEATURES], has_constant="add")
        model = sm.OLS(df["empirical_m"], X).fit(**fit_kw)
        df["predicted_m"] = model.predict(X)
        return model, df


def load_kpis() -> Optional[pd.DataFrame]:
    return pd.read_csv(KPI_PATH) if KPI_PATH.exists() else None


if __name__ == "__main__":
    macro_data, macro_source = load_macro()
    kpis = load_kpis()
    model_engine = SyntheticDTIModel(aov=320.0, active_loans=3)

    # 1. Benchmark borrower ($65,000 median income, 32% reported DTI, 3 stacked Pay-in-4 loans)
    metrics = model_engine.calculate_synthetic_dti(reported_dti=0.32, annual_gross_income=65000.0)
    print("=" * 60)
    print("STAGE 1: SYNTHETIC DTI EXPANSION METRICS")
    print(f"Reported DTI:                {metrics['reported_dti']*100:.2f}%")
    print(f"Monthly BNPL Add-On:         ${metrics['monthly_bnpl_pmt']:.2f}/mo")
    print(f"True (Synthetic) DTI:        {metrics['true_dti']*100:.2f}%")
    print(f"Underreported Debt Burden:   +{metrics['dti_surge_pct']*100:.2f}%")
    print("=" * 60)

    # 2. Estimator validation: the fit on a planted-coefficient target must recover TRUE_BETAS
    val_res, _ = model_engine.calibrate_default_multiplier(macro_data, kpis, target="synthetic")
    print("\nSTAGE 1a: ESTIMATOR VALIDATION (synthetic target, known betas)")
    print(pd.DataFrame({"true": TRUE_BETAS, "estimated": val_res.params, "cluster_se": val_res.bse}).round(4))

    # 3. Empirical calibration against Affirm's delinquency-implied multiplier
    if kpis is None:
        raise SystemExit("Run src/data/extract_affirm_kpis.py first to build bnpl_lender_kpis.csv")
    ols_res, reg_df = model_engine.calibrate_default_multiplier(macro_data, kpis, target="affirm")
    print(f"\nSTAGE 1b: AUDITABLE OLS REGRESSION SUMMARY (target: Affirm 30+ DPD multiplier, macro: {macro_source})")
    print(ols_res.summary())

    # 4. Export parameters for Track 3 Waterfall Engine
    reg_df["macro_source"] = macro_source
    reg_df["date"] = reg_df["date"].dt.strftime("%Y-%m-%d")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg_df[["date", "psavert", "aov", "active_loans", "true_dti", "dti_surge", "savings_contraction",
            "complaint_velocity", "dpd_30plus_pct", "empirical_m", "predicted_m", "macro_source"]].round(6).to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] Default multiplier exported to: {OUTPUT_PATH}")
    print(f"Current Calibrated Multiplier (Latest Quarter): {reg_df['predicted_m'].iloc[-1]:.3f}x")
    if macro_source == "mock":
        print("[WARN] Macro features are MOCK; coefficients on savings/complaints are placeholders until Track 1 merges.")
