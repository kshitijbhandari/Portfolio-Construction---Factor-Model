# Auto-generated from model.ipynb
# Core functions for factor model portfolio optimization

import pandas as pd
import numpy as np
import pulp
from scipy import stats
import warnings
import time
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

def estimate_betas_asof_nifty(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    asof: str | pd.Timestamp,          # e.g. "2022-06" or "2022-06-30"
    tickers_in_window: list[str] | None = None,
    universe_df: pd.DataFrame | None = None,
    year: int | None = None,
    lookback_months: int = 60,
    min_obs: int = 36,
    include_mom: bool = False,
    use_t_as_last_obs: bool = True,    # False => end at t-1
) -> pd.DataFrame:
    """
    Returns betas for ONE rebalance month (asof), for only the tickers in window.

    Regression:
      (RET - RF) ~ 1 + MF + SMB + HML (+ WML if include_mom)

    returns_df columns: Date, Ticker, RET  (RET in decimals)
    factors_df columns: Date, RF, MF, SMB, HML (+ WML) (often in #% -> auto scaled)
    universe_df (optional): Year + columns 0..N with tickers
    """

    # ----- tickers -----
    if tickers_in_window is None:
        if universe_df is None or year is None:
            raise ValueError("Provide tickers_in_window OR (universe_df and year).")
        row = universe_df.loc[universe_df["Year"] == year]
        if row.empty:
            raise ValueError(f"Year {year} not found in universe_df.")
        tickers_in_window = (
            row.drop(columns=["Year"])
               .iloc[0]
               .dropna()
               .astype(str)
               .tolist()
        )

    # ----- parse dates to month-end -----
    r = returns_df.copy()
    f = factors_df.copy()

    r["Date"] = pd.to_datetime(r["Date"].astype(str)).dt.to_period("M").dt.to_timestamp("M")
    f["Date"] = pd.to_datetime(f["Date"].astype(str)).dt.to_period("M").dt.to_timestamp("M")

    asof_ts = pd.to_datetime(str(asof)).to_period("M").to_timestamp("M")

    # ----- factor set -----
    base_factors = ["MF", "SMB", "HML"]
    if include_mom:
        base_factors.append("WML")

    needed_r = {"Date", "Ticker", "RET"}
    needed_f = {"Date", "RF", *base_factors}
    if not needed_r.issubset(r.columns):
        raise ValueError(f"returns_df missing columns: {sorted(needed_r - set(r.columns))}")
    if not needed_f.issubset(f.columns):
        raise ValueError(f"factors_df missing columns: {sorted(needed_f - set(f.columns))}")

    # ----- merge -----
    r = r.loc[r["Ticker"].isin(tickers_in_window), ["Date", "Ticker", "RET"]].copy()
    merged = (
        r.merge(f[["Date", "RF"] + base_factors], on="Date", how="inner")
         .sort_values(["Ticker", "Date"])
    )

    # ----- scale factors/RF if they are in percent -----
    def _maybe_percent_to_decimal(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.abs().median(skipna=True) > 1.5:  # percent-like
            return s / 100.0
        return s

    for col in ["RF"] + base_factors:
        merged[col] = _maybe_percent_to_decimal(merged[col])

    merged["ret_excess"] = merged["RET"] - merged["RF"]

    # ----- decide regression end date -----
    end_date = asof_ts if use_t_as_last_obs else (asof_ts - pd.offsets.MonthEnd(1))

    out = []
    for tic, g in merged.groupby("Ticker", sort=False):
        g = g.set_index("Date").sort_index()

        if end_date not in g.index:
            continue

        end_loc = g.index.get_loc(end_date)
        start_loc = end_loc - (lookback_months - 1)
        if start_loc < 0:
            continue

        window = g.iloc[start_loc:end_loc + 1][["ret_excess"] + base_factors].dropna()
        if len(window) < min_obs:
            continue

        y = window["ret_excess"].to_numpy()
        X = window[base_factors].to_numpy()
        X = np.column_stack([np.ones(len(X)), X])

        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        resid_vol = float(resid.std(ddof=len(base_factors) + 1))

        row = {
            "Date": asof_ts,   # betas "as of" rebalance month
            "Ticker": tic,
            "alpha": float(coef[0]),
            "resid_vol": resid_vol,
            "n_obs": int(len(window)),
        }
        for k, fac in enumerate(base_factors, start=1):
            row[f"beta_{fac}"] = float(coef[k])

        out.append(row)

    # Handle empty output
    if len(out) == 0:
        return pd.DataFrame(columns=["Date", "Ticker", "alpha", "resid_vol", "n_obs", "beta_MF", "beta_SMB", "beta_HML"])
    
    return pd.DataFrame(out).sort_values(["Date", "Ticker"]).reset_index(drop=True)


import numpy as np
import pandas as pd

def _to_month_end(x):
    return pd.to_datetime(str(x)).to_period("M").to_timestamp("M")

def get_factor_window_asof(
    factors_df: pd.DataFrame,
    asof: str | pd.Timestamp,
    lookback_months: int = 60,
    factor_cols: list[str] = ["MF", "SMB", "HML"],
) -> pd.DataFrame:
    f = factors_df.copy()
    f["Date"] = pd.to_datetime(f["Date"].astype(str)).dt.to_period("M").dt.to_timestamp("M")
    asof_ts = _to_month_end(asof)

    f = f.sort_values("Date")
    if asof_ts not in set(f["Date"]):
        raise ValueError(f"asof={asof_ts.date()} not found in factors_df['Date'].")

    end_loc = f.index[f["Date"] == asof_ts][0]
    start_loc = end_loc - (lookback_months - 1)
    if start_loc < 0:
        raise ValueError("Not enough factor history for lookback_months.")

    window = f.iloc[start_loc:end_loc+1][["Date"] + factor_cols].dropna()
    if len(window) < lookback_months:
        # allow some missing but warn via error for now
        raise ValueError("Factor window has missing rows; fix factor data gaps.")

    return window

def compute_sigma_f_and_lambda(
    factors_df: pd.DataFrame,
    asof: str | pd.Timestamp,
    lookback_months: int = 60,
    factor_cols: list[str] = ["MF", "SMB", "HML"],
) -> tuple[pd.DataFrame, pd.Series]:
    window = get_factor_window_asof(factors_df, asof, lookback_months, factor_cols)
    Sigma_f = window[factor_cols].cov()          # 3x3
    lam = window[factor_cols].mean()             # 3-vector (monthly premia)
    return Sigma_f, lam

def build_mu_and_sigma_from_betas(
    betas_asof: pd.DataFrame,
    Sigma_f: pd.DataFrame,
    lam: pd.Series,
    factor_cols: list[str] = ["MF", "SMB", "HML"],
    add_diag_jitter: float = 1e-10,  # numerical stability
) -> tuple[pd.Series, pd.DataFrame]:
    # ensure order + required cols
    beta_cols = [f"beta_{c}" for c in factor_cols]
    needed = {"Ticker", "resid_vol", *beta_cols}
    missing = needed - set(betas_asof.columns)
    if missing:
        raise ValueError(f"betas_asof missing columns: {sorted(missing)}")

    df = betas_asof[["Ticker", "resid_vol"] + beta_cols].dropna().copy()
    df = df.sort_values("Ticker").reset_index(drop=True)

    B = df[beta_cols].to_numpy()                              # N x K
    Sigma_f_np = Sigma_f.loc[factor_cols, factor_cols].to_numpy()  # K x K
    D = np.diag((df["resid_vol"].to_numpy())**2)              # N x N

    Sigma = B @ Sigma_f_np @ B.T + D
    Sigma = Sigma + np.eye(Sigma.shape[0]) * add_diag_jitter  # stabilize

    # expected stock returns: mu = B * lam
    lam_np = lam.loc[factor_cols].to_numpy()
    mu = B @ lam_np                                           # N

    mu_s = pd.Series(mu, index=df["Ticker"], name="mu")
    Sigma_df = pd.DataFrame(Sigma, index=df["Ticker"], columns=df["Ticker"])
    return mu_s, Sigma_df

import pulp
import pandas as pd

def compute_achievable_beta_bounds(
    R: pd.DataFrame,
    betas_asof: pd.DataFrame,
    K_max: int = 20,
    w_max: float = 0.10,
    solver: pulp.LpSolver | None = None,
) -> dict:
    factors = ["MF", "SMB", "HML"]
    need_cols = ["Ticker", "beta_MF", "beta_SMB", "beta_HML"]

    if solver is None:
        solver_use = pulp.PULP_CBC_CMD(msg=False)
    else:
        solver_use = solver

    # Universe tickers from R
    tickers = list(R.columns)

    # Align betas to tickers and drop missing-beta tickers
    bdf = (
        betas_asof[need_cols]
        .drop_duplicates("Ticker")
        .set_index("Ticker")
        .reindex(tickers)
    )
    good = bdf.dropna(axis=0, how="any")
    tickers = good.index.tolist()

    # --- ADDED: feasibility checks ---
    if len(tickers) == 0:
        return {"error": "No tickers with complete betas after alignment."}

    if K_max <= 0:
        return {"error": "K_max must be >= 1"}

    if K_max * w_max < 1.0 - 1e-12:
        return {"error": f"Infeasible constraints: K_max*w_max={K_max*w_max:.3f} < 1.0"}

    betas_by_ticker = {
        t: {"MF": float(good.loc[t, "beta_MF"]),
            "SMB": float(good.loc[t, "beta_SMB"]),
            "HML": float(good.loc[t, "beta_HML"])}
        for t in tickers
    }

    bounds = {}

    for factor in factors:
        # --- MIN ---
        prob_min = pulp.LpProblem(f"MinBeta_{factor}", pulp.LpMinimize)
        w = {t: pulp.LpVariable(f"wmin_{factor}_{t}", lowBound=0) for t in tickers}
        z = {t: pulp.LpVariable(f"zmin_{factor}_{t}", cat="Binary") for t in tickers}

        for t in tickers:
            prob_min += w[t] <= w_max * z[t]
        prob_min += pulp.lpSum(z[t] for t in tickers) <= int(K_max)
        prob_min += pulp.lpSum(w[t] for t in tickers) == 1.0

        expr_min = pulp.lpSum(w[t] * betas_by_ticker[t][factor] for t in tickers)
        prob_min += expr_min

        status_min = prob_min.solve(solver_use)
        min_beta = float(pulp.value(expr_min)) if pulp.LpStatus[status_min] == "Optimal" else None

        # --- MAX ---
        prob_max = pulp.LpProblem(f"MaxBeta_{factor}", pulp.LpMaximize)
        w2 = {t: pulp.LpVariable(f"wmax_{factor}_{t}", lowBound=0) for t in tickers}
        z2 = {t: pulp.LpVariable(f"zmax_{factor}_{t}", cat="Binary") for t in tickers}

        for t in tickers:
            prob_max += w2[t] <= w_max * z2[t]
        prob_max += pulp.lpSum(z2[t] for t in tickers) <= int(K_max)
        prob_max += pulp.lpSum(w2[t] for t in tickers) == 1.0

        expr_max = pulp.lpSum(w2[t] * betas_by_ticker[t][factor] for t in tickers)
        prob_max += expr_max

        status_max = prob_max.solve(solver_use)
        max_beta = float(pulp.value(expr_max)) if pulp.LpStatus[status_max] == "Optimal" else None

        bounds[factor] = {"min": min_beta, "max": max_beta}

    return bounds

def compute_conditional_beta_bounds(
    R: pd.DataFrame,
    betas_asof: pd.DataFrame,
    target_betas: dict,
    beta_tolerances: dict,
    K_max: int = 20,
    w_max: float = 0.10,
    solver: pulp.LpSolver | None = None,
) -> dict:
    factors = ["MF", "SMB", "HML"]
    need_cols = ["Ticker", "beta_MF", "beta_SMB", "beta_HML"]

    if solver is None:
        solver_use = pulp.PULP_CBC_CMD(msg=False)
    else:
        solver_use = solver

    tickers = list(R.columns)
    bdf = (
        betas_asof[need_cols]
        .drop_duplicates("Ticker")
        .set_index("Ticker")
        .reindex(tickers)
    )
    good = bdf.dropna(axis=0, how="any")
    tickers = good.index.tolist()

    if len(tickers) == 0:
        return {"error": "No tickers with complete betas after alignment."}
    if K_max * w_max < 1.0 - 1e-12:
        return {"error": f"Infeasible constraints: K_max*w_max={K_max*w_max:.3f} < 1.0"}

    betas_by_ticker = {
        t: {"MF": float(good.loc[t, "beta_MF"]),
            "SMB": float(good.loc[t, "beta_SMB"]),
            "HML": float(good.loc[t, "beta_HML"])}
        for t in tickers
    }

    out = {}

    for focus in factors:
        # Build a base model (constraints same for min/max; only objective differs)
        def solve_for(sense: str):
            prob = pulp.LpProblem(f"{sense}_{focus}_conditional", pulp.LpMinimize if sense=="min" else pulp.LpMaximize)
            w = {t: pulp.LpVariable(f"w_{sense}_{focus}_{t}", lowBound=0) for t in tickers}
            z = {t: pulp.LpVariable(f"z_{sense}_{focus}_{t}", cat="Binary") for t in tickers}

            for t in tickers:
                prob += w[t] <= w_max * z[t]
            prob += pulp.lpSum(z[t] for t in tickers) <= int(K_max)
            prob += pulp.lpSum(w[t] for t in tickers) == 1.0

            # Enforce OTHER factor bands (and you can include focus too if you want)
            for k in factors:
                if k == focus:
                    continue
                expr_k = pulp.lpSum(w[t] * betas_by_ticker[t][k] for t in tickers)
                tgt = float(target_betas[k])
                tol = float(beta_tolerances.get(k, 0.0))
                prob += expr_k >= tgt - tol
                prob += expr_k <= tgt + tol

            expr_focus = pulp.lpSum(w[t] * betas_by_ticker[t][focus] for t in tickers)
            prob += expr_focus

            status = prob.solve(solver_use)
            if pulp.LpStatus[status] == "Optimal":
                return float(pulp.value(expr_focus))
            return None

        out[focus] = {"min": solve_for("min"), "max": solve_for("max")}

    return out

def optimize_pulp_mad_targetbetas_cardinality(
    mu: pd.Series,                     # expected returns (index=tickers). Can be None if objective_mode="min_risk".
    R: pd.DataFrame,                   # scenario returns: rows=months, cols=tickers (decimals)
    betas_asof: pd.DataFrame,          # must have: Ticker, beta_MF, beta_SMB, beta_HML
    target_betas: dict,                # {"MF":1.0,"SMB":0.0,"HML":0.2}
    beta_tolerances: dict | None = None,  # {"MF":0.05,"SMB":0.05,"HML":0.05}. None => exact (tol=0)
    K_max: int = 20,                   # NEW: max number of stocks in portfolio
    risk_aversion: float = 5.0,        # lambda on MAD risk
    w_max: float = 0.10,
    w_min_if_selected: float = 0.0,    # optional: enforce w_i >= w_min * z_i to avoid tiny weights
    w_prev: pd.Series | None = None,   # previous weights for turnover
    turnover_cap: float | None = None, # e.g. 0.30
    objective_mode: str = "return_minus_risk",  # "return_minus_risk" or "min_risk"
    solver: pulp.LpSolver | None = None,
) -> pd.Series:
    """
    MILP (PuLP):
      maximize mu^T w - lambda * MAD
      subject to:
        sum(w)=1, 0<=w_i<=w_max*z_i, sum(z_i)<=K_max
        target betas within band (or exact)
        optional turnover constraint
    """
    factors = ["MF", "SMB", "HML"]
    need_cols = ["Ticker", "beta_MF", "beta_SMB", "beta_HML"]
    missing = set(need_cols) - set(betas_asof.columns)
    if missing:
        raise ValueError(f"betas_asof missing columns: {sorted(missing)}")

    # Decide universe tickers
    if mu is not None:
        tickers = list(mu.index)
    else:
        tickers = list(R.columns)

    # Align scenario returns to tickers and remove months with missing returns
    R = R.copy()
    R = R[tickers].dropna(axis=0, how="any")
    if R.shape[0] < 12:
        raise ValueError("Not enough scenario months in R after dropping NaNs (need ~12+).")

    # Feasibility checks
    if K_max <= 0:
        raise ValueError("K_max must be >= 1")
    if K_max * w_max < 1.0 - 1e-12:
        raise ValueError(f"Infeasible: K_max*w_max={K_max*w_max:.3f} < 1. Increase w_max or K_max.")

    # Build betas dict in ticker order
    bdf = (betas_asof[need_cols]
           .drop_duplicates("Ticker")
           .set_index("Ticker")
           .reindex(tickers))
    if bdf.isna().any().any():
        bad = bdf[bdf.isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing betas for tickers: {bad[:10]} (showing up to 10)")

    betas_by_ticker = {
        t: {"MF": float(bdf.loc[t, "beta_MF"]),
            "SMB": float(bdf.loc[t, "beta_SMB"]),
            "HML": float(bdf.loc[t, "beta_HML"])}
        for t in tickers
    }

    # Target beta tolerances (0 => exact)
    if beta_tolerances is None:
        beta_tolerances = {k: 0.0 for k in factors}
    for k in factors:
        if k not in target_betas:
            raise ValueError(f"target_betas must include '{k}'")
        beta_tolerances.setdefault(k, 0.0)

    # --- Build MILP ---
    prob = pulp.LpProblem("MAD_TargetBeta_Cardinality", pulp.LpMaximize)

    # Decision variables
    w = {t: pulp.LpVariable(f"w_{t}", lowBound=0) for t in tickers}                # continuous weights
    z = {t: pulp.LpVariable(f"z_{t}", lowBound=0, upBound=1, cat="Binary") for t in tickers}  # selection

    # Linking constraints: if z=0 => w=0; if z=1 => w<=w_max
    for t in tickers:
        prob += w[t] <= w_max * z[t]
        if w_min_if_selected > 0:
            prob += w[t] >= w_min_if_selected * z[t]

    # Cardinality constraint
    prob += pulp.lpSum(z[t] for t in tickers) <= int(K_max)

    # Fully invested
    prob += pulp.lpSum(w[t] for t in tickers) == 1.0

    # --- Target beta constraints (band) ---
    for k in factors:
        expr = pulp.lpSum(w[t] * betas_by_ticker[t][k] for t in tickers)
        tgt = float(target_betas[k])
        tol = float(beta_tolerances.get(k, 0.0))
        prob += expr >= (tgt - tol)
        prob += expr <= (tgt + tol)

    # --- MAD Risk construction ---
    T = R.shape[0]
    rp = []
    for tt in range(T):
        rp_tt = pulp.lpSum(w[t] * float(R.iloc[tt][t]) for t in tickers)  # portfolio return in scenario tt
        rp.append(rp_tt)

    mp = (1.0 / T) * pulp.lpSum(rp)  # mean portfolio return across scenarios

    u = [pulp.LpVariable(f"u_{tt}", lowBound=0) for tt in range(T)]
    for tt in range(T):
        prob += u[tt] >= rp[tt] - mp
        prob += u[tt] >= -(rp[tt] - mp)

    mad = (1.0 / T) * pulp.lpSum(u)

    # --- Optional turnover cap ---
    if turnover_cap is not None:
        if w_prev is None:
            raise ValueError("Provide w_prev if using turnover_cap.")
        w_prev = w_prev.reindex(tickers).fillna(0.0).astype(float)

        dpos = {t: pulp.LpVariable(f"dpos_{t}", lowBound=0) for t in tickers}
        dneg = {t: pulp.LpVariable(f"dneg_{t}", lowBound=0) for t in tickers}
        for t in tickers:
            prob += w[t] - float(w_prev[t]) == dpos[t] - dneg[t]
        prob += pulp.lpSum(dpos[t] + dneg[t] for t in tickers) <= float(turnover_cap)

    # --- Objective ---
    if objective_mode == "min_risk":
        prob += -mad
    elif objective_mode == "return_minus_risk":
        if mu is None:
            raise ValueError("mu must be provided when objective_mode='return_minus_risk'")
        mu_vals = mu.reindex(tickers).astype(float).to_dict()
        prob += pulp.lpSum(mu_vals[t] * w[t] for t in tickers) - float(risk_aversion) * mad
    else:
        raise ValueError("objective_mode must be 'return_minus_risk' or 'min_risk'")

    # Solve
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)

    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        # Compute achievable beta bounds to inform the user
        achievable = compute_achievable_beta_bounds(R, betas_asof, K_max, w_max, solver)
        
        # Format error message with achievable bounds
        error_msg = f"\n{'='*80}\nPuLP Optimization INFEASIBLE: {pulp.LpStatus[status]}\n{'='*80}\n"
        error_msg += f"\nTarget Betas Requested:\n"
        for factor in factors:
            tgt = float(target_betas.get(factor, 0.0))
            tol = float(beta_tolerances.get(factor, 0.0))
            error_msg += f"  {factor}: {tgt:.4f} ± {tol:.4f}  (range: [{tgt-tol:.4f}, {tgt+tol:.4f}])\n"
        
        error_msg += f"\nAchievable Beta Ranges (given available universe):\n"
        for factor in factors:
            bounds = achievable[factor]
            if bounds["min"] is not None and bounds["max"] is not None:
                error_msg += f"  {factor}: [{bounds['min']:.4f}, {bounds['max']:.4f}]\n"
            else:
                error_msg += f"  {factor}: Could not compute bounds\n"
        
        error_msg += f"\n{'='*80}\n"
        error_msg += "SOLUTION: Adjust target_betas or beta_tolerances to fit within achievable ranges.\n"
        error_msg += f"{'='*80}\n"
        
        raise RuntimeError(error_msg)

    w_out = pd.Series({t: float(pulp.value(w[t])) for t in tickers}, name="weight")
    return w_out[w_out > 0].sort_values(ascending=False)

import pandas as pd
import numpy as np
import pulp


def debug_optimization_failure(
    *,
    asof: pd.Timestamp | str | None,
    status: str,
    R: pd.DataFrame,
    betas_asof: pd.DataFrame,
    tickers_requested: list[str] | None,
    target_betas: dict,
    beta_tolerances: dict,
    K_max: int,
    w_max: float,
    w_min_if_selected: float = 0.0,
    turnover_cap: float | None = None,
    w_prev: pd.Series | None = None,
    solver_name: str = "CBC",
    objective_mode: str = "return_minus_risk",
    # bounds functions are optional (pass if you have them)
    compute_1d_bounds_fn=None,
    compute_conditional_bounds_fn=None,
    print_mismatch_samples: int = 8,
):
    """
    Prints a focused debug report for MILP failures.
    - Works even if bounds solvers fail; will print why.
    - Assumes betas_asof has columns: Ticker, beta_MF, beta_SMB, beta_HML
    """

    def _fmt_ts(x):
        if x is None:
            return "None"
        try:
            return pd.to_datetime(x).strftime("#%Y-#%m-#%d")
        except Exception:
            return str(x)

    factors = ["MF", "SMB", "HML"]
    need_cols = ["Ticker", "beta_MF", "beta_SMB", "beta_HML"]

    print("\n" + "=" * 95)
    print("OPTIMIZATION FAILURE DEBUG")
    print("=" * 95)
    print(f"asof: {_fmt_ts(asof)}")
    print(f"status: {status}")
    print(f"solver: {solver_name}")
    print(f"objective_mode: {objective_mode}")

    # --- Data sizes ---
    T = R.shape[0]
    nR = R.shape[1]
    print("\n[DATA]")
    print(f"Scenario months (T): {T}")
    print(f"Tickers in R.columns: {nR}")

    if tickers_requested is not None:
        print(f"Tickers requested (pre-filter): {len(tickers_requested)}")

    # --- Return sanity ---
    Rvals = R.to_numpy().ravel()
    Rvals = Rvals[~np.isnan(Rvals)]
    if Rvals.size > 0:
        print(f"R return stats: min={np.min(Rvals):.4f}, median={np.median(Rvals):.4f}, max={np.max(Rvals):.4f}")
        # quick scale hint
        if np.nanmedian(np.abs(Rvals)) > 0.5:
            print("WARNING: Median |return| > 0.5, returns may be in #% not decimals (scale issue).")
    else:
        print("R return stats: all NaN (this alone can cause failure).")

    # --- Beta availability ---
    print("\n[BETAS ALIGNMENT]")
    missing_cols = [c for c in need_cols if c not in betas_asof.columns]
    if missing_cols:
        print(f"ERROR: betas_asof missing columns: {missing_cols}")
        print("=" * 95 + "\n")
        return

    betas_df = (
        betas_asof[need_cols]
        .drop_duplicates("Ticker")
        .set_index("Ticker")
        .reindex(R.columns)
    )
    n_with_betas = int(betas_df.dropna(axis=0, how="any").shape[0])
    n_missing_betas = int(betas_df.isna().any(axis=1).sum())

    print(f"Tickers with complete betas (aligned to R): {n_with_betas}")
    print(f"Tickers missing any beta (aligned to R): {n_missing_betas}")

    # ticker mismatch samples (if tickers_requested provided)
    if tickers_requested is not None:
        set_req = set(tickers_requested)
        set_R = set(R.columns)
        in_req_not_R = sorted(list(set_req - set_R))[:print_mismatch_samples]
        in_R_not_req = sorted(list(set_R - set_req))[:print_mismatch_samples]
        print(f"Sample tickers in requested but NOT in R: {in_req_not_R}")
        print(f"Sample tickers in R but NOT in requested: {in_R_not_req}")

    # --- Hard feasibility checks ---
    print("\n[CONSTRAINT QUICK CHECKS]")
    print(f"K_max={K_max}, w_max={w_max:.4f}, K_max*w_max={K_max*w_max:.4f}")
    if K_max * w_max < 1.0 - 1e-12:
        print("HARD INFEASIBLE: K_max*w_max < 1.0 (cannot be fully invested).")

    print(f"w_min_if_selected={w_min_if_selected:.4f}")
    if w_min_if_selected > 0:
        print(f"K_max*w_min_if_selected={K_max*w_min_if_selected:.4f}")
        if K_max * w_min_if_selected > 1.0 + 1e-12:
            print("HARD INFEASIBLE: K_max*w_min_if_selected > 1.0 (cannot allocate min weights).")

    print(f"turnover_cap={turnover_cap}")
    if turnover_cap is not None and w_prev is not None:
        # best-effort: show prev exposure vs target
        w_prev_aligned = w_prev.reindex(R.columns).fillna(0.0).astype(float)
        # compute current exposures from betas_df (only where betas exist)
        good_idx = betas_df.dropna(axis=0, how="any")
        common = good_idx.index.intersection(w_prev_aligned.index)
        if len(common) > 0:
            w_prev_common = w_prev_aligned.reindex(common).fillna(0.0)
            # normalize in case it doesn't sum to 1
            s = float(w_prev_common.sum())
            if s > 0:
                w_prev_common = w_prev_common / s

            exp_prev = {}
            for f in factors:
                col = "beta_" + f
                exp_prev[f] = float((w_prev_common * good_idx.loc[common, col].astype(float)).sum())

            print("Prev exposures (normalized on available-beta names): " +
                  ", ".join([f"{k}={v:.4f}" for k, v in exp_prev.items()]))

            print("Target bands:")
            for f in factors:
                tgt = float(target_betas.get(f, 0.0))
                tol = float(beta_tolerances.get(f, 0.0))
                lo, hi = tgt - tol, tgt + tol
                dev = exp_prev[f] - tgt
                print(f"  {f}: target={tgt:.4f} band=[{lo:.4f},{hi:.4f}] prev={exp_prev[f]:.4f} dev={dev:+.4f}")
        else:
            print("Prev exposure debug skipped: no overlap between w_prev and beta-available tickers.")

    # --- Requested beta bands ---
    print("\n[REQUESTED BETA BANDS]")
    for f in factors:
        tgt = float(target_betas.get(f, 0.0))
        tol = float(beta_tolerances.get(f, 0.0))
        print(f"{f}: {tgt:.4f} ± {tol:.4f}  => [{tgt - tol:.4f}, {tgt + tol:.4f}]")

    # --- Beta distribution summary (helps spot impossible targets)
    print("\n[BETA DISTRIBUTIONS ACROSS UNIVERSE (aligned to R)]")
    good = betas_df.dropna(axis=0, how="any")
    if good.shape[0] > 0:
        for f in factors:
            col = "beta_" + f
            vals = good[col].astype(float).values
            print(f"{f}: min={np.min(vals):.4f}, median={np.median(vals):.4f}, max={np.max(vals):.4f}")
    else:
        print("No complete-beta tickers -> cannot compute beta distributions.")

    # --- Achievable bounds (1D)
    print("\n[ACHIEVABLE BOUNDS (1D per factor)]")
    if compute_1d_bounds_fn is None:
        print("1D bounds function not provided.")
    else:
        try:
            b1 = compute_1d_bounds_fn(R=R, betas_asof=betas_asof, K_max=K_max, w_max=w_max, solver=None)
            if isinstance(b1, dict) and "error" in b1:
                print(f"Could not compute 1D bounds: {b1['error']}")
            else:
                for f in factors:
                    if b1.get(f, {}).get("min") is not None and b1.get(f, {}).get("max") is not None:
                        print(f"{f}: [{b1[f]['min']:.4f}, {b1[f]['max']:.4f}]")
                    else:
                        print(f"{f}: could not compute")
        except Exception as e:
            print(f"1D bounds computation raised exception: {repr(e)}")

    # --- Conditional bounds (given other two factor bands)
    print("\n[CONDITIONAL BOUNDS (given other factor bands)]")
    if compute_conditional_bounds_fn is None:
        print("Conditional bounds function not provided.")
    else:
        try:
            bc = compute_conditional_bounds_fn(
                R=R, betas_asof=betas_asof, target_betas=target_betas, beta_tolerances=beta_tolerances,
                K_max=K_max, w_max=w_max, solver=None
            )
            if isinstance(bc, dict) and "error" in bc:
                print(f"Could not compute conditional bounds: {bc['error']}")
            else:
                for f in factors:
                    if bc.get(f, {}).get("min") is not None and bc.get(f, {}).get("max") is not None:
                        print(f"{f}: [{bc[f]['min']:.4f}, {bc[f]['max']:.4f}]")
                    else:
                        print(f"{f}: NO FEASIBLE RANGE under other factor bands")
        except Exception as e:
            print(f"Conditional bounds computation raised exception: {repr(e)}")

    print("=" * 95 + "\n")

def optimize_pulp_mad_targetbetas_cardinality(
    mu: pd.Series,
    R: pd.DataFrame,
    betas_asof: pd.DataFrame,
    target_betas: dict,
    beta_tolerances: dict | None = None,
    K_max: int = 20,
    risk_aversion: float = 5.0,
    w_max: float = 0.10,
    w_min_if_selected: float = 0.0,
    w_prev: pd.Series | None = None,
    turnover_cap: float | None = None,
    objective_mode: str = "return_minus_risk",
    solver: pulp.LpSolver | None = None,
) -> pd.Series:
    """
    MILP (PuLP):
      maximize mu^T w - lambda * MAD
      subject to:
        sum(w)=1, 0<=w_i<=w_max*z_i, sum(z_i)<=K_max
        target betas within band (or exact)
        optional turnover constraint

    On infeasibility:
      - prints 1D achievable beta ranges
      - prints conditional ranges given other factor bands
    """
    factors = ["MF", "SMB", "HML"]
    need_cols = ["Ticker", "beta_MF", "beta_SMB", "beta_HML"]
    missing = set(need_cols) - set(betas_asof.columns)
    if missing:
        raise ValueError(f"betas_asof missing columns: {sorted(missing)}")

    # Decide universe tickers
    tickers = list(mu.index) if mu is not None else list(R.columns)

    # Align scenario returns
    R = R.copy()
    R = R[tickers].dropna(axis=0, how="any")
    if R.shape[0] < 12:
        raise ValueError("Not enough scenario months in R after dropping NaNs (need ~12+).")

    # Feasibility checks
    if K_max <= 0:
        raise ValueError("K_max must be >= 1")
    if K_max * w_max < 1.0 - 1e-12:
        raise ValueError(f"Infeasible: K_max*w_max={K_max*w_max:.3f} < 1. Increase w_max or K_max.")

    # Betas dict
    bdf = (
        betas_asof[need_cols]
        .drop_duplicates("Ticker")
        .set_index("Ticker")
        .reindex(tickers)
    )
    if bdf.isna().any().any():
        bad = bdf[bdf.isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing betas for tickers: {bad[:10]} (showing up to 10)")

    betas_by_ticker = {
        t: {"MF": float(bdf.loc[t, "beta_MF"]),
            "SMB": float(bdf.loc[t, "beta_SMB"]),
            "HML": float(bdf.loc[t, "beta_HML"])}
        for t in tickers
    }

    # Tolerances
    if beta_tolerances is None:
        beta_tolerances = {k: 0.0 for k in factors}
    for k in factors:
        if k not in target_betas:
            raise ValueError(f"target_betas must include '{k}'")
        beta_tolerances.setdefault(k, 0.0)

    # Model
    prob = pulp.LpProblem("MAD_TargetBeta_Cardinality", pulp.LpMaximize)

    w = {t: pulp.LpVariable(f"w_{t}", lowBound=0) for t in tickers}
    z = {t: pulp.LpVariable(f"z_{t}", lowBound=0, upBound=1, cat="Binary") for t in tickers}

    for t in tickers:
        prob += w[t] <= w_max * z[t]
        if w_min_if_selected > 0:
            prob += w[t] >= w_min_if_selected * z[t]

    prob += pulp.lpSum(z[t] for t in tickers) <= int(K_max)
    prob += pulp.lpSum(w[t] for t in tickers) == 1.0

    # Target beta bands
    for k in factors:
        expr = pulp.lpSum(w[t] * betas_by_ticker[t][k] for t in tickers)
        tgt = float(target_betas[k])
        tol = float(beta_tolerances.get(k, 0.0))
        prob += expr >= (tgt - tol)
        prob += expr <= (tgt + tol)

    # MAD risk
    T = R.shape[0]
    rp = []
    for tt in range(T):
        rp_tt = pulp.lpSum(w[t] * float(R.iloc[tt][t]) for t in tickers)
        rp.append(rp_tt)

    mp = (1.0 / T) * pulp.lpSum(rp)

    u = [pulp.LpVariable(f"u_{tt}", lowBound=0) for tt in range(T)]
    for tt in range(T):
        prob += u[tt] >= rp[tt] - mp
        prob += u[tt] >= -(rp[tt] - mp)

    mad = (1.0 / T) * pulp.lpSum(u)

    # Turnover
    if turnover_cap is not None:
        if w_prev is None:
            raise ValueError("Provide w_prev if using turnover_cap.")
        w_prev = w_prev.reindex(tickers).fillna(0.0).astype(float)

        dpos = {t: pulp.LpVariable(f"dpos_{t}", lowBound=0) for t in tickers}
        dneg = {t: pulp.LpVariable(f"dneg_{t}", lowBound=0) for t in tickers}
        for t in tickers:
            prob += w[t] - float(w_prev[t]) == dpos[t] - dneg[t]
        prob += pulp.lpSum(dpos[t] + dneg[t] for t in tickers) <= float(turnover_cap)

    # Objective
    if objective_mode == "min_risk":
        prob += -mad
    elif objective_mode == "return_minus_risk":
        if mu is None:
            raise ValueError("mu must be provided when objective_mode='return_minus_risk'")
        mu_vals = mu.reindex(tickers).astype(float).to_dict()
        prob += pulp.lpSum(mu_vals[t] * w[t] for t in tickers) - float(risk_aversion) * mad
    else:
        raise ValueError("objective_mode must be 'return_minus_risk' or 'min_risk'")

    # Solve
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)

    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        status_str = pulp.LpStatus[status]

        debug_optimization_failure(
            asof=None,  # set this if you have an asof in the caller/backtest
            status=status_str,
            R=R,
            betas_asof=betas_asof,
            tickers_requested=tickers,
            target_betas=target_betas,
            beta_tolerances=beta_tolerances,
            K_max=K_max,
            w_max=w_max,
            w_min_if_selected=w_min_if_selected,
            turnover_cap=turnover_cap,
            w_prev=w_prev,
            solver_name=type(solver).__name__,
            objective_mode=objective_mode,
            compute_1d_bounds_fn=compute_achievable_beta_bounds,          # <- your function
            compute_conditional_bounds_fn=compute_conditional_beta_bounds # <- your function
        )

        raise RuntimeError(f"PuLP Optimization failed: {status_str}")

    w_out = pd.Series({t: float(pulp.value(w[t])) for t in tickers}, name="weight")
    return w_out[w_out > 0].sort_values(ascending=False)

import pandas as pd

def build_portfolio_asof_pulp(
    asof: str | pd.Timestamp,
    tickers_in_window: list[str],
    stock_returns_data: pd.DataFrame,   # columns: Date, Ticker, RET (monthly, decimals)
    fama_french_data: pd.DataFrame,     # Date, MF, SMB, HML, RF (decimals; you already /100)
    universe_df: pd.DataFrame | None = None,
    year: int | None = None,
    lookback_months: int = 60,
    min_obs: int = 36,
    use_t_as_last_obs: bool = False,

    # optimizer params
    target_betas: dict | None = None,       # {"MF":1.0,"SMB":0.0,"HML":0.2}
    beta_tolerances: dict | None = None,    # {"MF":0.05,"SMB":0.05,"HML":0.05}
    K_max: int = 20,
    risk_aversion: float = 5.0,
    w_max: float = 0.10,
    w_min_if_selected: float = 0.0,
    w_prev: pd.Series | None = None,
    turnover_cap: float | None = None,
    objective_mode: str = "return_minus_risk",
) -> dict:
    """
    End-to-end portfolio build at one rebalance month 'asof' using PuLP MILP (MAD risk + target betas + max K).
    Returns dict with weights and intermediate objects.
    """

    asof_ts = pd.to_datetime(str(asof)).to_period("M").to_timestamp("M")

    # 1) betas at asof (your function)
    betas_asof = estimate_betas_asof_nifty(
        returns_df=stock_returns_data,
        factors_df=fama_french_data,
        asof=asof_ts,
        tickers_in_window=tickers_in_window,
        universe_df=universe_df,
        year=year,
        lookback_months=lookback_months,
        min_obs=min_obs,
        include_mom=False,
        use_t_as_last_obs=use_t_as_last_obs
    )

    # 2) factor stats (Sigma_f, lambda)
    Sigma_f, lam = compute_sigma_f_and_lambda(
        factors_df=fama_french_data,
        asof=asof_ts,
        lookback_months=lookback_months,
        factor_cols=["MF","SMB","HML"]
    )

    # 3) expected returns mu from betas * lambda
    mu, _Sigma_unused = build_mu_and_sigma_from_betas(
        betas_asof=betas_asof,
        Sigma_f=Sigma_f,
        lam=lam,
        factor_cols=["MF","SMB","HML"],
        add_diag_jitter=1e-8
    )

    # 4) build scenario returns matrix R (MAD risk uses realized stock returns)
    panel = stock_returns_data.copy()
    panel["Date"] = pd.to_datetime(panel["Date"].astype(str)).dt.to_period("M").dt.to_timestamp("M")

    # use same universe tickers that actually have betas (avoid missing)
    tickers_used = betas_asof["Ticker"].unique().tolist()
    window = panel[(panel["Date"] <= asof_ts) & (panel["Ticker"].isin(tickers_used))]

    R = (window.pivot(index="Date", columns="Ticker", values="RET")
              .sort_index()
              .tail(lookback_months))

    # align mu to R columns (intersection)
    common = sorted(set(mu.index).intersection(R.columns))
    mu = mu.loc[common]
    R = R[common]
    betas_asof = betas_asof[betas_asof["Ticker"].isin(common)].copy()

    # 5) default target betas if not passed (optional)
    if target_betas is None:
        # if you don't want target betas, set loose tolerances around current exposures or skip calling this build
        target_betas = {"MF": 1.0, "SMB": 0.0, "HML": 0.0}
    if beta_tolerances is None:
        beta_tolerances = {"MF": 0.10, "SMB": 0.10, "HML": 0.10}

    # 6) optimize with PuLP MILP
    weights = optimize_pulp_mad_targetbetas_cardinality(
        mu=mu,
        R=R,
        betas_asof=betas_asof,
        target_betas=target_betas,
        beta_tolerances=beta_tolerances,
        K_max=K_max,
        risk_aversion=risk_aversion,
        w_max=w_max,
        w_min_if_selected=w_min_if_selected,
        w_prev=w_prev,
        turnover_cap=turnover_cap,
        objective_mode=objective_mode
    )

    return {
        "asof": asof_ts,
        "weights": weights,
        "mu": mu,
        "R": R,
        "betas_asof": betas_asof,
        "Sigma_f": Sigma_f,
        "lambda": lam
    }

def compute_portfolio_factor_exposures(weights: pd.Series,
                                       betas_asof: pd.DataFrame,
                                       factors=("MF", "SMB", "HML")):
    """
    Computes portfolio-level factor exposures given weights and betas.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by ticker.
    betas_asof : pd.DataFrame
        DataFrame containing columns:
        ['Ticker', 'beta_MF', 'beta_SMB', 'beta_HML']
    factors : tuple
        Factor names to compute.

    Returns
    -------
    pd.Series of portfolio factor exposures
    """

    # Align betas with weights
    beta_cols = [f"beta_{f}" for f in factors]
    beta_matrix = (
        betas_asof
        .set_index("Ticker")
        .loc[weights.index, beta_cols]
    )

    # Compute exposures
    exposures = beta_matrix.T.dot(weights)

    exposures.index = factors
    exposures.name = "Portfolio Exposure"

    return exposures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_returns_date_range(stock_returns_data: pd.DataFrame, date_col: str = "Date"):
    d = pd.to_datetime(stock_returns_data[date_col])
    dmin, dmax = d.min(), d.max()
    print(f"Returns date range: {dmin.date()} → {dmax.date()}")
    return dmin, dmax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _to_month_end(x):
    return pd.to_datetime(x).to_period("M").to_timestamp("M")

def portfolio_exposures_from_betas(weights: pd.Series,
                                  betas_asof: pd.DataFrame,
                                  factors=("MF","SMB","HML")) -> pd.Series:
    beta_cols = [f"beta_{f}" for f in factors]
    B = (betas_asof
         .drop_duplicates("Ticker")
         .set_index("Ticker")
         .reindex(weights.index)[beta_cols])

    # if a ticker is missing beta, treat as 0 exposure (or raise if you prefer)
    B = B.fillna(0.0)

    exp = B.T.dot(weights)
    exp.index = list(factors)
    exp.name = "exposure"
    return exp

def check_within_tolerance(exposures: pd.Series, target: dict, tol: dict) -> bool:
    for k, val in exposures.items():
        lo = target[k] - tol[k]
        hi = target[k] + tol[k]
        if not (lo <= float(val) <= hi):
            return False
    return True

def backtest_fixed_window_quarterly_rebalance_on_breach(
    stock_returns_data: pd.DataFrame,    # Date, Ticker, RET (monthly, decimals)
    fama_french_data: pd.DataFrame,      # Date, MF, SMB, HML, RF (monthly, decimals)
    index_returns,                       # <<< CHANGED >>> accept Series OR DataFrame
    universe_by_year: dict,              # {year: [tickers]}

    oos_start: str | pd.Timestamp,       # first OOS month (month-end)
    oos_months: int = 24,
    lookback_months: int = 36,
    rebalance_every: int = 3,            # quarterly
    initial_capital: float = 100000.0,

    # optimization knobs (passed into your builder)
    objective: str = "return_minus_risk",
    risk_aversion: float = 5.0,
    K_max: int = 15,
    w_max: float = 0.20,
    min_obs: int = 36,
    use_t_as_last_obs: bool = False,

    # factor controls
    target_betas: dict | None = None,    # {"MF":1,"SMB":1,"HML":1}
    beta_tolerances: dict | None = None, # {"MF":0.05,"SMB":0.05,"HML":0.05}

    # optional turnover control (only applies when you DO rebalance)
    turnover_cap: float | None = None,

    # <<< ADDED >>> progress controls
    show_progress: bool = True,
    progress_every: int = 1,             # update postfix every N months
    show_timing: bool = True,
):
    """
    Fixed-window backtest:
      - Build first portfolio using lookback_months before oos_start
      - Run OOS for oos_months
      - Every rebalance_every months: check exposure drift (using current betas at that date)
        - if within tolerance: skip rebalance
        - else: rebalance by re-optimizing
    """

    if target_betas is None:
        target_betas = {"MF": 1.0, "SMB": 0.0, "HML": 0.0}
    if beta_tolerances is None:
        beta_tolerances = {"MF": 0.10, "SMB": 0.10, "HML": 0.10}

    # Normalize dates
    sr = stock_returns_data.copy()
    sr["Date"] = pd.to_datetime(sr["Date"]).dt.to_period("M").dt.to_timestamp("M")

    ff = fama_french_data.copy()
    ff["Date"] = pd.to_datetime(ff["Date"]).dt.to_period("M").dt.to_timestamp("M")

    # ------------------------------------------------------------------
    # <<< CHANGED / ADDED >>> Robust normalization of index returns
    # Accept either:
    #   - Series with datetime index
    #   - DataFrame with columns: Date, MonthlyReturn
    # ------------------------------------------------------------------
    if isinstance(index_returns, pd.DataFrame):
        if ("Date" not in index_returns.columns) or ("MonthlyReturn" not in index_returns.columns):
            raise ValueError(
                "index_returns DataFrame must contain columns: 'Date' and 'MonthlyReturn'"
            )
        idx = index_returns.copy()
        idx["Date"] = pd.to_datetime(idx["Date"]).dt.to_period("M").dt.to_timestamp("M")
        idx = idx.set_index("Date")["MonthlyReturn"].astype(float).sort_index()
    elif isinstance(index_returns, pd.Series):
        idx = index_returns.copy()
        idx.index = pd.to_datetime(idx.index).to_period("M").to_timestamp("M")
        idx = idx.astype(float).sort_index()
    else:
        raise TypeError("index_returns must be a pandas Series or DataFrame")

    # Drop duplicate dates (keep last value if multiple dates map to same month-end)
    idx = idx[~idx.index.duplicated(keep="last")]

    # <<< ADDED >>> Optional: auto-detect #% vs decimal
    if len(idx) > 0:
        med_abs = float(idx.abs().median())
        if med_abs > 0.5:  # unrealistic for index monthly returns if already decimal
            idx = idx / 100.0

    # Pivot returns once
    R_full = sr.pivot(index="Date", columns="Ticker", values="RET").sort_index()

    oos_start = _to_month_end(oos_start)
    oos_end = _to_month_end(oos_start + pd.DateOffset(months=oos_months - 1))

    # OOS month grid
    all_months = R_full.index
    oos_month_grid = all_months[(all_months >= oos_start) & (all_months <= oos_end)]
    if len(oos_month_grid) != oos_months:
        raise ValueError(
            f"OOS months available = {len(oos_month_grid)}, expected {oos_months}. Check date range."
        )

    # First portfolio is built at asof = month-end BEFORE first OOS month
    first_asof = _to_month_end(oos_start - pd.DateOffset(months=1))

    # Universe for that year
    tickers_first = universe_by_year.get(int(first_asof.year), list(R_full.columns))

    # Build initial weights
    res0 = build_portfolio_asof_pulp(
        asof=first_asof,
        tickers_in_window=tickers_first,
        stock_returns_data=sr,
        fama_french_data=ff,
        lookback_months=lookback_months,
        min_obs=min_obs,
        use_t_as_last_obs=use_t_as_last_obs,
        # objective=objective,
        risk_aversion=risk_aversion,
        K_max=K_max,
        w_max=w_max,
        target_betas=target_betas,
        beta_tolerances=beta_tolerances,
        w_prev=None,
        turnover_cap=None
    )

    w = res0["weights"].copy()
    w_prev = w.copy()

    # Storage
    monthly_strategy_rets = []
    monthly_index_rets = []
    rebalance_log = {}  # keyed by rebalance asof date

    # We will check/rebalance at quarter boundaries:
    check_asofs = []
    cur = first_asof
    while cur <= _to_month_end(oos_end - pd.DateOffset(months=1)):
        check_asofs.append(cur)
        cur = _to_month_end(cur + pd.DateOffset(months=rebalance_every))

    check_set = set(check_asofs)

    # ------------------------------------------------------------------
    # <<< ADDED >>> progress bar setup
    # ------------------------------------------------------------------
    t0 = time.time()
    iterable = oos_month_grid
    if show_progress:
        pbar = tqdm(iterable, desc="Backtest OOS", unit="month")
    else:
        pbar = iterable

    for i, m in enumerate(pbar, start=1):
        # <<< ADDED >>> postfix updates
        if show_progress and (i % max(1, progress_every) == 0):
            postfix = {"month": m.strftime("%Y-%m")}
            if show_timing:
                elapsed = time.time() - t0
                rate = elapsed / max(1, i)
                eta = rate * (len(oos_month_grid) - i)
                postfix["eta_s"] = f"{eta:.0f}"
            pbar.set_postfix(postfix)

        # before earning return for month m, see if we are at a "new block start"
        prev_month_end = _to_month_end(m - pd.DateOffset(months=1))
        if prev_month_end in check_set:
            asof = prev_month_end

            # <<< ADDED >>> show we're doing a check
            if show_progress:
                pbar.set_postfix_str(f"{m.strftime('#%Y-#%m')} | check@{asof.strftime('#%Y-#%m')}")

            tickers_year = universe_by_year.get(int(asof.year), list(R_full.columns))

            betas_asof = estimate_betas_asof_nifty(
                returns_df=sr,
                factors_df=ff,
                asof=asof,
                tickers_in_window=tickers_year,
                universe_df=None,
                year=None,
                lookback_months=lookback_months,
                min_obs=min_obs,
                include_mom=False,
                use_t_as_last_obs=use_t_as_last_obs
            )

            current_exp = portfolio_exposures_from_betas(w, betas_asof)
            within = check_within_tolerance(current_exp, target_betas, beta_tolerances)

            did_rebalance = False
            if not within:
                # <<< ADDED >>> show rebalance beginning
                if show_progress:
                    pbar.set_postfix_str(f"{m.strftime('#%Y-#%m')} | REBAL@{asof.strftime('#%Y-#%m')}")

                res_new = build_portfolio_asof_pulp(
                    asof=asof,
                    tickers_in_window=tickers_year,
                    stock_returns_data=sr,
                    fama_french_data=ff,
                    lookback_months=lookback_months,
                    min_obs=min_obs,
                    use_t_as_last_obs=use_t_as_last_obs,
                    # objective=objective,
                    risk_aversion=risk_aversion,
                    K_max=K_max,
                    w_max=w_max,
                    target_betas=target_betas,
                    beta_tolerances=beta_tolerances,
                    w_prev=w_prev,
                    turnover_cap=turnover_cap
                )
                w = res_new["weights"].copy()
                w_prev = w.copy()
                did_rebalance = True

                betas_after = estimate_betas_asof_nifty(
                    returns_df=sr,
                    factors_df=ff,
                    asof=asof,
                    tickers_in_window=tickers_year,
                    universe_df=None,
                    year=None,
                    lookback_months=lookback_months,
                    min_obs=min_obs,
                    include_mom=False,
                    use_t_as_last_obs=use_t_as_last_obs
                )
                exp_after = portfolio_exposures_from_betas(w, betas_after)
            else:
                exp_after = current_exp

            # <<< ADDED >>> show final decision
            if show_progress:
                decision = "rebalance" if did_rebalance else "skip"
                pbar.set_postfix_str(
                    f"{m.strftime('#%Y-#%m')} | {decision}@{asof.strftime('#%Y-#%m')} | within={within}"
                )

            rebalance_log[str(asof.date())] = {
                "asof": asof,
                "within_tolerance": within,
                "rebalanced": did_rebalance,
                "exposure_before_or_current": current_exp,
                "exposure_after": exp_after,
                "weights": w
            }

        # Realized portfolio return for month m
        tickers = w.index
        r_vec = R_full.loc[m, tickers].fillna(0.0)
        rp = float((r_vec * w).sum())

        # Get index return for month m
        if m in idx.index:
            ri = float(idx.loc[m])
        else:
            prior_months = idx.index[idx.index <= m]
            if len(prior_months) > 0:
                ri = float(idx.loc[prior_months[-1]])
            else:
                ri = 0.0

        monthly_strategy_rets.append((m, rp))
        monthly_index_rets.append((m, ri))

    strat = pd.Series(
        [r for _, r in monthly_strategy_rets],
        index=[d for d, _ in monthly_strategy_rets],
        name="StrategyRet"
    )

    bench = pd.Series(
        [r for _, r in monthly_index_rets],
        index=[d for d, _ in monthly_index_rets],
        name="IndexRet"
    )

    # Capital paths
    strat_value = initial_capital * (1 + strat).cumprod()
    bench_value = initial_capital * (1 + bench).cumprod()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strat_value.index, strat_value.values, label="Customized Portfolio")
    ax.plot(bench_value.index, bench_value.values, label="Nifty 50")
    ax.set_title("OOS Backtest: Strategy vs Index (Value)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True)
    ax.legend()

    return {
        "oos_start": oos_start,
        "oos_end": oos_end,
        "strategy_returns": strat,
        "index_returns": bench,
        "strategy_value": strat_value,
        "index_value": bench_value,
        "rebalance_log": rebalance_log,
        "figure": fig
    }

