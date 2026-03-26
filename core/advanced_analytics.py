"""
Advanced analytics module: Monte Carlo simulation, Tornado sensitivity,
NPV/IRR, and 3-year decay projection.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import brentq


@dataclass
class TornadoResult:
    params: list
    param_labels: list
    roi_low: list
    roi_high: list
    base_roi: float
    delta_pct: float


@dataclass
class MonteCarloResult:
    roi_samples: np.ndarray
    p_positive: float
    p_payback_18: float
    p_payback_12: float
    pct10: float
    pct50: float
    pct90: float
    n_simulations: int
    impl_cost: float


@dataclass
class NPVResult:
    npv: float
    irr_pct: float | None
    cashflows: list
    wacc_pct: float
    yearly_benefits: list
    cumulative_npv: list


def _vectorized_roi(mh, ar, hr, eb, ea, cpe, vol, cb, ca, dpm, dvl, pb, pa, impl, pu):
    """Vectorized ROI calculation (works with arrays or scalars)."""
    time_saved     = mh * ar * 12.0 * hr
    error_saved    = ((eb - ea) / 100.0) * vol * 12.0 * cpe
    velocity       = np.where(cb > 0, (cb - ca) / cb, 0.0)
    revenue_impact = dpm * velocity * pu * 12.0 * dvl
    markov_gain    = dpm * (pa - pb) * dvl * 12.0
    total          = time_saved + error_saved + revenue_impact + markov_gain
    return total - impl, total


def run_monte_carlo(inp, n: int = 5000) -> MonteCarloResult:
    """
    Monte Carlo ROI simulation with ±15% uncertainty on all input parameters.
    Uses seeded RNG for reproducibility.
    """
    rng = np.random.default_rng(42)
    s = 0.15

    def ns(v, lo=None, hi=None):
        arr = rng.normal(float(v), abs(float(v)) * s, n)
        if lo is not None:
            arr = np.clip(arr, lo, None)
        if hi is not None:
            arr = np.clip(arr, None, hi)
        return arr

    mh  = ns(inp.manual_hours_per_month, 10)
    ar  = ns(inp.automation_rate, 0.05, 0.99)
    hr  = ns(inp.hour_rate_eur, 1)
    eb  = ns(inp.error_rate_before_pct, 0.2, 50.0)
    ea  = ns(inp.error_rate_after_pct, 0.01, 10.0)
    ea  = np.minimum(ea, eb * 0.8)
    cpe = ns(inp.cost_per_error_eur, 1)
    vol = ns(inp.monthly_volume, 1)
    cb  = ns(inp.deal_cycle_before_days, 1)
    ca  = ns(inp.deal_cycle_after_days, 0.5)
    ca  = np.minimum(ca, cb * 0.9)
    dpm = ns(inp.deals_per_month, 1)
    dvl = ns(inp.avg_deal_value_eur, 100)
    pb  = ns(inp.p_complete_before, 0.05, 0.95)
    pa  = ns(inp.p_complete_after, 0.10, 0.99)
    pa  = np.maximum(pa, pb + 0.01)
    pu  = getattr(inp, "pipeline_utilization_pct", 30) / 100.0

    net_roi, total = _vectorized_roi(
        mh, ar, hr, eb, ea, cpe, vol, cb, ca, dpm, dvl, pb, pa,
        inp.implementation_cost_eur, pu
    )
    payback = np.where(total > 0, inp.implementation_cost_eur / (total / 12.0), 999.0)

    return MonteCarloResult(
        roi_samples=net_roi,
        p_positive=float(np.mean(net_roi > 0)),
        p_payback_18=float(np.mean(payback < 18)),
        p_payback_12=float(np.mean(payback < 12)),
        pct10=float(np.percentile(net_roi, 10)),
        pct50=float(np.percentile(net_roi, 50)),
        pct90=float(np.percentile(net_roi, 90)),
        n_simulations=n,
        impl_cost=inp.implementation_cost_eur,
    )


def run_tornado(inp, delta: float = 0.20) -> TornadoResult:
    """
    Sensitivity analysis: vary each key parameter by ±delta and record ROI change.
    Returns results sorted by impact magnitude (largest first).
    """
    pu   = getattr(inp, "pipeline_utilization_pct", 30) / 100.0
    impl = inp.implementation_cost_eur

    def roi_pt(**ov):
        mh  = ov.get("mh",  inp.manual_hours_per_month)
        ar  = ov.get("ar",  inp.automation_rate)
        hr  = ov.get("hr",  inp.hour_rate_eur)
        eb  = ov.get("eb",  inp.error_rate_before_pct)
        ea  = ov.get("ea",  inp.error_rate_after_pct)
        cpe = ov.get("cpe", inp.cost_per_error_eur)
        vol = ov.get("vol", inp.monthly_volume)
        cb  = ov.get("cb",  inp.deal_cycle_before_days)
        ca  = ov.get("ca",  inp.deal_cycle_after_days)
        dpm = ov.get("dpm", inp.deals_per_month)
        dvl = ov.get("dvl", inp.avg_deal_value_eur)
        pb  = ov.get("pb",  inp.p_complete_before)
        pa  = ov.get("pa",  inp.p_complete_after)
        net, _ = _vectorized_roi(mh, ar, hr, eb, ea, cpe, vol, cb, ca, dpm, dvl, pb, pa, impl, pu)
        return float(net)

    base = roi_pt()

    cfg = [
        ("manual_hours",      "mh",  inp.manual_hours_per_month,    None),
        ("automation_rate",   "ar",  inp.automation_rate,           None),
        ("hour_rate",         "hr",  inp.hour_rate_eur,             None),
        ("cost_per_error",    "cpe", inp.cost_per_error_eur,        None),
        ("deal_value",        "dvl", inp.avg_deal_value_eur,        None),
        ("deals_per_month",   "dpm", inp.deals_per_month,           None),
        ("cycle_improvement", None,  None,                          "cycle"),
        ("p_uplift",          None,  None,                          "p"),
    ]

    labels, lows, highs = [], [], []
    for label, key, val, special in cfg:
        if special == "cycle":
            ca_orig = inp.deal_cycle_after_days
            lo = roi_pt(ca=min(ca_orig * (1 + delta), inp.deal_cycle_before_days * 0.95))
            hi = roi_pt(ca=max(ca_orig * (1 - delta), 0.5))
        elif special == "p":
            uplift = inp.p_complete_after - inp.p_complete_before
            lo = roi_pt(pa=inp.p_complete_before + uplift * (1 - delta))
            hi = roi_pt(pa=min(inp.p_complete_before + uplift * (1 + delta), 0.99))
        else:
            lo = roi_pt(**{key: val * (1 - delta)})
            hi = roi_pt(**{key: val * (1 + delta)})
        labels.append(label)
        lows.append(lo)
        highs.append(hi)

    impacts = [abs(h - l) for h, l in zip(highs, lows)]
    order   = sorted(range(len(labels)), key=lambda i: impacts[i], reverse=True)

    return TornadoResult(
        params      =[labels[i] for i in order],
        param_labels=[labels[i] for i in order],
        roi_low     =[lows[i]   for i in order],
        roi_high    =[highs[i]  for i in order],
        base_roi    =base,
        delta_pct   =delta,
    )


def compute_npv_irr(
    total_benefit: float,
    impl_cost: float,
    wacc_pct: float = 12.0,
    decay: tuple = (1.0, 0.80, 0.65),
) -> NPVResult:
    """
    Compute NPV and IRR for a multi-year projection with benefit decay.
    decay[t] = fraction of Year-1 benefit in Year t+1.
    """
    r       = wacc_pct / 100.0
    yearly  = [total_benefit * d for d in decay]
    cashflows = [-impl_cost] + yearly
    npv     = sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))

    cumulative_npv = []
    running = -impl_cost
    for t, y in enumerate(yearly, start=1):
        running += y / (1 + r) ** t
        cumulative_npv.append(round(running, 2))

    try:
        def npv_fn(r_try):
            return sum(cf / (1 + r_try) ** t for t, cf in enumerate(cashflows))
        irr_raw = brentq(npv_fn, -0.99, 100.0)
        irr = round(irr_raw * 100, 1)
    except Exception:
        irr = None

    return NPVResult(
        npv=round(npv, 2),
        irr_pct=irr,
        cashflows=cashflows,
        wacc_pct=wacc_pct,
        yearly_benefits=[round(y, 2) for y in yearly],
        cumulative_npv=cumulative_npv,
    )
