import sys
import os
import io
import json
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from db.database import (
    load_history as db_load_history,
    save_audit   as db_save_audit,
    delete_audit as db_delete_audit,
    db_available,
)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

import dataclasses

from core.math_engine import MathEngine, build_markov_graph, MarkovGraphResult
from core.roi_engine import ROIEngine, ROIInput, ROIResult
from core.advanced_analytics import run_monte_carlo, run_tornado, compute_npv_irr
from etl.extractor import MatrixExtractor
from exports.pdf_generator import build_roi_passport_pdf
from ui.i18n import TRANSLATIONS, LANG_NAMES, t


# ── Robust CSV loader (handles garbage rows before real header) ────────────────
def load_and_clean_csv(uploaded_file) -> "pd.DataFrame | None":
    """
    Bulletproof Universal CSV Parser for enterprise exports (Jira, Salesforce, Zendesk).

    Handles files that start with metadata garbage before the real column headers:
        "Report generated on 2025-01-15"   ← discarded
        "Table 1: Support Tickets"          ← discarded
        ""                                  ← discarded
        "Ticket_ID,Status,Days,Assignee"    ← TRUE HEADER  ← parsing starts here
        "T-001,Open,3,Alice"

    Algorithm
    ---------
    Step 1 — Encoding resilience
        Read raw bytes from the uploaded file object and decode using a cascade:
        utf-8-sig (Excel BOM) → utf-8 → cp1251 (Cyrillic Windows) → latin-1 (fallback).
        First successful decode wins; returns None if all four fail.

    Step 2 — Dynamic header detection (heuristic)
        Split the decoded text into individual lines.
        Iterate line-by-line from the top and test each line:
          a) Split by comma to get candidate columns.
          b) Count non-empty parts (parts with at least one printable character).
          c) Accept the line as the TRUE HEADER when BOTH conditions hold:
               • non_empty_parts > 2        — guarantees multi-column structure
               • len(line.strip()) > 10     — rejects stray single-word rows
             The first line satisfying both criteria becomes the header.

    Step 3 — Garbage removal
        Discard ALL lines above the detected header row index.
        Re-join header + data lines into a clean CSV block (no metadata noise).

    Step 4 — Pandas normalisation
        a) Parse the clean block with pd.read_csv via io.StringIO (no temp files).
        b) Strip leading/trailing whitespace from every column name (export artefact).
        c) Drop any column whose name contains 'Unnamed' (caused by trailing commas
           in the source row, e.g., "ID,Status,,," → Unnamed: 2, Unnamed: 3).
        d) Drop rows where ALL values are NaN or empty strings — leftover blank lines.

    Step 5 — @st.cache_data bypass
        This function is deliberately NOT decorated with @st.cache_data.
        Cache invalidation is handled upstream in session_state via _etl_fkey:
        when the uploaded file name/size changes, mapped_df and col_mapping are
        cleared from st.session_state before this function is called, so stale
        data from a previous upload can never pollute the current analysis.
    """
    # ── Step 1: Encoding resilience ───────────────────────────────────────────
    raw_bytes = uploaded_file.read()
    text: str | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        return None  # Unreadable binary — cannot proceed

    # ── Step 2: Dynamic header detection (heuristic) ─────────────────────────
    lines = text.splitlines()
    header_idx: int | None = None
    for i, line in enumerate(lines):
        # Criterion a: count non-empty comma-separated segments
        parts = line.split(",")
        non_empty_parts = sum(1 for p in parts if p.strip())
        # Criterion b: the line itself must be long enough to be a real header
        substantial_length = len(line.strip()) > 10
        if non_empty_parts > 2 and substantial_length:
            header_idx = i
            break  # First qualifying row is the true header; stop scanning

    if header_idx is None:
        return None  # No valid header row found anywhere in the file

    # ── Step 3: Garbage removal ───────────────────────────────────────────────
    # Everything above header_idx is metadata / report noise — discard it.
    clean_text = "\n".join(lines[header_idx:])

    # ── Step 4: Pandas normalisation ──────────────────────────────────────────
    try:
        df = pd.read_csv(io.StringIO(clean_text))
    except Exception:
        return None  # Malformed CSV even after cleaning

    # 4b — Strip whitespace from column names (common in Salesforce/Zendesk exports)
    df.columns = [str(c).strip() for c in df.columns]

    # 4c — Drop Unnamed columns (artefact of trailing commas: "a,b,c,,,")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

    # 4d — Drop fully blank rows (empty lines that survived past the header)
    df = df.dropna(how="all")
    df = df[~df.apply(lambda row: row.astype(str).str.strip().eq("").all(), axis=1)]
    df = df.reset_index(drop=True)

    return df if not df.empty else None


# Backward-compat alias so any other call-site using the old name still works
load_cleaned_csv = load_and_clean_csv


# ── Cached Monte Carlo (Sprint 3) ──────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def _cached_mc(
    manual_hours, automation_rate, hour_rate,
    error_before, error_after, cost_per_error, volume,
    cycle_before, cycle_after, deals, deal_value,
    p_before, p_after, impl_cost, pipeline_util,
    n: int = 5000,
):
    _inp = ROIInput(
        company_name="",
        manual_hours_per_month=float(manual_hours),
        automation_rate=automation_rate / 100.0,
        hour_rate_eur=float(hour_rate),
        error_rate_before_pct=float(error_before),
        error_rate_after_pct=float(error_after),
        cost_per_error_eur=float(cost_per_error),
        monthly_volume=int(volume),
        deal_cycle_before_days=float(cycle_before),
        deal_cycle_after_days=float(cycle_after),
        deals_per_month=int(deals),
        avg_deal_value_eur=float(deal_value),
        p_complete_before=p_before / 100.0,
        p_complete_after=p_after / 100.0,
        implementation_cost_eur=float(impl_cost),
        positive_signals=4, total_signals=5,
        pipeline_utilization_pct=float(pipeline_util),
    )
    return run_monte_carlo(_inp, n=n)


# ── Currency ───────────────────────────────────────────────────────────────────
_CURRENCIES = {
    "EUR": {"sym": "€",    "rate": 1.0,   "label": "EUR"},
    "RUB": {"sym": "₽",    "rate": 100.0, "label": "RUB"},
    "RSD": {"sym": "дин.", "rate": 117.0, "label": "RSD"},
}

def _fmt(val_eur: float, currency: str) -> str:
    cur = _CURRENCIES.get(currency, _CURRENCIES["EUR"])
    converted = val_eur * cur["rate"]
    if currency == "RSD":
        if abs(converted) >= 1_000_000:
            s = "{:,.0f}".format(converted / 1_000_000).replace(",", ".")
            return "{} {}M".format(s, cur["sym"])
        s = "{:,.0f}".format(converted).replace(",", ".")
        return "{} {}".format(s, cur["sym"])
    if currency == "RUB":
        if abs(converted) >= 1_000_000:
            s = "{:,.0f}".format(converted / 1_000_000).replace(",", "\u202f")
            return "{} {}M".format(s, cur["sym"])
        s = "{:,.0f}".format(converted).replace(",", "\u202f")
        return "{} {}".format(s, cur["sym"])
    if abs(converted) >= 1_000_000:
        return "{:,.0f} {}".format(converted / 1_000_000, cur["sym"] + "M")
    return "{:,.0f} {}".format(converted, cur["sym"])


# ── Industry benchmarks (avg ROI %, avg payback months) ───────────────────────
_BENCHMARKS = {
    "logistics": {"roi_pct": 280, "payback": 2.8, "net_roi_mult": 6.2},
    "agency":    {"roi_pct": 420, "payback": 1.9, "net_roi_mult": 8.5},
    "retail":    {"roi_pct": 190, "payback": 3.5, "net_roi_mult": 4.1},
    None:        {"roi_pct": 300, "payback": 2.5, "net_roi_mult": 6.0},
}


# ── Client history ─────────────────────────────────────────────────────────────
_PARAM_KEYS = ["manual_hours", "automation_rate", "hour_rate",
               "error_before", "error_after", "cost_per_error", "volume",
               "cycle_before", "cycle_after", "deals_month", "deal_value",
               "p_before", "p_after", "impl_cost"]

def _load_history() -> list:
    return db_load_history()

def _save_to_history(company_name: str, extra: dict | None = None) -> bool:
    params = {k: st.session_state.get(k) for k in _PARAM_KEYS}
    kw = extra or {}
    return db_save_audit(
        company_name        = company_name,
        params              = params,
        friction_tax_usd    = kw.get("friction_tax_usd"),
        adjusted_confidence_pct = kw.get("adjusted_confidence_pct"),
        bottleneck_stage    = kw.get("bottleneck_stage"),
        roi_pct             = kw.get("roi_pct"),
        rework_rate_pct     = kw.get("rework_rate_pct"),
        total_transitions   = kw.get("total_transitions"),
        total_rework        = kw.get("total_rework"),
    )

def _restore_from_history(entry: dict):
    st.session_state["company_name"] = entry.get("company_name", "")
    raw_params = entry.get("params") or {}
    if isinstance(raw_params, str):
        try:
            raw_params = json.loads(raw_params)
        except Exception:
            raw_params = {}
    for k, v in raw_params.items():
        if v is not None:
            st.session_state[k] = v


# ── Demo presets ───────────────────────────────────────────────────────────────
DEMO_PRESETS = {
    "logistics": {
        "labels": {"en": "Logistics", "ru": "Логистика", "sr": "Logistika"},
        "desc": {
            "en": "3PL operator Moscow — 14 managers, 520 manual hrs/mo, deal avg 12 000 €",
            "ru": "3PL-оператор Москва — 14 менеджеров, 520 ручных ч/мес, ср. контракт 12 000 €",
            "sr": "3PL operator Moskva — 14 menadžera, 520 ručnih h/mes., ugovor 12 000 €",
        },
        "company_name": "ТрансЛогик МСК",
        "manual_hours": 520, "automation_rate": 75, "hour_rate": 15,
        "error_before": 14.0, "error_after": 1.8, "cost_per_error": 85, "volume": 250,
        "cycle_before": 14, "cycle_after": 6, "deals_month": 7, "deal_value": 2800,
        "p_before": 68, "p_after": 82, "impl_cost": 28000,
    },
    "agency": {
        "labels": {"en": "Agency", "ru": "Агентство", "sr": "Agencija"},
        "desc": {
            "en": "Performance agency Moscow — 18 clients, 320 manual hrs/mo reporting, avg account 8 500 €",
            "ru": "Performance-агентство Москва — 18 клиентов, 320 ч/мес на отчёты, ср. аккаунт 8 500 €",
            "sr": "Performance agencija Moskva — 18 klijenata, 320 h/mes. izveštaji, prosečan nalog 8 500 €",
        },
        "company_name": "MOKO Digital",
        "manual_hours": 320, "automation_rate": 75, "hour_rate": 22,
        "error_before": 8.5, "error_after": 0.9, "cost_per_error": 100, "volume": 80,
        "cycle_before": 18, "cycle_after": 6, "deals_month": 4, "deal_value": 4500,
        "p_before": 71, "p_after": 88, "impl_cost": 15000,
    },
    "retail": {
        "labels": {"en": "Retail", "ru": "Ритейл", "sr": "Maloprodaja"},
        "desc": {
            "en": "Retail chain Belgrade — 12 stores, 380 manual hrs/mo, 2 200 invoices",
            "ru": "Сеть магазинов Белград — 12 точек, 380 ч/мес, 2 200 накладных",
            "sr": "Maloprodajni lanac Beograd — 12 prodavnica, 380 h/mes., 2 200 faktura",
        },
        "company_name": "МегаМаркет d.o.o.",
        "manual_hours": 380, "automation_rate": 65, "hour_rate": 10,
        "error_before": 9.2, "error_after": 1.5, "cost_per_error": 55, "volume": 400,
        "cycle_before": 8, "cycle_after": 3, "deals_month": 8, "deal_value": 1200,
        "p_before": 72, "p_after": 87, "impl_cost": 14000,
    },
}
_DEMO_KEYS = ["manual_hours", "automation_rate", "hour_rate",
              "error_before", "error_after", "cost_per_error", "volume",
              "cycle_before", "cycle_after", "deals_month", "deal_value",
              "p_before", "p_after", "impl_cost"]
_DEMO_BANNER = {
    "en": ("Demo mode", "{desc}", "Adjust sliders on the left or pick another preset to explore."),
    "ru": ("Демо-режим", "{desc}", "Меняйте параметры слева или выберите другой кейс для сравнения."),
    "sr": ("Demo režim", "{desc}", "Prilagodite parametre levo ili izaberite drugi slučaj."),
}

def _apply_preset(preset_key):
    p = DEMO_PRESETS[preset_key]
    st.session_state["demo_preset"] = preset_key
    st.session_state["company_name"] = p["company_name"]
    for k in _DEMO_KEYS:
        st.session_state[k] = p[k]

def _clear_demo():
    st.session_state.pop("demo_preset", None)


# ── Chart colour palette ───────────────────────────────────────────────────────
_C = dict(
    navy="#0071E3", green="#34C759", gold="#FF9F0A",
    red="#FF3B30",  purple="#AF52DE", amber="#FF9F0A",
    blue="#0071E3", teal="#5AC8FA",
)


def _apply_confidence(res: ROIResult, factor: float) -> ROIResult:
    if factor == 1.0:
        return res
    ts    = res.time_saved_annual      * factor
    er    = res.error_reduction_annual * factor
    ri    = res.revenue_impact_annual  * factor
    mg    = res.markov_gain_annual     * factor
    total = ts + er + ri + mg
    impl  = res.total_benefit - res.net_roi
    net   = total - impl
    roi_pct = (net / impl * 100) if impl else 0.0
    payback = impl / (total / 12) if total else 0.0
    return dataclasses.replace(
        res,
        time_saved_annual=round(ts, 2),
        error_reduction_annual=round(er, 2),
        revenue_impact_annual=round(ri, 2),
        markov_gain_annual=round(mg, 2),
        total_benefit=round(total, 2),
        net_roi=round(net, 2),
        roi_pct=round(roi_pct, 2),
        payback_months=round(payback, 1),
    )


CHART_LAYOUT = dict(
    plot_bgcolor="rgba(255,255,255,0)",
    paper_bgcolor="rgba(255,255,255,0)",
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif",
              color="#6E6E73", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
        linecolor="rgba(0,0,0,0.06)", zeroline=False,
        tickfont=dict(color="#AEAEB2", size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
        linecolor="rgba(0,0,0,0.06)", zeroline=False,
        tickfont=dict(color="#AEAEB2", size=11),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        font=dict(color="#6E6E73", size=11),
        bordercolor="rgba(0,0,0,0.06)", borderwidth=1,
    ),
)


def run_dashboard():

    # ── CSS ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    *, *::before, *::after { box-sizing: border-box; }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text',
                     'Inter', 'Helvetica Neue', Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased !important;
    }
    .stApp { background: #F5F5F7 !important; }
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid rgba(0,0,0,0.08) !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {
        color: #6E6E73 !important; font-size: 13px !important; line-height: 1.5 !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #1D1D1F !important; font-size: 17px !important;
        font-weight: 600 !important; letter-spacing: -0.02em !important;
    }
    [data-testid="metric-container"],
    [data-testid="stMetric"] {
        background: #FFFFFF !important; border-radius: 18px !important;
        padding: 24px 26px !important; border: none !important;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08) !important;
        transition: box-shadow 0.3s ease, transform 0.3s ease !important;
    }
    [data-testid="metric-container"]:hover,
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 32px rgba(0,0,0,0.13) !important; transform: scale(1.015) !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #6E6E73 !important; font-size: 13px !important; font-weight: 400 !important;
        letter-spacing: 0 !important; text-transform: none !important; font-family: inherit !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div {
        color: #1D1D1F !important; font-size: 30px !important;
        font-weight: 700 !important; letter-spacing: -0.03em !important; font-family: inherit !important;
    }
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] > div { font-size: 13px !important; font-weight: 400 !important; font-family: inherit !important; }
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(0,0,0,0.10) !important; gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: #6E6E73 !important;
        font-size: 15px !important; font-weight: 400 !important;
        border: none !important; border-radius: 0 !important; padding: 12px 22px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0071E3 !important; font-weight: 500 !important;
        border-bottom: 2px solid #0071E3 !important; background: transparent !important;
    }
    [data-testid="stMarkdownContainer"] h1 {
        font-size: 28px !important; font-weight: 700 !important;
        color: #1D1D1F !important; letter-spacing: -0.035em !important;
    }
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-size: 13px !important; font-weight: 600 !important;
        color: #6E6E73 !important; letter-spacing: -0.01em !important;
    }
    [data-testid="stMarkdownContainer"] p { color: #1D1D1F !important; font-size: 15px !important; }
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3 {
        color: #6E6E73 !important; font-size: 13px !important; font-weight: 600 !important;
    }
    div.stDownloadButton > button, div.stButton > button {
        background: #0071E3 !important; color: #FFFFFF !important; border: none !important;
        border-radius: 980px !important; font-size: 15px !important; font-weight: 400 !important;
        padding: 10px 22px !important; min-width: 120px !important;
        transition: background 0.2s ease !important;
    }
    div.stDownloadButton > button:hover, div.stButton > button:hover {
        background: #0077ED !important; opacity: 0.92 !important;
    }
    [data-testid="stSidebar"] div.stButton > button {
        background: #F5F5F7 !important; color: #1D1D1F !important;
        border: 1px solid rgba(0,0,0,0.10) !important; border-radius: 8px !important;
        font-size: 13px !important; font-weight: 500 !important; padding: 6px 10px !important;
        min-width: unset !important; min-height: unset !important;
        white-space: nowrap !important; overflow: hidden !important;
        text-overflow: ellipsis !important; box-shadow: none !important;
    }
    [data-testid="stSidebar"] div.stButton > button:hover { background: #E8E8ED !important; }
    div[data-testid="stRadio"] > label { color: #6E6E73 !important; font-size: 13px !important; }
    [data-testid="stSlider"] > div > div > div > div { background: #0071E3 !important; }
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
        border: 1px solid rgba(0,0,0,0.12) !important; border-radius: 10px !important;
        font-size: 15px !important; background: #FFFFFF !important; color: #1D1D1F !important;
    }
    [data-testid="stDataFrame"] {
        background: #FFFFFF !important; border-radius: 16px !important;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08) !important;
    }
    [data-testid="stAlert"] {
        background: rgba(0,113,227,0.06) !important;
        border: 1px solid rgba(0,113,227,0.18) !important; border-radius: 12px !important;
    }
    hr { border: none !important; border-top: 1px solid rgba(0,0,0,0.08) !important; }
    [data-testid="stExpander"] {
        border: none !important; border-radius: 16px !important;
        background: #FFFFFF !important; box-shadow: 0 2px 16px rgba(0,0,0,0.07) !important;
    }
    /* Presentation mode */
    body.pres-mode [data-testid="stSidebar"] { display: none !important; }

    /* Hide Streamlit toolbar, header, footer */
    header, footer, #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stHeader"],
    [data-testid="baseButton-headerNoPadding"],
    .stAppDeployButton {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Demo / auth gate ───────────────────────────────────────────────────────
    is_demo = st.session_state.get("demo_only", False) and \
              not st.session_state.get("authenticated", False)

    # ── Presentation mode CSS toggle ───────────────────────────────────────────
    if st.session_state.get("presentation_mode"):
        st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { padding-left: 2rem !important; padding-right: 2rem !important; max-width: 100% !important; }
        </style>
        """, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    st.session_state.setdefault("lang_select", "ru")
    st.session_state.setdefault("currency_select", "EUR")
    st.session_state.setdefault("presentation_mode", False)
    st.session_state.setdefault("auditor_name", "Andrew | AI Product Advisor")
    st.session_state.setdefault("contact_url", "https://t.me/weerowoolf")
    st.session_state.setdefault("scenario_confidence", 0.75)

    with st.sidebar:
        lang = st.radio(
            "Language / Язык / Jezik",
            options=list(LANG_NAMES.keys()),
            format_func=lambda k: LANG_NAMES[k],
            horizontal=True,
            key="lang_select",
        )

        # Presentation mode toggle
        pres_label = t(lang, "presentation_off") if st.session_state["presentation_mode"] \
                     else t(lang, "presentation_on")
        if st.button(pres_label, key="btn_pres", use_container_width=True):
            st.session_state["presentation_mode"] = not st.session_state["presentation_mode"]
            st.rerun()

        st.markdown("---")

        # Scenario confidence (2)
        _sc_title = {"en": "Scenario", "ru": "Сценарий", "sr": "Scenario"}
        _sc_opts  = {
            "en": ("Pessimist.", "Realistic", "Optimist."),
            "ru": ("Пессим.",   "Реалист.",  "Оптимист."),
            "sr": ("Pesimist.", "Realista",  "Optimist."),
        }
        _cur_conf = st.session_state["scenario_confidence"]
        st.markdown(
            f'<div style="font-size:11px;font-weight:600;color:#AEAEB2;'
            f'letter-spacing:0.08em;text-transform:uppercase;margin:0 0 6px;">'
            f'{_sc_title[lang]}</div>',
            unsafe_allow_html=True,
        )
        _sca, _scb, _scc = st.columns(3)
        _sc_style = lambda active: (
            "background:#0071E3!important;color:#fff!important;"
            if active else ""
        )
        if _sca.button(
            _sc_opts[lang][0], key="btn_sc_pess", use_container_width=True,
            type="primary" if _cur_conf == 0.50 else "secondary",
        ):
            st.session_state["scenario_confidence"] = 0.50; st.rerun()
        if _scb.button(
            _sc_opts[lang][1], key="btn_sc_real", use_container_width=True,
            type="primary" if _cur_conf == 0.75 else "secondary",
        ):
            st.session_state["scenario_confidence"] = 0.75; st.rerun()
        if _scc.button(
            _sc_opts[lang][2], key="btn_sc_opti", use_container_width=True,
            type="primary" if _cur_conf == 1.00 else "secondary",
        ):
            st.session_state["scenario_confidence"] = 1.00; st.rerun()

        st.markdown("---")

        # Currency selector (3)
        currency = st.radio(
            t(lang, "currency_label"),
            options=list(_CURRENCIES.keys()),
            horizontal=True,
            key="currency_select",
        )

        st.markdown("---")
        st.markdown("## " + t(lang, "sidebar_title"))

        st.session_state.setdefault("company_name", "Marteco Digital Services")
        company_name = st.text_input(t(lang, "company_label"), key="company_name")

        # CSV upload is now in the main area — no sliders in sidebar
        csv_file = None

        st.markdown("---")

        # Auditor settings (8)
        st.markdown(t(lang, "auditor_section"))
        auditor_name = st.text_input(t(lang, "auditor_name_label"), key="auditor_name")
        contact_url  = st.text_input(t(lang, "contact_url_label"),  key="contact_url")

        st.markdown("---")

        # Client history (1)
        st.markdown(t(lang, "history_section"))
        h_col1, h_col2 = st.columns(2)
        if h_col1.button(t(lang, "save_client"), key="btn_save_hist", use_container_width=True):
            _extra = {
                "friction_tax_usd":         st.session_state.get("_saved_friction_tax"),
                "adjusted_confidence_pct":  st.session_state.get("_saved_confidence"),
                "bottleneck_stage":         st.session_state.get("_saved_bottleneck"),
                "roi_pct":                  st.session_state.get("_saved_roi_pct"),
                "rework_rate_pct":          st.session_state.get("_saved_rework_rate"),
                "total_transitions":        st.session_state.get("_saved_total_transitions"),
                "total_rework":             st.session_state.get("_saved_total_rework"),
            }
            if _save_to_history(company_name, _extra):
                st.toast(t(lang, "saved_ok"))

        history = _load_history()
        if history:
            _options = [""] + ["{} ({})".format(h["company_name"], h["saved_at"]) for h in history]
            _sel = h_col2.selectbox(t(lang, "load_label"), _options, key="hist_sel",
                                    label_visibility="collapsed")
            if _sel:
                _idx = _options.index(_sel) - 1
                _restore_from_history(history[_idx])
                st.rerun()
        else:
            h_col2.caption(t(lang, "no_history"))

        st.markdown("---")
        st.caption(t(lang, "footer"))
        if db_available():
            st.caption("🟢 PostgreSQL")
        else:
            st.caption("🟡 Local JSON (no DATABASE_URL)")

    # ── CURRENCY (always read from session state) ──────────────────────────────
    currency = st.session_state.get("currency_select", "EUR")

    math_eng = MathEngine()
    roi_eng  = ROIEngine()

    # ── DERIVE ROI PARAMETERS from mapped DataFrame or use defaults ────────────
    # All computation is data-driven: CSV upload → column mapping → derived params.
    # When no data is loaded we fall back to neutral demonstration defaults.
    _mapped_df = st.session_state.get("mapped_df")

    # Financial inputs (stored by the ETL section below; read here for compute)
    _cost_per_hour = float(st.session_state.get("fin_cost_per_hour", 12.0))
    _hours_per_day = float(st.session_state.get("fin_hours_per_day", 8.0))

    # ── Absorbing-state dictionaries ─────────────────────────────────────────
    # POSITIVE: successful final states → count toward pos_signals and p_before.
    _POSITIVE_KW = {
        # English — generic
        "done", "completed", "complete", "finished", "closed", "won",
        "delivered", "shipped", "approved", "deployed", "production",
        # English — support / ITSM (Jira, Zendesk, ServiceNow)
        "resolved", "resolution", "fixed", "fulfilled",
        "accepted", "verified", "released",
        # Russian
        "завершена", "завершено", "выполнено", "продано", "закрыта",
        "закрыто", "доставлено",
        # Serbian
        "završeno", "zatvoreno", "isporučeno",
    }
    # NEGATIVE: failed / neutral final states → absorbing, but NOT positive signals.
    _NEGATIVE_KW = {
        "refunded", "cancelled", "canceled", "rejected", "archived",
        "seized", "confiscated", "lost", "failed", "expired", "returned",
        "отклонено", "отменено", "возврат", "изъято", "потеряно",
        "odbijeno", "otkazano", "oduzeto",
    }
    # All absorbing states = union (used to detect that a ticket has actually ended)
    _ABSORBING_KW = _POSITIVE_KW | _NEGATIVE_KW

    if _mapped_df is not None and len(_mapped_df) > 0:
        # ── Derive cycle time per entity ────────────────────────────────────
        _entity_times = _mapped_df.groupby("entity_id")["time_spent"].sum()
        _unique_ent   = max(1, _mapped_df["entity_id"].nunique())
        _avg_cycle    = max(1.0, float(_entity_times.mean()))

        # ── Derive volume & deal cadence ────────────────────────────────────
        # Assume dataset represents ~3 months of activity (conservative)
        volume      = max(10, _unique_ent)
        cycle_before = round(_avg_cycle)
        cycle_after  = max(1, round(cycle_before * 0.45))
        deals_month  = max(1, int(volume / 3))

        # ── Derive manual hours from time spent * working hours per day ─────
        manual_hours = max(1.0, _avg_cycle * _hours_per_day * deals_month / 30.0)
        hour_rate    = _cost_per_hour

        # ── Estimate completion probability from absorbing states ───────────
        _next_lower   = _mapped_df["next_stage"].astype(str).str.lower().str.strip()
        # Positive completions: successful endings (Resolved, Done, Closed, …)
        _pos_done     = _mapped_df[_next_lower.isin(_POSITIVE_KW)]["entity_id"].nunique()
        # All endings: positive + negative (Refunded, Cancelled, Rejected, …)
        _all_done     = _mapped_df[_next_lower.isin(_ABSORBING_KW)]["entity_id"].nunique()
        # p_before = success rate = positive / total entities observed
        p_before      = max(30, min(95, int(_pos_done / _unique_ent * 100)))
        p_after       = min(99, int(p_before * 1.28))
        # Bayesian signals: how many succeeded out of all that finished
        pos_signals   = max(1, _pos_done)
        tot_signals   = max(2, max(_all_done, _unique_ent))
    else:
        # ── Neutral demonstration defaults (no CSV loaded) ──────────────────
        volume, cycle_before, cycle_after = 600, 21, 9
        deals_month, manual_hours         = 25, 320
        hour_rate                         = _cost_per_hour
        p_before, p_after                 = 74, 96
        pos_signals, tot_signals          = 4, 5

    # Constants not derivable from process-log data — sensible industry defaults
    automation_rate = 80
    error_before    = 8.5
    error_after     = 1.2
    cost_per_error  = 95
    deal_value      = 1000
    impl_cost       = 15000
    pipeline_util   = 30
    deals           = deals_month

    # Seed Bayesian tab sliders.
    # When CSV is loaded, always overwrite so sliders reflect actual data.
    # When no CSV (else branch), only set on first run — don't clobber manual adjustments.
    if _mapped_df is not None and len(_mapped_df) > 0:
        st.session_state["pos_signals"]    = pos_signals
        st.session_state["tot_signals"]    = tot_signals
        # Prior = empirical completion rate from CSV (p_before already clamped 30-95)
        st.session_state["bayes_prior_pct"] = p_before
    else:
        st.session_state.setdefault("pos_signals",    pos_signals)
        st.session_state.setdefault("tot_signals",    tot_signals)
        st.session_state.setdefault("bayes_prior_pct", 34)

    inp = ROIInput(
        company_name=company_name,
        manual_hours_per_month=float(manual_hours),
        automation_rate=automation_rate / 100.0,
        hour_rate_eur=float(hour_rate),
        error_rate_before_pct=float(error_before),
        error_rate_after_pct=float(error_after),
        cost_per_error_eur=float(cost_per_error),
        monthly_volume=int(volume),
        deal_cycle_before_days=float(cycle_before),
        deal_cycle_after_days=float(cycle_after),
        deals_per_month=int(deals),
        avg_deal_value_eur=float(deal_value),
        p_complete_before=p_before / 100.0,
        p_complete_after=p_after / 100.0,
        implementation_cost_eur=float(impl_cost),
        positive_signals=int(pos_signals),
        total_signals=int(tot_signals),
        pipeline_utilization_pct=float(pipeline_util),
    )

    _bayes_prior_rate = st.session_state.get("bayes_prior_pct", 34) / 100.0
    bayes_res = math_eng.bayesian_update(
        inp.positive_signals, inp.total_signals, prior_rate=_bayes_prior_rate
    )
    res = roi_eng.calculate(inp, bayes_result=bayes_res)
    res = _apply_confidence(res, st.session_state.get("scenario_confidence", 0.75))
    st.session_state["_saved_roi_pct"] = res.roi_pct

    process_log = None
    _default_edges = [
        ("Lead", "In Review", 3.0),
        ("In Review", "Approved", 0.8),
        ("In Review", "Revision", 0.6),
        ("Revision", "In Review", 0.7),
        ("Revision", "Rejected", 0.4),
    ]
    _active_edges = _default_edges
    graph_res = math_eng.graph_bottleneck(_active_edges)

    # ── Data-driven Markov graph from ETL-mapped DataFrame ──────────────────
    _mapped_df: pd.DataFrame | None = st.session_state.get("mapped_df")
    _mkv_graph_res: MarkovGraphResult | None = None
    if _mapped_df is not None and len(_mapped_df) > 0:
        try:
            _mkv_graph_res = build_markov_graph(_mapped_df)
        except Exception as _e:
            _mkv_graph_res = None

    _default_Q      = np.array([[0.2, 0.3], [0.1, 0.4]])
    _default_states = ["Qualification", "Proposal"] if lang == "en" else (
                      ["Квалификация", "Предложение"] if lang == "ru" else
                      ["Kvalifikacija", "Ponuda"])
    Q_mat    = _default_Q
    m_states = _default_states

    # ── When CSV is loaded: derive Q_mat + m_states from real transition data ──
    if _mkv_graph_res is not None:
        try:
            _G = _mkv_graph_res.G
            # Absorbing nodes = terminal states (no outgoing edges)
            _absorbing = {n for n in _G.nodes() if _G.out_degree(n) == 0}
            # Also treat next_stage-only nodes with absorbing keywords as absorbing
            _absorbing |= {
                n for n in _G.nodes()
                if any(kw in str(n).lower()
                       for kw in ("delivered","seized","done","closed","resolved",
                                  "won","rejected","cancelled","canceled","lost"))
                and _G.out_degree(n) == 0
            }
            # Transient nodes: have outgoing edges and are NOT absorbing
            _transient = [n for n in _G.nodes()
                          if n not in _absorbing and _G.out_degree(n) > 0]
            if len(_transient) >= 2:
                _n = len(_transient)
                _Q = np.zeros((_n, _n))
                for _i, _fi in enumerate(_transient):
                    for _j, _tj in enumerate(_transient):
                        _Q[_i][_j] = _mkv_graph_res.transition_probs.get(
                            (_fi, _tj), 0.0)
                # Validate: I-Q must be invertible (spectral radius of Q < 1)
                _eigs = np.abs(np.linalg.eigvals(_Q))
                if _eigs.max() < 1.0:
                    Q_mat    = _Q
                    m_states = _transient
        except Exception:
            pass  # Fall back to demo Q_mat silently

    if process_log is not None and process_log.avg_time_per_state:
        _fallback_h = float(cycle_before) * 24 / max(len(m_states), 1)
        state_times = np.array([
            process_log.avg_time_per_state.get(s, _fallback_h) or _fallback_h
            for s in m_states
        ])
    else:
        state_times = np.full(len(m_states), float(cycle_before) * 24 / max(len(m_states), 1))
    try:
        markov_res = math_eng.markov_absorbing(
            Q_mat, state_times, m_states,
            p_complete_before=p_before / 100.0,
            p_complete_after=p_after / 100.0,
        )
        I_mat = np.eye(Q_mat.shape[0])
        N_mat = np.linalg.inv(I_mat - Q_mat)
    except Exception:
        markov_res = None
        N_mat = None

    # ── Benchmarks (generic — no industry preset) ──────────────────────────────
    _bench = _BENCHMARKS[None]
    _bench_roi_eur = impl_cost * _bench["net_roi_mult"]

    # ── HEADER ─────────────────────────────────────────────────────────────────
    if st.session_state.get("presentation_mode"):
        if st.button(t(lang, "presentation_off"), key="pres_exit_top"):
            st.session_state["presentation_mode"] = False
            st.rerun()

    st.markdown(
        '<div style="margin-bottom:4px;">'
        '<span style="font-size:28px;font-weight:700;color:#1D1D1F;letter-spacing:-0.035em;'
        'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
        + t(lang, "app_title") +
        '</span></div>'
        '<div style="font-size:15px;color:#6E6E73;font-weight:400;letter-spacing:-0.01em;'
        'margin-bottom:24px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
        + company_name + '&ensp;·&ensp;' + t(lang, "app_subtitle") +
        '</div>',
        unsafe_allow_html=True,
    )

    # ── ETL: DATA UPLOAD & COLUMN MAPPING ──────────────────────────────────────
    # Localized labels for the ETL section
    _etl_lbl = {
        "en": {
            "section":    "Data Upload",
            "cost_label": "Cost per hour of team work ($)",
            "hours_label":"Average working hours per day",
            "upload_label": "Upload process data (.csv)",
            "col_entity":  "Column for Entity ID (Task_ID, Deal_ID, …)",
            "col_current": "Column for Current Stage",
            "col_next":    "Column for Next Stage",
            "col_time":    "Column for Time Spent (Days)",
            "preview":     "Data preview",
            "no_data":     "Upload a CSV file to begin. The engine will derive all ROI parameters automatically.",
            "data_ok":     "Data loaded",
            "lock":        "Sign in to upload your own CSV data",
            "sample":      "Download sample CSV",
        },
        "ru": {
            "section":    "Загрузка данных",
            "cost_label": "Стоимость часа работы команды ($)",
            "hours_label":"Рабочих часов в день (среднее)",
            "upload_label": "Загрузите файл процессных данных (.csv)",
            "col_entity":  "Колонка — ID сущности (Task_ID, Deal_ID, …)",
            "col_current": "Колонка — Текущий этап",
            "col_next":    "Колонка — Следующий этап",
            "col_time":    "Колонка — Время на этапе (дней)",
            "preview":     "Предпросмотр данных",
            "no_data":     "Загрузите CSV-файл, чтобы начать. Движок автоматически вычислит все ROI-параметры.",
            "data_ok":     "Данные загружены",
            "lock":        "Войдите, чтобы загрузить свои CSV-данные",
            "sample":      "Скачать пример CSV",
        },
        "sr": {
            "section":    "Učitavanje podataka",
            "cost_label": "Trošak sata rada tima ($)",
            "hours_label":"Prosečni radni sati po danu",
            "upload_label": "Učitajte procesne podatke (.csv)",
            "col_entity":  "Kolona za ID entiteta (Task_ID, Deal_ID, …)",
            "col_current": "Kolona za trenutnu fazu",
            "col_next":    "Kolona za sledeću fazu",
            "col_time":    "Kolona za vreme (dani)",
            "preview":     "Pregled podataka",
            "no_data":     "Učitajte CSV fajl da biste počeli. Motor će automatski izračunati ROI parametre.",
            "data_ok":     "Podaci učitani",
            "lock":        "Prijavite se da biste učitali sopstvene CSV podatke",
            "sample":      "Preuzmi primer CSV",
        },
    }
    _el = _etl_lbl.get(lang, _etl_lbl["en"])

    with st.container():
        st.markdown(
            f'<div style="font-size:11px;font-weight:600;color:#AEAEB2;'
            f'letter-spacing:0.08em;text-transform:uppercase;margin:0 0 10px;">'
            f'{_el["section"]}</div>',
            unsafe_allow_html=True,
        )

        # ── Financial inputs (always visible) ────────────────────────────────
        st.session_state.setdefault("fin_cost_per_hour", 12.0)
        st.session_state.setdefault("fin_hours_per_day", 8.0)
        _fin1, _fin2, _fin_pad = st.columns([2, 2, 5])
        _fin1.number_input(
            _el["cost_label"], min_value=1.0, max_value=999.0,
            step=1.0, format="%.0f", key="fin_cost_per_hour",
        )
        _fin2.number_input(
            _el["hours_label"], min_value=1.0, max_value=24.0,
            step=0.5, format="%.1f", key="fin_hours_per_day",
        )

        # ── File uploader ─────────────────────────────────────────────────────
        if is_demo:
            st.info(_el["lock"])
            try:
                with open("data/mock_client_data.csv", "rb") as _sf:
                    st.download_button(
                        _el["sample"], _sf.read(),
                        file_name="sample_audit_data.csv", mime="text/csv",
                        key="dl_sample_etl",
                    )
            except Exception:
                pass
            _etl_file = None
        else:
            _etl_file = st.file_uploader(
                _el["upload_label"], type=["csv"], key="etl_csv_file",
            )

        if _etl_file is not None:
            # ── Detect new file to reset column-mapping dropdowns ─────────────
            _etl_fname = getattr(_etl_file, "name", "")
            _etl_fsize = getattr(_etl_file, "size", 0)
            _etl_fkey  = f"{_etl_fname}:{_etl_fsize}"
            if st.session_state.get("_etl_fkey") != _etl_fkey:
                st.session_state["_etl_fkey"] = _etl_fkey
                # Clear column-mapping dropdowns AND any previously mapped data
                # so a new upload never shows stale results from the old file.
                for _k in ("_etl_idx_entity", "_etl_idx_current",
                           "_etl_idx_next", "_etl_idx_time",
                           "mapped_df", "col_mapping", "_etl_computed_fkey"):
                    st.session_state.pop(_k, None)

            # ── Read CSV via bulletproof universal parser ─────────────────
            # load_and_clean_csv: detects true header, strips garbage rows,
            # normalises column names, drops Unnamed + all-NaN rows.
            _etl_file.seek(0)
            _raw_df = load_and_clean_csv(_etl_file)
            if _raw_df is None:
                st.error("Не удалось распознать CSV — убедитесь, что файл "
                         "содержит строку заголовков с 3+ непустыми столбцами. / "
                         "Could not detect a valid header row (need 3+ non-empty columns).")
            else:
                _cols = list(_raw_df.columns)

                # ── Smart auto-detect default columns ─────────────────────────
                def _best_idx(hints: list, cols: list) -> int:
                    for h in hints:
                        for i, c in enumerate(cols):
                            if h.lower() in c.lower():
                                return i
                    return 0

                st.session_state.setdefault(
                    "_etl_idx_entity",
                    _best_idx(["entity_id","id","deal","task","project","client"], _cols),
                )
                st.session_state.setdefault(
                    "_etl_idx_current",
                    _best_idx(["current","from","stage","phase","status","cur"], _cols),
                )
                st.session_state.setdefault(
                    "_etl_idx_next",
                    _best_idx(["next","to","target","dest"], _cols),
                )
                st.session_state.setdefault(
                    "_etl_idx_time",
                    _best_idx(["time","days","duration","spent","hours"], _cols),
                )

                # ── 4 side-by-side selectboxes for column mapping ─────────────
                _sm1, _sm2, _sm3, _sm4 = st.columns(4)
                col_entity  = _sm1.selectbox(
                    _el["col_entity"],  _cols,
                    index=min(st.session_state["_etl_idx_entity"],  len(_cols)-1),
                    key="_etl_sel_entity",
                )
                col_current = _sm2.selectbox(
                    _el["col_current"], _cols,
                    index=min(st.session_state["_etl_idx_current"], len(_cols)-1),
                    key="_etl_sel_current",
                )
                col_next    = _sm3.selectbox(
                    _el["col_next"],    _cols,
                    index=min(st.session_state["_etl_idx_next"],    len(_cols)-1),
                    key="_etl_sel_next",
                )
                col_time    = _sm4.selectbox(
                    _el["col_time"],    _cols,
                    index=min(st.session_state["_etl_idx_time"],    len(_cols)-1),
                    key="_etl_sel_time",
                )

                # ── Normalize: rename mapped columns to internal standard names ─
                _mapped = _raw_df[[col_entity, col_current,
                                   col_next, col_time]].copy()
                _mapped.columns = ["entity_id", "current_stage",
                                   "next_stage", "time_spent"]
                _mapped["time_spent"] = (
                    pd.to_numeric(_mapped["time_spent"], errors="coerce").fillna(0)
                )
                _mapped["entity_id"] = _mapped["entity_id"].astype(str).str.strip()
                _mapped = _mapped[_mapped["time_spent"] >= 0].reset_index(drop=True)

                # ── Persist to session_state for the compute block above ───────
                st.session_state["mapped_df"]   = _mapped
                st.session_state["col_mapping"] = {
                    "entity_id":    col_entity,
                    "current_stage": col_current,
                    "next_stage":   col_next,
                    "time_spent":   col_time,
                }
                # ── Force re-run so the compute block (rendered before this
                #    section) picks up the newly mapped data immediately.
                #    Guard with a fkey so we only rerun once per new file. ──
                if st.session_state.get("_etl_computed_fkey") != _etl_fkey:
                    st.session_state["_etl_computed_fkey"] = _etl_fkey
                    st.rerun()

                # ── Data summary badge ─────────────────────────────────────────
                _n_ent = _mapped["entity_id"].nunique()
                _n_row = len(_mapped)
                _n_stg = _mapped["current_stage"].nunique()
                st.markdown(
                    f'<div style="display:flex;gap:12px;align-items:center;'
                    f'margin:8px 0 4px;flex-wrap:wrap;">'
                    f'<span style="background:#34C75914;border:1px solid #34C75944;'
                    f'border-radius:980px;padding:3px 12px;font-size:12px;'
                    f'font-weight:600;color:#34C759;">'
                    f'✓ {_el["data_ok"]}</span>'
                    f'<span style="font-size:12px;color:#AEAEB2;">'
                    f'{_n_row:,} rows · {_n_ent:,} entities · {_n_stg} stages</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Collapsible data preview ───────────────────────────────────
                with st.expander(f"📋 {_el['preview']} ({_n_row:,} rows)", expanded=False):
                    st.dataframe(_mapped.head(30), use_container_width=True)

        elif st.session_state.get("mapped_df") is None:
            # No file and no cached data — show upload prompt
            st.info(_el["no_data"])

    st.markdown("---")

    # ── Scenario badge ──────────────────────────────────────────────────────────
    _conf_val = st.session_state.get("scenario_confidence", 0.75)
    _conf_badge = {
        0.50: {"en": "Pessimistic · 50%", "ru": "Пессимистичный · 50%", "sr": "Pesimistički · 50%", "color": "#FF9F0A"},
        0.75: {"en": "Realistic · 75%",   "ru": "Реалистичный · 75%",   "sr": "Realistički · 75%",  "color": "#0071E3"},
        1.00: {"en": "Optimistic · 100%", "ru": "Оптимистичный · 100%", "sr": "Optimistički · 100%", "color": "#34C759"},
    }.get(_conf_val, {"en": "Realistic · 75%", "ru": "Реалистичный · 75%", "sr": "Realistički · 75%", "color": "#0071E3"})
    st.markdown(
        f'<div style="display:inline-block;background:{_conf_badge["color"]}18;'
        f'border:1px solid {_conf_badge["color"]}44;border-radius:980px;'
        f'padding:3px 14px;margin-bottom:12px;">'
        f'<span style="font-size:12px;font-weight:600;color:{_conf_badge["color"]};'
        f'letter-spacing:0.02em;">{_conf_badge[lang]}</span></div>',
        unsafe_allow_html=True,
    )

    # ── KPI cards with benchmark delta ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    _roi_delta = res.net_roi - _bench_roi_eur
    _pay_delta = _bench["payback"] - res.payback_months
    _cur_sym = _CURRENCIES.get(currency, _CURRENCIES["EUR"])["sym"]
    _sign_roi = "+" if _roi_delta >= 0 else ""
    c1.metric(
        t(lang, "metric_net_roi"),
        _fmt(res.net_roi, currency),
        delta="{}{} {}".format(_sign_roi, _fmt(_roi_delta, currency), t(lang, "vs_industry")),
        delta_color="normal",
    )
    c2.metric(
        t(lang, "metric_payback"),
        "{:.1f} {}".format(res.payback_months, t(lang, "months")),
        delta="{:+.1f} {} {}".format(_pay_delta, t(lang, "months"), t(lang, "vs_industry")),
        delta_color="normal",
    )
    c3.metric(t(lang, "metric_bayes"), "{:.1f}%".format(res.bayesian_posterior_pct))
    c4.metric(t(lang, "metric_impl"), _fmt(impl_cost, currency))

    _risk_adj_val = res.net_roi * (res.bayesian_posterior_pct / 100.0)
    st.markdown(
        '<div style="background:linear-gradient(90deg,#F5F5F7,#EEF5FF);'
        'border-radius:12px;padding:10px 18px;margin:8px 0 0 0;'
        'border-left:4px solid #0071E3;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
        '<span style="font-size:12px;font-weight:600;color:#6E6E73;text-transform:uppercase;'
        'letter-spacing:0.07em;">' + t(lang, "risk_adj_roi") + '</span>&nbsp;&nbsp;'
        '<span style="font-size:20px;font-weight:700;color:#0071E3;">'
        + _fmt(_risk_adj_val, currency) + '</span>&nbsp;'
        '<span style="font-size:12px;color:#AEAEB2;">'
        + {
            "en": "Net ROI × Bayesian confidence",
            "ru": "Чистый ROI × Байес. доверие",
            "sr": "Neto ROI × Bajesovsko poverenje",
        }.get(lang, "Net ROI × Bayesian confidence")
        + ' = ' + _fmt(_risk_adj_val, currency) + '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        t(lang, "tab_roi"), t(lang, "tab_graph"), t(lang, "tab_markov"),
        t(lang, "tab_bayes"), t(lang, "tab_passport"), t(lang, "tab_precision"),
        t(lang, "tab_about"),
    ])

    # ── TAB 1: ROI BREAKDOWN ───────────────────────────────────────────────────
    with tab1:
        st.subheader(t(lang, "waterfall_title"))
        _wf_rate = _CURRENCIES[currency]["rate"]
        _wf_sym  = _CURRENCIES[currency]["sym"]
        labels   = [t(lang, "time_saved"), t(lang, "error_saved"),
                    t(lang, "revenue_speed"), t(lang, "revenue_conv"),
                    t(lang, "investment"), t(lang, "net_roi")]
        values   = [v * _wf_rate for v in [
            res.time_saved_annual, res.error_reduction_annual,
            res.revenue_impact_annual, res.markov_gain_annual,
            -impl_cost, res.net_roi]]
        measures = ["relative", "relative", "relative", "relative", "relative", "total"]

        fig_wf = go.Figure(go.Waterfall(
            name="ROI", orientation="v",
            measure=measures, x=labels, y=values,
            connector=dict(line=dict(color="rgba(26,50,113,0.18)", width=1)),
            increasing=dict(marker_color=_C["green"]),
            decreasing=dict(marker_color=_C["red"]),
            totals=dict(marker_color=_C["navy"]),
            texttemplate="%{y:,.0f} " + _wf_sym, textposition="outside",
            textfont=dict(color="#7a8499", size=10),
        ))
        fig_wf.update_layout(showlegend=False, height=420, **CHART_LAYOUT)
        st.plotly_chart(fig_wf, width="stretch")

        col_p, col_t = st.columns(2)
        with col_p:
            st.subheader(t(lang, "pie_title"))
            pie_labels = [t(lang, "time_saved"), t(lang, "error_saved"),
                          t(lang, "revenue_speed"), t(lang, "revenue_conv")]
            pie_vals   = [res.time_saved_annual, res.error_reduction_annual,
                          res.revenue_impact_annual, res.markov_gain_annual]
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels, values=pie_vals, hole=0.5,
                marker=dict(
                    colors=["#34C759", "#5AC8FA", "#0071E3", "#AF52DE"],
                    line=dict(color="rgba(240,236,228,1)", width=2),
                ),
                textinfo="label+percent", textfont=dict(color="#4a5168", size=11),
            ))
            fig_pie.update_layout(height=360, **CHART_LAYOUT)
            st.plotly_chart(fig_pie, width="stretch")

        with col_t:
            st.subheader("")
            st.markdown("<br>", unsafe_allow_html=True)
            _cur_sym = _CURRENCIES[currency]["sym"]
            _rate    = _CURRENCIES[currency]["rate"]
            df_bd = pd.DataFrame({
                t(lang, "component"): [t(lang, "time_saved"), t(lang, "error_saved"),
                                       t(lang, "revenue_speed"), t(lang, "revenue_conv"),
                                       t(lang, "total"), t(lang, "investment"), t(lang, "net_roi")],
                f"{_cur_sym}/год":  [round(v * _rate) for v in [
                    res.time_saved_annual, res.error_reduction_annual,
                    res.revenue_impact_annual, res.markov_gain_annual,
                    res.total_benefit, -impl_cost, res.net_roi]],
            })
            st.dataframe(df_bd, height=300)

        # ── Gauge + Radar row ────────────────────────────────────────────────
        _g1, _g2 = st.columns(2)

        # Gauge: ROI % vs industry benchmark
        with _g1:
            _bench_roi = _bench["roi_pct"]
            _gauge_max = max(int(res.roi_pct * 1.5), 600)
            _gauge_title = {
                "en": f"ROI % vs industry avg ({_bench_roi:.0f}%)",
                "ru": f"ROI % vs ср. по рынку ({_bench_roi:.0f}%)",
                "sr": f"ROI % vs prosek grane ({_bench_roi:.0f}%)",
            }.get(lang, f"ROI % vs benchmark ({_bench_roi:.0f}%)")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=res.roi_pct,
                delta={"reference": _bench_roi, "valueformat": ".0f",
                       "increasing": {"color": "#34C759"}, "decreasing": {"color": "#FF3B30"}},
                number={"suffix": "%", "font": {"size": 32, "color": "#1D1D1F", "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, _gauge_max], "tickcolor": "#AEAEB2",
                             "tickfont": {"size": 10, "color": "#AEAEB2"}},
                    "bar": {"color": "#0071E3", "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, _bench_roi * 0.5], "color": "rgba(255,59,48,0.08)"},
                        {"range": [_bench_roi * 0.5, _bench_roi], "color": "rgba(255,159,10,0.08)"},
                        {"range": [_bench_roi, _gauge_max], "color": "rgba(52,199,89,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": "#FF9F0A", "width": 2},
                        "thickness": 0.75,
                        "value": _bench_roi,
                    },
                },
                title={"text": _gauge_title, "font": {"size": 12, "color": "#6E6E73"}},
            ))
            fig_gauge.update_layout(
                height=280, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_gauge, width="stretch")

        # Radar: 4-component ROI structure
        with _g2:
            _radar_labels = [
                t(lang, "time_saved"), t(lang, "error_saved"),
                t(lang, "revenue_speed"), t(lang, "revenue_conv"),
            ]
            _radar_vals = [
                res.time_saved_annual, res.error_reduction_annual,
                res.revenue_impact_annual, res.markov_gain_annual,
            ]
            _max_r = max(_radar_vals) if any(_radar_vals) else 1
            _radar_pct = [v / _max_r * 100 for v in _radar_vals]
            _radar_labels_closed = _radar_labels + [_radar_labels[0]]
            _radar_pct_closed    = _radar_pct    + [_radar_pct[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=_radar_pct_closed, theta=_radar_labels_closed,
                fill="toself", fillcolor="rgba(0,113,227,0.10)",
                line=dict(color="#0071E3", width=2.5),
                name="ROI",
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=9, color="#AEAEB2"),
                                   gridcolor="rgba(0,0,0,0.06)", linecolor="rgba(0,0,0,0.06)"),
                    angularaxis=dict(tickfont=dict(size=11, color="#1D1D1F"),
                                     gridcolor="rgba(0,0,0,0.06)"),
                ),
                showlegend=False, height=280,
                margin=dict(l=40, r=40, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif"),
                title=dict(
                    text={
                        "en": "Benefit structure (relative)",
                        "ru": "Структура выгод (относительно)",
                        "sr": "Struktura koristi (relativno)",
                    }.get(lang, "Benefit structure"),
                    font=dict(size=12, color="#6E6E73"),
                ),
            )
            st.plotly_chart(fig_radar, width="stretch")

        # ── Bullet chart: Payback Progress ───────────────────────────────────
        _pay_val   = res.payback_months
        _pay_max   = 36
        _pay_tgt   = 12
        _pay_warn  = 18
        _pay_color = "#34C759" if _pay_val <= _pay_tgt else ("#FF9F0A" if _pay_val <= _pay_warn else "#FF3B30")
        _pay_rate  = _CURRENCIES[currency]["rate"]
        _pay_label = _CURRENCIES[currency]["label"]
        _bullet_x_label = {
            "en": "Months to payback",
            "ru": "Месяцев до окупаемости",
            "sr": "Meseci do povrata",
        }.get(lang, "Months to payback")
        fig_bullet = go.Figure()
        fig_bullet.add_trace(go.Bar(
            x=[_pay_max], y=[t(lang, "bullet_title")], orientation="h",
            marker_color="rgba(200,200,200,0.15)", width=0.22, showlegend=False, hoverinfo="none",
        ))
        fig_bullet.add_trace(go.Bar(
            x=[_pay_warn], y=[t(lang, "bullet_title")], orientation="h",
            marker_color="rgba(255,159,10,0.13)", width=0.22, showlegend=False, hoverinfo="none",
        ))
        fig_bullet.add_trace(go.Bar(
            x=[_pay_tgt], y=[t(lang, "bullet_title")], orientation="h",
            marker_color="rgba(52,199,89,0.13)", width=0.22, showlegend=False, hoverinfo="none",
        ))
        fig_bullet.add_trace(go.Bar(
            x=[_pay_val], y=[t(lang, "bullet_title")], orientation="h",
            marker_color=_pay_color, width=0.12, showlegend=True,
            name="{:.1f} {}".format(_pay_val, t(lang, "months")),
        ))
        fig_bullet.add_vline(
            x=_pay_tgt, line_dash="dash", line_color="#34C759", line_width=2,
            annotation_text=t(lang, "bullet_target"),
            annotation_position="top right",
            annotation_font=dict(size=10, color="#34C759"),
        )
        fig_bullet.update_layout(
            barmode="overlay",
            height=130,
            margin=dict(l=10, r=30, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            showlegend=True,
            legend=dict(x=0.75, y=1.2, font=dict(size=11)),
            xaxis=dict(
                range=[0, _pay_max], title=_bullet_x_label,
                gridcolor="rgba(0,0,0,0.05)", tickfont=dict(color="#AEAEB2", size=10),
                zeroline=False,
            ),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig_bullet, width="stretch")

        # Scenario comparison (5)
        with st.expander(t(lang, "scenario_section")):
            sc1, sc2, sc3 = st.columns(3)
            st.session_state.setdefault("scen_b_auto", min(automation_rate + 10, 95))
            st.session_state.setdefault("scen_b_cost", max(impl_cost - 3000, 5000))
            st.session_state.setdefault("scen_b_cycle", max(cycle_after - 2, 1))
            scen_b_auto  = sc1.slider(t(lang, "scen_b_automation"), 50, 95, key="scen_b_auto")
            scen_b_cost  = sc2.slider(t(lang, "scen_b_impl_cost"), 5000, 100000, step=1000, key="scen_b_cost")
            scen_b_cycle = sc3.slider(t(lang, "scen_b_cycle"), 1, 30, key="scen_b_cycle")

            inp_b = ROIInput(
                company_name=company_name,
                manual_hours_per_month=float(manual_hours),
                automation_rate=scen_b_auto / 100.0,
                hour_rate_eur=float(hour_rate),
                error_rate_before_pct=float(error_before),
                error_rate_after_pct=float(error_after),
                cost_per_error_eur=float(cost_per_error),
                monthly_volume=int(volume),
                deal_cycle_before_days=float(cycle_before),
                deal_cycle_after_days=float(scen_b_cycle),
                deals_per_month=int(deals),
                avg_deal_value_eur=float(deal_value),
                p_complete_before=p_before / 100.0,
                p_complete_after=p_after / 100.0,
                implementation_cost_eur=float(scen_b_cost),
                positive_signals=4, total_signals=5,
            )
            res_b = roi_eng.calculate(inp_b)

            df_cmp = pd.DataFrame({
                "": [t(lang, "net_roi"), t(lang, "metric_payback"), "ROI %"],
                t(lang, "scenario_a"): [
                    _fmt(res.net_roi, currency),
                    "{:.1f} {}".format(res.payback_months, t(lang, "months")),
                    "{:.0f}%".format(res.roi_pct),
                ],
                t(lang, "scenario_b"): [
                    _fmt(res_b.net_roi, currency),
                    "{:.1f} {}".format(res_b.payback_months, t(lang, "months")),
                    "{:.0f}%".format(res_b.roi_pct),
                ],
            })
            st.dataframe(df_cmp, hide_index=True)

    # ── TAB 2: GRAPH ──────────────────────────────────────────────────────────
    with tab2:
        _g_title = {"en": "Process Flow & Bottleneck Analysis",
                    "ru": "Граф процессов и анализ узких мест",
                    "sr": "Graf procesa i analiza uskih grla"}.get(lang, "Process Graph")
        st.subheader(_g_title)

        if _mkv_graph_res is None:
            _g_upload_hint = {
                "en": "Upload and map a CSV in the **ETL Pipeline** tab to activate the Markov process graph.",
                "ru": "Загрузите и замапьте CSV во вкладке **ETL Pipeline**, чтобы активировать граф процессов.",
                "sr": "Postavite i mapirajte CSV u tabu **ETL Pipeline** da aktivirate grafikon procesa.",
            }.get(lang, "Upload a CSV to activate the process graph.")
            st.info(_g_upload_hint)

            # ── Fallback: old static graph ────────────────────────────────────
            nodes = list(graph_res.betweenness.keys())
            node_colors = [_C["red"] if n == graph_res.bottleneck_node else _C["navy"] for n in nodes]
            node_sizes  = [22 + graph_res.betweenness[n] * 200 for n in nodes]
            angle_step  = 2 * np.pi / max(len(nodes), 1)
            pos = {n: (np.cos(i * angle_step), np.sin(i * angle_step)) for i, n in enumerate(nodes)}
            edge_x, edge_y = [], []
            for frm, to, _ in _active_edges:
                if frm in pos and to in pos:
                    edge_x += [pos[frm][0], pos[to][0], None]
                    edge_y += [pos[frm][1], pos[to][1], None]
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                       line=dict(color="rgba(26,50,113,0.15)", width=1.5),
                                       hoverinfo="none"))
            fig_g.add_trace(go.Scatter(
                x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                mode="markers+text", text=nodes, textposition="top center",
                textfont=dict(color="#1a2744", size=11),
                marker=dict(color=node_colors, size=node_sizes,
                            line=dict(color="rgba(255,255,255,0.9)", width=2)),
                hovertemplate="%{text}<br>" + t(lang, "centrality_col") + ": %{customdata:.4f}<extra></extra>",
                customdata=[graph_res.betweenness[n] for n in nodes],
            ))
            _no_axes = {k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")}
            fig_g.update_layout(showlegend=False, height=380,
                                xaxis=dict(visible=False), yaxis=dict(visible=False),
                                **_no_axes)
            st.plotly_chart(fig_g, width="stretch")

        else:
            # ── FULL DATA-DRIVEN MARKOV GRAPH ─────────────────────────────────
            _mgr = _mkv_graph_res
            _G   = _mgr.G

            # ── Corporate CSS injection ───────────────────────────────────────
            st.markdown("""
<style>
.clevel-block {
    background: #1D1D1F;
    border-radius: 14px;
    padding: 28px 32px 22px;
    margin-bottom: 24px;
}
.clevel-rule {
    border: none;
    border-top: 1px solid #3A3A3C;
    margin: 0 0 22px;
}
.clevel-section-label {
    font-family: 'SF Mono', 'Courier New', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    color: #48484A;
    text-transform: uppercase;
    margin-bottom: 18px;
}
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border-radius: 18px !important;
    padding: 24px 26px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.08) !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 13px !important;
    letter-spacing: 0 !important;
    color: #6E6E73 !important;
    text-transform: none !important;
    font-family: inherit !important;
}
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    color: #1D1D1F !important;
    font-weight: 700 !important;
    font-family: inherit !important;
}
[data-testid="stMetricDelta"] {
    font-size: 13px !important;
    font-family: inherit !important;
}
</style>""", unsafe_allow_html=True)

            # ── Prior probability slider ──────────────────────────────────────
            _prior_label = {
                "en": "Prior Probability of Success — Target %",
                "ru": "Априорная вероятность успеха — целевой %",
                "sr": "Apriorni procenat uspešnosti — cilj %",
            }.get(lang, "Prior Probability of Success (%)")

            _prior_help = {
                "en": ("Your initial confidence that the process will succeed "
                       "before observing the rework data. "
                       "The model updates this using the Bayesian likelihood ratio."),
                "ru": ("Начальная уверенность в успехе процесса до анализа данных. "
                       "Модель скорректирует её через коэффициент правдоподобия Байеса."),
                "sr": ("Vaša početna uvernost pre analize podataka. "
                       "Model je koriguje bayesovskim koeficijentom."),
            }.get(lang, "")

            _prior_pct = st.slider(
                _prior_label,
                min_value=10, max_value=99, value=85, step=1,
                format="%d%%",
                help=_prior_help,
                key="clevel_prior_pct",
            )

            # ── Friction Tax computation ──────────────────────────────────────
            _bt_node       = _mgr.bottleneck_node
            _lat_days      = _mgr.decision_latency.get(_bt_node, 0.0)
            # Fallback: use avg_days × rework_rate when no pure latency captured
            if _lat_days <= 0.0:
                _lat_days = _mgr.bottleneck_avg_days * _mgr.bottleneck_rework_rate

            _friction_usd  = math_eng.compute_friction_tax(
                decision_latency_days=_lat_days,
                cost_per_hour=_cost_per_hour,
                hours_per_day=_hours_per_day,
            )

            # ── Bayesian Posterior computation ────────────────────────────────
            _conf = math_eng.compute_process_confidence(
                prior_pct=float(_prior_pct),
                bottleneck_rework_rate=_mgr.bottleneck_rework_rate,
            )
            _posterior_pct = _conf["posterior_pct"]
            _delta_pct     = _conf["delta_pct"]
            _lr            = _conf["likelihood_ratio"]

            # ── Persist metrics in session state for "Save client" button ─────
            st.session_state["_saved_friction_tax"]       = _friction_usd
            st.session_state["_saved_confidence"]         = _posterior_pct
            st.session_state["_saved_bottleneck"]         = _bt_node
            st.session_state["_saved_rework_rate"]        = _mgr.bottleneck_rework_rate * 100.0
            if _mkv_graph_res is not None:
                st.session_state["_saved_total_transitions"] = int(_mkv_graph_res.total_transitions)
                st.session_state["_saved_total_rework"]      = int(_mkv_graph_res.total_rework_transitions)

            # ── C-Level metric cards ──────────────────────────────────────────
            _lbl_bt  = {"en": "Bottleneck Stage",
                        "ru": "Узкое место",
                        "sr": "Usko grlo"}.get(lang, "Bottleneck Stage")
            _lbl_ft  = {"en": "Friction Tax / Entity",
                        "ru": "Налог на трение / ед.",
                        "sr": "Porez trenja / entitet"}.get(lang, "Friction Tax")
            _lbl_ac  = {"en": "Adjusted Confidence",
                        "ru": "Скорр. уверенность",
                        "sr": "Korigovana pouzdanost"}.get(lang, "Adjusted Confidence")

            _delta_bt = {
                "en": f"Rework rate {_mgr.bottleneck_rework_rate:.0%}  |  Score {_mgr.bottleneck_score:.3f}",
                "ru": f"Доработок {_mgr.bottleneck_rework_rate:.0%}  |  Оценка {_mgr.bottleneck_score:.3f}",
                "sr": f"Povrat {_mgr.bottleneck_rework_rate:.0%}  |  Ocena {_mgr.bottleneck_score:.3f}",
            }.get(lang, "")

            _delta_ft = {
                "en": f"{_lat_days:.1f} latency days  ×  ${_cost_per_hour:.0f}/h  ×  {_hours_per_day:.0f} h/d",
                "ru": f"{_lat_days:.1f} дн. задержки  ×  ${_cost_per_hour:.0f}/ч  ×  {_hours_per_day:.0f} ч/д",
                "sr": f"{_lat_days:.1f} dana kašnjenja  ×  ${_cost_per_hour:.0f}/h  ×  {_hours_per_day:.0f} h/d",
            }.get(lang, "")

            _delta_ac_sign = f"{_delta_pct:+.1f}%" if _delta_pct != 0 else "—"
            _delta_ac = {
                "en": f"Prior {_prior_pct}%  →  LR {_lr:.2f}  →  {_delta_ac_sign}",
                "ru": f"Апр. {_prior_pct}%  →  LR {_lr:.2f}  →  {_delta_ac_sign}",
                "sr": f"Apriori {_prior_pct}%  →  LR {_lr:.2f}  →  {_delta_ac_sign}",
            }.get(lang, "")

            _col_bt, _col_ft, _col_ac = st.columns(3)
            with _col_bt:
                st.metric(
                    label=_lbl_bt,
                    value=_bt_node,
                    delta=_delta_bt,
                    delta_color="off",
                )
            with _col_ft:
                st.metric(
                    label=_lbl_ft,
                    value=f"${_friction_usd:,.0f}",
                    delta=_delta_ft,
                    delta_color="inverse" if _friction_usd > 0 else "off",
                )
            with _col_ac:
                st.metric(
                    label=_lbl_ac,
                    value=f"{_posterior_pct:.1f}%",
                    delta=_delta_ac,
                    delta_color="normal" if _delta_pct >= 0 else "inverse",
                )

            st.markdown("<hr style='border:none;border-top:1px solid #E5E5EA;margin:18px 0 22px;'>",
                        unsafe_allow_html=True)

            # ── Methodology note (expandable) ────────────────────────────────
            _meth_label = {"en": "Methodology", "ru": "Методология",
                           "sr": "Metodologija"}.get(lang, "Methodology")
            with st.expander(_meth_label, expanded=False):
                _meth_txt = {
                    "en": (
                        f"**Friction Tax** = Decision Latency × Cost/h × h/day\n\n"
                        f"Decision Latency = average days spent in the bottleneck stage "
                        f"(**{_bt_node}**) before a rework loop is triggered. "
                        f"Computed as: `{_lat_days:.2f} d × ${_cost_per_hour:.0f}/h "
                        f"× {_hours_per_day:.0f} h = ${_friction_usd:,.0f}`\n\n"
                        f"**Adjusted Confidence** uses Bayesian odds-form update:\n\n"
                        f"  1. Evidence = observed rework rate at bottleneck "
                        f"({_mgr.bottleneck_rework_rate:.1%})\n"
                        f"  2. P(evidence | success) = {1-_mgr.bottleneck_rework_rate:.2f} "
                        f"(low rework in healthy process)\n"
                        f"  3. P(evidence | failure) = {_mgr.bottleneck_rework_rate:.2f} "
                        f"(high rework in broken process)\n"
                        f"  4. Likelihood Ratio = {_lr:.4f}\n"
                        f"  5. Posterior odds = Prior odds × LR → {_posterior_pct:.1f}%"
                    ),
                    "ru": (
                        f"**Налог на трение** = Задержка × Стоимость/ч × Часов/день\n\n"
                        f"Задержка решения = среднее число дней в стадии (**{_bt_node}**) "
                        f"перед запуском петли доработки. "
                        f"Расчёт: `{_lat_days:.2f} дн × ${_cost_per_hour:.0f}/ч "
                        f"× {_hours_per_day:.0f} ч = ${_friction_usd:,.0f}`\n\n"
                        f"**Скорр. уверенность** — байесовское обновление (форма шансов):\n\n"
                        f"  1. Свидетельство = ставка доработки на узком месте "
                        f"({_mgr.bottleneck_rework_rate:.1%})\n"
                        f"  2. P(св | успех) = {1-_mgr.bottleneck_rework_rate:.2f}\n"
                        f"  3. P(св | провал) = {_mgr.bottleneck_rework_rate:.2f}\n"
                        f"  4. Коэф. правдоподобия (LR) = {_lr:.4f}\n"
                        f"  5. Апостериорные шансы = априорные × LR → {_posterior_pct:.1f}%"
                    ),
                    "sr": (
                        f"**Porez trenja** = Kašnjenje × Cena/h × h/dan\n\n"
                        f"Kašnjenje odluke = prosečni dani u fazi (**{_bt_node}**) "
                        f"pre povratne petlje. "
                        f"Izračun: `{_lat_days:.2f} d × ${_cost_per_hour:.0f}/h "
                        f"× {_hours_per_day:.0f} h = ${_friction_usd:,.0f}`\n\n"
                        f"**Korigovana pouzdanost** — bayesovsko ažuriranje (u obliku šansi):\n\n"
                        f"  1. Dokaz = stopa povrata na uskom grlu "
                        f"({_mgr.bottleneck_rework_rate:.1%})\n"
                        f"  2. P(dokaz | uspeh) = {1-_mgr.bottleneck_rework_rate:.2f}\n"
                        f"  3. P(dokaz | neuspeh) = {_mgr.bottleneck_rework_rate:.2f}\n"
                        f"  4. Koeficijent verodostojnosti (LR) = {_lr:.4f}\n"
                        f"  5. Posteriorne šanse = apriorne × LR → {_posterior_pct:.1f}%"
                    ),
                }.get(lang, "")
                st.markdown(_meth_txt)

            # ── Spring layout (deterministic seed)
            try:
                import networkx as nx_local
                _pos = nx_local.spring_layout(_G, k=2.2, seed=42, iterations=80)
            except Exception:
                _n_list = list(_G.nodes)
                _ang = 2 * np.pi / max(len(_n_list), 1)
                _pos = {n: (np.cos(i * _ang), np.sin(i * _ang)) for i, n in enumerate(_n_list)}

            # ── Node classification ───────────────────────────────────────────
            _bt_node   = _mgr.bottleneck_node
            _rw_thresh = 0.30   # ≥30 % rework outgoing → orange warning node
            _node_color_map, _node_size_map = {}, {}
            _max_traffic = max(_mgr.stage_total.values()) if _mgr.stage_total else 1
            for _nd in _G.nodes:
                _traffic = _mgr.stage_total.get(_nd, 0)
                _base_sz = 18 + 28 * (_traffic / _max_traffic)
                _rr      = _mgr.rework_rate.get(_nd, 0.0)
                if _nd == _bt_node:
                    _node_color_map[_nd] = "#FF3B30"   # Apple red — bottleneck
                    _node_size_map[_nd]  = max(_base_sz + 8, 42)
                elif _rr >= _rw_thresh:
                    _node_color_map[_nd] = "#FF9F0A"   # Apple orange — high rework
                    _node_size_map[_nd]  = _base_sz + 4
                else:
                    _node_color_map[_nd] = "#0071E3"   # Apple blue — normal
                    _node_size_map[_nd]  = _base_sz

            # ── Edge traces (normal vs rework) ────────────────────────────────
            _norm_ex, _norm_ey = [], []
            _rw_ex,   _rw_ey   = [], []
            _ann_list = []      # arrowhead annotations

            _max_cnt = max(_mgr.transition_counts.values()) if _mgr.transition_counts else 1

            for _frm, _to, _edata in _G.edges(data=True):
                if _frm not in _pos or _to not in _pos:
                    continue
                _x0, _y0 = _pos[_frm]
                _x1, _y1 = _pos[_to]

                # Offset for bidirectional edges to avoid overlap
                _is_bidi = _G.has_edge(_to, _frm)
                if _is_bidi:
                    _dx, _dy = (_y1 - _y0) * 0.08, -(_x1 - _x0) * 0.08
                    _x0b, _y0b = _x0 + _dx, _y0 + _dy
                    _x1b, _y1b = _x1 + _dx, _y1 + _dy
                else:
                    _x0b, _y0b, _x1b, _y1b = _x0, _y0, _x1, _y1

                _prob  = _edata.get("weight", 0.0)
                _cnt   = _edata.get("count", 0)
                _is_rw = _edata.get("is_rework", False)
                _lw    = 1.2 + 3.5 * (_cnt / _max_cnt)

                if _is_rw:
                    _rw_ex  += [_x0b, _x1b, None]
                    _rw_ey  += [_y0b, _y1b, None]
                else:
                    _norm_ex += [_x0b, _x1b, None]
                    _norm_ey += [_y0b, _y1b, None]

                # Arrow annotation pointing to target node
                _ann_list.append(dict(
                    ax=_x0b, ay=_y0b, x=_x1b, y=_y1b,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2, arrowsize=1.1,
                    arrowwidth=_lw,
                    arrowcolor="#FF3B30" if _is_rw else "rgba(0,113,227,0.55)",
                    text=f"{_prob:.0%}", font=dict(size=9, color="#6E6E73"),
                ))

            # ── Plotly figure ─────────────────────────────────────────────────
            _fig_mk = go.Figure()

            # Normal edges
            if _norm_ex:
                _fig_mk.add_trace(go.Scatter(
                    x=_norm_ex, y=_norm_ey, mode="lines",
                    line=dict(color="rgba(0,113,227,0.28)", width=1.5),
                    hoverinfo="none", name="Forward",
                ))

            # Rework edges
            if _rw_ex:
                _fig_mk.add_trace(go.Scatter(
                    x=_rw_ex, y=_rw_ey, mode="lines",
                    line=dict(color="rgba(255,59,48,0.45)", width=2.2, dash="dot"),
                    hoverinfo="none", name="Rework loop",
                ))

            # Nodes
            _all_nodes = list(_G.nodes)
            _nx_arr  = [_pos[n][0] for n in _all_nodes]
            _ny_arr  = [_pos[n][1] for n in _all_nodes]
            _nc_arr  = [_node_color_map[n] for n in _all_nodes]
            _ns_arr  = [_node_size_map[n]  for n in _all_nodes]
            _bt_lbl  = {"en": "BOTTLENECK", "ru": "УЗКОЕ МЕСТО", "sr": "USKO GRLO"}.get(lang, "BOTTLENECK")
            _rw_lbl  = {"en": "high rework", "ru": "высокий rework", "sr": "visok rework"}.get(lang, "high rework")
            _hover_lines = []
            for _nd in _all_nodes:
                _rr  = _mgr.rework_rate.get(_nd, 0.0)
                _bt  = _mgr.betweenness.get(_nd, 0.0)
                _lat = _mgr.decision_latency.get(_nd, 0.0)
                _tr  = _mgr.stage_total.get(_nd, 0)
                _role = (f"<b>⚠ {_bt_lbl}</b>" if _nd == _bt_node
                         else (f"⚡ {_rw_lbl.capitalize()}" if _rr >= _rw_thresh else ""))
                _hover_lines.append(
                    f"<b>{_nd}</b>{(' — ' + _role) if _role else ''}<br>"
                    f"Transitions: {_tr}<br>"
                    f"Rework rate: {_rr:.0%}<br>"
                    f"Betweenness: {_bt:.4f}<br>"
                    f"Decision latency: {_lat:.1f} d"
                )

            _fig_mk.add_trace(go.Scatter(
                x=_nx_arr, y=_ny_arr,
                mode="markers+text",
                text=_all_nodes,
                textposition="top center",
                textfont=dict(family="Inter, sans-serif", size=11, color="#1D1D1F"),
                marker=dict(
                    color=_nc_arr, size=_ns_arr,
                    line=dict(color="rgba(255,255,255,0.9)", width=2.5),
                    symbol="circle",
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=_hover_lines,
                name="Stages",
            ))

            _no_axes = {k: v for k, v in CHART_LAYOUT.items()
                        if k not in ("xaxis", "yaxis", "legend")}
            _fig_mk.update_layout(
                annotations=_ann_list,
                showlegend=True,
                legend=dict(orientation="h", y=-0.07, x=0.5, xanchor="center",
                            font=dict(size=11, color="#6E6E73")),
                height=520,
                xaxis=dict(visible=False, range=[-1.6, 1.6]),
                yaxis=dict(visible=False, range=[-1.6, 1.6]),
                **_no_axes,
            )
            st.plotly_chart(_fig_mk, width="stretch", key="markov_graph_plotly")

            # ── Legend chips ──────────────────────────────────────────────────
            _lc1, _lc2, _lc3 = st.columns(3)
            with _lc1:
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:8px;">'
                    '<div style="width:14px;height:14px;border-radius:50%;background:#FF3B30;flex-shrink:0;"></div>'
                    f'<span style="font-size:12px;color:#6E6E73;">{_bt_lbl} — {_bt_node}</span></div>',
                    unsafe_allow_html=True,
                )
            with _lc2:
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:8px;">'
                    '<div style="width:14px;height:14px;border-radius:50%;background:#FF9F0A;flex-shrink:0;"></div>'
                    f'<span style="font-size:12px;color:#6E6E73;">{_rw_lbl.capitalize()} (&ge;30%)</span></div>',
                    unsafe_allow_html=True,
                )
            with _lc3:
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:8px;">'
                    '<div style="width:14px;height:14px;border-radius:2px;background:rgba(255,59,48,0.45);flex-shrink:0;"></div>'
                    f'<span style="font-size:12px;color:#6E6E73;">Rework edge (dotted)</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Bottleneck alert ──────────────────────────────────────────────
            _bt_days_label = {"en": "avg days in stage",
                              "ru": "ср. дней в стадии",
                              "sr": "prosek dana u fazi"}.get(lang, "avg days in stage")
            _rw_label = {"en": "rework rate",
                         "ru": "процент доработки",
                         "sr": "stopa dorade"}.get(lang, "rework rate")

            _bt_days = _mgr.bottleneck_avg_days
            _bt_rr   = _mgr.bottleneck_rework_rate

            if _bt_rr >= 0.3 or _bt_days >= 5:
                _msg_bt = {
                    "en": (f"**Bottleneck detected:** stage **{_bt_node}** "
                           f"has a {_bt_rr:.0%} {_rw_label} "
                           f"and tasks spend on average **{_bt_days:.1f} days** here "
                           f"before looping back."),
                    "ru": (f"**Узкое место обнаружено:** стадия **{_bt_node}** "
                           f"имеет {_bt_rr:.0%} доработок; "
                           f"задачи проводят здесь в среднем **{_bt_days:.1f} дн.** "
                           f"перед возвратом."),
                    "sr": (f"**Usko grlo detektovano:** faza **{_bt_node}** "
                           f"ima {_bt_rr:.0%} povratnih prelaza; "
                           f"zadaci provode prosečno **{_bt_days:.1f} dana** ovde "
                           f"pre povratka."),
                }.get(lang, f"Bottleneck: {_bt_node} | rework {_bt_rr:.0%} | {_bt_days:.1f} days")
                st.error(_msg_bt)
            else:
                _msg_bt_ok = {
                    "en": (f"**Primary focus stage: {_bt_node}** — "
                           f"{_bt_rr:.0%} {_rw_label}, "
                           f"{_bt_days:.1f} {_bt_days_label}. "
                           f"No critical bottleneck detected."),
                    "ru": (f"**Приоритетная стадия: {_bt_node}** — "
                           f"{_bt_rr:.0%} доработок, "
                           f"{_bt_days:.1f} {_bt_days_label}. "
                           f"Критических узких мест не обнаружено."),
                    "sr": (f"**Prioritetna faza: {_bt_node}** — "
                           f"{_bt_rr:.0%} {_rw_label}, "
                           f"{_bt_days:.1f} {_bt_days_label}. "
                           f"Nije detektovano kritično usko grlo."),
                }.get(lang, f"Focus: {_bt_node} | {_bt_rr:.0%} rework | {_bt_days:.1f} days")
                st.warning(_msg_bt_ok)

            # ── Rework pairs table ────────────────────────────────────────────
            if _mgr.rework_pairs:
                _rw_hdr = {"en": "Detected Rework Loops",
                           "ru": "Обнаруженные петли доработки",
                           "sr": "Detektovane petlje dorade"}.get(lang, "Rework Loops")
                st.subheader(_rw_hdr)

                _col_a  = {"en": "Stage A", "ru": "Стадия A", "sr": "Faza A"}.get(lang, "Stage A")
                _col_b  = {"en": "Stage B", "ru": "Стадия B", "sr": "Faza B"}.get(lang, "Stage B")
                _col_p1 = {"en": "P(A→B)", "ru": "P(A→B)", "sr": "P(A→B)"}[lang]
                _col_p2 = {"en": "P(B→A rework)", "ru": "P(B→A доработка)", "sr": "P(B→A povratak)"}[lang]
                _col_d  = {"en": "Decision Latency (days)", "ru": "Задержка решения (дн.)", "sr": "Kašnjenje odluke (dana)"}.get(lang, "Decision Latency (d)")
                _col_n  = {"en": "# Rework Events", "ru": "Кол-во возвратов", "sr": "Br. povrata"}.get(lang, "# Events")

                _rw_rows = [
                    {
                        _col_a: rp.stage_a,
                        _col_b: rp.stage_b,
                        _col_p1: f"{rp.prob_a_to_b:.1%}",
                        _col_p2: f"{rp.prob_b_to_a:.1%}",
                        _col_d: rp.avg_days_in_b,
                        _col_n: rp.count_rework,
                    }
                    for rp in _mgr.rework_pairs
                ]
                _df_rw = pd.DataFrame(_rw_rows)
                st.dataframe(
                    _df_rw.style.background_gradient(subset=[_col_d], cmap="Reds"),
                    height=min(60 + 38 * len(_rw_rows), 340),
                )

                _total_rw_label = {
                    "en": f"Total rework transitions in dataset: **{_mgr.total_rework_transitions}** of {_mgr.total_transitions} ({_mgr.total_rework_transitions / max(_mgr.total_transitions, 1):.1%})",
                    "ru": f"Всего переходов-доработок в данных: **{_mgr.total_rework_transitions}** из {_mgr.total_transitions} ({_mgr.total_rework_transitions / max(_mgr.total_transitions, 1):.1%})",
                    "sr": f"Ukupno povratnih prelaza: **{_mgr.total_rework_transitions}** od {_mgr.total_transitions} ({_mgr.total_rework_transitions / max(_mgr.total_transitions, 1):.1%})",
                }.get(lang, "")
                st.caption(_total_rw_label)

            # ── Transition probability matrix (expandable) ────────────────────
            _tp_hdr = {"en": "Transition Probability Matrix",
                       "ru": "Матрица переходных вероятностей",
                       "sr": "Matrica verovatnoća prelaza"}.get(lang, "Transition Matrix")
            with st.expander(_tp_hdr, expanded=False):
                _all_stages_tp = set(_G.nodes) if _G else set()
                _stages_sorted = sorted(_all_stages_tp)
                _tp_data = {}
                for _s in _stages_sorted:
                    _row = {}
                    for _t in _stages_sorted:
                        _p = _mgr.transition_probs.get((_s, _t), 0.0)
                        _row[_t] = _p
                    _tp_data[_s] = _row
                _df_tp = pd.DataFrame(_tp_data).T.fillna(0.0)
                if not _df_tp.empty:
                    st.dataframe(
                        _df_tp.style.background_gradient(cmap="Blues").format("{:.2%}"),
                        height=min(80 + 38 * len(_stages_sorted), 400),
                    )

    # ── TAB 3: MARKOV ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader(t(lang, "markov_title"))
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(t(lang, "p_before_metric"), "{:.0f}%".format(p_before))
        mc2.metric(t(lang, "p_after_metric"),  "{:.0f}%".format(p_after))
        if markov_res:
            mc3.metric(t(lang, "expected_time"),
                       "{:.1f} {}".format(markov_res.expected_lead_time_hours, t(lang, "hours")))
        st.markdown(t(lang, "matrix_q"))
        df_Q = pd.DataFrame(Q_mat, index=m_states, columns=m_states)
        st.dataframe(df_Q.style.format("{:.4f}"))
        if N_mat is not None:
            st.markdown(t(lang, "matrix_n"))
            df_N = pd.DataFrame(N_mat, index=m_states, columns=m_states)
            st.dataframe(df_N.style.format("{:.4f}"))
        # ── Funnel chart: Pipeline BEFORE vs AFTER ────────────────────────────
        st.markdown("---")
        st.subheader(t(lang, "funnel_title"))
        _fn1, _fn2 = st.columns(2)
        _fn_stages = m_states + (
            ["Won"] if lang == "en" else (["Выиграна"] if lang == "ru" else ["Zatvoren"])
        )
        _fn_n = len(_fn_stages)
        _fn_before = [round(deals * (p_before / 100.0) ** (i / max(_fn_n - 1, 1)), 1)
                      for i in range(_fn_n)]
        _fn_after  = [round(deals * (p_after  / 100.0) ** (i / max(_fn_n - 1, 1)), 1)
                      for i in range(_fn_n)]
        with _fn1:
            fig_fn_b = go.Figure(go.Funnel(
                y=_fn_stages, x=_fn_before, textinfo="value+percent previous",
                marker=dict(color=[_C["navy"], "#5AC8FA", _C["blue"], "#34C759"][:_fn_n]),
                connector=dict(line=dict(color="rgba(26,50,113,0.10)", width=1)),
                name=t(lang, "funnel_before"),
            ))
            fig_fn_b.update_layout(
                title=dict(text=t(lang, "funnel_before"), font=dict(size=13, color="#6E6E73")),
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"),
                showlegend=False,
            )
            st.plotly_chart(fig_fn_b, width="stretch")
        with _fn2:
            fig_fn_a = go.Figure(go.Funnel(
                y=_fn_stages, x=_fn_after, textinfo="value+percent previous",
                marker=dict(color=["#34C759", "#5AC8FA", _C["blue"], _C["navy"]][:_fn_n]),
                connector=dict(line=dict(color="rgba(52,199,89,0.10)", width=1)),
                name=t(lang, "funnel_after"),
            ))
            fig_fn_a.update_layout(
                title=dict(text=t(lang, "funnel_after"), font=dict(size=13, color="#34C759")),
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"),
                showlegend=False,
            )
            st.plotly_chart(fig_fn_a, width="stretch")

        st.markdown("---")
        st.markdown(t(lang, "timeline_title"))
        _tl_rate     = _CURRENCIES[currency]["rate"]
        _tl_label    = _CURRENCIES[currency]["label"]
        months_range  = list(range(0, 13))
        monthly_gain  = res.total_benefit / 12
        cumulative    = [(monthly_gain * m - impl_cost) * _tl_rate for m in months_range]
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(x=months_range, y=cumulative, mode="lines+markers",
                                    name=t(lang, "cumulative_roi"),
                                    line=dict(color=_C["navy"], width=2.5),
                                    marker=dict(size=5, color=_C["navy"], line=dict(color="#ffffff", width=2)),
                                    fill="tozeroy", fillcolor="rgba(26,50,113,0.07)"))
        fig_tl.add_trace(go.Scatter(x=months_range, y=[0]*13, mode="lines",
                                    name=t(lang, "breakeven"),
                                    line=dict(color=_C["gold"], width=1.5, dash="dot")))
        fig_tl.update_layout(xaxis_title=t(lang, "month_label"), yaxis_title=_tl_label, height=400, **CHART_LAYOUT)
        st.plotly_chart(fig_tl, width="stretch")

    # ── TAB 4: BAYES ──────────────────────────────────────────────────────────
    with tab4:
        st.subheader(t(lang, "bayes_title"))
        b1, b2 = st.columns(2)
        with b1:
            pos_signals = st.slider(t(lang, "positive_signals"), 1, 50, key="pos_signals")
        with b2:
            tot_signals = st.slider(t(lang, "total_signals"), 2, 100, key="tot_signals")

        # Prior slider — seeded from CSV completion rate when data is uploaded,
        # otherwise defaults to 34 (conservative demo prior).
        # session_state["bayes_prior_pct"] is always pre-populated above
        # (p_before from CSV, or 34 via setdefault), so value= is not needed.
        _prior_label_b = {"en": "Prior belief (%)", "ru": "Априорная уверенность (%)",
                          "sr": "Apriorna verovatnoća (%)"}.get(lang, "Prior belief (%)")
        _prior_help_b  = {"en": "Starting probability before observing signals. Auto-set from CSV when data is loaded.",
                          "ru": "Начальная вероятность до наблюдения сигналов. Авто-заполняется из CSV.",
                          "sr": "Početna verovatnoća pre posmatranja signala. Automatski iz CSV-a."
                         }.get(lang, "")
        bayes_prior_pct = st.slider(
            _prior_label_b, min_value=1, max_value=99,
            step=1, format="%d%%",
            help=_prior_help_b, key="bayes_prior_pct",
        )
        _bpr = bayes_prior_pct / 100.0  # float prior rate for calculations

        bayes_live = math_eng.bayesian_update(pos_signals, tot_signals, prior_rate=_bpr)
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric(t(lang, "prior"),     "{:.1f}%".format(bayes_live.prior_pct))
        bc2.metric(t(lang, "posterior"), "{:.1f}%".format(bayes_live.posterior_pct))
        bc3.metric(t(lang, "ci_80"), "{}% – {}%".format(bayes_live.ci_80_low, bayes_live.ci_80_high))
        prior_a  = _bpr * 10; prior_b  = (1 - _bpr) * 10
        post_a   = prior_a + pos_signals; post_b   = prior_b + (tot_signals - pos_signals)
        x        = np.linspace(0.01, 0.99, 300)
        y_prior  = stats.beta.pdf(x, prior_a, prior_b)
        y_post   = stats.beta.pdf(x, post_a,  post_b)
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=x*100, y=y_prior, mode="lines", name=t(lang, "prior_label"),
                                   line=dict(color=_C["gold"], width=2),
                                   fill="tozeroy", fillcolor="rgba(192,160,98,0.10)"))
        fig_b.add_trace(go.Scatter(x=x*100, y=y_post, mode="lines", name=t(lang, "posterior_label"),
                                   line=dict(color=_C["navy"], width=2.5),
                                   fill="tozeroy", fillcolor="rgba(26,50,113,0.10)"))
        fig_b.update_layout(xaxis_title=t(lang, "probability_pct"), yaxis_title=t(lang, "density"),
                            height=400, **CHART_LAYOUT)
        st.plotly_chart(fig_b, width="stretch")
        risk = math_eng.bayesian_contextual_risk(0.05, 0.80, 0.20)
        st.info(t(lang, "contextual_risk", risk=risk))

    # ── TAB 5: PASSPORT ───────────────────────────────────────────────────────
    with tab5:
        st.subheader(t(lang, "passport_title"))
        _pass_sym  = _CURRENCIES[currency]["sym"]
        _pass_rate = _CURRENCIES[currency]["rate"]
        passport = roi_eng.passport_text(inp, res,
                                         currency_sym=_pass_sym,
                                         currency_rate=_pass_rate,
                                         lang=lang,
                                         auditor_name=auditor_name)
        st.code(passport, language="")

        # Meeting notes (4)
        st.session_state.setdefault("meeting_notes", "")
        meeting_notes = st.text_area(
            t(lang, "notes_label"),
            key="meeting_notes",
            help=t(lang, "notes_help"),
            height=100,
            placeholder={"en": "Key points from the meeting...",
                         "ru": "Ключевые моменты встречи...",
                         "sr": "Ključne tačke sastanka..."}.get(lang, ""),
        )

        if is_demo:
            _lock_msg = {
                "en": ("Download TXT — available after login",
                       "Download PDF — available after login",
                       "Sign in to export your ROI Passport"),
                "ru": ("Скачать TXT — доступно после входа",
                       "Скачать PDF — доступно после входа",
                       "Войдите, чтобы экспортировать ROI-паспорт"),
                "sr": ("Preuzmi TXT — dostupno nakon prijave",
                       "Preuzmi PDF — dostupno nakon prijave",
                       "Prijavite se da biste izvezli ROI pasoš"),
            }
            dl1, dl2 = st.columns(2)
            with dl1:
                st.button(_lock_msg[lang][0], key="dl_txt_lock", disabled=True, use_container_width=True)
            with dl2:
                st.button(_lock_msg[lang][1], key="dl_pdf_lock", disabled=True, use_container_width=True)
            st.caption(_lock_msg[lang][2])
        else:
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label=t(lang, "download_txt"),
                    data=passport.encode("utf-8"),
                    file_name="roi_passport_{}.txt".format(company_name.replace(" ", "_")),
                    mime="text/plain", key="dl_txt",
                )
            with dl2:
                try:
                    pdf_bytes = build_roi_passport_pdf(
                        company_name=company_name,
                        auditor_name=auditor_name,
                        contact_url=contact_url,
                        meeting_notes=meeting_notes,
                        time_saved=res.time_saved_annual,
                        error_reduction=res.error_reduction_annual,
                        revenue_impact=res.revenue_impact_annual,
                        markov_gain=res.markov_gain_annual,
                        implementation_cost=float(impl_cost),
                        manual_hours_before=float(manual_hours),
                        automation_rate_pct=float(automation_rate),
                        error_rate_before=float(error_before),
                        error_rate_after=float(error_after),
                        deal_cycle_before=float(cycle_before),
                        deal_cycle_after=float(cycle_after),
                        p_complete_before_pct=float(p_before),
                        p_complete_after_pct=float(p_after),
                        bayes_prior=bayes_res.prior_pct,
                        bayes_posterior=bayes_res.posterior_pct,
                        bayes_ci="{}%-{}%".format(bayes_res.ci_80_low, bayes_res.ci_80_high),
                        bottleneck_node=(_mkv_graph_res.bottleneck_node
                                         if _mkv_graph_res is not None
                                         else graph_res.bottleneck_node),
                        bottleneck_score=(_mkv_graph_res.bottleneck_score
                                          if _mkv_graph_res is not None
                                          else graph_res.bottleneck_score),
                        net_roi=res.net_roi,
                        roi_pct=res.roi_pct,
                        payback_months=res.payback_months,
                        currency_sym=_pass_sym,
                        currency_rate=_pass_rate,
                        lang=lang,
                    )
                except Exception as _pdf_err:
                    st.error("PDF error: {}".format(_pdf_err))
                    import traceback; st.code(traceback.format_exc())
                    pdf_bytes = b""
                st.download_button(
                    label=t(lang, "download_pdf"),
                    data=pdf_bytes,
                    file_name="roi_passport_{}.pdf".format(company_name.replace(" ", "_")),
                    mime="application/pdf", key="dl_pdf",
                )

        linkedin_text = t(lang, "linkedin_text",
                          company=company_name,
                          roi_str=_fmt(res.net_roi, currency),
                          roi_pct=res.roi_pct,
                          payback=res.payback_months)
        st.text_area(t(lang, "linkedin_label"), value=linkedin_text, height=110, key="linkedin_ta")

    # ── TAB 6: PRECISION ANALYSIS ─────────────────────────────────────────────
    with tab6:
        _prec_sym   = _CURRENCIES[currency]["sym"]
        _prec_label = _CURRENCIES[currency]["label"]
        _prec_rate  = _CURRENCIES[currency]["rate"]

        # ── TORNADO CHART ────────────────────────────────────────────────────
        st.subheader(t(lang, "tornado_title"))
        _tornado = run_tornado(inp, delta=0.20)

        _param_label_map = {
            "en": {
                "manual_hours": "Manual hours/mo",
                "automation_rate": "Automation %",
                "hour_rate": "Hour rate €",
                "cost_per_error": "Cost per error €",
                "deal_value": "Avg deal value €",
                "deals_per_month": "Deals/month",
                "cycle_improvement": "Cycle improvement",
                "p_uplift": "Completion prob. uplift",
            },
            "ru": {
                "manual_hours": "Ручные часы/мес",
                "automation_rate": "Автоматизация %",
                "hour_rate": "Ставка €/ч",
                "cost_per_error": "Стоимость ошибки €",
                "deal_value": "Ср. сделка €",
                "deals_per_month": "Сделок/мес",
                "cycle_improvement": "Улучшение цикла",
                "p_uplift": "Прирост вероятности",
            },
            "sr": {
                "manual_hours": "Ručni sati/mes.",
                "automation_rate": "Automatizacija %",
                "hour_rate": "Stopa €/h",
                "cost_per_error": "Trošak greške €",
                "deal_value": "Prosečan posao €",
                "deals_per_month": "Poslova/mes.",
                "cycle_improvement": "Poboljšanje ciklusa",
                "p_uplift": "Porast verovatnoće",
            },
        }
        _pm = _param_label_map.get(lang, _param_label_map["en"])
        _t_labels = [_pm.get(p, p) for p in _tornado.params]

        _t_fig = go.Figure()
        _t_fig.add_trace(go.Bar(
            y=_t_labels,
            x=[l - _tornado.base_roi for l in _tornado.roi_low],
            orientation="h",
            name="-20%",
            marker_color="#FF3B30",
            base=_tornado.base_roi,
        ))
        _t_fig.add_trace(go.Bar(
            y=_t_labels,
            x=[h - _tornado.base_roi for h in _tornado.roi_high],
            orientation="h",
            name="+20%",
            marker_color="#34C759",
            base=_tornado.base_roi,
        ))
        _t_fig.add_vline(
            x=_tornado.base_roi,
            line_dash="dot", line_color="#1D1D1F", line_width=1.5,
            annotation_text="base", annotation_position="top",
        )
        _tornado_x_labels = {
            "en": "ROI Impact ({})".format(_prec_label),
            "ru": "Влияние на ROI ({})".format(_prec_label),
            "sr": "Uticaj na ROI ({})".format(_prec_label),
        }
        _t_fig.update_layout(
            barmode="overlay",
            xaxis_title=_tornado_x_labels.get(lang, _tornado_x_labels["en"]),
            yaxis_title=t(lang, "tornado_param"),
            height=380,
            **CHART_LAYOUT,
        )
        st.plotly_chart(_t_fig, width="stretch")

        st.markdown("---")

        # ── MONTE CARLO ──────────────────────────────────────────────────────
        st.subheader(t(lang, "mc_title"))
        with st.spinner("Running simulation…"):
            _mc = _cached_mc(
                manual_hours, automation_rate, hour_rate,
                error_before, error_after, cost_per_error, volume,
                cycle_before, cycle_after, deals, deal_value,
                p_before, p_after, impl_cost, pipeline_util,
            )

        _mc_c1, _mc_c2, _mc_c3 = st.columns(3)
        _mc_c1.metric(t(lang, "mc_p_positive"),  "{:.1f}%".format(_mc.p_positive  * 100))
        _mc_c2.metric(t(lang, "mc_p_payback18"), "{:.1f}%".format(_mc.p_payback_18 * 100))
        _mc_c3.metric(t(lang, "mc_p_payback12"), "{:.1f}%".format(_mc.p_payback_12 * 100))

        _mc_c4, _mc_c5 = st.columns(2)
        _mc_c4.metric(t(lang, "mc_median"), _fmt(_mc.pct50, currency))
        _mc_c5.metric(
            t(lang, "mc_range"),
            "{} – {}".format(_fmt(_mc.pct10, currency), _fmt(_mc.pct90, currency)),
        )

        _mc_samples_conv = [v * _prec_rate for v in _mc.roi_samples]
        _fig_mc = go.Figure(go.Histogram(
            x=_mc_samples_conv,
            nbinsx=60,
            marker_color="#0071E3",
            opacity=0.75,
            name="{:,} {}".format(_mc.n_simulations, t(lang, "mc_runs")),
        ))
        _fig_mc.add_vline(x=0, line_dash="dash", line_color="#FF3B30", line_width=2,
                          annotation_text="ROI=0", annotation_position="top right")
        _fig_mc.add_vline(x=_mc.pct10 * _prec_rate, line_dash="dot", line_color="#AEAEB2", line_width=1)
        _fig_mc.add_vline(x=_mc.pct50 * _prec_rate, line_dash="solid", line_color="#34C759", line_width=2,
                          annotation_text="p50", annotation_position="top")
        _fig_mc.add_vline(x=_mc.pct90 * _prec_rate, line_dash="dot", line_color="#AEAEB2", line_width=1)
        _mc_x_labels = {
            "en": "Net ROI ({})".format(_prec_label),
            "ru": "Чистый ROI ({})".format(_prec_label),
            "sr": "Neto ROI ({})".format(_prec_label),
        }
        _fig_mc.update_layout(
            xaxis_title=_mc_x_labels.get(lang, _mc_x_labels["en"]),
            yaxis_title=t(lang, "mc_hist_y"),
            height=340,
            **CHART_LAYOUT,
        )
        st.plotly_chart(_fig_mc, width="stretch")

        # ── Scatter: Risk / Return Cloud ──────────────────────────────────────
        st.subheader(t(lang, "scatter_title"))
        _sc_rate   = _prec_rate
        _sc_median = _mc.pct50
        _sc_step   = max(len(_mc.roi_samples) // 800, 1)
        _sc_sample = _mc.roi_samples[::_sc_step]
        _sc_x_raw  = [((v - _sc_median) / max(abs(_sc_median), 1)) * 100 for v in _sc_sample]
        _sc_y_raw  = [v * _sc_rate for v in _sc_sample]
        _sc_colors = ["#34C759" if v >= 0 else "#FF3B30" for v in _sc_sample]
        _sc_labels = [t(lang, "scatter_positive") if v >= 0 else t(lang, "scatter_negative")
                      for v in _sc_sample]
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=[x for x, v in zip(_sc_x_raw, _sc_sample) if v >= 0],
            y=[y for y, v in zip(_sc_y_raw, _sc_sample) if v >= 0],
            mode="markers",
            name=t(lang, "scatter_positive"),
            marker=dict(color="#34C759", size=4, opacity=0.45,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
        ))
        fig_scatter.add_trace(go.Scatter(
            x=[x for x, v in zip(_sc_x_raw, _sc_sample) if v < 0],
            y=[y for y, v in zip(_sc_y_raw, _sc_sample) if v < 0],
            mode="markers",
            name=t(lang, "scatter_negative"),
            marker=dict(color="#FF3B30", size=4, opacity=0.45,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
        ))
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="#FF3B30", line_width=1.5,
                              annotation_text="ROI=0", annotation_position="bottom right")
        fig_scatter.add_vline(x=0, line_dash="dot", line_color="#AEAEB2", line_width=1)
        fig_scatter.update_layout(
            xaxis_title=t(lang, "scatter_x"),
            yaxis_title=t(lang, "scatter_y") + f" ({_prec_label})",
            height=320,
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_scatter, width="stretch")

        st.markdown("---")

        # ── NPV / IRR ────────────────────────────────────────────────────────
        st.subheader(t(lang, "npv_title"))
        _wacc = st.slider(t(lang, "wacc_label"), 1, 25, value=12, key="wacc_slider")
        _npv_res = compute_npv_irr(
            total_benefit=res.total_benefit,
            impl_cost=float(impl_cost),
            wacc_pct=float(_wacc),
            decay=(1.0, 0.80, 0.65),
        )

        _npv_c1, _npv_c2 = st.columns(2)
        _npv_c1.metric(t(lang, "npv_label"), _fmt(_npv_res.npv, currency))
        _irr_str = (
            "{:.1f}%".format(_npv_res.irr_pct)
            if _npv_res.irr_pct is not None
            else t(lang, "irr_na")
        )
        _npv_c2.metric(t(lang, "irr_label"), _irr_str)

        st.markdown("---")

        # ── 3-YEAR PROJECTION ────────────────────────────────────────────────
        st.subheader(t(lang, "proj_title"))
        st.caption(t(lang, "proj_decay_note"))

        _years = [
            "{} 1".format(t(lang, "proj_year")),
            "{} 2".format(t(lang, "proj_year")),
            "{} 3".format(t(lang, "proj_year")),
        ]
        _benefits_eur = _npv_res.yearly_benefits
        _cum_npvs_eur = _npv_res.cumulative_npv
        _benefits     = [v * _prec_rate for v in _benefits_eur]
        _cum_npvs     = [v * _prec_rate for v in _cum_npvs_eur]

        _fig_proj = go.Figure()
        _fig_proj.add_trace(go.Bar(
            x=_years,
            y=_benefits,
            name=t(lang, "proj_benefit"),
            marker_color=[_C["green"], _C["blue"], _C["navy"]],
            text=[_fmt(v, currency) for v in _benefits_eur],
            textposition="outside",
            textfont=dict(size=11, color="#6E6E73"),
        ))
        _fig_proj.add_trace(go.Scatter(
            x=_years,
            y=_cum_npvs,
            mode="lines+markers+text",
            name=t(lang, "proj_cum_npv"),
            line=dict(color=_C["gold"], width=2.5),
            marker=dict(size=8, color=_C["gold"]),
            text=[_fmt(v, currency) for v in _cum_npvs_eur],
            textposition="top center",
            textfont=dict(size=10, color=_C["gold"]),
            yaxis="y2",
        ))
        _proj_layout = {k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        _fig_proj.update_layout(
            yaxis=dict(
                title=t(lang, "proj_benefit"),
                gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
                linecolor="rgba(0,0,0,0.06)", zeroline=False,
                tickfont=dict(color="#AEAEB2", size=11),
            ),
            yaxis2=dict(
                title=t(lang, "proj_cum_npv"),
                overlaying="y", side="right", showgrid=False,
                tickfont=dict(color="#AEAEB2", size=11),
            ),
            barmode="group",
            height=380,
            **_proj_layout,
        )
        st.plotly_chart(_fig_proj, width="stretch")

    # ── TAB 7: ABOUT ──────────────────────────────────────────────────────────
    with tab7:
        _ab = {
            "en": {
                "subtitle": "Professional ROI audit tool for automation projects",
                "author_title": "About the Author",
                "author_body": (
                    "**Andrew** — AI Product Advisor & Automation Consultant, Serbia/EU. "
                    "20+ years in enterprise software, CRM/ERP integrations, and process intelligence. "
                    "Trusted by mid-market companies across Eastern Europe and the Balkans."
                ),
                "method_title": "Methodology",
                "method_body": (
                    "OmniCore ROI Auditor combines 5 independent quantitative models:\n\n"
                    "**1. Labor Savings** — Hours freed × automation rate × hourly cost\n\n"
                    "**2. Error Reduction** — Volume × (rate_before − rate_after) × cost_per_error × 12\n\n"
                    "**3. Cycle Acceleration** — Days saved → more deal completions per year\n\n"
                    "**4. Markov Chain (Conversion Lift)** — Absorbing Markov chains model each "
                    "pipeline stage. P(complete) uplift drives Markov-weighted annual gain.\n\n"
                    "**5. Bayesian Confidence** — Beta-Binomial update on observed signals adjusts "
                    "the ROI output by real-world confidence."
                ),
                "tech_title": "Technical Stack",
                "tech_body": "Python 3.12 · Streamlit 1.44 · NumPy · SciPy · NetworkX · Plotly · ReportLab",
                "changelog_title": "Changelog",
                "changelog": [
                    ("v3.2", "Sprint 1–3: expanders, help texts, gauge/radar, funnel, scatter, bullet, @cache, About page"),
                    ("v3.1", "Multilingual PDF export (EN/RU/SR), number formatting, LinkedIn passport"),
                    ("v3.0", "Tornado sensitivity, Monte Carlo 5 000 runs, NPV/IRR, 3-year projection"),
                    ("v2.0", "NetworkX bottleneck graph, Bayesian update, ETL CSV pipeline"),
                    ("v1.0", "Core ROI engine, Markov chains, freemium demo mode"),
                ],
            },
            "ru": {
                "subtitle": "Профессиональный инструмент ROI-аудита для проектов автоматизации",
                "author_title": "Об авторе",
                "author_body": (
                    "**Андрей** — AI Product Advisor и консультант по автоматизации, Сербия/ЕС. "
                    "20+ лет в enterprise-разработке, CRM/ERP-интеграциях и process intelligence. "
                    "Работает с компаниями среднего рынка в Восточной Европе и на Балканах."
                ),
                "method_title": "Методология",
                "method_body": (
                    "OmniCore ROI Auditor сочетает 5 независимых количественных моделей:\n\n"
                    "**1. Экономия труда** — Освобождённые часы × ставку автоматизации × стоимость часа\n\n"
                    "**2. Снижение ошибок** — Объём × (до − после) × стоимость ошибки × 12\n\n"
                    "**3. Ускорение цикла** — Сэкономленные дни → больше сделок в год\n\n"
                    "**4. Цепи Маркова** — Поглощающие цепи Маркова моделируют каждую стадию пайплайна. "
                    "Прирост P(завершение) даёт взвешенный годовой выигрыш.\n\n"
                    "**5. Байесовское доверие** — Beta-Binomial-обновление по сигналам корректирует "
                    "ROI с учётом реального уровня уверенности."
                ),
                "tech_title": "Технический стек",
                "tech_body": "Python 3.12 · Streamlit 1.44 · NumPy · SciPy · NetworkX · Plotly · ReportLab",
                "changelog_title": "История версий",
                "changelog": [
                    ("v3.2", "Спринты 1–3: expanders, help, gauge/radar, воронка, scatter, bullet, @cache, страница О приложении"),
                    ("v3.1", "Многоязычный PDF-экспорт (EN/RU/SR), форматирование чисел, LinkedIn-паспорт"),
                    ("v3.0", "Tornado, Монте-Карло 5 000 итераций, NPV/IRR, 3-летний прогноз"),
                    ("v2.0", "NetworkX граф узких мест, байесовское обновление, ETL CSV-пайплайн"),
                    ("v1.0", "Ядро ROI-движка, цепи Маркова, демо-режим freemium"),
                ],
            },
            "sr": {
                "subtitle": "Profesionalni alat za ROI reviziju projekata automatizacije",
                "author_title": "O autoru",
                "author_body": (
                    "**Andrej** — AI Product Advisor i konsultant za automatizaciju, Srbija/EU. "
                    "20+ godina u enterprise softveru, CRM/ERP integracijama i process intelligence. "
                    "Sarađuje sa kompanijama srednje veličine u Istočnoj Evropi i na Balkanu."
                ),
                "method_title": "Metodologija",
                "method_body": (
                    "OmniCore ROI Auditor kombinuje 5 nezavisnih kvantitativnih modela:\n\n"
                    "**1. Ušteda rada** — Oslobođeni sati × stopa automatizacije × cena sata\n\n"
                    "**2. Smanjenje grešaka** — Obim × (pre − posle) × trošak greške × 12\n\n"
                    "**3. Ubrzanje ciklusa** — Ušteda dana → više poslova godišnje\n\n"
                    "**4. Markovljevi lanci** — Apsorbujući Markovljevi lanci modeluju svaku fazu pipeline-a. "
                    "Porast P(završetak) daje Markovljevski ponderisani godišnji dobitak.\n\n"
                    "**5. Bajesovsko poverenje** — Beta-Binomial ažuriranje na osnovu signala koriguje "
                    "ROI stvarnim nivoom pouzdanosti."
                ),
                "tech_title": "Tehnički stek",
                "tech_body": "Python 3.12 · Streamlit 1.44 · NumPy · SciPy · NetworkX · Plotly · ReportLab",
                "changelog_title": "Istorija verzija",
                "changelog": [
                    ("v3.2", "Sprintovi 1–3: expanders, help, gauge/radar, levak, scatter, bullet, @cache, O aplikaciji"),
                    ("v3.1", "Višejezični PDF izvoz (EN/RU/SR), formatiranje brojeva, LinkedIn pasoš"),
                    ("v3.0", "Tornado, Monte Karlo 5 000 iteracija, NPV/IRR, 3-godišnja projekcija"),
                    ("v2.0", "NetworkX graf uskih grla, bajesovsko ažuriranje, ETL CSV pipeline"),
                    ("v1.0", "Jezgro ROI motora, Markovljevi lanci, freemium demo mod"),
                ],
            },
        }.get(lang, {}).copy() or {
            "en": {"subtitle": "Professional ROI audit tool"}
        }["en"]

        # Hero card
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1D1D1F 0%,#2C2C2E 100%);'
            f'border-radius:18px;padding:32px 36px;margin-bottom:24px;">'
            f'<h2 style="color:#fff;font-size:24px;font-weight:700;margin:0 0 6px;">'
            f'{t(lang, "about_title")}</h2>'
            f'<p style="color:#AEAEB2;font-size:14px;margin:0;">{_ab.get("subtitle","")}</p>'
            f'<p style="color:#6E6E73;font-size:12px;margin:12px 0 0;">'
            f'{t(lang, "about_version")} · 2024–2025</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        _card = (
            "background:#F5F5F7;border-radius:14px;padding:20px 24px;margin-bottom:16px;"
        )
        _h3 = "font-size:16px;font-weight:700;color:#1D1D1F;margin:0 0 10px;"
        _p  = "font-size:13px;color:#3A3A3C;margin:0;line-height:1.6;"

        ab1, ab2 = st.columns([1, 1])
        with ab1:
            st.markdown(
                f'<div style="{_card}"><p style="{_h3}">{_ab.get("author_title","")}</p>'
                f'<p style="{_p}">{_ab.get("author_body","").replace("**", "")}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="{_card}"><p style="{_h3}">{_ab.get("tech_title","")}</p>'
                f'<p style="{_p}">{_ab.get("tech_body","")}</p></div>',
                unsafe_allow_html=True,
            )

        with ab2:
            with st.expander(_ab.get("method_title", "Methodology"), expanded=True):
                st.markdown(_ab.get("method_body", ""), unsafe_allow_html=False)

        st.markdown(f'<div style="{_card}"><p style="{_h3}">{_ab.get("changelog_title","")}</p>', unsafe_allow_html=True)
        for _ver, _desc in _ab.get("changelog", []):
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:7px 0;border-bottom:1px solid rgba(0,0,0,0.06);">'
                f'<span style="background:#0071E3;color:#fff;border-radius:6px;padding:2px 9px;'
                f'font-size:11px;font-weight:600;white-space:nowrap;min-width:42px;text-align:center;">'
                f'{_ver}</span>'
                f'<span style="font-size:13px;color:#3A3A3C;">{_desc}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── FAQ / METHODOLOGY ──────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📚 " + t(lang, "faq_title"), expanded=False):
        _faq_style = (
            "background:#F5F5F7;border-radius:12px;padding:14px 18px;"
            "margin:6px 0;border-left:3px solid #0071E3;"
        )
        _q_style = "font-size:14px;font-weight:600;color:#1D1D1F;margin:0 0 4px;"
        _a_style = "font-size:13px;color:#6E6E73;margin:0;"
        for _qi in range(1, 6):
            _q = t(lang, f"faq_q{_qi}")
            _a = t(lang, f"faq_a{_qi}")
            st.markdown(
                f'<div style="{_faq_style}">'
                f'<p style="{_q_style}">{_q}</p>'
                f'<p style="{_a_style}">{_a}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
