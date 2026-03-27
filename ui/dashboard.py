import sys
import os
import json
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

import dataclasses

from core.math_engine import MathEngine
from core.roi_engine import ROIEngine, ROIInput, ROIResult
from core.advanced_analytics import run_monte_carlo, run_tornado, compute_npv_irr
from etl.extractor import MatrixExtractor
from exports.pdf_generator import build_roi_passport_pdf
from ui.i18n import TRANSLATIONS, LANG_NAMES, t


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
_HISTORY_FILE = os.path.join(parent_dir, "data", "clients.json")

_PARAM_KEYS = ["manual_hours", "automation_rate", "hour_rate",
               "error_before", "error_after", "cost_per_error", "volume",
               "cycle_before", "cycle_after", "deals_month", "deal_value",
               "p_before", "p_after", "impl_cost"]

def _load_history() -> list:
    try:
        if os.path.exists(_HISTORY_FILE):
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_to_history(company_name: str) -> bool:
    try:
        params = {k: st.session_state.get(k) for k in _PARAM_KEYS}
        history = _load_history()
        history = [h for h in history if h.get("company_name") != company_name]
        history.insert(0, {
            "company_name": company_name,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "params": params,
        })
        history = history[:15]
        os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def _restore_from_history(entry: dict):
    st.session_state["company_name"] = entry.get("company_name", "")
    for k, v in entry.get("params", {}).items():
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
    [data-testid="metric-container"] {
        background: #FFFFFF !important; border-radius: 18px !important;
        padding: 24px 26px !important; border: none !important;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08) !important;
        transition: box-shadow 0.3s ease, transform 0.3s ease !important;
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 32px rgba(0,0,0,0.13) !important; transform: scale(1.015) !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #6E6E73 !important; font-size: 13px !important; font-weight: 400 !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #34C759 !important; font-size: 32px !important;
        font-weight: 700 !important; letter-spacing: -0.04em !important;
    }
    [data-testid="stMetricDelta"] > div { font-size: 13px !important; font-weight: 400 !important; }
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
            if _save_to_history(company_name):
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

    # Absorbing-state keywords (multi-language)
    _ABSORBING_KW = {
        "done", "completed", "won", "closed", "delivered", "shipped",
        "approved", "deployed", "production", "complete", "finished",
        "завершена", "выполнено", "продано", "закрыта", "доставлено",
        "završeno", "zatvoreno", "isporučeno",
    }

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
        _next_lower  = _mapped_df["next_stage"].astype(str).str.lower().str.strip()
        _completed   = _mapped_df[_next_lower.isin(_ABSORBING_KW)]["entity_id"].nunique()
        p_before     = max(30, min(95, int(_completed / _unique_ent * 100)))
        p_after      = min(99, int(p_before * 1.28))

        pos_signals  = max(1, min(_completed, _unique_ent))
        tot_signals  = max(2, _unique_ent)
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

    # Seed Bayesian tab sliders with derived values (setdefault = only on first run)
    st.session_state.setdefault("pos_signals", pos_signals)
    st.session_state.setdefault("tot_signals", tot_signals)

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

    bayes_res = math_eng.bayesian_update(inp.positive_signals, inp.total_signals)
    res = roi_eng.calculate(inp, bayes_result=bayes_res)
    res = _apply_confidence(res, st.session_state.get("scenario_confidence", 0.75))

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

    _default_Q      = np.array([[0.2, 0.3], [0.1, 0.4]])
    _default_states = ["Qualification", "Proposal"] if lang == "en" else (
                      ["Квалификация", "Предложение"] if lang == "ru" else
                      ["Kvalifikacija", "Ponuda"])
    if csv_file is not None:
        extractor = MatrixExtractor()
        process_log = extractor.from_csv(csv_file)
        _q = process_log.matrix_Q
        _s = process_log.states_transient
        if len(_s) > 0 and _q.shape[0] == len(_s) and _q.shape[1] == len(_s):
            Q_mat    = _q
            m_states = _s

            # ── Fix 1: build graph edges from real CSV transitions ──────────────
            _csv_edges = [
                (frm, to, float(count))
                for frm, targets in process_log.raw_counts.items()
                for to, count in targets.items()
            ]
            if _csv_edges:
                _active_edges = _csv_edges
                graph_res = math_eng.graph_bottleneck(_active_edges)

            # sliders (cycle_before, volume, pos_signals, tot_signals) are
            # pre-populated from CSV in the pre-scan block above (before widgets)
        else:
            _err_detail = getattr(extractor, "last_error", "")
            _err_msg = {
                "en": "⚠️ CSV format not recognised — demo data loaded.",
                "ru": "⚠️ CSV не распознан — загружены демо-данные.",
                "sr": "⚠️ CSV nije prepoznat — učitani demo podaci.",
            }[lang]
            if _err_detail:
                _err_msg += f"\n\n`{_err_detail}`"
            st.warning(_err_msg)
            Q_mat    = _default_Q
            m_states = _default_states
    else:
        Q_mat    = _default_Q
        m_states = _default_states

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
                for _k in ("_etl_idx_entity", "_etl_idx_current",
                           "_etl_idx_next", "_etl_idx_time"):
                    st.session_state.pop(_k, None)

            # ── Read CSV (multi-encoding fallback) ────────────────────────────
            _raw_df = None
            for _enc in ("utf-8", "cp1251", "latin-1"):
                try:
                    _etl_file.seek(0)
                    _raw_df = pd.read_csv(_etl_file, encoding=_enc)
                    break
                except Exception:
                    continue
            if _raw_df is None:
                st.error("Could not parse CSV — try saving as UTF-8.")
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
        st.subheader(t(lang, "graph_title"))
        st.info(t(lang, "bottleneck_info", node=graph_res.bottleneck_node, score=graph_res.bottleneck_score))
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
                                   line=dict(color="rgba(26,50,113,0.15)", width=1.5), hoverinfo="none"))
        fig_g.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text", text=nodes, textposition="top center",
            textfont=dict(color="#1a2744", size=11),
            marker=dict(color=node_colors, size=node_sizes, line=dict(color="rgba(255,255,255,0.9)", width=2)),
            hovertemplate="%{text}<br>" + t(lang, "centrality_col") + ": %{customdata:.4f}<extra></extra>",
            customdata=[graph_res.betweenness[n] for n in nodes],
        ))
        _no_axes = {k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        fig_g.update_layout(showlegend=False, height=420,
                            xaxis=dict(visible=False), yaxis=dict(visible=False), **_no_axes)
        st.plotly_chart(fig_g, width="stretch")
        st.subheader(t(lang, "centrality_table"))
        df_bt = pd.DataFrame(graph_res.all_nodes_ranked, columns=[t(lang, "node_col"), t(lang, "centrality_col")])
        st.dataframe(df_bt)

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
            pos_signals = st.slider(t(lang, "positive_signals"), 1, 50, 4, key="pos_signals")
        with b2:
            tot_signals = st.slider(t(lang, "total_signals"), 2, 100, 5, key="tot_signals")
        bayes_live = math_eng.bayesian_update(pos_signals, tot_signals)
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric(t(lang, "prior"),     "{:.1f}%".format(bayes_live.prior_pct))
        bc2.metric(t(lang, "posterior"), "{:.1f}%".format(bayes_live.posterior_pct))
        bc3.metric(t(lang, "ci_80"), "{}% – {}%".format(bayes_live.ci_80_low, bayes_live.ci_80_high))
        prior_a  = 0.34 * 10; prior_b  = (1 - 0.34) * 10
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
                        bottleneck_node=graph_res.bottleneck_node,
                        bottleneck_score=graph_res.bottleneck_score,
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
