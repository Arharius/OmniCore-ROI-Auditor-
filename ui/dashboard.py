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


# ── Currency ───────────────────────────────────────────────────────────────────
_CURRENCIES = {
    "EUR": {"sym": "€",    "rate": 1.0,   "label": "EUR"},
    "RUB": {"sym": "₽",    "rate": 100.0, "label": "RUB"},
    "RSD": {"sym": "дин.", "rate": 117.0, "label": "RSD"},
}

def _fmt(val_eur: float, currency: str) -> str:
    cur = _CURRENCIES.get(currency, _CURRENCIES["EUR"])
    converted = val_eur * cur["rate"]
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

        # Demo presets
        _demo_labels = {
            "en": ("Demo cases", "Run live audit"),
            "ru": ("Демо-кейсы", "Живой аудит"),
            "sr": ("Demo slučajevi", "Živi audit"),
        }
        st.markdown(
            f'<div style="font-size:11px;font-weight:600;color:#AEAEB2;'
            f'letter-spacing:0.08em;text-transform:uppercase;margin:8px 0 6px;">'
            f'{_demo_labels[lang][0]}</div>',
            unsafe_allow_html=True,
        )
        _active = st.session_state.get("demo_preset")
        _dc1, _dc2, _dc3 = st.columns(3)
        if _dc1.button(DEMO_PRESETS["logistics"]["labels"][lang], key="btn_logistics", use_container_width=True):
            _apply_preset("logistics"); st.rerun()
        if _dc2.button(DEMO_PRESETS["agency"]["labels"][lang], key="btn_agency", use_container_width=True):
            _apply_preset("agency"); st.rerun()
        if _dc3.button(DEMO_PRESETS["retail"]["labels"][lang], key="btn_retail", use_container_width=True):
            _apply_preset("retail"); st.rerun()
        if _active:
            if st.button(_demo_labels[lang][1], key="btn_live", use_container_width=True):
                _clear_demo(); st.rerun()

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

        if is_demo:
            _lock_csv = {
                "en": "Sign in to upload your own CSV data",
                "ru": "Войдите, чтобы загрузить свои CSV-данные",
                "sr": "Prijavite se da učitate sopstvene CSV podatke",
            }
            st.info(_lock_csv[lang])
            _sample_label = {"en": "Download sample CSV", "ru": "Скачать пример CSV", "sr": "Preuzmi primer CSV"}
            try:
                with open("data/mock_client_data.csv", "rb") as _sf:
                    st.download_button(
                        label=_sample_label[lang],
                        data=_sf.read(),
                        file_name="sample_audit_data.csv",
                        mime="text/csv",
                        key="dl_sample_csv",
                        use_container_width=True,
                    )
            except Exception:
                pass
            csv_file = None
        else:
            csv_file = st.file_uploader(
                t(lang, "csv_label"), type=["csv"], help=t(lang, "csv_help"), key="csv_file"
            )

        _slider_defaults = {
            "manual_hours": 320, "automation_rate": 86, "hour_rate": 12,
            "error_before": 8.5, "error_after": 1.2, "cost_per_error": 95, "volume": 600,
            "cycle_before": 21, "cycle_after": 9, "deals_month": 25, "deal_value": 700,
            "p_before": 74, "p_after": 96, "impl_cost": 14000, "pipeline_util": 30,
        }
        for _k, _v in _slider_defaults.items():
            st.session_state.setdefault(_k, _v)

        st.markdown("---")
        st.markdown(t(lang, "labor_section"))
        manual_hours    = st.slider(t(lang, "manual_hours"),   50,   600,       key="manual_hours")
        automation_rate = st.slider(t(lang, "automation_pct"), 50,    95,       key="automation_rate")
        hour_rate       = st.slider(t(lang, "hour_rate"),       8,    30,       key="hour_rate")

        st.markdown(t(lang, "errors_section"))
        error_before   = st.slider(t(lang, "error_before"),  1.0, 20.0, step=0.1, key="error_before")
        error_after    = st.slider(t(lang, "error_after"),   0.1,  5.0, step=0.1, key="error_after")
        cost_per_error = st.slider(t(lang, "cost_per_error"), 20,  500,          key="cost_per_error")
        volume         = st.slider(t(lang, "volume"),        100, 2000,          key="volume")

        st.markdown(t(lang, "cycle_section"))
        cycle_before = st.slider(t(lang, "cycle_before"),  5,  60,              key="cycle_before")
        cycle_after  = st.slider(t(lang, "cycle_after"),   1,  30,              key="cycle_after")
        deals        = st.slider(t(lang, "deals_month"),   5, 200,              key="deals_month")
        deal_value   = st.slider(t(lang, "deal_value"),  100, 15000, step=100,  key="deal_value")

        st.markdown(t(lang, "proba_section"))
        p_before = st.slider(t(lang, "p_before"), 50,  95, key="p_before")
        p_after  = st.slider(t(lang, "p_after"),  70,  99, key="p_after")

        st.markdown(t(lang, "invest_section"))
        impl_cost    = st.slider(t(lang, "impl_cost"), 5000, 100000, step=1000, key="impl_cost")
        pipeline_util = st.slider(
            t(lang, "pipeline_util"), 10, 60, key="pipeline_util",
            help=t(lang, "pipeline_util_help"),
        )

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

    # ── COMPUTE ────────────────────────────────────────────────────────────────
    math_eng = MathEngine()
    roi_eng  = ROIEngine()

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
        positive_signals=4,
        total_signals=5,
        pipeline_utilization_pct=float(pipeline_util),
    )

    bayes_res = math_eng.bayesian_update(inp.positive_signals, inp.total_signals)
    res = roi_eng.calculate(inp, bayes_result=bayes_res)
    res = _apply_confidence(res, st.session_state.get("scenario_confidence", 0.75))

    default_edges = [
        ("Lead", "In Review", 3.0),
        ("In Review", "Approved", 0.8),
        ("In Review", "Revision", 0.6),
        ("Revision", "In Review", 0.7),
        ("Revision", "Rejected", 0.4),
    ]
    graph_res = math_eng.graph_bottleneck(default_edges)

    if csv_file is not None:
        extractor = MatrixExtractor()
        process_log = extractor.from_csv(csv_file)
        Q_mat   = process_log.matrix_Q
        m_states = process_log.states_transient
    else:
        Q_mat   = np.array([[0.2, 0.3], [0.1, 0.4]])
        m_states = ["Qualification", "Proposal"] if lang == "en" else (
                   ["Квалификация", "Предложение"] if lang == "ru" else
                   ["Kvalifikacija", "Ponuda"])

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

    # ── Benchmarks ─────────────────────────────────────────────────────────────
    _preset_key = st.session_state.get("demo_preset")
    _bench = _BENCHMARKS.get(_preset_key, _BENCHMARKS[None])
    _bench_roi_eur = impl_cost * _bench["net_roi_mult"]

    # ── HEADER ─────────────────────────────────────────────────────────────────
    if st.session_state.get("presentation_mode"):
        # Presentation mode: clean exit button
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

    # Demo banner
    _active_preset = st.session_state.get("demo_preset")
    if _active_preset and _active_preset in DEMO_PRESETS:
        _p_data = DEMO_PRESETS[_active_preset]
        _b_title, _b_desc_tpl, _b_hint = _DEMO_BANNER[lang]
        _b_desc = _p_data["desc"].get(lang, _p_data["desc"]["ru"])
        _label  = _p_data["labels"][lang]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'background:linear-gradient(135deg,rgba(0,113,227,0.07) 0%,rgba(52,199,89,0.06) 100%);'
            f'border:1px solid rgba(0,113,227,0.18);border-radius:16px;'
            f'padding:12px 18px;margin-bottom:16px;">'
            f'<div style="background:#0071E3;color:#fff;font-size:11px;font-weight:700;'
            f'letter-spacing:0.06em;padding:3px 10px;border-radius:980px;white-space:nowrap;">DEMO</div>'
            f'<div>'
            f'<div style="font-size:14px;font-weight:600;color:#0071E3;">{_label} — {_b_desc}</div>'
            f'<div style="font-size:12px;color:#AEAEB2;margin-top:2px;">{_b_hint}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Scenario badge (2) ──────────────────────────────────────────────────────
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

    # ── KPI cards with benchmark delta (6) ─────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    _roi_delta = res.net_roi - _bench_roi_eur
    _pay_delta = _bench["payback"] - res.payback_months
    c1.metric(
        t(lang, "metric_net_roi"),
        _fmt(res.net_roi, currency),
        delta="{:+,.0f}€ {}".format(_roi_delta, t(lang, "vs_industry")),
        delta_color="normal",
    )
    c2.metric(
        t(lang, "metric_payback"),
        "{:.1f} {}".format(res.payback_months, t(lang, "months")),
        delta="{:+.1f} мес {}".format(_pay_delta, t(lang, "vs_industry")),
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
        + t(lang, "risk_adj_formula", val=_risk_adj_val) + '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        t(lang, "tab_roi"), t(lang, "tab_graph"), t(lang, "tab_markov"),
        t(lang, "tab_bayes"), t(lang, "tab_passport"), t(lang, "tab_precision"),
    ])

    # ── TAB 1: ROI BREAKDOWN ───────────────────────────────────────────────────
    with tab1:
        st.subheader(t(lang, "waterfall_title"))
        labels   = [t(lang, "time_saved"), t(lang, "error_saved"),
                    t(lang, "revenue_speed"), t(lang, "revenue_conv"),
                    t(lang, "investment"), t(lang, "net_roi")]
        values   = [res.time_saved_annual, res.error_reduction_annual,
                    res.revenue_impact_annual, res.markov_gain_annual,
                    -impl_cost, res.net_roi]
        measures = ["relative", "relative", "relative", "relative", "relative", "total"]

        fig_wf = go.Figure(go.Waterfall(
            name="ROI", orientation="v",
            measure=measures, x=labels, y=values,
            connector=dict(line=dict(color="rgba(26,50,113,0.18)", width=1)),
            increasing=dict(marker_color=_C["green"]),
            decreasing=dict(marker_color=_C["red"]),
            totals=dict(marker_color=_C["navy"]),
            texttemplate="%{y:,.0f} €", textposition="outside",
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
        for frm, to, _ in default_edges:
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
        st.markdown(t(lang, "timeline_title"))
        months_range  = list(range(0, 13))
        monthly_gain  = res.total_benefit / 12
        cumulative    = [monthly_gain * m - impl_cost for m in months_range]
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(x=months_range, y=cumulative, mode="lines+markers",
                                    name=t(lang, "cumulative_roi"),
                                    line=dict(color=_C["navy"], width=2.5),
                                    marker=dict(size=5, color=_C["navy"], line=dict(color="#ffffff", width=2)),
                                    fill="tozeroy", fillcolor="rgba(26,50,113,0.07)"))
        fig_tl.add_trace(go.Scatter(x=months_range, y=[0]*13, mode="lines",
                                    name=t(lang, "breakeven"),
                                    line=dict(color=_C["gold"], width=1.5, dash="dot")))
        fig_tl.update_layout(xaxis_title=t(lang, "month_label"), yaxis_title="EUR", height=400, **CHART_LAYOUT)
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
        passport = roi_eng.passport_text(inp, res)
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
                          company=company_name, roi=res.net_roi,
                          roi_pct=res.roi_pct, payback=res.payback_months)
        st.text_area(t(lang, "linkedin_label"), value=linkedin_text, height=110, key="linkedin_ta")

    # ── TAB 6: PRECISION ANALYSIS ─────────────────────────────────────────────
    with tab6:

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
        _t_fig.update_layout(
            barmode="overlay",
            xaxis_title=t(lang, "tornado_impact_eur"),
            yaxis_title=t(lang, "tornado_param"),
            height=380,
            **CHART_LAYOUT,
        )
        st.plotly_chart(_t_fig, width="stretch")

        st.markdown("---")

        # ── MONTE CARLO ──────────────────────────────────────────────────────
        st.subheader(t(lang, "mc_title"))
        with st.spinner("Running simulation…"):
            _mc = run_monte_carlo(inp, n=5000)

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

        _fig_mc = go.Figure(go.Histogram(
            x=_mc.roi_samples,
            nbinsx=60,
            marker_color="#0071E3",
            opacity=0.75,
            name="{:,} {}".format(_mc.n_simulations, t(lang, "mc_runs")),
        ))
        _fig_mc.add_vline(x=0, line_dash="dash", line_color="#FF3B30", line_width=2,
                          annotation_text="ROI=0", annotation_position="top right")
        _fig_mc.add_vline(x=_mc.pct10, line_dash="dot", line_color="#AEAEB2", line_width=1)
        _fig_mc.add_vline(x=_mc.pct50, line_dash="solid", line_color="#34C759", line_width=2,
                          annotation_text="p50", annotation_position="top")
        _fig_mc.add_vline(x=_mc.pct90, line_dash="dot", line_color="#AEAEB2", line_width=1)
        _fig_mc.update_layout(
            xaxis_title=t(lang, "mc_hist_x"),
            yaxis_title=t(lang, "mc_hist_y"),
            height=340,
            **CHART_LAYOUT,
        )
        st.plotly_chart(_fig_mc, width="stretch")

        st.markdown("---")

        # ── NPV / IRR ────────────────────────────────────────────────────────
        st.subheader(t(lang, "npv_title"))
        _wacc = st.slider(t(lang, "wacc_label"), 6, 25, value=12, key="wacc_slider")
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
        _benefits  = _npv_res.yearly_benefits
        _cum_npvs  = _npv_res.cumulative_npv

        _fig_proj = go.Figure()
        _fig_proj.add_trace(go.Bar(
            x=_years,
            y=_benefits,
            name=t(lang, "proj_benefit"),
            marker_color=[_C["green"], _C["blue"], _C["navy"]],
            text=[_fmt(v, currency) for v in _benefits],
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
            text=[_fmt(v, currency) for v in _cum_npvs],
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
