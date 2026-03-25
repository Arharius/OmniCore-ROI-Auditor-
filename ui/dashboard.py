import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

from core.math_engine import MathEngine
from core.roi_engine import ROIEngine, ROIInput
from etl.extractor import MatrixExtractor
from exports.pdf_generator import build_roi_passport_pdf
from ui.i18n import TRANSLATIONS, LANG_NAMES, t

st.set_page_config(
    page_title="OmniCore ROI Auditor",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── App background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0b1628 0%, #060a11 55%, #050810 100%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a111e 0%, #060d18 100%) !important;
    border-right: 1px solid rgba(192,160,98,0.12) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption { color: #8b92a5 !important; font-size: 12px; }
[data-testid="stSidebar"] h2 { color: #c0a062 !important; font-size: 14px !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0e1828 0%, #0b1320 100%);
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid rgba(192,160,98,0.18);
    box-shadow: 0 4px 24px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: rgba(192,160,98,0.38); }
[data-testid="stMetricLabel"] > div {
    color: #6b7485 !important; font-size: 11px !important;
    font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase;
}
[data-testid="stMetricValue"] > div {
    color: #00c48c !important; font-size: 28px !important;
    font-weight: 700 !important; letter-spacing: -0.02em !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #555e72 !important;
    font-size: 12px !important; font-weight: 600 !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important;
    border: none !important; border-radius: 0 !important;
    padding: 12px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #c0a062 !important;
    border-bottom: 2px solid #c0a062 !important;
    background: transparent !important;
}

/* ── Section headings ── */
[data-testid="stMarkdownContainer"] h1 {
    font-size: 22px !important; font-weight: 700 !important;
    color: #e8eaf0 !important; letter-spacing: -0.02em !important;
}
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-size: 13px !important; font-weight: 600 !important;
    color: #c0a062 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Subheaders ── */
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {
    color: #9aa3b5 !important; font-size: 13px !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important; text-transform: uppercase;
}

/* ── Download / action buttons ── */
div.stDownloadButton > button, div.stButton > button {
    background: linear-gradient(135deg, #1a2d50 0%, #162645 100%) !important;
    color: #c0a062 !important; border: 1px solid rgba(192,160,98,0.35) !important;
    border-radius: 8px; font-weight: 600; font-size: 12px;
    letter-spacing: 0.06em; padding: 10px 24px; width: 100%;
    transition: all 0.2s;
}
div.stDownloadButton > button:hover, div.stButton > button:hover {
    background: linear-gradient(135deg, #1f3560 0%, #1a2e52 100%) !important;
    border-color: rgba(192,160,98,0.65) !important;
}

/* ── Radio (language switcher) ── */
div[data-testid="stRadio"] > label {
    color: #6b7485 !important; font-size: 11px !important;
    font-weight: 600 !important; letter-spacing: 0.08em; text-transform: uppercase;
}
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: #9aa3b5 !important; }

/* ── Sliders ── */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #4a5168 !important; }
[data-testid="stSlider"] > div > div > div > div { background: #c0a062 !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] th { background: #0b1320 !important; color: #6b7485 !important;
    font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stDataFrame"] td { color: #c0c8d8 !important; font-size: 13px !important; }

/* ── Code blocks (passport) ── */
[data-testid="stCode"] { background: #0a111e !important; border: 1px solid rgba(192,160,98,0.12); border-radius: 8px; }
[data-testid="stCode"] code { color: #8fbc9a !important; font-size: 12px !important; }

/* ── Horizontal rule ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Gold accent strip on top ── */
.stApp::before {
    content: "";
    position: fixed; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c0a062, #d4b87a, #c0a062, transparent);
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

# ── Chart colour palette (institutional dark) ──────────────────────────────
_C = dict(
    gold="#c0a062", blue="#1565d8", green="#00c48c",
    red="#d63231",  purple="#7c5cbf", amber="#e8a020",
    teal="#00968a",
)
CHART_LAYOUT = dict(
    plot_bgcolor="rgba(8,12,22,0)",
    paper_bgcolor="rgba(8,12,22,0)",
    font=dict(family="Inter, sans-serif", color="#8b92a5", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)", gridwidth=1,
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#555e72", size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)", gridwidth=1,
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#555e72", size=10),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7485", size=10),
        bordercolor="rgba(255,255,255,0.06)", borderwidth=1,
    ),
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
if "lang_select" not in st.session_state:
    st.session_state["lang_select"] = "ru"

with st.sidebar:
    lang = st.radio(
        "🌐 Language / Язык / Jezik",
        options=list(LANG_NAMES.keys()),
        format_func=lambda k: LANG_NAMES[k],
        horizontal=True,
        key="lang_select",
    )

    st.markdown("## " + t(lang, "sidebar_title"))

    company_name = st.text_input(t(lang, "company_label"), value="Marteco Digital Services",
                                 key="company_name")

    csv_file = st.file_uploader(
        t(lang, "csv_label"), type=["csv"], help=t(lang, "csv_help"), key="csv_file"
    )

    st.markdown("---")
    st.markdown(t(lang, "labor_section"))
    manual_hours    = st.slider(t(lang, "manual_hours"),   50,   500, 320,       key="manual_hours")
    automation_rate = st.slider(t(lang, "automation_pct"), 50,    95,  86,       key="automation_rate")
    hour_rate       = st.slider(t(lang, "hour_rate"),       8,    30,  12,       key="hour_rate")

    st.markdown(t(lang, "errors_section"))
    error_before   = st.slider(t(lang, "error_before"),  1.0, 20.0,  8.5, step=0.1, key="error_before")
    error_after    = st.slider(t(lang, "error_after"),   0.1,  5.0,  1.2, step=0.1, key="error_after")
    cost_per_error = st.slider(t(lang, "cost_per_error"), 20,  500,   95,       key="cost_per_error")
    volume         = st.slider(t(lang, "volume"),        100, 2000,  600,       key="volume")

    st.markdown(t(lang, "cycle_section"))
    cycle_before = st.slider(t(lang, "cycle_before"),  5,  60, 21,             key="cycle_before")
    cycle_after  = st.slider(t(lang, "cycle_after"),   1,  30,  9,             key="cycle_after")
    deals        = st.slider(t(lang, "deals_month"),   5, 200, 25,             key="deals_month")
    deal_value   = st.slider(t(lang, "deal_value"),  100, 5000, 650,           key="deal_value")

    st.markdown(t(lang, "proba_section"))
    p_before = st.slider(t(lang, "p_before"), 50,  95, 74, key="p_before")
    p_after  = st.slider(t(lang, "p_after"),  70,  99, 96, key="p_after")

    st.markdown(t(lang, "invest_section"))
    impl_cost = st.slider(t(lang, "impl_cost"), 5000, 100000, 14000, step=1000, key="impl_cost")

    st.markdown("---")
    st.caption(t(lang, "footer"))

# ── COMPUTE ───────────────────────────────────────────────────────────────────
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
)

bayes_res = math_eng.bayesian_update(inp.positive_signals, inp.total_signals)
res = roi_eng.calculate(inp, bayes_result=bayes_res)

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

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px;">'
    '<span style="font-size:22px;font-weight:700;color:#e8eaf0;letter-spacing:-0.02em;">'
    + t(lang, "app_title") +
    '</span>'
    '<span style="font-size:12px;font-weight:600;color:#c0a062;letter-spacing:0.12em;'
    'text-transform:uppercase;margin-left:8px;">INSTITUTIONAL</span>'
    '</div>'
    '<div style="font-size:12px;color:#555e72;font-weight:500;letter-spacing:0.04em;margin-bottom:20px;">'
    '<span style="color:#c0a062;">◆</span>&nbsp;&nbsp;'
    + company_name +
    '&nbsp;&nbsp;<span style="color:#2a3145;">|</span>&nbsp;&nbsp;'
    + t(lang, "app_subtitle") +
    '</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(t(lang, "metric_net_roi"),  "{:,.0f} €".format(res.net_roi))
c2.metric(t(lang, "metric_payback"),  "{:.1f} {}".format(res.payback_months, t(lang, "months")))
c3.metric(t(lang, "metric_bayes"),    "{:.1f}%".format(res.bayesian_posterior_pct))
c4.metric(t(lang, "metric_impl"),     "{:,.0f} €".format(impl_cost))

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t(lang, "tab_roi"),
    t(lang, "tab_graph"),
    t(lang, "tab_markov"),
    t(lang, "tab_bayes"),
    t(lang, "tab_passport"),
])

# ────────────────────────────────────────────────────────────────────
# TAB 1 — ROI BREAKDOWN
# ────────────────────────────────────────────────────────────────────
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
        connector=dict(line=dict(color="rgba(192,160,98,0.25)", width=1)),
        increasing=dict(marker_color=_C["green"]),
        decreasing=dict(marker_color=_C["red"]),
        totals=dict(marker_color=_C["blue"]),
        texttemplate="%{y:,.0f} €",
        textposition="outside",
        textfont=dict(color="#8b92a5", size=10),
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
                colors=[_C["green"], _C["blue"], _C["purple"], _C["amber"]],
                line=dict(color="rgba(8,12,22,0.8)", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(color="#9aa3b5", size=11),
        ))
        fig_pie.update_layout(height=360, **CHART_LAYOUT)
        st.plotly_chart(fig_pie, width="stretch")

    with col_t:
        st.subheader("")
        st.markdown("<br>", unsafe_allow_html=True)
        df_bd = pd.DataFrame({
            t(lang, "component"): [t(lang, "time_saved"), t(lang, "error_saved"),
                                   t(lang, "revenue_speed"), t(lang, "revenue_conv"),
                                   t(lang, "total"), t(lang, "investment"), t(lang, "net_roi")],
            t(lang, "eur_year"):  [res.time_saved_annual, res.error_reduction_annual,
                                   res.revenue_impact_annual, res.markov_gain_annual,
                                   res.total_benefit, -impl_cost, res.net_roi],
        })
        st.dataframe(df_bd, height=300)

# ────────────────────────────────────────────────────────────────────
# TAB 2 — GRAPH
# ────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(t(lang, "graph_title"))
    st.info(t(lang, "bottleneck_info",
              node=graph_res.bottleneck_node, score=graph_res.bottleneck_score))

    nodes = list(graph_res.betweenness.keys())
    node_colors = [_C["red"] if n == graph_res.bottleneck_node else _C["blue"] for n in nodes]
    node_sizes  = [22 + graph_res.betweenness[n] * 200 for n in nodes]

    angle_step = 2 * np.pi / max(len(nodes), 1)
    pos = {n: (np.cos(i * angle_step), np.sin(i * angle_step)) for i, n in enumerate(nodes)}

    edge_x, edge_y = [], []
    for frm, to, _ in default_edges:
        if frm in pos and to in pos:
            edge_x += [pos[frm][0], pos[to][0], None]
            edge_y += [pos[frm][1], pos[to][1], None]

    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(192,160,98,0.18)", width=1.5),
        hoverinfo="none",
    ))
    fig_g.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(color="#c0c8d8", size=11),
        marker=dict(color=node_colors, size=node_sizes,
                    line=dict(color="rgba(255,255,255,0.15)", width=1.5)),
        hovertemplate="%{text}<br>" + t(lang, "centrality_col") + ": %{customdata:.4f}<extra></extra>",
        customdata=[graph_res.betweenness[n] for n in nodes],
    ))
    _no_axes = {k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig_g.update_layout(
        showlegend=False, height=420,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        **_no_axes,
    )
    st.plotly_chart(fig_g, width="stretch")

    st.subheader(t(lang, "centrality_table"))
    df_bt = pd.DataFrame(graph_res.all_nodes_ranked,
                         columns=[t(lang, "node_col"), t(lang, "centrality_col")])
    st.dataframe(df_bt, )

# ────────────────────────────────────────────────────────────────────
# TAB 3 — MARKOV
# ────────────────────────────────────────────────────────────────────
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
    st.dataframe(df_Q.style.format("{:.4f}"), )

    if N_mat is not None:
        st.markdown(t(lang, "matrix_n"))
        df_N = pd.DataFrame(N_mat, index=m_states, columns=m_states)
        st.dataframe(df_N.style.format("{:.4f}"), )

    st.markdown(t(lang, "timeline_title"))
    months_range  = list(range(0, 13))
    monthly_gain  = res.total_benefit / 12
    cumulative    = [monthly_gain * m - impl_cost for m in months_range]
    breakeven_line = [0] * 13

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=months_range, y=cumulative, mode="lines+markers",
        name=t(lang, "cumulative_roi"),
        line=dict(color=_C["blue"], width=2.5),
        marker=dict(size=5, color=_C["gold"],
                    line=dict(color=_C["blue"], width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(21,101,216,0.08)",
    ))
    fig_tl.add_trace(go.Scatter(
        x=months_range, y=breakeven_line, mode="lines",
        name=t(lang, "breakeven"),
        line=dict(color=_C["gold"], width=1.5, dash="dash"),
    ))
    fig_tl.update_layout(
        xaxis_title=t(lang, "month_label"),
        yaxis_title="EUR",
        height=400,
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_tl, width="stretch")

# ────────────────────────────────────────────────────────────────────
# TAB 4 — BAYES
# ────────────────────────────────────────────────────────────────────
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
    bc3.metric(t(lang, "ci_80"),
               "{}% – {}%".format(bayes_live.ci_80_low, bayes_live.ci_80_high))

    prior_a  = 0.34 * 10
    prior_b  = (1 - 0.34) * 10
    post_a   = prior_a + pos_signals
    post_b   = prior_b + (tot_signals - pos_signals)
    x        = np.linspace(0.01, 0.99, 300)
    y_prior  = stats.beta.pdf(x, prior_a, prior_b)
    y_post   = stats.beta.pdf(x, post_a,  post_b)

    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(
        x=x * 100, y=y_prior, mode="lines",
        name=t(lang, "prior_label"),
        line=dict(color=_C["gold"], width=2),
        fill="tozeroy", fillcolor="rgba(192,160,98,0.07)",
    ))
    fig_b.add_trace(go.Scatter(
        x=x * 100, y=y_post, mode="lines",
        name=t(lang, "posterior_label"),
        line=dict(color=_C["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(21,101,216,0.12)",
    ))
    fig_b.update_layout(
        xaxis_title=t(lang, "probability_pct"),
        yaxis_title=t(lang, "density"),
        height=400,
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_b, width="stretch")

    risk = math_eng.bayesian_contextual_risk(0.05, 0.80, 0.20)
    st.info(t(lang, "contextual_risk", risk=risk))

# ────────────────────────────────────────────────────────────────────
# TAB 5 — PASSPORT
# ────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader(t(lang, "passport_title"))

    passport = roi_eng.passport_text(inp, res)
    st.code(passport, language="")

    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            label=t(lang, "download_txt"),
            data=passport.encode("utf-8"),
            file_name="roi_passport_{}.txt".format(company_name.replace(" ", "_")),
            mime="text/plain",
            key="dl_txt",
        )

    with dl2:
        try:
            pdf_bytes = build_roi_passport_pdf(
                company_name=company_name,
                auditor_name="Andrew | AI Product Advisor",
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
            import traceback
            st.code(traceback.format_exc())
            pdf_bytes = b""
        st.download_button(
            label=t(lang, "download_pdf"),
            data=pdf_bytes,
            file_name="roi_passport_{}.pdf".format(company_name.replace(" ", "_")),
            mime="application/pdf",
            key="dl_pdf",
        )

    linkedin_text = t(lang, "linkedin_text",
                      company=company_name,
                      roi=res.net_roi,
                      roi_pct=res.roi_pct,
                      payback=res.payback_months)
    st.text_area(t(lang, "linkedin_label"), value=linkedin_text, height=110, key="linkedin_ta")
