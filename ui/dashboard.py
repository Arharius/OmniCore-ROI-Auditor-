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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── App background — warm ivory ── */
.stApp {
    background: #f0ece4 !important;
}

/* ── Sidebar — white panel ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(0,0,0,0.07) !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.05);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #7a8499 !important; font-size: 12px;
}
[data-testid="stSidebar"] h2 {
    color: #1a3271 !important; font-size: 13px !important;
    font-weight: 700 !important; letter-spacing: 0.09em !important; text-transform: uppercase;
}

/* ── Metric cards — white with shadow ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px 26px;
    border: 1px solid rgba(0,0,0,0.07);
    box-shadow: 0 2px 12px rgba(26,50,113,0.07), 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.15s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 6px 24px rgba(26,50,113,0.12);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] > div {
    color: #9aa3b8 !important; font-size: 11px !important;
    font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase;
}
[data-testid="stMetricValue"] > div {
    color: #009a5c !important; font-size: 28px !important;
    font-weight: 800 !important; letter-spacing: -0.03em !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,0,0,0.08) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #9aa3b8 !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.09em !important; text-transform: uppercase !important;
    border: none !important; border-radius: 0 !important;
    padding: 12px 22px !important;
}
.stTabs [aria-selected="true"] {
    color: #1a3271 !important;
    border-bottom: 2px solid #1a3271 !important;
    background: transparent !important;
}

/* ── Headings ── */
[data-testid="stMarkdownContainer"] h1 {
    font-size: 24px !important; font-weight: 800 !important;
    color: #1a2744 !important; letter-spacing: -0.03em !important;
}
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-size: 11px !important; font-weight: 700 !important;
    color: #9aa3b8 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {
    color: #9aa3b8 !important; font-size: 11px !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important; text-transform: uppercase;
}

/* ── Buttons ── */
div.stDownloadButton > button, div.stButton > button {
    background: #1a3271 !important;
    color: #ffffff !important; border: none !important;
    border-radius: 8px; font-weight: 600; font-size: 12px;
    letter-spacing: 0.05em; padding: 10px 24px; width: 100%;
    box-shadow: 0 2px 8px rgba(26,50,113,0.25);
    transition: all 0.2s;
}
div.stDownloadButton > button:hover, div.stButton > button:hover {
    background: #1e3d8f !important;
    box-shadow: 0 4px 16px rgba(26,50,113,0.35) !important;
    transform: translateY(-1px);
}

/* ── Radio (language) ── */
div[data-testid="stRadio"] > label {
    color: #7a8499 !important; font-size: 11px !important;
    font-weight: 700 !important; letter-spacing: 0.09em; text-transform: uppercase;
}
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: #1a2744 !important; }

/* ── Sliders ── */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #b0b8cc !important; }
[data-testid="stSlider"] > div > div > div > div { background: #1a3271 !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,0,0,0.07); border-radius: 10px;
    overflow: hidden; background: #fff;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
[data-testid="stDataFrame"] th {
    background: #f8f6f2 !important; color: #9aa3b8 !important;
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 0.1em; text-transform: uppercase;
}
[data-testid="stDataFrame"] td { color: #1a2744 !important; font-size: 13px !important; }

/* ── Code (passport) ── */
[data-testid="stCode"] {
    background: #f8f6f2 !important;
    border: 1px solid rgba(0,0,0,0.07); border-radius: 10px;
}
[data-testid="stCode"] code { color: #1a3271 !important; font-size: 12px !important; }

/* ── Info boxes ── */
[data-testid="stAlert"] {
    background: #eef2fb !important; border: 1px solid rgba(26,50,113,0.15) !important;
    border-radius: 10px; color: #1a3271 !important;
}

/* ── Horizontal rule ── */
hr { border-color: rgba(0,0,0,0.07) !important; }

/* ── Navy top accent bar ── */
.stApp::before {
    content: "";
    position: fixed; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1a3271 0%, #2a52a0 50%, #c0a062 100%);
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

# ── Chart colour palette (light institutional) ─────────────────────────────
_C = dict(
    navy="#1a3271", green="#009a5c", gold="#c0a062",
    red="#c0392b",  purple="#6c5ce7", amber="#d4860a",
    blue="#2563eb", teal="#0d9488",
)
CHART_LAYOUT = dict(
    plot_bgcolor="rgba(255,255,255,0)",
    paper_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Inter, sans-serif", color="#9aa3b8", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
        linecolor="rgba(0,0,0,0.08)", zeroline=False,
        tickfont=dict(color="#b0b8cc", size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
        linecolor="rgba(0,0,0,0.08)", zeroline=False,
        tickfont=dict(color="#b0b8cc", size=10),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        font=dict(color="#7a8499", size=10),
        bordercolor="rgba(0,0,0,0.06)", borderwidth=1,
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
    '<div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">'
    '<span style="font-size:24px;font-weight:800;color:#1a2744;letter-spacing:-0.03em;">'
    + t(lang, "app_title") +
    '</span>'
    '<span style="font-size:10px;font-weight:700;color:#ffffff;letter-spacing:0.12em;'
    'text-transform:uppercase;background:#1a3271;border-radius:4px;padding:3px 8px;">'
    'INSTITUTIONAL</span>'
    '</div>'
    '<div style="font-size:12px;color:#9aa3b8;font-weight:500;letter-spacing:0.03em;margin-bottom:20px;">'
    '<span style="color:#c0a062;font-size:10px;">◆</span>&nbsp;&nbsp;'
    '<span style="color:#1a2744;font-weight:600;">' + company_name + '</span>'
    '&nbsp;&nbsp;<span style="color:#dde0e8;">|</span>&nbsp;&nbsp;'
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
        connector=dict(line=dict(color="rgba(26,50,113,0.18)", width=1)),
        increasing=dict(marker_color=_C["green"]),
        decreasing=dict(marker_color=_C["red"]),
        totals=dict(marker_color=_C["navy"]),
        texttemplate="%{y:,.0f} €",
        textposition="outside",
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
                colors=[_C["green"], _C["navy"], _C["purple"], _C["amber"]],
                line=dict(color="rgba(240,236,228,1)", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(color="#4a5168", size=11),
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
    node_colors = [_C["red"] if n == graph_res.bottleneck_node else _C["navy"] for n in nodes]
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
        line=dict(color="rgba(26,50,113,0.15)", width=1.5),
        hoverinfo="none",
    ))
    fig_g.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(color="#1a2744", size=11),
        marker=dict(color=node_colors, size=node_sizes,
                    line=dict(color="rgba(255,255,255,0.9)", width=2)),
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
        line=dict(color=_C["navy"], width=2.5),
        marker=dict(size=5, color=_C["navy"],
                    line=dict(color="#ffffff", width=2)),
        fill="tozeroy",
        fillcolor="rgba(26,50,113,0.07)",
    ))
    fig_tl.add_trace(go.Scatter(
        x=months_range, y=breakeven_line, mode="lines",
        name=t(lang, "breakeven"),
        line=dict(color=_C["gold"], width=1.5, dash="dot"),
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
        fill="tozeroy", fillcolor="rgba(192,160,98,0.10)",
    ))
    fig_b.add_trace(go.Scatter(
        x=x * 100, y=y_post, mode="lines",
        name=t(lang, "posterior_label"),
        line=dict(color=_C["navy"], width=2.5),
        fill="tozeroy", fillcolor="rgba(26,50,113,0.10)",
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
