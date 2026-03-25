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
/* ── Full dark theme via CSS (no Streamlit theme config needed) ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.main, .block-container {
    background-color: #0e1117 !important;
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] {
    background-color: #1e2130 !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Inputs, sliders, selectboxes */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
.stSelectbox div[data-baseweb="select"] {
    background-color: #1e2130 !important;
    color: #e0e0e0 !important;
    border-color: #636EFA44 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1e2130 !important;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid rgba(99,110,250,0.35);
    margin-bottom: 4px;
}
[data-testid="stMetricLabel"] > div { color: #aaaacc !important; font-size: 13px; }
[data-testid="stMetricValue"] > div {
    color: #00CC96 !important;
    font-size: 26px !important;
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #1e2130 !important; }
.stTabs [data-baseweb="tab"] {
    background: #1e2130 !important;
    color: #cccccc !important;
    border-radius: 6px 6px 0 0;
    padding: 8px 18px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #636EFA !important;
    color: white !important;
}

/* Buttons */
div.stDownloadButton > button, div.stButton > button {
    background: linear-gradient(90deg, #636EFA, #00CC96) !important;
    color: white !important; border: none !important;
    border-radius: 8px; font-weight: 600; padding: 10px 24px; width: 100%;
}
div.stDownloadButton > button:hover, div.stButton > button:hover { opacity: 0.85; }

/* Text */
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, label, .stMarkdown { color: #dddddd !important; }

/* DataFrames */
[data-testid="stDataFrame"] { background: #1e2130 !important; }
[data-testid="stDataFrame"] th { background: #0e1117 !important; color: #cccccc !important; }
[data-testid="stDataFrame"] td { background: #1e2130 !important; color: #e0e0e0 !important; }

/* Info / alerts */
[data-testid="stInfo"] { background: #1e2130 !important; border-color: #636EFA44 !important; color: #cccccc !important; }

/* Code blocks */
[data-testid="stCode"] { background: #1e2130 !important; }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(14,17,23,0.8)",
    paper_bgcolor="rgba(14,17,23,0)",
    font_color="#dddddd",
    margin=dict(l=20, r=20, t=40, b=20),
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    lang = st.selectbox(
        "🌐 Language / Язык / Jezik",
        options=list(LANG_NAMES.keys()),
        format_func=lambda k: LANG_NAMES[k],
        index=1,
    )

    st.markdown("## " + t(lang, "sidebar_title"))

    company_name = st.text_input(t(lang, "company_label"), value="Marteco Digital Services")

    csv_file = st.file_uploader(
        t(lang, "csv_label"), type=["csv"], help=t(lang, "csv_help")
    )

    st.markdown("---")
    st.markdown(t(lang, "labor_section"))
    manual_hours    = st.slider(t(lang, "manual_hours"),   50,   500, 320)
    automation_rate = st.slider(t(lang, "automation_pct"), 50,    95,  86)
    hour_rate       = st.slider(t(lang, "hour_rate"),       8,    30,  12)

    st.markdown(t(lang, "errors_section"))
    error_before   = st.slider(t(lang, "error_before"),  1.0, 20.0,  8.5, step=0.1)
    error_after    = st.slider(t(lang, "error_after"),   0.1,  5.0,  1.2, step=0.1)
    cost_per_error = st.slider(t(lang, "cost_per_error"), 20,  500,   95)
    volume         = st.slider(t(lang, "volume"),        100, 2000,  600)

    st.markdown(t(lang, "cycle_section"))
    cycle_before = st.slider(t(lang, "cycle_before"),  5,  60, 21)
    cycle_after  = st.slider(t(lang, "cycle_after"),   1,  30,  9)
    deals        = st.slider(t(lang, "deals_month"),   5, 200, 25)
    deal_value   = st.slider(t(lang, "deal_value"),  100, 5000, 650)

    st.markdown(t(lang, "proba_section"))
    p_before = st.slider(t(lang, "p_before"), 50,  95, 74)
    p_after  = st.slider(t(lang, "p_after"),  70,  99, 96)

    st.markdown(t(lang, "invest_section"))
    impl_cost = st.slider(t(lang, "impl_cost"), 5000, 100000, 14000, step=1000)

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
st.markdown("# " + t(lang, "app_title"))
st.markdown("**{}** — {}".format(company_name, t(lang, "app_subtitle")))
st.markdown("---")

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
        connector=dict(line=dict(color="#636EFA", width=1)),
        increasing=dict(marker_color="#00CC96"),
        decreasing=dict(marker_color="#EF553B"),
        totals=dict(marker_color="#636EFA"),
        texttemplate="%{y:,.0f} €",
        textposition="outside",
    ))
    fig_wf.update_layout(showlegend=False, height=400, **CHART_LAYOUT)
    st.plotly_chart(fig_wf, use_container_width=True)

    col_p, col_t = st.columns(2)

    with col_p:
        st.subheader(t(lang, "pie_title"))
        pie_labels = [t(lang, "time_saved"), t(lang, "error_saved"),
                      t(lang, "revenue_speed"), t(lang, "revenue_conv")]
        pie_vals   = [res.time_saved_annual, res.error_reduction_annual,
                      res.revenue_impact_annual, res.markov_gain_annual]
        fig_pie = go.Figure(go.Pie(
            labels=pie_labels, values=pie_vals, hole=0.45,
            marker=dict(colors=["#00CC96", "#636EFA", "#AB63FA", "#FFA15A"]),
            textinfo="label+percent",
        ))
        fig_pie.update_layout(height=340, **CHART_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)

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
        st.dataframe(df_bd, use_container_width=True, height=300)

# ────────────────────────────────────────────────────────────────────
# TAB 2 — GRAPH
# ────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(t(lang, "graph_title"))
    st.info(t(lang, "bottleneck_info",
              node=graph_res.bottleneck_node, score=graph_res.bottleneck_score))

    nodes = list(graph_res.betweenness.keys())
    node_colors = ["#EF553B" if n == graph_res.bottleneck_node else "#636EFA" for n in nodes]
    node_sizes  = [20 + graph_res.betweenness[n] * 180 for n in nodes]

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
        line=dict(color="rgba(99,110,250,0.35)", width=1.5),
        hoverinfo="none",
    ))
    fig_g.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(color="#ffffff", size=12),
        marker=dict(color=node_colors, size=node_sizes,
                    line=dict(color="white", width=1.5)),
        hovertemplate="%{text}<br>" + t(lang, "centrality_col") + ": %{customdata:.4f}<extra></extra>",
        customdata=[graph_res.betweenness[n] for n in nodes],
    ))
    fig_g.update_layout(
        showlegend=False, height=400,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_g, use_container_width=True)

    st.subheader(t(lang, "centrality_table"))
    df_bt = pd.DataFrame(graph_res.all_nodes_ranked,
                         columns=[t(lang, "node_col"), t(lang, "centrality_col")])
    st.dataframe(df_bt, use_container_width=True)

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
    st.dataframe(df_Q.style.format("{:.4f}"), use_container_width=True)

    if N_mat is not None:
        st.markdown(t(lang, "matrix_n"))
        df_N = pd.DataFrame(N_mat, index=m_states, columns=m_states)
        st.dataframe(df_N.style.format("{:.4f}"), use_container_width=True)

    st.markdown(t(lang, "timeline_title"))
    months_range  = list(range(0, 13))
    monthly_gain  = res.total_benefit / 12
    cumulative    = [monthly_gain * m - impl_cost for m in months_range]
    breakeven_line = [0] * 13

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=months_range, y=cumulative, mode="lines+markers",
        name=t(lang, "cumulative_roi"),
        line=dict(color="#00CC96", width=2.5),
        marker=dict(size=6, color="#00CC96"),
        fill="tozeroy",
        fillcolor="rgba(0,204,150,0.1)",
    ))
    fig_tl.add_trace(go.Scatter(
        x=months_range, y=breakeven_line, mode="lines",
        name=t(lang, "breakeven"),
        line=dict(color="#EF553B", width=1.5, dash="dash"),
    ))
    fig_tl.update_layout(
        xaxis_title=t(lang, "month_label"),
        yaxis_title="EUR",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=380,
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_tl, use_container_width=True)

# ────────────────────────────────────────────────────────────────────
# TAB 4 — BAYES
# ────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(t(lang, "bayes_title"))

    b1, b2 = st.columns(2)
    with b1:
        pos_signals = st.slider(t(lang, "positive_signals"), 1, 50, 4)
    with b2:
        tot_signals = st.slider(t(lang, "total_signals"), 2, 100, 5)

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
        line=dict(color="#636EFA", width=2.5),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.15)",
    ))
    fig_b.add_trace(go.Scatter(
        x=x * 100, y=y_post, mode="lines",
        name=t(lang, "posterior_label"),
        line=dict(color="#00CC96", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,204,150,0.15)",
    ))
    fig_b.update_layout(
        xaxis_title=t(lang, "probability_pct"),
        yaxis_title=t(lang, "density"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=380,
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_b, use_container_width=True)

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
        )

    with dl2:
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
        st.download_button(
            label=t(lang, "download_pdf"),
            data=pdf_bytes,
            file_name="roi_passport_{}.pdf".format(company_name.replace(" ", "_")),
            mime="application/pdf",
        )

    linkedin_text = t(lang, "linkedin_text",
                      company=company_name,
                      roi=res.net_roi,
                      roi_pct=res.roi_pct,
                      payback=res.payback_months)
    st.text_area(t(lang, "linkedin_label"), value=linkedin_text, height=110)
