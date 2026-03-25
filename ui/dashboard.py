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

st.set_page_config(
    page_title="OmniCore ROI Auditor",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] { background: #0e1117; }
div[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #636EFA44;
}
div[data-testid="metric-container"] label { color: #aaaacc; font-size: 13px; }
div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #00CC96;
    font-size: 28px;
    font-weight: 700;
}
div.stButton > button {
    background: linear-gradient(90deg, #636EFA, #00CC96);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
}
div.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="white",
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Параметры аудита")

    company_name = st.text_input("Компания", value="Marteco Digital Services")

    csv_file = st.file_uploader(
        "Загрузить CSV сделок",
        type=["csv"],
        help="Колонки: Deal_ID, Status, Timestamp, Has_Error, Deal_Value",
    )

    st.markdown("---")
    st.markdown("**Трудозатраты**")
    manual_hours   = st.slider("Ручные часы/мес",      50,   500,  320)
    automation_rate = st.slider("Автоматизация %",      50,    95,   86)
    hour_rate      = st.slider("Ставка €/ч",            8,    30,   12)

    st.markdown("**Ошибки**")
    error_before   = st.slider("Ошибки ДО %",          1.0,  20.0,  8.5, step=0.1)
    error_after    = st.slider("Ошибки ПОСЛЕ %",        0.1,   5.0,  1.2, step=0.1)
    cost_per_error = st.slider("Стоимость ошибки €",   20,   500,   95)
    volume         = st.slider("Объём сделок/мес",    100,  2000,  600)

    st.markdown("**Цикл сделки**")
    cycle_before   = st.slider("Цикл ДО (дн.)",         5,    60,   21)
    cycle_after    = st.slider("Цикл ПОСЛЕ (дн.)",       1,    30,    9)
    deals          = st.slider("Сделок/мес",             5,   200,   25)
    deal_value     = st.slider("Ср. сделка €",         100,  5000,  650)

    st.markdown("**Вероятности завершения**")
    p_before       = st.slider("P(завершение) ДО %",   50,    95,   74)
    p_after        = st.slider("P(завершение) ПОСЛЕ %", 70,   99,   96)

    st.markdown("**Инвестиции**")
    impl_cost      = st.slider("Бюджет внедрения €", 5000, 100000, 14000, step=1000)

    st.markdown("---")
    st.caption("Andrew | AI Product Advisor | Fractional TPM | Serbia/EU")

# ── COMPUTE ───────────────────────────────────────────────────────────────────
math   = MathEngine()
engine = ROIEngine()

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

bayes_res = math.bayesian_update(inp.positive_signals, inp.total_signals)
res = engine.calculate(inp, bayes_result=bayes_res)

# ── GRAPH BOTTLENECK ──────────────────────────────────────────────────────────
default_edges = [
    ("Лид", "Квалификация", 3.0),
    ("Квалификация", "Предложение", 2.0),
    ("Лид", "Предложение", 1.0),
    ("Предложение", "Закрыт", 4.0),
    ("Квалификация", "Закрыт", 1.5),
    ("Предложение", "Потерян", 2.5),
]
graph_res = math.graph_bottleneck(default_edges)

# ── MARKOV ────────────────────────────────────────────────────────────────────
if csv_file is not None:
    extractor = MatrixExtractor()
    process_log = extractor.from_csv(csv_file)
    Q_mat = process_log.matrix_Q
    m_states = process_log.states_transient
else:
    Q_mat = np.array([[0.2, 0.3], [0.1, 0.4]])
    m_states = ["Квалификация", "Предложение"]

state_times = np.array([float(cycle_before) * 24 / max(len(m_states), 1)] * len(m_states))

try:
    markov_res = math.markov_absorbing(Q_mat, state_times, m_states,
                                       p_complete_before=p_before / 100.0,
                                       p_complete_after=p_after / 100.0)
    I = np.eye(Q_mat.shape[0])
    N_mat = np.linalg.inv(I - Q_mat)
except Exception:
    markov_res = None
    N_mat = None

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 OmniCore ROI Auditor")
st.markdown("**{0}** — Аудит эффекта автоматизации".format(company_name))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Чистый ROI", "{:,.0f} €".format(res.net_roi))
c2.metric("Окупаемость", "{:.1f} мес.".format(res.payback_months))
c3.metric("Байес. доверие", "{:.1f}%".format(res.bayesian_posterior_pct))
c4.metric("Бюджет внедрения", "{:,.0f} €".format(impl_cost))

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 ROI Breakdown",
    "🕸️ Граф",
    "🔗 Маркова",
    "🎲 Байес",
    "📄 Паспорт",
])

# ── TAB 1: ROI BREAKDOWN ──────────────────────────────────────────────────────
with tab1:
    col_w, col_p = st.columns(2)

    with col_w:
        st.subheader("Waterfall ROI")
        labels = [
            "Экономия времени",
            "Снижение ошибок",
            "Скорость сделок",
            "Конверсия",
            "Внедрение",
            "Чистый ROI",
        ]
        values = [
            res.time_saved_annual,
            res.error_reduction_annual,
            res.revenue_impact_annual,
            res.markov_gain_annual,
            -impl_cost,
            res.net_roi,
        ]
        measures = ["relative", "relative", "relative", "relative", "relative", "total"]
        bar_colors = ["#00CC96", "#00CC96", "#AB63FA", "#AB63FA", "#EF553B", "#636EFA"]

        fig_wf = go.Figure(go.Waterfall(
            name="ROI",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector=dict(line=dict(color="#636EFA", width=1)),
            increasing=dict(marker_color="#00CC96"),
            decreasing=dict(marker_color="#EF553B"),
            totals=dict(marker_color="#636EFA"),
        ))
        fig_wf.update_layout(showlegend=False, **CHART_LAYOUT)
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_p:
        st.subheader("Структура выгод")
        pie_labels = ["Время", "Ошибки", "Скорость", "Конверсия"]
        pie_vals   = [
            res.time_saved_annual,
            res.error_reduction_annual,
            res.revenue_impact_annual,
            res.markov_gain_annual,
        ]
        fig_pie = go.Figure(go.Pie(
            labels=pie_labels,
            values=pie_vals,
            hole=0.4,
            marker=dict(colors=["#00CC96", "#636EFA", "#AB63FA", "#FFA15A"]),
        ))
        fig_pie.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)

    df_breakdown = pd.DataFrame({
        "Компонент": ["Время", "Ошибки", "Скорость", "Конверсия", "ИТОГО", "Инвестиции", "Чистый ROI"],
        "EUR/год":   [
            res.time_saved_annual, res.error_reduction_annual,
            res.revenue_impact_annual, res.markov_gain_annual,
            res.total_benefit, -impl_cost, res.net_roi,
        ],
    })
    st.dataframe(df_breakdown, use_container_width=True)

# ── TAB 2: ГРАФ ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Анализ узких мест процесса")
    st.info("🔴 Узкое место: **{0}** (центральность {1:.4f})".format(
        graph_res.bottleneck_node, graph_res.bottleneck_score))

    nodes = list(graph_res.betweenness.keys())
    node_colors = [
        "#EF553B" if n == graph_res.bottleneck_node else "#636EFA"
        for n in nodes
    ]
    node_sizes = [
        30 + graph_res.betweenness[n] * 200
        for n in nodes
    ]

    angle_step = 2 * np.pi / max(len(nodes), 1)
    pos = {n: (np.cos(i * angle_step), np.sin(i * angle_step))
           for i, n in enumerate(nodes)}

    edge_x, edge_y = [], []
    for frm, to, _ in default_edges:
        if frm in pos and to in pos:
            edge_x += [pos[frm][0], pos[to][0], None]
            edge_y += [pos[frm][1], pos[to][1], None]

    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(99,110,250,0.33)", width=1.5), hoverinfo="none",
    ))
    fig_g.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=nodes,
        textposition="top center",
        marker=dict(color=node_colors, size=node_sizes, line=dict(color="white", width=1)),
        hovertemplate="%{text}<br>Центральность: %{customdata:.4f}<extra></extra>",
        customdata=[graph_res.betweenness[n] for n in nodes],
    ))
    fig_g.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        **CHART_LAYOUT)
    st.plotly_chart(fig_g, use_container_width=True)

    df_bt = pd.DataFrame(graph_res.all_nodes_ranked, columns=["Узел", "Центральность"])
    st.dataframe(df_bt, use_container_width=True)

# ── TAB 3: МАРКОВА ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Поглощающая цепь Маркова")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("P(завершение) ДО",    "{:.0f}%".format(p_before))
    mc2.metric("P(завершение) ПОСЛЕ", "{:.0f}%".format(p_after))
    if markov_res:
        mc3.metric("Ожидаемое время",
                   "{:.1f} ч".format(markov_res.expected_lead_time_hours))

    st.markdown("**Матрица переходов Q**")
    df_Q = pd.DataFrame(Q_mat, index=m_states, columns=m_states)
    st.dataframe(df_Q.style.format("{:.4f}"), use_container_width=True)

    if N_mat is not None:
        st.markdown("**Фундаментальная матрица N = (I-Q)⁻¹**")
        df_N = pd.DataFrame(N_mat, index=m_states, columns=m_states)
        st.dataframe(df_N.style.format("{:.4f}"), use_container_width=True)

    st.markdown("**Динамика накопленного ROI (0–12 мес.)**")
    months = list(range(0, 13))
    monthly_benefit = res.total_benefit / 12
    cumulative = [monthly_benefit * m - impl_cost for m in months]
    breakeven_y = [0] * 13

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=months, y=cumulative, mode="lines+markers",
        name="Накопленный ROI",
        line=dict(color="#00CC96", width=2),
        marker=dict(size=6),
    ))
    fig_tl.add_trace(go.Scatter(
        x=months, y=breakeven_y, mode="lines",
        name="Точка безубыточности",
        line=dict(color="#EF553B", width=1.5, dash="dash"),
    ))
    fig_tl.update_layout(
        xaxis_title="Месяц", yaxis_title="EUR",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_tl, use_container_width=True)

# ── TAB 4: БАЙЕС ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Байесовское обновление")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        pos_signals = st.slider("Положительных сигналов", 1, 50, 4)
    with b_col2:
        tot_signals = st.slider("Всего сигналов", 2, 100, 5)

    bayes_live = math.bayesian_update(pos_signals, tot_signals)

    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Априорная",     "{:.1f}%".format(bayes_live.prior_pct))
    bc2.metric("Апостериорная", "{:.1f}%".format(bayes_live.posterior_pct))
    bc3.metric("80% ДИ",        bayes_live.ci_80_low.__str__() + "% – " + bayes_live.ci_80_high.__str__() + "%")

    prior_rate = 0.34
    alpha_pr = prior_rate * 10
    beta_pr  = (1 - prior_rate) * 10
    alpha_po = alpha_pr + pos_signals
    beta_po  = beta_pr + (tot_signals - pos_signals)

    x = np.linspace(0.01, 0.99, 300)
    y_prior = stats.beta.pdf(x, alpha_pr, beta_pr)
    y_post  = stats.beta.pdf(x, alpha_po, beta_po)

    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(
        x=x * 100, y=y_prior, mode="lines", name="Априорная",
        line=dict(color="#636EFA", width=2),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.15)",
    ))
    fig_b.add_trace(go.Scatter(
        x=x * 100, y=y_post, mode="lines", name="Апостериорная",
        line=dict(color="#00CC96", width=2),
        fill="tozeroy", fillcolor="rgba(0,204,150,0.15)",
    ))
    fig_b.update_layout(
        xaxis_title="Вероятность (%)", yaxis_title="Плотность",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_b, use_container_width=True)

    risk = math.bayesian_contextual_risk(
        prior_error_rate=0.05,
        prob_condition_given_error=0.80,
        prob_condition=0.20,
    )
    st.info("⚠️ Контекстуальный риск ошибки: **{:.2%}** (P(Ошибка|Условие) по Байесу)".format(risk))

# ── TAB 5: ПАСПОРТ ───────────────────────────────────────────────────────────
with tab5:
    st.subheader("ROI Паспорт")

    passport = engine.passport_text(inp, res)
    st.code(passport, language="")

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.download_button(
            label="⬇️ Скачать TXT",
            data=passport.encode("utf-8"),
            file_name="roi_passport_{0}.txt".format(company_name.replace(" ", "_")),
            mime="text/plain",
        )

    with p_col2:
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
            bayes_ci=bayes_res.ci_80_low.__str__() + "%-" + bayes_res.ci_80_high.__str__() + "%",
            bottleneck_node=graph_res.bottleneck_node,
            bottleneck_score=graph_res.bottleneck_score,
            net_roi=res.net_roi,
            roi_pct=res.roi_pct,
            payback_months=res.payback_months,
        )
        st.download_button(
            label="⬇️ Скачать PDF",
            data=pdf_bytes,
            file_name="roi_passport_{0}.pdf".format(company_name.replace(" ", "_")),
            mime="application/pdf",
        )

    linkedin_hook = (
        "Провёл ROI-аудит для {company}. "
        "Чистый ROI: {roi:,.0f} EUR ({roi_pct:.0f}%), окупаемость — {payback:.1f} мес. "
        "Использовал граф-анализ, цепи Маркова и байесовское обновление. "
        "Если хотите такой же разбор для своего процесса — пишите."
    ).format(
        company=company_name,
        roi=res.net_roi,
        roi_pct=res.roi_pct,
        payback=res.payback_months,
    )
    st.text_area("💼 LinkedIn Hook", value=linkedin_hook, height=100)
