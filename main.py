import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from core.math_engine import MathEngine
from core.roi_engine import ROIEngine, ROIInput
from etl.extractor import MatrixExtractor


def run_demo():
    math = MathEngine()
    engine = ROIEngine()

    print("=" * 60)
    print("  OMNICORE ROI AUDITOR — DEMO MODE")
    print("=" * 60)

    print("\n--- [ГРАФ] graph_bottleneck ---")
    graph_res = math.graph_bottleneck([
        ("Онбординг", "KYC/AML",    0.9),
        ("KYC/AML",   "Скоринг",    0.8),
        ("Скоринг",   "Документы",  0.7),
        ("Документы", "Аудит",      0.4),
        ("KYC/AML",   "Аудит",      0.6),
    ])
    print("Узкое место   :", graph_res.bottleneck_node)
    print("Оценка        :", graph_res.bottleneck_score)
    print("Рейтинг узлов :", graph_res.all_nodes_ranked)
    print("PageRank      :", {k: round(v, 4) for k, v in graph_res.pagerank.items()})

    print("\n--- [МАРКОВ] markov_absorbing ---")
    Q = np.array([[0.10, 0.20], [0.50, 0.10]])
    times = np.array([4.0, 8.0])
    markov_res = math.markov_absorbing(
        Q=Q,
        state_times=times,
        states=["Проверка", "Доработка"],
        p_complete_before=0.74,
        p_complete_after=0.96,
    )
    print("Ожидаемое время  :", round(markov_res.expected_lead_time_hours, 2), "ч")
    print("P(завершение)    :", markov_res.p_complete)
    print("P(потеря)        :", markov_res.p_lost)
    print("Состояния        :", markov_res.states)
    print("Фундам. матрица N:")
    print(markov_res.fundamental_matrix)

    print("\n--- [БАЙЕС] bayesian_update ---")
    bayes_res = math.bayesian_update(4, 5)
    print("Априорная        :", bayes_res.prior_pct, "%")
    print("Апостериорная    :", bayes_res.posterior_pct, "%")
    print("80% ДИ           :", bayes_res.ci_80_low, "% –", bayes_res.ci_80_high, "%")

    print("\n--- [БАЙЕС] bayesian_contextual_risk ---")
    risk = math.bayesian_contextual_risk(
        prior_error_rate=0.05,
        prob_condition_given_error=0.80,
        prob_condition=0.15,
    )
    print("P(Ошибка|Условие):", round(risk, 4))

    print("\n--- [ROI] ROIEngine.calculate ---")
    inp = ROIInput(
        company_name="Marteco Digital Services",
        manual_hours_per_month=320,
        automation_rate=0.86,
        hour_rate_eur=12,
        error_rate_before_pct=8.5,
        error_rate_after_pct=1.2,
        cost_per_error_eur=95,
        monthly_volume=600,
        deal_cycle_before_days=21,
        deal_cycle_after_days=9,
        deals_per_month=25,
        avg_deal_value_eur=650,
        p_complete_before=0.74,
        p_complete_after=0.96,
        implementation_cost_eur=14000,
        positive_signals=4,
        total_signals=5,
    )
    res = engine.calculate(inp, bayes_result=bayes_res)

    print("Экономия времени  :", res.time_saved_annual, "EUR/год")
    print("Снижение ошибок   :", res.error_reduction_annual, "EUR/год")
    print("Выручка (скорость):", res.revenue_impact_annual, "EUR/год")
    print("Выручка (конверсия):", res.markov_gain_annual, "EUR/год")
    print("Суммарная выгода  :", res.total_benefit, "EUR/год")
    print("Чистый ROI        :", res.net_roi, "EUR")
    print("ROI %             :", res.roi_pct, "%")
    print("Окупаемость       :", res.payback_months, "мес.")

    print("\n--- ПАСПОРТ ---")
    print(engine.passport_text(inp, res))


def run_from_csv(filepath, company):
    math = MathEngine()
    engine = ROIEngine()
    extractor = MatrixExtractor()

    print("=" * 60)
    print("  OMNICORE ROI AUDITOR — CSV MODE")
    print("  Файл    :", filepath)
    print("  Компания:", company)
    print("=" * 60)

    process_log = extractor.from_csv(filepath)

    print("\n--- ProcessLog ---")
    print("Всего сделок     :", process_log.total_deals)
    print("Все состояния    :", process_log.states_all)
    print("Поглощающие      :", process_log.absorbing_states)
    print("Переходные       :", process_log.states_transient)
    print("Доля ошибок      :", "{:.2%}".format(process_log.error_rate))
    print("Средний цикл     :", process_log.avg_cycle_days, "дн.")
    print("Средняя сделка   :", process_log.avg_deal_value, "EUR")
    print("Матрица Q        :")
    print(process_log.matrix_Q)

    n_transient = len(process_log.states_transient)
    if n_transient > 0:
        state_times = np.array(
            [process_log.avg_time_per_state.get(s, 8.0) for s in process_log.states_transient]
        )
        try:
            markov_res = math.markov_absorbing(
                Q=process_log.matrix_Q,
                state_times=state_times,
                states=process_log.states_transient,
            )
            print("\n--- [МАРКОВ] markov_absorbing ---")
            print("Ожидаемое время  :", round(markov_res.expected_lead_time_hours, 2), "ч")
            print("P(завершение)    :", markov_res.p_complete)
        except ValueError as e:
            print("\n[МАРКОВ] Ошибка:", e)

    bayes_res = math.bayesian_update(4, 5)

    print("\n--- [БАЙЕС] bayesian_update ---")
    print("Априорная        :", bayes_res.prior_pct, "%")
    print("Апостериорная    :", bayes_res.posterior_pct, "%")

    inp = ROIInput(
        company_name=company,
        manual_hours_per_month=320,
        automation_rate=0.86,
        hour_rate_eur=12,
        error_rate_before_pct=process_log.error_rate * 100,
        error_rate_after_pct=max(process_log.error_rate * 100 * 0.15, 0.1),
        cost_per_error_eur=95,
        monthly_volume=max(process_log.total_deals, 1),
        deal_cycle_before_days=process_log.avg_cycle_days if process_log.avg_cycle_days > 0 else 21,
        deal_cycle_after_days=max(process_log.avg_cycle_days * 0.43, 1),
        deals_per_month=max(process_log.total_deals, 1),
        avg_deal_value_eur=process_log.avg_deal_value if process_log.avg_deal_value > 0 else 650,
        p_complete_before=0.74,
        p_complete_after=0.96,
        implementation_cost_eur=14000,
        positive_signals=4,
        total_signals=5,
    )
    res = engine.calculate(inp, bayes_result=bayes_res)

    print("\n--- [ROI] ---")
    print("Чистый ROI  :", res.net_roi, "EUR")
    print("ROI %       :", res.roi_pct, "%")
    print("Окупаемость :", res.payback_months, "мес.")

    print("\n--- ПАСПОРТ ---")
    print(engine.passport_text(inp, res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniCore ROI Auditor CLI")
    parser.add_argument("--input",   type=str,  default=None,            help="Путь к CSV-файлу сделок")
    parser.add_argument("--company", type=str,  default="Demo Company",  help="Название компании")
    parser.add_argument("--demo",    action="store_true",                 help="Запустить демо-режим")
    args = parser.parse_args()

    if args.demo or args.input is None:
        run_demo()
    else:
        run_from_csv(args.input, args.company)
