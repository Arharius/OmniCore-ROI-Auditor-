from dataclasses import dataclass, field
from datetime import date
from core.math_engine import MathEngine
import numpy as np


@dataclass
class ROIInput:
    company_name: str
    manual_hours_per_month: float
    automation_rate: float
    hour_rate_eur: float
    error_rate_before_pct: float
    error_rate_after_pct: float
    cost_per_error_eur: float
    monthly_volume: int
    deal_cycle_before_days: float
    deal_cycle_after_days: float
    deals_per_month: int
    avg_deal_value_eur: float
    p_complete_before: float
    p_complete_after: float
    implementation_cost_eur: float
    positive_signals: int = 4
    total_signals: int = 5


@dataclass
class ROIResult:
    time_saved_annual: float
    error_reduction_annual: float
    revenue_impact_annual: float
    markov_gain_annual: float
    total_benefit: float
    net_roi: float
    roi_pct: float
    payback_months: float
    bayesian_prior_pct: float
    bayesian_posterior_pct: float
    bayesian_ci: str


class ROIEngine:
    """Движок расчёта ROI для оценки эффекта автоматизации."""

    def calculate(self, inp: ROIInput, bayes_result=None) -> ROIResult:
        """
        Вычисляет полный ROI на основе входных параметров.

        Параметры:
            inp: ROIInput — входные данные компании
            bayes_result: результат байесовского обновления (опционально)

        Возвращает:
            ROIResult с полным набором финансовых метрик.
        """
        time_saved = inp.manual_hours_per_month * inp.automation_rate * 12 * inp.hour_rate_eur
        error_saved = ((inp.error_rate_before_pct - inp.error_rate_after_pct) / 100) * inp.monthly_volume * 12 * inp.cost_per_error_eur
        velocity = (inp.deal_cycle_before_days - inp.deal_cycle_after_days) / inp.deal_cycle_before_days
        revenue_impact = inp.deals_per_month * velocity * 0.30 * 12 * inp.avg_deal_value_eur
        markov_gain = inp.deals_per_month * ((1 - inp.p_complete_before) - (1 - inp.p_complete_after)) * inp.avg_deal_value_eur * 12
        total = time_saved + error_saved + revenue_impact + markov_gain
        net_roi = total - inp.implementation_cost_eur
        roi_pct = (net_roi / inp.implementation_cost_eur) * 100
        payback = inp.implementation_cost_eur / (total / 12)

        if bayes_result is None:
            engine = MathEngine()
            bayes_result = engine.bayesian_update(inp.positive_signals, inp.total_signals)

        bayesian_ci = f"{bayes_result.ci_80_low}%–{bayes_result.ci_80_high}%"

        return ROIResult(
            time_saved_annual=round(time_saved, 2),
            error_reduction_annual=round(error_saved, 2),
            revenue_impact_annual=round(revenue_impact, 2),
            markov_gain_annual=round(markov_gain, 2),
            total_benefit=round(total, 2),
            net_roi=round(net_roi, 2),
            roi_pct=round(roi_pct, 2),
            payback_months=round(payback, 1),
            bayesian_prior_pct=bayes_result.prior_pct,
            bayesian_posterior_pct=bayes_result.posterior_pct,
            bayesian_ci=bayesian_ci,
        )

    def passport_text(self, inp: ROIInput, res: ROIResult) -> str:
        """
        Генерирует ASCII-паспорт ROI с ключевыми метриками.

        Параметры:
            inp: ROIInput — входные данные компании
            res: ROIResult — результаты расчёта

        Возвращает:
            str: форматированный текстовый паспорт ROI.
        """
        today = date.today().strftime("%d.%m.%Y")
        lines = [
            "=" * 60,
            "        ПАСПОРТ ROI — OMNICORE AUDITOR",
            "=" * 60,
            f"  Компания   : {inp.company_name}",
            f"  Дата       : {today}",
            "-" * 60,
            "  КЛЮЧЕВЫЕ МЕТРИКИ",
            "-" * 60,
            f"  [ГРАФ]   Экономия времени (год)    : {res.time_saved_annual:>12,.2f} EUR",
            f"  [ГРАФ]   Снижение ошибок (год)     : {res.error_reduction_annual:>12,.2f} EUR",
            f"  [МАРКОВ] Выручка — скорость сделок : {res.revenue_impact_annual:>12,.2f} EUR",
            f"  [МАРКОВ] Выручка — конверсия       : {res.markov_gain_annual:>12,.2f} EUR",
            f"  [БАЙЕС]  Доверие до обновления     : {res.bayesian_prior_pct:>11.1f}%",
            f"  [БАЙЕС]  Доверие после обновления  : {res.bayesian_posterior_pct:>11.1f}%",
            f"  [БАЙЕС]  80% ДИ                    : {res.bayesian_ci:>12}",
            "-" * 60,
            f"  Суммарная выгода (год)    : {res.total_benefit:>16,.2f} EUR",
            f"  Чистый ROI                : {res.net_roi:>16,.2f} EUR",
            f"  ROI %                     : {res.roi_pct:>15.2f}%",
            f"  Срок окупаемости          : {res.payback_months:>12.1f} мес.",
            "=" * 60,
            "  Следующий шаг: передать отчёт команде внедрения",
            "  и согласовать план-график автоматизации.",
            "-" * 60,
            "  Аудитор: OmniCore ROI Engine v1.0",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    inp = ROIInput(
        company_name="Marteco Digital",
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

    engine = ROIEngine()
    result = engine.calculate(inp)

    print(engine.passport_text(inp, result))
    print()
    print(f"ROI %          : {result.roi_pct}%")
    print(f"Payback        : {result.payback_months} мес.")
    print(f"Bayesian CI    : {result.bayesian_ci}")
