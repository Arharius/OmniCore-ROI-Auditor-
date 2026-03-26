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
    pipeline_utilization_pct: float = 30.0


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
        velocity = (inp.deal_cycle_before_days - inp.deal_cycle_after_days) / inp.deal_cycle_before_days if inp.deal_cycle_before_days else 0.0
        pipeline_util = getattr(inp, "pipeline_utilization_pct", 30.0) / 100.0
        revenue_impact = inp.deals_per_month * velocity * pipeline_util * 12 * inp.avg_deal_value_eur
        markov_gain = inp.deals_per_month * ((1 - inp.p_complete_before) - (1 - inp.p_complete_after)) * inp.avg_deal_value_eur * 12
        total = time_saved + error_saved + revenue_impact + markov_gain
        net_roi = total - inp.implementation_cost_eur
        roi_pct = (net_roi / inp.implementation_cost_eur) * 100 if inp.implementation_cost_eur else 0.0
        payback = inp.implementation_cost_eur / (total / 12) if total else 0.0

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

    def passport_text(self, inp: ROIInput, res: ROIResult,
                      currency_sym: str = "EUR", currency_rate: float = 1.0,
                      lang: str = "ru",
                      auditor_name: str = "") -> str:
        """
        Generates an ASCII ROI passport in the selected language.

        Args:
            inp: ROIInput
            res: ROIResult
            currency_sym: currency symbol (€, ₽, RSD …)
            currency_rate: multiplier from EUR
            lang: "en" | "ru" | "sr"
        """
        _L = {
            "en": {
                "title":       "ROI PASSPORT — OMNICORE AUDITOR",
                "company":     "Company  ",
                "date":        "Date     ",
                "metrics":     "KEY METRICS",
                "time_saved":  "[GRAPH]   Time savings (year)        ",
                "error_saved": "[GRAPH]   Error reduction (year)     ",
                "rev_speed":   "[MARKOV]  Revenue — deal speed       ",
                "rev_conv":    "[MARKOV]  Revenue — conversion       ",
                "b_prior":     "[BAYES]   Confidence (prior)         ",
                "b_post":      "[BAYES]   Confidence (posterior)     ",
                "b_ci":        "[BAYES]   80% CI                     ",
                "total_b":     "Total benefit (year)      ",
                "net_roi":     "Net ROI                   ",
                "roi_pct":     "ROI %                     ",
                "payback":     "Payback period            ",
                "months":      "mo.",
                "next":        "Next step: pass the report to the implementation team",
                "next2":       "and agree on the automation roadmap.",
                "auditor_lbl": "Auditor",
            },
            "ru": {
                "title":       "ПАСПОРТ ROI — OMNICORE AUDITOR",
                "company":     "Компания  ",
                "date":        "Дата      ",
                "metrics":     "КЛЮЧЕВЫЕ МЕТРИКИ",
                "time_saved":  "[ГРАФ]   Экономия времени (год)    ",
                "error_saved": "[ГРАФ]   Снижение ошибок (год)     ",
                "rev_speed":   "[МАРКОВ] Выручка — скорость сделок ",
                "rev_conv":    "[МАРКОВ] Выручка — конверсия       ",
                "b_prior":     "[БАЙЕС]  Доверие до обновления     ",
                "b_post":      "[БАЙЕС]  Доверие после обновления  ",
                "b_ci":        "[БАЙЕС]  80% ДИ                    ",
                "total_b":     "Суммарная выгода (год)    ",
                "net_roi":     "Чистый ROI                ",
                "roi_pct":     "ROI %                     ",
                "payback":     "Срок окупаемости          ",
                "months":      "мес.",
                "next":        "Следующий шаг: передать отчёт команде внедрения",
                "next2":       "и согласовать план-график автоматизации.",
                "auditor_lbl": "Аудитор",
            },
            "sr": {
                "title":       "ROI PASOŠ — OMNICORE AUDITOR",
                "company":     "Kompanija ",
                "date":        "Datum     ",
                "metrics":     "KLJUČNE METRIKE",
                "time_saved":  "[GRAF]   Ušteda vremena (god.)      ",
                "error_saved": "[GRAF]   Smanjenje grešaka (god.)   ",
                "rev_speed":   "[MARKOV] Prihod — brzina poslova    ",
                "rev_conv":    "[MARKOV] Prihod — konverzija        ",
                "b_prior":     "[BAJES]  Poverenje (apriorno)       ",
                "b_post":      "[BAJES]  Poverenje (aposteriorno)   ",
                "b_ci":        "[BAJES]  80% IP                     ",
                "total_b":     "Ukupna korist (god.)      ",
                "net_roi":     "Neto ROI                  ",
                "roi_pct":     "ROI %                     ",
                "payback":     "Period povrata            ",
                "months":      "mes.",
                "next":        "Sledeći korak: proslediti izveštaj timu za implementaciju",
                "next2":       "i dogovoriti plan automatizacije.",
                "auditor_lbl": "Revizor",
            },
        }
        l = _L.get(lang, _L["en"])
        r = currency_rate
        s = currency_sym
        today = date.today().strftime("%d.%m.%Y")

        def _n(val: float) -> str:
            """Format number with locale-appropriate thousands separator."""
            if lang == "sr":
                return "{:,.0f}".format(val).replace(",", ".")
            if lang == "ru":
                return "{:,.0f}".format(val).replace(",", "\u202f")
            return "{:,.0f}".format(val)

        SEP = "=" * 64
        sep = "-" * 64
        lines = [
            SEP,
            "  {}".format(l["title"]),
            SEP,
            "  {}: {}".format(l["company"], inp.company_name),
            "  {}: {}".format(l["date"], today),
            sep,
            "  {}".format(l["metrics"]),
            sep,
            "  {}: {:>16} {}".format(l["time_saved"],  _n(res.time_saved_annual * r),      s),
            "  {}: {:>16} {}".format(l["error_saved"], _n(res.error_reduction_annual * r),  s),
            "  {}: {:>16} {}".format(l["rev_speed"],   _n(res.revenue_impact_annual * r),   s),
            "  {}: {:>16} {}".format(l["rev_conv"],    _n(res.markov_gain_annual * r),       s),
            "  {}: {:>16.1f}%".format(l["b_prior"],  res.bayesian_prior_pct),
            "  {}: {:>16.1f}%".format(l["b_post"],   res.bayesian_posterior_pct),
            "  {}: {:>17}".format(l["b_ci"],         res.bayesian_ci),
            sep,
            "  {}: {:>16} {}".format(l["total_b"], _n(res.total_benefit * r), s),
            "  {}: {:>16} {}".format(l["net_roi"], _n(res.net_roi * r),        s),
            "  {}: {:>16.2f}%".format(l["roi_pct"],   res.roi_pct),
            "  {}: {:>12.1f} {}".format(l["payback"],  res.payback_months, l["months"]),
            SEP,
            "  {}".format(l["next"]),
            "  {}".format(l["next2"]),
            sep,
            "  {}: {} | OmniCore ROI Engine v1.0 | {}".format(
                l["auditor_lbl"],
                auditor_name if auditor_name else "OmniCore",
                date.today().strftime("%d.%m.%Y"),
            ),
            SEP,
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
