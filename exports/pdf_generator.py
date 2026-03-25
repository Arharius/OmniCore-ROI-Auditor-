import io
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

DARK_BG = HexColor("#0e1117")
ACCENT  = HexColor("#636EFA")
GREEN   = HexColor("#00CC96")
PURPLE  = HexColor("#AB63FA")
GRAY_D  = HexColor("#1e2130")
WHITE   = HexColor("#FFFFFF")
LIGHT   = HexColor("#cccccc")
YELLOW  = HexColor("#FFA15A")


def _style(name, **kw):
    base = dict(
        fontName="Helvetica",
        fontSize=9,
        textColor=WHITE,
        leading=13,
        alignment=TA_LEFT,
    )
    base.update(kw)
    return ParagraphStyle(name, **base)


S_CENTER   = _style("sc", alignment=TA_CENTER)
S_LEFT     = _style("sl")
S_BOLD_C   = _style("sbc", fontName="Helvetica-Bold", alignment=TA_CENTER)
S_BOLD_L   = _style("sbl", fontName="Helvetica-Bold")
S_HERO     = _style("sh", fontName="Helvetica-Bold", fontSize=22, textColor=GREEN, alignment=TA_CENTER)
S_HERO_SUB = _style("shs", fontSize=8, textColor=LIGHT, alignment=TA_CENTER)
S_GREEN    = _style("sg", fontName="Helvetica-Bold", textColor=GREEN, alignment=TA_RIGHT)
S_ACCENT   = _style("sa", fontName="Helvetica-Bold", textColor=ACCENT)
S_PURPLE   = _style("sp", fontName="Helvetica-Bold", textColor=PURPLE)
S_SMALL_C  = _style("smc", fontSize=7, textColor=LIGHT, alignment=TA_CENTER)
S_LARGE_G  = _style("slg", fontName="Helvetica-Bold", fontSize=14, textColor=GREEN, alignment=TA_CENTER)


def _p(text, style=None):
    if style is None:
        style = S_LEFT
    return Paragraph(str(text), style)


def _ts_base(bg=GRAY_D, grid=ACCENT):
    return [
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("TEXTCOLOR",  (0, 0), (-1, -1), WHITE),
        ("GRID",       (0, 0), (-1, -1), 0.4, grid),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [bg, DARK_BG]),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]


def build_roi_passport_pdf(
    company_name,
    auditor_name,
    time_saved,
    error_reduction,
    revenue_impact,
    markov_gain,
    implementation_cost,
    manual_hours_before,
    automation_rate_pct,
    error_rate_before,
    error_rate_after,
    deal_cycle_before,
    deal_cycle_after,
    p_complete_before_pct,
    p_complete_after_pct,
    bayes_prior,
    bayes_posterior,
    bayes_ci,
    bottleneck_node,
    bottleneck_score,
    net_roi,
    roi_pct,
    payback_months,
    success_fee_threshold=50000,
    success_fee_pct=10,
):
    """Build a dark-themed ROI Passport PDF and return its bytes."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )
    W = A4[0] - 24 * mm
    story = []

    today_str = date.today().strftime("%d.%m.%Y")
    total_benefit = time_saved + error_reduction + revenue_impact + markov_gain
    year1_fee = max(0.0, (net_roi - success_fee_threshold) * (success_fee_pct / 100.0))

    # ── 1. HEADER ──────────────────────────────────────────────────────────────
    hdr_data = [[
        _p("AI PRODUCT AUDIT", S_BOLD_C),
        _p("ROI PASSPORT", S_BOLD_C),
        _p(company_name, S_BOLD_C),
        _p(today_str, S_CENTER),
    ]]
    hdr_ts = _ts_base(bg=DARK_BG, grid=ACCENT) + [
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BG),
        ("LINEBELOW",  (0, 0), (-1, 0), 1.5, ACCENT),
        ("FONTSIZE",   (1, 0), (1, 0), 12),
        ("TEXTCOLOR",  (1, 0), (1, 0), ACCENT),
        ("TEXTCOLOR",  (2, 0), (2, 0), GREEN),
    ]
    story.append(Table(hdr_data, colWidths=[W * 0.25] * 4,
                       style=TableStyle(hdr_ts)))
    story.append(Spacer(1, 4 * mm))

    # ── 2. ROI HERO ────────────────────────────────────────────────────────────
    hero_data = [
        [
            _p("{:,.0f} EUR".format(net_roi), S_HERO),
            _p("{:.1f}%".format(roi_pct), S_HERO),
            _p("{:.1f} мес.".format(payback_months), S_HERO),
        ],
        [
            _p("Чистый ROI", S_HERO_SUB),
            _p("ROI %", S_HERO_SUB),
            _p("Окупаемость", S_HERO_SUB),
        ],
    ]
    hero_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND",  (0, 0), (-1, -1), GRAY_D),
        ("BOX",         (0, 0), (-1, -1), 1.5, ACCENT),
        ("LINEABOVE",   (0, 0), (-1, 0), 2.0, GREEN),
    ]
    story.append(Table(hero_data, colWidths=[W / 3] * 3,
                       style=TableStyle(hero_ts)))
    story.append(Spacer(1, 4 * mm))

    # ── 3. 4-METRICS TABLE ────────────────────────────────────────────────────
    m_hdr = [
        _p("Компонент", S_BOLD_C),
        _p("ДО", S_BOLD_C),
        _p("ПОСЛЕ", S_BOLD_C),
        _p("EUR / год", S_BOLD_C),
        _p("Источник", S_BOLD_C),
    ]
    m_rows = [
        [
            _p("Трудозатраты", S_LEFT),
            _p("{} ч/мес".format(manual_hours_before), S_CENTER),
            _p("{:.0f} ч/мес".format(manual_hours_before * (1 - automation_rate_pct / 100)), S_CENTER),
            _p("{:,.0f}".format(time_saved), S_GREEN),
            _p("[ГРАФ]", _style("src", textColor=ACCENT, alignment=TA_CENTER)),
        ],
        [
            _p("Ошибки процесса", S_LEFT),
            _p("{:.1f}%".format(error_rate_before), S_CENTER),
            _p("{:.1f}%".format(error_rate_after), S_CENTER),
            _p("{:,.0f}".format(error_reduction), S_GREEN),
            _p("[ГРАФ]", _style("src2", textColor=ACCENT, alignment=TA_CENTER)),
        ],
        [
            _p("Скорость сделок", S_LEFT),
            _p("{:.0f} дн.".format(deal_cycle_before), S_CENTER),
            _p("{:.0f} дн.".format(deal_cycle_after), S_CENTER),
            _p("{:,.0f}".format(revenue_impact), S_GREEN),
            _p("[МАРКОВ]", _style("srm", textColor=PURPLE, alignment=TA_CENTER)),
        ],
        [
            _p("Конверсия сделок", S_LEFT),
            _p("{:.0f}%".format(p_complete_before_pct), S_CENTER),
            _p("{:.0f}%".format(p_complete_after_pct), S_CENTER),
            _p("{:,.0f}".format(markov_gain), S_GREEN),
            _p("[МАРКОВ]", _style("srm2", textColor=PURPLE, alignment=TA_CENTER)),
        ],
    ]
    m_data = [m_hdr] + m_rows
    m_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BG),
        ("LINEBELOW",   (0, 0), (-1, 0), 1.0, ACCENT),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ]
    story.append(Table(m_data, colWidths=[W * 0.28, W * 0.15, W * 0.14, W * 0.18, W * 0.25],
                       style=TableStyle(m_ts)))
    story.append(Spacer(1, 4 * mm))

    # ── 4. FINANCIAL MODEL ────────────────────────────────────────────────────
    fin_label = _style("fl", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE)
    fin_val   = _style("fv", fontName="Helvetica-Bold", fontSize=10, textColor=GREEN, alignment=TA_RIGHT)
    fin_neg   = _style("fn", fontName="Helvetica-Bold", fontSize=10, textColor=HexColor("#FF6692"), alignment=TA_RIGHT)
    fin_net_l = _style("fnl", fontName="Helvetica-Bold", fontSize=13, textColor=GREEN)
    fin_net_v = _style("fnv", fontName="Helvetica-Bold", fontSize=13, textColor=GREEN, alignment=TA_RIGHT)

    fin_data = [
        [_p("+ Экономия времени",      fin_label), _p("{:,.0f} EUR".format(time_saved),      fin_val)],
        [_p("+ Снижение ошибок",       fin_label), _p("{:,.0f} EUR".format(error_reduction),  fin_val)],
        [_p("+ Выручка (скорость)",    fin_label), _p("{:,.0f} EUR".format(revenue_impact),   fin_val)],
        [_p("+ Выручка (конверсия)",   fin_label), _p("{:,.0f} EUR".format(markov_gain),      fin_val)],
        [_p("= Суммарная выгода",      fin_label), _p("{:,.0f} EUR".format(total_benefit),    fin_val)],
        [_p("− Инвестиции",            fin_label), _p("{:,.0f} EUR".format(implementation_cost), fin_neg)],
        [_p("= ЧИСТЫЙ ROI",            fin_net_l), _p("{:,.0f} EUR".format(net_roi),          fin_net_v)],
    ]
    fin_ts = _ts_base(bg=GRAY_D, grid=GRAY_D) + [
        ("BACKGROUND",  (0, 4), (-1, 4), DARK_BG),
        ("LINEABOVE",   (0, 4), (-1, 4), 0.5, ACCENT),
        ("BACKGROUND",  (0, 6), (-1, 6), DARK_BG),
        ("BOX",         (0, 6), (-1, 6), 1.5, GREEN),
        ("LINEABOVE",   (0, 6), (-1, 6), 1.5, GREEN),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [GRAY_D, DARK_BG]),
    ]
    story.append(Table(fin_data, colWidths=[W * 0.6, W * 0.4],
                       style=TableStyle(fin_ts)))
    story.append(Spacer(1, 4 * mm))

    # ── 5. MATH MODELS ────────────────────────────────────────────────────────
    math_hdr = [
        _p("Модель", S_BOLD_C),
        _p("Параметр", S_BOLD_C),
        _p("Результат", S_BOLD_C),
        _p("Интерпретация", S_BOLD_C),
    ]
    math_rows = [
        [
            _p("[ГРАФ]", _style("mg", textColor=ACCENT, fontName="Helvetica-Bold")),
            _p("Узкое место", S_LEFT),
            _p("{} / {:.4f}".format(bottleneck_node, bottleneck_score), S_CENTER),
            _p("Приоритет оптимизации процесса", S_LEFT),
        ],
        [
            _p("[МАРКОВ]", _style("mm", textColor=PURPLE, fontName="Helvetica-Bold")),
            _p("P(завершение)", S_LEFT),
            _p("{:.0f}% → {:.0f}%".format(p_complete_before_pct, p_complete_after_pct), S_CENTER),
            _p("Рост конверсии через цепи Маркова", S_LEFT),
        ],
        [
            _p("[БАЙЕС]", _style("mb", textColor=GREEN, fontName="Helvetica-Bold")),
            _p("Доверие к результату", S_LEFT),
            _p("{:.1f}% → {:.1f}%".format(bayes_prior, bayes_posterior), S_CENTER),
            _p("80% ДИ: {}".format(bayes_ci), S_LEFT),
        ],
    ]
    math_data = [math_hdr] + math_rows
    math_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BG),
        ("LINEBELOW",   (0, 0), (-1, 0), 1.0, ACCENT),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ]
    story.append(Table(math_data, colWidths=[W * 0.12, W * 0.20, W * 0.22, W * 0.46],
                       style=TableStyle(math_ts)))
    story.append(Spacer(1, 4 * mm))

    # ── 6. SUCCESS FEE ────────────────────────────────────────────────────────
    fee_data = [
        [
            _p("Формат оплаты", S_BOLD_C),
            _p("Порог", S_BOLD_C),
            _p("Доля", S_BOLD_C),
            _p("Оценка Year-1", S_BOLD_C),
        ],
        [
            _p("Подписка + Success Fee", S_CENTER),
            _p("{:,.0f} EUR".format(success_fee_threshold), S_CENTER),
            _p("{}%".format(success_fee_pct), S_CENTER),
            _p("{:,.0f} EUR".format(year1_fee), _style("sfy", textColor=YELLOW, fontName="Helvetica-Bold", alignment=TA_CENTER)),
        ],
    ]
    fee_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BG),
        ("LINEBELOW",   (0, 0), (-1, 0), 1.0, ACCENT),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX",         (0, 0), (-1, -1), 1.0, PURPLE),
    ]
    story.append(Table(fee_data, colWidths=[W * 0.35, W * 0.20, W * 0.15, W * 0.30],
                       style=TableStyle(fee_ts)))
    story.append(Spacer(1, 4 * mm))

    # ── 7. FOOTER ─────────────────────────────────────────────────────────────
    foot_data = [
        [_p("Следующий шаг:", S_BOLD_L), _p("Передать отчёт команде внедрения и согласовать план-график автоматизации.", S_LEFT)],
        [_p("Аудитор:", S_BOLD_L),       _p("{} | OmniCore ROI Auditor v1.0 | {}".format(auditor_name, today_str), S_LEFT)],
    ]
    foot_ts = _ts_base(bg=DARK_BG, grid=ACCENT) + [
        ("BACKGROUND",  (0, 0), (-1, -1), DARK_BG),
        ("LINEABOVE",   (0, 0), (-1, 0), 1.0, ACCENT),
        ("TEXTCOLOR",   (0, 0), (0, -1), ACCENT),
    ]
    story.append(Table(foot_data, colWidths=[W * 0.18, W * 0.82],
                       style=TableStyle(foot_ts)))

    doc.build(story)
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    pdf_bytes = build_roi_passport_pdf(
        company_name="Marteco Digital",
        auditor_name="OmniCore Team",
        time_saved=39628.80,
        error_reduction=49932.00,
        revenue_impact=33428.57,
        markov_gain=42900.00,
        implementation_cost=14000.0,
        manual_hours_before=320,
        automation_rate_pct=86,
        error_rate_before=8.5,
        error_rate_after=1.2,
        deal_cycle_before=21,
        deal_cycle_after=9,
        p_complete_before_pct=74,
        p_complete_after_pct=96,
        bayes_prior=34.0,
        bayes_posterior=49.3,
        bayes_ci="33.0%-65.7%",
        bottleneck_node="Proposal",
        bottleneck_score=0.3333,
        net_roi=151889.37,
        roi_pct=1084.92,
        payback_months=1.0,
        success_fee_threshold=50000,
        success_fee_pct=10,
    )

    with open("test_passport.pdf", "wb") as f:
        f.write(pdf_bytes)

    print("Saved: test_passport.pdf ({} bytes)".format(len(pdf_bytes)))
