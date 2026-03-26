import io
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
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
    base = dict(fontName="Helvetica", fontSize=9, textColor=WHITE, leading=13, alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


S_CENTER   = _style("sc",  alignment=TA_CENTER)
S_LEFT     = _style("sl")
S_BOLD_C   = _style("sbc", fontName="Helvetica-Bold", alignment=TA_CENTER)
S_BOLD_L   = _style("sbl", fontName="Helvetica-Bold")
S_HERO     = _style("sh",  fontName="Helvetica-Bold", fontSize=22, textColor=GREEN, alignment=TA_CENTER)
S_HERO_SUB = _style("shs", fontSize=8, textColor=LIGHT, alignment=TA_CENTER)
S_GREEN    = _style("sg",  fontName="Helvetica-Bold", textColor=GREEN, alignment=TA_RIGHT)
S_ACCENT   = _style("sa",  fontName="Helvetica-Bold", textColor=ACCENT)
S_PURPLE   = _style("sp",  fontName="Helvetica-Bold", textColor=PURPLE)
S_SMALL_C  = _style("smc", fontSize=7, textColor=LIGHT, alignment=TA_CENTER)
S_LARGE_G  = _style("slg", fontName="Helvetica-Bold", fontSize=14, textColor=GREEN, alignment=TA_CENTER)
S_SRC      = _style("src",  textColor=ACCENT,  alignment=TA_CENTER)
S_SRC2     = _style("src2", textColor=ACCENT,  alignment=TA_CENTER)
S_SRM      = _style("srm",  textColor=PURPLE,  alignment=TA_CENTER)
S_SRM2     = _style("srm2", textColor=PURPLE,  alignment=TA_CENTER)
S_FL       = _style("fl",   fontName="Helvetica-Bold", fontSize=10, textColor=WHITE)
S_FV       = _style("fv",   fontName="Helvetica-Bold", fontSize=10, textColor=GREEN, alignment=TA_RIGHT)
S_FN       = _style("fn",   fontName="Helvetica-Bold", fontSize=10, textColor=HexColor("#FF6692"), alignment=TA_RIGHT)
S_FNL      = _style("fnl",  fontName="Helvetica-Bold", fontSize=13, textColor=GREEN)
S_FNV      = _style("fnv",  fontName="Helvetica-Bold", fontSize=13, textColor=GREEN, alignment=TA_RIGHT)
S_SFY      = _style("sfy",  textColor=YELLOW,  fontName="Helvetica-Bold", alignment=TA_CENTER)
S_MG       = _style("mg",   textColor=ACCENT,  fontName="Helvetica-Bold")
S_MM       = _style("mm",   textColor=PURPLE,  fontName="Helvetica-Bold")
S_MB       = _style("mb",   textColor=GREEN,   fontName="Helvetica-Bold")
S_NOTES    = _style("nt",   fontSize=8, textColor=LIGHT, leading=12)


def _p(text, style=None):
    if style is None:
        style = S_LEFT
    return Paragraph(str(text), style)


def _ts_base(bg=GRAY_D, grid=ACCENT):
    return [
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("TEXTCOLOR",    (0, 0), (-1, -1), WHITE),
        ("GRID",         (0, 0), (-1, -1), 0.4, grid),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [bg, DARK_BG]),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]


def _make_qr(url: str, size_cm: float = 2.8):
    try:
        import qrcode as _qr
        qr = _qr.QRCode(
            version=1,
            error_correction=_qr.constants.ERROR_CORRECT_L,
            box_size=4, border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="white", back_color="#0e1117")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        s = size_cm * cm
        return RLImage(buf, width=s, height=s)
    except Exception:
        return None


def build_roi_passport_pdf(
    company_name,
    auditor_name="Andrew | AI Product Advisor",
    contact_url="https://t.me/weerowoolf",
    meeting_notes="",
    time_saved=0,
    error_reduction=0,
    revenue_impact=0,
    markov_gain=0,
    implementation_cost=0,
    manual_hours_before=0,
    automation_rate_pct=0,
    error_rate_before=0,
    error_rate_after=0,
    deal_cycle_before=0,
    deal_cycle_after=0,
    p_complete_before_pct=0,
    p_complete_after_pct=0,
    bayes_prior=0,
    bayes_posterior=0,
    bayes_ci="",
    bottleneck_node="",
    bottleneck_score=0,
    net_roi=0,
    roi_pct=0,
    payback_months=0,
    success_fee_threshold=50000,
    success_fee_pct=10,
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=10*mm, bottomMargin=10*mm,
    )
    W = A4[0] - 24*mm
    story = []

    today_str   = date.today().strftime("%d.%m.%Y")
    total_benefit = time_saved + error_reduction + revenue_impact + markov_gain
    year1_fee   = max(0.0, (net_roi - success_fee_threshold) * (success_fee_pct / 100.0))

    # ── 1. HEADER ──────────────────────────────────────────────────────────────
    hdr_data = [[
        _p("AI PRODUCT AUDIT", S_BOLD_C),
        _p("ROI PASSPORT",     S_BOLD_C),
        _p(company_name,       S_BOLD_C),
        _p(today_str,          S_CENTER),
    ]]
    hdr_ts = _ts_base(bg=DARK_BG, grid=ACCENT) + [
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BG),
        ("LINEBELOW",  (0, 0), (-1, 0), 1.5, ACCENT),
        ("FONTSIZE",   (1, 0), (1, 0), 12),
        ("TEXTCOLOR",  (1, 0), (1, 0), ACCENT),
        ("TEXTCOLOR",  (2, 0), (2, 0), GREEN),
    ]
    story.append(Table(hdr_data, colWidths=[W*0.25]*4, style=TableStyle(hdr_ts)))
    story.append(Spacer(1, 4*mm))

    # ── 2. ROI HERO ────────────────────────────────────────────────────────────
    hero_data = [
        [_p("{:,.0f} EUR".format(net_roi), S_HERO),
         _p("{:.1f}%".format(roi_pct), S_HERO),
         _p("{:.1f} мес.".format(payback_months), S_HERO)],
        [_p("Чистый ROI", S_HERO_SUB),
         _p("ROI %",      S_HERO_SUB),
         _p("Окупаемость",S_HERO_SUB)],
    ]
    hero_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND", (0, 0), (-1, -1), GRAY_D),
        ("BOX",        (0, 0), (-1, -1), 1.5, ACCENT),
        ("LINEABOVE",  (0, 0), (-1, 0),  2.0, GREEN),
    ]
    story.append(Table(hero_data, colWidths=[W/3]*3, style=TableStyle(hero_ts)))
    story.append(Spacer(1, 4*mm))

    # ── 3. 4-METRICS TABLE ─────────────────────────────────────────────────────
    m_hdr = [_p("Компонент", S_BOLD_C), _p("ДО", S_BOLD_C), _p("ПОСЛЕ", S_BOLD_C),
             _p("EUR / год",  S_BOLD_C), _p("Источник", S_BOLD_C)]
    m_rows = [
        [_p("Трудозатраты",    S_LEFT),
         _p("{} ч/мес".format(manual_hours_before), S_CENTER),
         _p("{:.0f} ч/мес".format(manual_hours_before*(1-automation_rate_pct/100)), S_CENTER),
         _p("{:,.0f}".format(time_saved),      S_GREEN), _p("[ГРАФ]", S_SRC)],
        [_p("Ошибки процесса", S_LEFT),
         _p("{:.1f}%".format(error_rate_before), S_CENTER),
         _p("{:.1f}%".format(error_rate_after),  S_CENTER),
         _p("{:,.0f}".format(error_reduction),   S_GREEN), _p("[ГРАФ]", S_SRC2)],
        [_p("Скорость сделок", S_LEFT),
         _p("{:.0f} дн.".format(deal_cycle_before), S_CENTER),
         _p("{:.0f} дн.".format(deal_cycle_after),  S_CENTER),
         _p("{:,.0f}".format(revenue_impact),   S_GREEN), _p("[МАРКОВ]", S_SRM)],
        [_p("Конверсия сделок",S_LEFT),
         _p("{:.0f}%".format(p_complete_before_pct), S_CENTER),
         _p("{:.0f}%".format(p_complete_after_pct),  S_CENTER),
         _p("{:,.0f}".format(markov_gain),      S_GREEN), _p("[МАРКОВ]", S_SRM2)],
    ]
    m_data = [m_hdr] + m_rows
    m_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BG),
        ("LINEBELOW",  (0, 0), (-1, 0), 1.0, ACCENT),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
    ]
    story.append(Table(m_data, colWidths=[W*0.28, W*0.15, W*0.14, W*0.18, W*0.25],
                       style=TableStyle(m_ts)))
    story.append(Spacer(1, 4*mm))

    # ── 4. FINANCIAL MODEL ─────────────────────────────────────────────────────
    fin_data = [
        [_p("+ Экономия времени",    S_FL), _p("{:,.0f} EUR".format(time_saved),           S_FV)],
        [_p("+ Снижение ошибок",     S_FL), _p("{:,.0f} EUR".format(error_reduction),       S_FV)],
        [_p("+ Выручка (скорость)",  S_FL), _p("{:,.0f} EUR".format(revenue_impact),        S_FV)],
        [_p("+ Выручка (конверсия)", S_FL), _p("{:,.0f} EUR".format(markov_gain),           S_FV)],
        [_p("= Суммарная выгода",    S_FL), _p("{:,.0f} EUR".format(total_benefit),         S_FV)],
        [_p("− Инвестиции",          S_FL), _p("{:,.0f} EUR".format(implementation_cost),   S_FN)],
        [_p("= ЧИСТЫЙ ROI",          S_FNL),_p("{:,.0f} EUR".format(net_roi),               S_FNV)],
    ]
    fin_ts = _ts_base(bg=GRAY_D, grid=GRAY_D) + [
        ("BACKGROUND",    (0, 4), (-1, 4), DARK_BG),
        ("LINEABOVE",     (0, 4), (-1, 4), 0.5, ACCENT),
        ("BACKGROUND",    (0, 6), (-1, 6), DARK_BG),
        ("BOX",           (0, 6), (-1, 6), 1.5, GREEN),
        ("LINEABOVE",     (0, 6), (-1, 6), 1.5, GREEN),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [GRAY_D, DARK_BG]),
    ]
    story.append(Table(fin_data, colWidths=[W*0.6, W*0.4], style=TableStyle(fin_ts)))
    story.append(Spacer(1, 4*mm))

    # ── 5. MATH MODELS ─────────────────────────────────────────────────────────
    math_hdr = [_p("Модель", S_BOLD_C), _p("Параметр", S_BOLD_C),
                _p("Результат", S_BOLD_C), _p("Интерпретация", S_BOLD_C)]
    math_rows = [
        [_p("[ГРАФ]",    S_MG), _p("Узкое место", S_LEFT),
         _p("{} / {:.4f}".format(bottleneck_node, bottleneck_score), S_CENTER),
         _p("Приоритет оптимизации процесса", S_LEFT)],
        [_p("[МАРКОВ]",  S_MM), _p("P(завершение)", S_LEFT),
         _p("{:.0f}% → {:.0f}%".format(p_complete_before_pct, p_complete_after_pct), S_CENTER),
         _p("Рост конверсии через цепи Маркова", S_LEFT)],
        [_p("[БАЙЕС]",   S_MB), _p("Доверие к результату", S_LEFT),
         _p("{:.1f}% → {:.1f}%".format(bayes_prior, bayes_posterior), S_CENTER),
         _p("80% ДИ: {}".format(bayes_ci), S_LEFT)],
    ]
    math_data = [math_hdr] + math_rows
    math_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BG),
        ("LINEBELOW",  (0, 0), (-1, 0), 1.0, ACCENT),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
    ]
    story.append(Table(math_data, colWidths=[W*0.12, W*0.20, W*0.22, W*0.46],
                       style=TableStyle(math_ts)))
    story.append(Spacer(1, 4*mm))

    # ── 6. SUCCESS FEE ─────────────────────────────────────────────────────────
    fee_data = [
        [_p("Формат оплаты", S_BOLD_C), _p("Порог", S_BOLD_C),
         _p("Доля", S_BOLD_C), _p("Оценка Year-1", S_BOLD_C)],
        [_p("Подписка + Success Fee", S_CENTER),
         _p("{:,.0f} EUR".format(success_fee_threshold), S_CENTER),
         _p("{}%".format(success_fee_pct), S_CENTER),
         _p("{:,.0f} EUR".format(year1_fee), S_SFY)],
    ]
    fee_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BG),
        ("LINEBELOW",  (0, 0), (-1, 0), 1.0, ACCENT),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX",        (0, 0), (-1, -1), 1.0, PURPLE),
    ]
    story.append(Table(fee_data, colWidths=[W*0.35, W*0.20, W*0.15, W*0.30],
                       style=TableStyle(fee_ts)))
    story.append(Spacer(1, 4*mm))

    # ── 7. MEETING NOTES ───────────────────────────────────────────────────────
    if meeting_notes and meeting_notes.strip():
        notes_data = [
            [_p("Заметки встречи", S_BOLD_L)],
            [_p(meeting_notes.strip(), S_NOTES)],
        ]
        notes_ts = _ts_base(bg=DARK_BG, grid=ACCENT) + [
            ("BACKGROUND",  (0, 0), (-1, 0), HexColor("#1a1f2e")),
            ("LINEBELOW",   (0, 0), (-1, 0), 0.8, ACCENT),
            ("TEXTCOLOR",   (0, 0), (-1, 0), ACCENT),
            ("TOPPADDING",  (0, 1), (-1, 1), 8),
            ("BOTTOMPADDING",(0, 1), (-1, 1), 8),
        ]
        story.append(Table(notes_data, colWidths=[W], style=TableStyle(notes_ts)))
        story.append(Spacer(1, 4*mm))

    # ── 8. FOOTER + QR ─────────────────────────────────────────────────────────
    qr_img = _make_qr(contact_url, size_cm=2.5) if contact_url else None

    footer_text = [
        [_p("Следующий шаг:", S_BOLD_L),
         _p("Передать отчёт команде внедрения и согласовать план-график автоматизации.", S_LEFT)],
        [_p("Аудитор:", S_BOLD_L),
         _p("{} | OmniCore ROI Auditor v1.0 | {}".format(auditor_name, today_str), S_LEFT)],
    ]
    foot_ts = _ts_base(bg=DARK_BG, grid=ACCENT) + [
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BG),
        ("LINEABOVE",  (0, 0), (-1, 0),  1.0, ACCENT),
        ("TEXTCOLOR",  (0, 0), (0, -1),  ACCENT),
    ]

    if qr_img is not None:
        qr_w = 2.5 * cm
        foot_col_w = W - qr_w - 4*mm
        foot_table = Table(
            [[Table(footer_text, colWidths=[foot_col_w*0.22, foot_col_w*0.78],
                    style=TableStyle(foot_ts)), qr_img]],
            colWidths=[foot_col_w, qr_w],
        )
        foot_outer_ts = [
            ("BACKGROUND", (0, 0), (-1, -1), DARK_BG),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",      (1, 0), (1, 0),   "CENTER"),
            ("LEFTPADDING",(0, 0), (-1, -1), 0),
            ("RIGHTPADDING",(0,0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING",(0, 0),(-1, -1),0),
        ]
        foot_table.setStyle(TableStyle(foot_outer_ts))
        story.append(foot_table)
    else:
        story.append(Table(footer_text, colWidths=[W*0.18, W*0.82],
                           style=TableStyle(foot_ts)))

    doc.build(story)
    buf.seek(0)
    return buf.read()
