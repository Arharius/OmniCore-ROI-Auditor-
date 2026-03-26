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
    currency_sym="EUR",
    currency_rate=1.0,
    lang="ru",
):
    _T = {
        "en": {
            "net_roi_lbl":   "Net ROI",
            "roi_pct_lbl":   "ROI %",
            "payback_lbl":   "Payback",
            "months":        "mo.",
            "component":     "Component",
            "before":        "BEFORE",
            "after":         "AFTER",
            "source":        "Source",
            "labor":         "Labor",
            "h_mo":          "h/mo",
            "errors":        "Process errors",
            "deal_speed":    "Deal speed",
            "days":          "d.",
            "conversion":    "Deal conversion",
            "t_time_saved":  "+ Time savings",
            "t_error_red":   "+ Error reduction",
            "t_rev_speed":   "+ Revenue (speed)",
            "t_rev_conv":    "+ Revenue (conv.)",
            "t_total":       "= Total benefit",
            "t_invest":      "\u2212 Investment",
            "t_net_roi":     "= NET ROI",
            "model":         "Model",
            "param":         "Parameter",
            "result":        "Result",
            "interpretation":"Interpretation",
            "bottleneck_p":  "Bottleneck",
            "bottleneck_i":  "Process optimization priority",
            "p_complete_p":  "P(completion)",
            "markov_i":      "Conversion growth via Markov chains",
            "bayes_p":       "Result confidence",
            "ci_label":      "80% CI: {}",
            "fee_format":    "Payment format",
            "fee_threshold": "Threshold",
            "fee_share":     "Share",
            "fee_year1":     "Year-1 estimate",
            "fee_model":     "Subscription + Success Fee",
            "notes_title":   "Meeting notes",
            "next_step":     "Next step:",
            "next_body":     "Pass the report to the implementation team and agree on the roadmap.",
            "auditor_lbl":   "Auditor:",
        },
        "ru": {
            "net_roi_lbl":   "Чистый ROI",
            "roi_pct_lbl":   "ROI %",
            "payback_lbl":   "Окупаемость",
            "months":        "мес.",
            "component":     "Компонент",
            "before":        "ДО",
            "after":         "ПОСЛЕ",
            "source":        "Источник",
            "labor":         "Трудозатраты",
            "h_mo":          "ч/мес",
            "errors":        "Ошибки процесса",
            "deal_speed":    "Скорость сделок",
            "days":          "дн.",
            "conversion":    "Конверсия сделок",
            "t_time_saved":  "+ Экономия времени",
            "t_error_red":   "+ Снижение ошибок",
            "t_rev_speed":   "+ Выручка (скорость)",
            "t_rev_conv":    "+ Выручка (конверсия)",
            "t_total":       "= Суммарная выгода",
            "t_invest":      "\u2212 Инвестиции",
            "t_net_roi":     "= ЧИСТЫЙ ROI",
            "model":         "Модель",
            "param":         "Параметр",
            "result":        "Результат",
            "interpretation":"Интерпретация",
            "bottleneck_p":  "Узкое место",
            "bottleneck_i":  "Приоритет оптимизации процесса",
            "p_complete_p":  "P(завершение)",
            "markov_i":      "Рост конверсии через цепи Маркова",
            "bayes_p":       "Доверие к результату",
            "ci_label":      "80% ДИ: {}",
            "fee_format":    "Формат оплаты",
            "fee_threshold": "Порог",
            "fee_share":     "Доля",
            "fee_year1":     "Оценка Year-1",
            "fee_model":     "Подписка + Success Fee",
            "notes_title":   "Заметки встречи",
            "next_step":     "Следующий шаг:",
            "next_body":     "Передать отчёт команде внедрения и согласовать план-график автоматизации.",
            "auditor_lbl":   "Аудитор:",
        },
        "sr": {
            "net_roi_lbl":   "Neto ROI",
            "roi_pct_lbl":   "ROI %",
            "payback_lbl":   "Povrat",
            "months":        "mes.",
            "component":     "Komponenta",
            "before":        "PRE",
            "after":         "POSLE",
            "source":        "Izvor",
            "labor":         "Radni sati",
            "h_mo":          "h/mes.",
            "errors":        "Greške procesa",
            "deal_speed":    "Brzina poslova",
            "days":          "dana",
            "conversion":    "Konverzija poslova",
            "t_time_saved":  "+ Ušteda vremena",
            "t_error_red":   "+ Smanjenje grešaka",
            "t_rev_speed":   "+ Prihod (brzina)",
            "t_rev_conv":    "+ Prihod (konverzija)",
            "t_total":       "= Ukupna korist",
            "t_invest":      "\u2212 Investicija",
            "t_net_roi":     "= NETO ROI",
            "model":         "Model",
            "param":         "Parametar",
            "result":        "Rezultat",
            "interpretation":"Interpretacija",
            "bottleneck_p":  "Usko grlo",
            "bottleneck_i":  "Prioritet optimizacije procesa",
            "p_complete_p":  "P(završetak)",
            "markov_i":      "Rast konverzije kroz Markovljeve lance",
            "bayes_p":       "Poverenje u rezultat",
            "ci_label":      "80% IP: {}",
            "fee_format":    "Format plaćanja",
            "fee_threshold": "Prag",
            "fee_share":     "Udeo",
            "fee_year1":     "Procena Year-1",
            "fee_model":     "Pretplata + Success Fee",
            "notes_title":   "Beleške sa sastanka",
            "next_step":     "Sledeći korak:",
            "next_body":     "Proslediti izveštaj timu za implementaciju i dogovoriti plan automatizacije.",
            "auditor_lbl":   "Revizor:",
        },
    }
    T = _T.get(lang, _T["en"])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=10*mm, bottomMargin=10*mm,
    )
    W = A4[0] - 24*mm
    story = []

    r  = currency_rate
    cs = currency_sym

    today_str   = date.today().strftime("%d.%m.%Y")
    total_benefit       = (time_saved + error_reduction + revenue_impact + markov_gain) * r
    time_saved_c        = time_saved          * r
    error_reduction_c   = error_reduction     * r
    revenue_impact_c    = revenue_impact      * r
    markov_gain_c       = markov_gain         * r
    implementation_c    = implementation_cost * r
    net_roi_c           = net_roi             * r
    fee_threshold_c     = success_fee_threshold * r
    year1_fee           = max(0.0, (net_roi_c - fee_threshold_c) * (success_fee_pct / 100.0))

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
        [_p("{:,.0f} {}".format(net_roi_c, cs), S_HERO),
         _p("{:.1f}%".format(roi_pct), S_HERO),
         _p("{:.1f} {}".format(payback_months, T["months"]), S_HERO)],
        [_p(T["net_roi_lbl"], S_HERO_SUB),
         _p(T["roi_pct_lbl"], S_HERO_SUB),
         _p(T["payback_lbl"], S_HERO_SUB)],
    ]
    hero_ts = _ts_base(bg=GRAY_D, grid=ACCENT) + [
        ("BACKGROUND", (0, 0), (-1, -1), GRAY_D),
        ("BOX",        (0, 0), (-1, -1), 1.5, ACCENT),
        ("LINEABOVE",  (0, 0), (-1, 0),  2.0, GREEN),
    ]
    story.append(Table(hero_data, colWidths=[W/3]*3, style=TableStyle(hero_ts)))
    story.append(Spacer(1, 4*mm))

    # ── 3. 4-METRICS TABLE ─────────────────────────────────────────────────────
    m_hdr = [_p(T["component"], S_BOLD_C), _p(T["before"], S_BOLD_C), _p(T["after"], S_BOLD_C),
             _p("{} / {}".format(cs, {"en":"yr","ru":"год","sr":"god."}.get(lang,"yr")), S_BOLD_C),
             _p(T["source"], S_BOLD_C)]
    m_rows = [
        [_p(T["labor"],      S_LEFT),
         _p("{} {}".format(manual_hours_before, T["h_mo"]), S_CENTER),
         _p("{:.0f} {}".format(manual_hours_before*(1-automation_rate_pct/100), T["h_mo"]), S_CENTER),
         _p("{:,.0f}".format(time_saved_c),      S_GREEN), _p("[GRAPH]", S_SRC)],
        [_p(T["errors"],     S_LEFT),
         _p("{:.1f}%".format(error_rate_before), S_CENTER),
         _p("{:.1f}%".format(error_rate_after),  S_CENTER),
         _p("{:,.0f}".format(error_reduction_c), S_GREEN), _p("[GRAPH]", S_SRC2)],
        [_p(T["deal_speed"], S_LEFT),
         _p("{:.0f} {}".format(deal_cycle_before, T["days"]), S_CENTER),
         _p("{:.0f} {}".format(deal_cycle_after,  T["days"]), S_CENTER),
         _p("{:,.0f}".format(revenue_impact_c),  S_GREEN), _p("[MARKOV]", S_SRM)],
        [_p(T["conversion"], S_LEFT),
         _p("{:.0f}%".format(p_complete_before_pct), S_CENTER),
         _p("{:.0f}%".format(p_complete_after_pct),  S_CENTER),
         _p("{:,.0f}".format(markov_gain_c),     S_GREEN), _p("[MARKOV]", S_SRM2)],
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
        [_p(T["t_time_saved"], S_FL), _p("{:,.0f} {}".format(time_saved_c,      cs), S_FV)],
        [_p(T["t_error_red"],  S_FL), _p("{:,.0f} {}".format(error_reduction_c, cs), S_FV)],
        [_p(T["t_rev_speed"],  S_FL), _p("{:,.0f} {}".format(revenue_impact_c,  cs), S_FV)],
        [_p(T["t_rev_conv"],   S_FL), _p("{:,.0f} {}".format(markov_gain_c,     cs), S_FV)],
        [_p(T["t_total"],      S_FL), _p("{:,.0f} {}".format(total_benefit,     cs), S_FV)],
        [_p(T["t_invest"],     S_FL), _p("{:,.0f} {}".format(implementation_c,  cs), S_FN)],
        [_p(T["t_net_roi"],    S_FNL),_p("{:,.0f} {}".format(net_roi_c,         cs), S_FNV)],
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
    math_hdr = [_p(T["model"], S_BOLD_C), _p(T["param"], S_BOLD_C),
                _p(T["result"], S_BOLD_C), _p(T["interpretation"], S_BOLD_C)]
    math_rows = [
        [_p("[GRAPH]",   S_MG), _p(T["bottleneck_p"], S_LEFT),
         _p("{} / {:.4f}".format(bottleneck_node, bottleneck_score), S_CENTER),
         _p(T["bottleneck_i"], S_LEFT)],
        [_p("[MARKOV]",  S_MM), _p(T["p_complete_p"], S_LEFT),
         _p("{:.0f}% → {:.0f}%".format(p_complete_before_pct, p_complete_after_pct), S_CENTER),
         _p(T["markov_i"], S_LEFT)],
        [_p("[BAYES]",   S_MB), _p(T["bayes_p"], S_LEFT),
         _p("{:.1f}% → {:.1f}%".format(bayes_prior, bayes_posterior), S_CENTER),
         _p(T["ci_label"].format(bayes_ci), S_LEFT)],
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
        [_p(T["fee_format"], S_BOLD_C), _p(T["fee_threshold"], S_BOLD_C),
         _p(T["fee_share"], S_BOLD_C), _p(T["fee_year1"], S_BOLD_C)],
        [_p(T["fee_model"], S_CENTER),
         _p("{:,.0f} {}".format(fee_threshold_c, cs), S_CENTER),
         _p("{}%".format(success_fee_pct), S_CENTER),
         _p("{:,.0f} {}".format(year1_fee, cs), S_SFY)],
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
            [_p(T["notes_title"], S_BOLD_L)],
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
        [_p(T["next_step"], S_BOLD_L),
         _p(T["next_body"], S_LEFT)],
        [_p(T["auditor_lbl"], S_BOLD_L),
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
