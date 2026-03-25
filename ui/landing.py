import streamlit as st
from auth.credentials import authenticate

_LANG_NAMES = {"en": "🇬🇧 EN", "ru": "🇷🇺 RU", "sr": "🇷🇸 SR"}

_COPY = {
    "en": {
        "hero_label": "AI-powered ROI Audit Platform",
        "hero_h1": "Know the value of automation — before you invest",
        "hero_sub": "Graph analysis · Markov chains · Bayesian confidence · PDF report — in 5 minutes.",
        "stat1_v": "847%", "stat1_l": "Average ROI",
        "stat2_v": "1.4 mo", "stat2_l": "Average payback",
        "stat3_v": "3 models", "stat3_l": "Independent validation",
        "f1_t": "🕸️ Graph bottleneck", "f1_b": "Find which stage of your sales or ops process loses the most money — using NetworkX centrality analysis.",
        "f2_t": "🔗 Markov chains",   "f2_b": "Model deal flow as an absorbing Markov chain. Get mathematically expected conversion and lead time.",
        "f3_t": "🎲 Bayesian update", "f3_b": "Quantify confidence in your forecast. Prior → posterior probability with 80% credible intervals.",
        "f4_t": "📄 PDF Passport",    "f4_b": "One-click professional report ready to send to your client immediately after the meeting.",
        "login_h": "Sign in to access the auditor",
        "login_user": "Username", "login_pass": "Password",
        "login_btn": "Enter →",
        "login_err": "Incorrect username or password.",
        "request": "Request demo access",
        "request_sub": "Telegram: @weerowoolf",
    },
    "ru": {
        "hero_label": "Платформа ROI-аудита на базе ИИ",
        "hero_h1": "Узнай ценность автоматизации — до того как вложить деньги",
        "hero_sub": "Граф-анализ · Цепи Маркова · Байесовская уверенность · PDF-отчёт — за 5 минут.",
        "stat1_v": "847%", "stat1_l": "Средний ROI",
        "stat2_v": "1.4 мес", "stat2_l": "Средняя окупаемость",
        "stat3_v": "3 модели", "stat3_l": "Независимая верификация",
        "f1_t": "🕸️ Граф узких мест",  "f1_b": "Находит, на какой стадии продаж или операций теряется больше всего денег — через анализ центральности NetworkX.",
        "f2_t": "🔗 Цепи Маркова",      "f2_b": "Моделирует воронку как поглощающую цепь Маркова. Математически ожидаемая конверсия и время сделки.",
        "f3_t": "🎲 Байесовское обновление", "f3_b": "Квантифицирует уверенность в прогнозе. Априорная → апостериорная вероятность с 80% доверительным интервалом.",
        "f4_t": "📄 PDF Паспорт",        "f4_b": "Один клик — профессиональный документ, готовый к отправке клиенту сразу после встречи.",
        "login_h": "Войти в аудитор",
        "login_user": "Логин", "login_pass": "Пароль",
        "login_btn": "Войти →",
        "login_err": "Неверный логин или пароль.",
        "request": "Запросить доступ",
        "request_sub": "Telegram: @weerowoolf",
    },
    "sr": {
        "hero_label": "AI platforma za ROI reviziju",
        "hero_h1": "Saznajte vrednost automatizacije — pre nego što investirate",
        "hero_sub": "Analiza grafova · Markovljevi lanci · Bajesovsko poverenje · PDF izveštaj — za 5 minuta.",
        "stat1_v": "847%", "stat1_l": "Prosečan ROI",
        "stat2_v": "1.4 mes.", "stat2_l": "Prosečan povrat",
        "stat3_v": "3 modela", "stat3_l": "Nezavisna verifikacija",
        "f1_t": "🕸️ Graf uskih grla",   "f1_b": "Pronalazi koja faza procesa gubi najviše novca — NetworkX analiza centralnosti.",
        "f2_t": "🔗 Markovljevi lanci", "f2_b": "Modeluje tok poslova kao apsorbujući Markovljev lanac. Matematički očekivana konverzija i vreme.",
        "f3_t": "🎲 Bajesovo ažuriranje","f3_b": "Kvantifikuje poverenje u prognozu. Apriorna → aposteriorna verovatnoća, 80% interval pouzdanosti.",
        "f4_t": "📄 PDF Pasoš",          "f4_b": "Jedan klik — profesionalni dokument spreman za slanje klijentu odmah posle sastanka.",
        "login_h": "Prijavite se za pristup",
        "login_user": "Korisničko ime", "login_pass": "Lozinka",
        "login_btn": "Ulaz →",
        "login_err": "Pogrešno korisničko ime ili lozinka.",
        "request": "Zatražite pristup",
        "request_sub": "Telegram: @weerowoolf",
    },
}


def _stats_html(c: dict) -> str:
    parts = []
    for i in range(1, 4):
        v = c.get(f"stat{i}_v", "")
        label = c.get(f"stat{i}_l", "")
        parts.append(
            '<div style="text-align:center;">'
            f'<div style="color:#FFFFFF;font-size:32px;font-weight:800;'
            f'letter-spacing:-0.04em;line-height:1;">{v}</div>'
            f'<div style="color:rgba(255,255,255,0.75);font-size:13px;'
            f'font-weight:500;margin-top:4px;">{label}</div>'
            '</div>'
        )
    return "".join(parts)


def show_landing():
    if "land_lang" not in st.session_state:
        st.session_state["land_lang"] = "ru"
    lang = st.session_state["land_lang"]
    c = _COPY[lang]

    # ── Global CSS ───────────────────────────────────────────────────────────
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter',
                 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}
.stApp { background: #F5F5F7 !important; }
[data-testid="stSidebar"] { display: none !important; }
footer, header { visibility: hidden !important; }
.block-container { padding-top: 0 !important; max-width: 900px !important; }

/* hide streamlit top bar */
#MainMenu { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

div[data-testid="stTextInput"] input {
    border: 1.5px solid rgba(0,0,0,0.12) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
    background: #FFFFFF !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #0071E3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important;
}
div[data-testid="stFormSubmitButton"] button,
button[kind="primary"] {
    background: #0071E3 !important;
    color: #FFFFFF !important;
    border-radius: 980px !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 28px !important;
    width: 100% !important;
    transition: background 0.2s ease !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #0060C0 !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Lang switcher (top right) ─────────────────────────────────────────────
    _, lc = st.columns([6, 1])
    with lc:
        chosen = st.selectbox("Language", list(_LANG_NAMES.keys()),
                              format_func=lambda k: _LANG_NAMES[k],
                              index=list(_LANG_NAMES.keys()).index(lang),
                              key="land_lang_sel", label_visibility="collapsed")
        if chosen != lang:
            st.session_state["land_lang"] = chosen
            st.rerun()

    # ── HERO ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="
    background: linear-gradient(135deg,#0071E3 0%,#34C759 100%);
    border-radius: 24px;
    padding: 48px 40px 40px;
    margin-bottom: 28px;
    text-align: center;
">
  <div style="display:inline-block;background:rgba(255,255,255,0.18);
       color:#fff;font-size:12px;font-weight:700;letter-spacing:0.1em;
       text-transform:uppercase;padding:4px 14px;border-radius:980px;
       margin-bottom:18px;">{c['hero_label']}</div>
  <h1 style="color:#FFFFFF;font-size:clamp(24px,4vw,40px);font-weight:800;
       letter-spacing:-0.03em;line-height:1.1;margin:0 0 14px;">
    {c['hero_h1']}
  </h1>
  <p style="color:rgba(255,255,255,0.85);font-size:16px;font-weight:400;
       margin:0 0 32px;max-width:560px;margin-left:auto;margin-right:auto;">
    {c['hero_sub']}
  </p>
  <div style="display:flex;justify-content:center;gap:32px;flex-wrap:wrap;">
    {_stats_html(c)}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── FEATURES ─────────────────────────────────────────────────────────────
    f1, f2 = st.columns(2)
    f3, f4 = st.columns(2)
    for col, fi in [(f1, "f1"), (f2, "f2"), (f3, "f3"), (f4, "f4")]:
        with col:
            st.markdown(f"""
<div style="background:#FFFFFF;border-radius:18px;padding:22px 22px 20px;
     box-shadow:0 2px 20px rgba(0,0,0,0.07);margin-bottom:4px;height:130px;">
  <div style="font-size:15px;font-weight:700;color:#1D1D1F;
       letter-spacing:-0.01em;margin-bottom:8px;">{c[fi+'_t']}</div>
  <div style="font-size:13px;color:#6E6E73;line-height:1.55;">{c[fi+'_b']}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── LOGIN CARD ───────────────────────────────────────────────────────────
    _, lcc, _ = st.columns([1, 2, 1])
    with lcc:
        st.markdown(f"""
<div style="background:#FFFFFF;border-radius:20px;padding:32px 28px 24px;
     box-shadow:0 4px 40px rgba(0,0,0,0.10);margin-bottom:8px;">
  <div style="text-align:center;margin-bottom:20px;">
    <span style="font-size:22px;font-weight:700;color:#1D1D1F;
         letter-spacing:-0.025em;">📊 OmniCore ROI Auditor</span><br>
    <span style="font-size:13px;color:#6E6E73;font-weight:400;">
      {c['login_h']}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(c["login_user"], placeholder="username", key="l_user")
            password = st.text_input(c["login_pass"], type="password", placeholder="••••••••", key="l_pass")
            submitted = st.form_submit_button(c["login_btn"], use_container_width=True)

        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state["auth_user"] = user
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error(c["login_err"])

        st.markdown(f"""
<div style="text-align:center;margin-top:12px;">
  <span style="font-size:13px;color:#AEAEB2;">{c['request']} · {c['request_sub']}</span>
</div>
""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;margin-top:32px;padding-bottom:24px;">
  <span style="font-size:12px;color:#AEAEB2;">
    Andrew · AI Product Advisor · Fractional TPM · Serbia / EU
  </span>
</div>
""", unsafe_allow_html=True)
