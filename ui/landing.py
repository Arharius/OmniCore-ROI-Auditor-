import streamlit as st
from auth.credentials import authenticate

_LANG_NAMES = {"en": "EN", "ru": "RU", "sr": "SR"}

_COPY = {
    "en": {
        "eyebrow":   "ROI Audit Platform",
        "hero_h1":   "Know the value\nof automation.",
        "hero_sub":  "Markov chains · Graph analysis · Bayesian inference · PDF export.",
        "stat1_v": "847%",    "stat1_l": "Average ROI",
        "stat2_v": "1.4 mo",  "stat2_l": "Average payback",
        "stat3_v": "3",       "stat3_l": "Independent models",
        "f1_t": "Graph bottleneck",
        "f1_b": "Identifies which stage of your sales or ops process loses the most money — using NetworkX centrality analysis.",
        "f2_t": "Markov chains",
        "f2_b": "Models your deal flow as an absorbing Markov chain. Mathematically expected conversion and cycle time.",
        "f3_t": "Bayesian inference",
        "f3_b": "Quantifies forecast confidence. Prior to posterior with 80% credible intervals, updated on real signals.",
        "f4_t": "PDF Passport",
        "f4_b": "One click produces a professional ROI report ready to hand to a client immediately after the meeting.",
        "demo_btn":   "Try Demo",
        "login_h":    "Sign in",
        "login_user": "Username",
        "login_pass": "Password",
        "login_btn":  "Continue",
        "login_err":  "Incorrect username or password.",
        "request":    "Request access",
        "request_sub": "@weerowoolf",
    },
    "ru": {
        "eyebrow":   "Платформа ROI-аудита",
        "hero_h1":   "Узнай ценность\nавтоматизации.",
        "hero_sub":  "Цепи Маркова · Граф-анализ · Байесовский вывод · PDF-экспорт.",
        "stat1_v": "847%",    "stat1_l": "Средний ROI",
        "stat2_v": "1.4 мес", "stat2_l": "Средняя окупаемость",
        "stat3_v": "3",       "stat3_l": "Независимые модели",
        "f1_t": "Граф узких мест",
        "f1_b": "Находит, на какой стадии продаж или операций теряется больше всего денег — через анализ центральности NetworkX.",
        "f2_t": "Цепи Маркова",
        "f2_b": "Моделирует воронку как поглощающую цепь Маркова. Математически ожидаемая конверсия и время сделки.",
        "f3_t": "Байесовский вывод",
        "f3_b": "Квантифицирует уверенность в прогнозе. Апостериорная вероятность с 80% доверительным интервалом.",
        "f4_t": "PDF Паспорт",
        "f4_b": "Один клик — профессиональный документ готов к отправке клиенту сразу после встречи.",
        "demo_btn":   "Демо без входа",
        "login_h":    "Войти",
        "login_user": "Логин",
        "login_pass": "Пароль",
        "login_btn":  "Продолжить",
        "login_err":  "Неверный логин или пароль.",
        "request":    "Запросить доступ",
        "request_sub": "@weerowoolf",
    },
    "sr": {
        "eyebrow":   "Platforma za ROI reviziju",
        "hero_h1":   "Saznajte vrednost\nautomatizacije.",
        "hero_sub":  "Markovljevi lanci · Analiza grafova · Bajesov zaključak · PDF izvoz.",
        "stat1_v": "847%",      "stat1_l": "Prosečan ROI",
        "stat2_v": "1.4 mes.",  "stat2_l": "Prosečan povrat",
        "stat3_v": "3",         "stat3_l": "Nezavisnih modela",
        "f1_t": "Graf uskih grla",
        "f1_b": "Pronalazi koja faza prodaje ili operacija gubi najviše novca — NetworkX analiza centralnosti.",
        "f2_t": "Markovljevi lanci",
        "f2_b": "Modeluje tok poslova kao apsorbujući Markovljev lanac. Matematički očekivana konverzija i vreme.",
        "f3_t": "Bajesov zaključak",
        "f3_b": "Kvantifikuje poverenje u prognozu. Apriorna ka aposteriornoj sa 80% intervalom pouzdanosti.",
        "f4_t": "PDF Pasoš",
        "f4_b": "Jedan klik — profesionalni izveštaj spreman za slanje klijentu odmah posle sastanka.",
        "demo_btn":   "Demo bez prijave",
        "login_h":    "Prijavite se",
        "login_user": "Korisničko ime",
        "login_pass": "Lozinka",
        "login_btn":  "Nastavi",
        "login_err":  "Pogrešno korisničko ime ili lozinka.",
        "request":    "Zatražite pristup",
        "request_sub": "@weerowoolf",
    },
}


def show_landing():
    if "land_lang" not in st.session_state:
        st.session_state["land_lang"] = "ru"
    lang = st.session_state["land_lang"]
    c = _COPY[lang]

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display',
                 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

.stApp { background: #F5F5F7 !important; }
[data-testid="stSidebar"] { display: none !important; }
footer, header, #MainMenu,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stHeader"],
[data-testid="baseButton-headerNoPadding"],
.stAppDeployButton { display: none !important; visibility: hidden !important; height: 0 !important; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 860px !important;
}

/* ── lang picker ── */
div[data-testid="stSelectbox"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: transparent !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    color: #1D1D1F !important;
    padding: 2px 10px !important;
    min-height: unset !important;
}

/* ── inputs ── */
div[data-testid="stTextInput"] input {
    border: 1px solid rgba(0,0,0,0.15) !important;
    border-radius: 10px !important;
    padding: 11px 14px !important;
    font-size: 15px !important;
    background: #FAFAFA !important;
    color: #1D1D1F !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #0071E3 !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.12) !important;
    outline: none !important;
}

/* ── form submit → Apple blue pill ── */
div[data-testid="stFormSubmitButton"] button {
    background: #0071E3 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 980px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    padding: 11px 28px !important;
    width: 100% !important;
    transition: background 0.2s ease !important;
    cursor: pointer !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #0064CC !important;
}

/* ── secondary button (Try Demo) ── */
div.stButton > button {
    background: #1D1D1F !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 980px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    padding: 11px 28px !important;
    width: 100% !important;
    transition: background 0.2s ease !important;
    cursor: pointer !important;
}
div.stButton > button:hover {
    background: #3A3A3C !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Lang switcher ───────────────────────────────────────────────────────
    _, lc = st.columns([8, 1])
    with lc:
        chosen = st.selectbox(
            "lang", list(_LANG_NAMES.keys()),
            format_func=lambda k: _LANG_NAMES[k],
            index=list(_LANG_NAMES.keys()).index(lang),
            key="land_lang_sel", label_visibility="collapsed",
        )
        if chosen != lang:
            st.session_state["land_lang"] = chosen
            st.rerun()

    # ── HERO — dark Apple-style ─────────────────────────────────────────────
    hero_h1 = c["hero_h1"].replace("\n", "<br>")
    st.markdown(f"""
<div style="
    background: #1D1D1F;
    border-radius: 22px;
    padding: 56px 52px 52px;
    margin-bottom: 3px;
    text-align: left;
">
  <div style="
    display: inline-block;
    color: rgba(255,255,255,0.55);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 20px;
  ">{c['eyebrow']}</div>

  <h1 style="
    color: #F5F5F7;
    font-size: clamp(36px, 5vw, 56px);
    font-weight: 800;
    letter-spacing: -0.045em;
    line-height: 1.05;
    margin: 0 0 18px;
  ">{hero_h1}</h1>

  <p style="
    color: rgba(255,255,255,0.5);
    font-size: 15px;
    font-weight: 400;
    letter-spacing: -0.01em;
    line-height: 1.6;
    margin: 0 0 44px;
    max-width: 480px;
  ">{c['hero_sub']}</p>

  <div style="display:flex; gap:48px; flex-wrap:wrap; align-items:flex-start;">
    <div>
      <div style="color:#F5F5F7;font-size:40px;font-weight:800;
           letter-spacing:-0.05em;line-height:1;">{c['stat1_v']}</div>
      <div style="color:rgba(255,255,255,0.4);font-size:12px;
           font-weight:500;margin-top:5px;letter-spacing:0.02em;">{c['stat1_l'].upper()}</div>
    </div>
    <div>
      <div style="color:#F5F5F7;font-size:40px;font-weight:800;
           letter-spacing:-0.05em;line-height:1;">{c['stat2_v']}</div>
      <div style="color:rgba(255,255,255,0.4);font-size:12px;
           font-weight:500;margin-top:5px;letter-spacing:0.02em;">{c['stat2_l'].upper()}</div>
    </div>
    <div>
      <div style="color:#F5F5F7;font-size:40px;font-weight:800;
           letter-spacing:-0.05em;line-height:1;">{c['stat3_v']}</div>
      <div style="color:rgba(255,255,255,0.4);font-size:12px;
           font-weight:500;margin-top:5px;letter-spacing:0.02em;">{c['stat3_l'].upper()}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── FEATURES 2×2 ───────────────────────────────────────────────────────
    st.markdown("<div style='height:3px'></div>", unsafe_allow_html=True)
    fa, fb = st.columns(2, gap="small")
    fc, fd = st.columns(2, gap="small")

    _feats = [
        (fa, "f1"), (fb, "f2"),
        (fc, "f3"), (fd, "f4"),
    ]
    for col, fi in _feats:
        with col:
            st.markdown(f"""
<div style="
    background: #FFFFFF;
    border-radius: 18px;
    padding: 28px 26px 26px;
    margin-bottom: 3px;
    height: 160px;
">
  <div style="
    font-size: 15px;
    font-weight: 700;
    color: #1D1D1F;
    letter-spacing: -0.02em;
    margin-bottom: 10px;
    line-height: 1.2;
  ">{c[fi + '_t']}</div>
  <div style="
    font-size: 13px;
    color: #6E6E73;
    line-height: 1.6;
  ">{c[fi + '_b']}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── AUTH CARD ──────────────────────────────────────────────────────────
    _, lcc, _ = st.columns([1, 2, 1])
    with lcc:
        st.markdown("""
<div style="
    background: #FFFFFF;
    border-radius: 20px;
    padding: 36px 32px 28px;
">
  <div style="
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #AEAEB2;
    margin-bottom: 6px;
  ">OmniCore ROI Auditor</div>
</div>
""", unsafe_allow_html=True)

        # Demo CTA
        _demo_labels = {
            "en": "Try Demo",
            "ru": "Демо без входа",
            "sr": "Demo bez prijave",
        }
        if st.button(_demo_labels[lang], key="btn_try_demo", use_container_width=True):
            st.session_state["demo_only"]    = True
            st.session_state["demo_preset"]  = "logistics"
            st.session_state["company_name"] = "ТрансЛогик МСК"
            from ui.dashboard import DEMO_PRESETS, _DEMO_KEYS
            p = DEMO_PRESETS["logistics"]
            for k in _DEMO_KEYS:
                st.session_state[k] = p[k]
            st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Login form — clean, no card wrapper (card is above)
        with st.form("login_form", clear_on_submit=False):
            st.text_input(c["login_user"], placeholder="username", key="l_user",
                          label_visibility="collapsed")
            st.text_input(c["login_pass"], type="password", placeholder="••••••••",
                          key="l_pass", label_visibility="collapsed")
            submitted = st.form_submit_button(c["login_btn"], use_container_width=True)

        if submitted:
            username = st.session_state.get("l_user", "")
            password = st.session_state.get("l_pass", "")
            user = authenticate(username, password)
            if user:
                st.session_state["auth_user"]    = user
                st.session_state["authenticated"] = True
                st.session_state.pop("demo_only", None)
                # Store token for deferred cookie write on next render (avoids blocking the login transition)
                from core.session_cookie import make_token
                st.session_state["_auth_token_pending"] = make_token(
                    user.get("username", username),
                    user.get("role", "user"),
                )
                st.rerun()
            else:
                st.error(c["login_err"])

        st.markdown(f"""
<div style="text-align:center; margin-top:16px;">
  <a href="https://t.me/weerowoolf" target="_blank"
     style="font-size:12px; color:#AEAEB2; text-decoration:none;
            letter-spacing:0.01em;">
    {c['request']} · {c['request_sub']}
  </a>
</div>
""", unsafe_allow_html=True)

    # ── FOOTER ─────────────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center; margin-top:40px; padding-bottom:32px;">
  <span style="font-size:11px; color:#C7C7CC; letter-spacing:0.02em;">
    Andrew · AI Product Advisor · Serbia / EU
  </span>
</div>
""", unsafe_allow_html=True)
