# OmniCore ROI Auditor

Платформа ROI-аудита автоматизации на базе ИИ — Python/Streamlit приложение с цепями Маркова, байесовскими обновлениями, граф-анализом (NetworkX), PDF-отчётами, ETL-конвейером, поддержкой 3 языков (EN/RU/SR), демо-режимом и системой аутентификации с маркетинговым лэндингом.

## Project Structure

```
omnicore-roi-auditor/
├── app.py                    # Entry point: auth gate + routing
├── requirements.txt
├── auth/
│   ├── credentials.py        # SHA256 hashing, user management, authenticate()
│   └── users.json            # Auto-created on first run
├── core/
│   ├── roi_engine.py         # ROIEngine, ROIInput, ROIResult
│   └── math_engine.py        # MathEngine: markov_absorbing, bayesian_update, graph_bottleneck
├── etl/
│   └── extractor.py          # MatrixExtractor: from_csv, from_dict
├── exports/
│   └── pdf_generator.py      # build_roi_passport_pdf()
├── ui/
│   ├── landing.py            # Marketing landing page + login form (3 languages)
│   ├── dashboard.py          # Main ROI auditor dashboard, run_dashboard()
│   ├── admin.py              # Superadmin user management panel, show_admin()
│   └── i18n.py               # Translations: TRANSLATIONS, LANG_NAMES, t()
├── docs/
│   ├── PRODUCT.md            # Value proposition
│   ├── MANUAL.md             # User manual
│   └── GTM_SERBIA_MOSCOW.md  # Go-to-market strategy
└── data/                     # Data directory (CSV uploads)
```

## Auth Flow

1. `app.py` checks `st.session_state["authenticated"]`
2. If not authenticated → shows `ui/landing.show_landing()` (login form + marketing)
3. On successful login → sets `st.session_state["authenticated"]=True`, `auth_user`
4. Sidebar shows username badge, logout button, admin panel toggle (for superadmin)
5. Routes to `ui/admin.show_admin()` or `ui/dashboard.run_dashboard()`

## Superadmin Credentials

- Login: `weerowoolf`
- Password: `Aa20052006Aa`
- Hash (SHA256): `abfbf0b9fee161b9a2b9ccf1dea82c8aa774c35746c095976d3efc6b16331e65`

## Demo Mode

Three industry presets in dashboard sidebar:
- 🚛 Логистика (LogiTrans)
- 🎯 Агентство (Creative Agency Moscow)
- 💻 B2B SaaS (TechFlow Solutions)

## Tech Stack

- **Frontend/UI**: Streamlit (port 5000), Apple HIG design system
- **Data Processing**: Pandas, NumPy, SciPy
- **Graph Analysis**: NetworkX
- **Visualization**: Plotly
- **PDF Export**: ReportLab
- **Auth**: SHA256 hashing, JSON user store
- **Languages**: Python 3.11

## Running the App

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

## Design System

- Background: `#F5F5F7` (Apple grey)
- Text: `#1D1D1F`
- Blue: `#0071E3`
- Green KPIs: `#34C759`
- Cards: `border-radius: 18px`, white bg, subtle shadow
- Buttons: `border-radius: 980px` (pill)
- Font: SF Pro / Inter

## Test Suite

12/12 tests passing (`python -m pytest tests/ -v`)
