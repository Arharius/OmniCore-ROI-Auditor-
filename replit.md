# OmniCore ROI Auditor

Платформа ROI-аудита автоматизации на базе ИИ — Python/Streamlit приложение с цепями Маркова, байесовскими обновлениями, граф-анализом (NetworkX), PDF-отчётами, ETL-конвейером, поддержкой 3 языков (EN/RU/SR), демо-режимом и системой аутентификации с маркетинговым лэндингом.

## Project Structure

```
omnicore-roi-auditor/
├── app.py                    # Entry point: auth gate + routing
├── requirements.txt          # numpy, networkx, pandas, scipy, streamlit, plotly, reportlab
├── .streamlit/config.toml    # Streamlit server config (port 5000, headless, Apple theme)
├── auth/
│   ├── credentials.py        # SHA256 hashing, user management, authenticate()
│   └── users.json            # Auto-created on first run (superadmin seeded)
├── core/
│   ├── roi_engine.py         # ROIEngine, ROIInput, ROIResult
│   └── math_engine.py        # MathEngine: markov_absorbing, bayesian_update, graph_bottleneck
├── etl/
│   └── extractor.py          # MatrixExtractor: from_csv, from_dict
├── exports/
│   └── pdf_generator.py      # build_roi_passport_pdf()
├── ui/
│   ├── landing.py            # Apple-style dark-hero landing + login (3 languages)
│   ├── dashboard.py          # Main ROI auditor dashboard, run_dashboard()
│   ├── admin.py              # Superadmin user management panel, show_admin()
│   └── i18n.py               # Translations: TRANSLATIONS, LANG_NAMES, t()
├── docs/
│   ├── CASES_RU.md           # 4 detailed Russian case studies
│   ├── PRODUCT.md            # Value proposition
│   ├── MANUAL.md             # User manual
│   └── GTM_SERBIA_MOSCOW.md  # Go-to-market strategy
└── data/                     # CSV uploads (ephemeral per session)
```

## Auth Flow

1. `app.py` checks `st.session_state["authenticated"]` OR `demo_only`
2. If neither → shows `ui/landing.show_landing()` (Apple dark hero + login + demo CTA)
3. Demo mode: sets `demo_only=True`, loads Logistics preset → full dashboard, export locked
4. On successful login → sets `authenticated=True`, clears `demo_only`
5. Sidebar: username + logout for auth users; "Sign in" block for demo users
6. Routes to `ui/admin.show_admin()` or `ui/dashboard.run_dashboard()`

## Superadmin Credentials

- Login: `weerowoolf`
- Password: `Aa20052006Aa`
- Hash (SHA256): `abfbf0b9fee161b9a2b9ccf1dea82c8aa774c35746c095976d3efc6b16331e65`

## Demo Mode (Freemium Gates)

Free (demo without login):
- All 3 presets: Логистика, Агентство, Ритейл
- All sliders and parameter controls
- All calculations: ROI, Markov, Bayesian, Graph
- Passport text visible

Locked (requires login):
- CSV upload (own data)
- TXT download
- PDF download

## Demo Presets

- **Логистика** — ТрансЛогик МСК: 3PL, 520 ч/мес, 85 сделок, deal 12 000€
- **Агентство** — MOKO Digital: performance agency, 320 ч/мес, 18 клиентов, deal 8 500€
- **Ритейл** — МегаМаркет d.o.o.: retail chain Belgrade, 380 ч/мес, 45 сделок, deal 1 800€

## Tech Stack

- **Frontend/UI**: Streamlit (port 5000), Apple HIG design system (#1D1D1F hero, #F5F5F7 bg)
- **Data Processing**: Pandas, NumPy, SciPy
- **Graph Analysis**: NetworkX (betweenness centrality, PageRank)
- **Visualization**: Plotly
- **PDF Export**: ReportLab
- **Auth**: SHA256 hashing, JSON user store (auto-seeded superadmin)
- **Languages**: Python 3.11

## Running the App

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

## Design System (Apple HIG)

- Background: `#F5F5F7`
- Hero: `#1D1D1F` (dark, like Apple product pages)
- Text: `#1D1D1F` / secondary `#6E6E73`
- Blue: `#0071E3`
- Green KPIs: `#34C759`
- Cards: `border-radius: 18px`, white bg, subtle shadow
- Buttons main: `border-radius: 980px` pill, blue
- Buttons sidebar: `border-radius: 8px`, light gray, no emoji
- Font: SF Pro / Inter / system sans-serif
- Zero emoji in UI chrome

## Deployment

- Target: **vm** (always-running — required for session state + users.json writes)
- Run: `streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true`
- Port: 5000 → external 80

## Test Suite

82 mathematical/logical tests passing across 7 blocks:
- ROI Engine: 21 tests (all formulas verified manually)
- Bayesian: 14 tests (Beta distribution, CI ordering, Bayes theorem)
- Markov: 10 tests (N=(I-Q)⁻¹, fundamental matrix, degenerate cases)
- Graph: 7 tests (betweenness centrality, PageRank)
- ETL: 15 tests (Q matrix, stochasticity, cycle days, error rates)
- Integration: 12 tests (all 3 presets full pipeline)
- Edge cases: 10 tests (overflow, minimum values, large graphs)
