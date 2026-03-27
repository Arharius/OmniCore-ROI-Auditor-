# OmniCore ROI Auditor

Платформа ROI-аудита автоматизации на базе ИИ — Python/Streamlit приложение с цепями Маркова, байесовскими обновлениями, граф-анализом (NetworkX), PDF-отчётами, динамическим ETL CSV-конвейером, поддержкой 3 языков (EN/RU/SR), демо-режимом и системой аутентификации.

## C-Level Metrics Block — Friction Tax + Bayesian Confidence (2026-03-27)

### core/math_engine.py — новые методы MathEngine

**`compute_friction_tax(decision_latency_days, cost_per_hour, hours_per_day)`**
- Возвращает прямые потери в $ на одну сущность из-за петель доработки
- Формула: `latency_days × hours_per_day × cost_per_hour`
- Защита от нулей: возвращает 0.0 при любом нулевом параметре

**`compute_process_confidence(prior_pct, bottleneck_rework_rate)`**
- Байесовское обновление через форму шансов (odds-form Bayes)
- Evidence = наблюдаемая ставка rework на узком месте
- `P(ev|success) = 1 − rework_rate` / `P(ev|failure) = rework_rate`
- `LR = P(ev|success) / P(ev|failure)`
- `Posterior_odds = Prior_odds × LR → Posterior_pct`
- Возвращает dict: `prior_pct, likelihood_ratio, posterior_pct, delta_pct`
- Проверенные граничные случаи: rework=50% → LR=1 (нейтр.); rework=20% → LR=4; rework=80% → LR=0.25

### ui/dashboard.py — Tab 2 (данные из CSV) добавлено ПЕРЕД графом

1. **CSS-инъекция** — тёмные метрики в mono-шрифте:
   - `[data-testid="stMetric"]` → `#2C2C2E` фон, border-radius 10px
   - Labels: `SF Mono` 9px, letter-spacing 2.5px, uppercase, `#8E8E93`
   - Values: `SF Mono` 26px, `#F5F5F7`, weight 600

2. **Prior slider** — `st.slider` "Априорная вероятность успеха", 10–99%, default 85%, key `clevel_prior_pct`

3. **Friction Tax** — `decision_latency[bottleneck_node]` (fallback: `avg_days × rework_rate`) × `fin_cost_per_hour` × `fin_hours_per_day`

4. **3 `st.metric` в колонках**:
   - **Bottleneck Stage** — название + delta: `rework% | score`
   - **Friction Tax / Entity** — `$xxx` + delta: формула в словах
   - **Adjusted Confidence** — `xx.x%` + delta: `Prior → LR → Δ%`

5. **Expander "Методология"** — пошаговый вывод формул (EN/RU/SR)

6. `<hr>` разделитель → граф ниже

## Markov Graph — build_markov_graph (2026-03-27)

### core/math_engine.py — новые компоненты
- **`ReworkPair` dataclass** — одна обнаруженная петля: stage_a, stage_b, P(A→B), P(B→A), avg_days_in_b, count_rework
- **`MarkovGraphResult` dataclass** — полный результат: transition_probs, transition_counts, rework_pairs, rework_edges, rework_rate, decision_latency, bottleneck_node/rate/days/score, G (nx.DiGraph), betweenness, pagerank, combo_score
- **`build_markov_graph(df)`** — standalone-функция, принимает нормализованный DataFrame (entity_id/current_stage/next_stage/time_spent):
  1. Строит матрицу переходных вероятностей из реальных данных
  2. Обнаруживает петли: (A→B) рework если (B→A) тоже существует
  3. Decision Latency = avg `time_spent` в стадии B перед возвратом в A
  4. NetworkX DiGraph с атрибутами `weight=probability`, `count`, `is_rework`
  5. Betweenness centrality + PageRank
  6. Composite bottleneck score: 0.55×norm_betweenness + 0.45×rework_rate

### ui/dashboard.py — Tab 2 полностью переписан
- **Без CSV** → static fallback граф + подсказка загрузить ETL
- **С CSV** → полный интерактивный Plotly граф:
  - Spring layout (nx.spring_layout, k=2.2, seed=42)
  - Обычные рёбра: синие, толщина пропорциональна count
  - Rework рёбра: красные пунктирные (`dash="dot"`)
  - Стрелки-аннотации с % вероятности на каждом переходе
  - Узлы: RED=bottleneck, ORANGE=rework≥30%, BLUE=normal
  - Размер узла пропорционален трафику
  - Hover: transitions count, rework%, betweenness, decision latency
  - Цветовые чипы-легенды
  - `st.error` (если rework≥30% или avg_days≥5) или `st.warning` — явный репорт bottleneck
  - Таблица петель с `background_gradient` по Decision Latency
  - Раскрывающаяся матрица переходных вероятностей (heatmap)

### Тест на реальных данных
- 20 строк CSV: Lead→In Review→Revision→In Review (петля)
- Обнаружен rework: In Review↔Revision, P(возврат)=80%, latency=5.08 дн.
- Bottleneck: In Review (score=0.775, rework 50%)
- 40% всех переходов — rework-переходы

## ETL Refactor (2026-03-27) — Data-Driven Pipeline

### Архитектурные изменения
- **Удалены** демо-пресеты (Retail/Agency/Logistics) из сайдбара
- **Удалены** все ручные слайдеры (Labor/Errors/Cycle/Probability/Investment)
- **Добавлен** `st.file_uploader` в главной области (строго `.csv`)
- **Добавлены** 4 динамических `st.selectbox` для маппинга колонок:
  - Entity ID (Deal_ID, Task_ID, …)
  - Current Stage
  - Next Stage
  - Time Spent (Days)
- **Добавлены** 2 финансовых инпута: "Стоимость часа ($)" и "Рабочих часов/день"
- **Нормализация**: маппированные колонки переименовываются в `entity_id`, `current_stage`, `next_stage`, `time_spent` и сохраняются в `st.session_state["mapped_df"]`
- **Smart auto-detect**: движок угадывает правильные колонки по именам (hints-based)
- **Все ROI-параметры** выводятся автоматически из загруженных данных:
  - `volume` = unique entities
  - `cycle_before` = средний суммарный цикл на entity
  - `cycle_after` = 45% от cycle_before
  - `manual_hours` = avg_cycle × hours_per_day × deals_month / 30
  - `p_before/after` = % сущностей, достигших absorbing state
  - `pos/tot_signals` = completed / total entities

## UX Sprint 1+2+3 (2026-03-26)

### Sprint 1 — Sidebar + Validation
- Sidebar разбит на 5 collapsible expanders (💼 Labor / ⚠️ Errors / 🔄 Cycle / 📐 Probability / 💰 Investment)
- Help-тексты для всех 14 полей ввода (EN/RU/SR)
- `deal_value` и `impl_cost` — теперь `number_input` вместо слайдера
- Inline-валидация: error_after < error_before, cycle_after < cycle_before, p_after > p_before
- Gauge chart (ROI% vs отраслевой бенчмарк с delta и цветовыми зонами) в Tab 1
- Radar chart (4 компонента ROI: время/ошибки/цикл/конверсия) в Tab 1
- FAQ/Методология — 5 Q&A в expandable карточках в конце страницы

### Sprint 2 — Новые графики
- **Funnel chart** (До/После) в Markov tab — показывает воронку сделок по стадиям
- **Scatter plot** (облако риск/доходность) в Monte Carlo — зелёные/красные точки для всех симуляций
- **Bullet chart** прогресса окупаемости в Tab 1 — цветовые зоны 0-12 (зелёный), 12-18 (жёлтый), 18-36 (красный)

### Sprint 3 — Архитектура + About tab
- `@st.cache_data(ttl=300)` для Monte Carlo — ускорение повторных запросов с теми же параметрами
- **Tab 7 "О приложении"** (EN/RU/SR): hero-карточка, bio автора, методология (5 моделей), тех.стек, changelog

## 8 Feature Updates (2026-03-26)
1. **История клиентов** — сохранение/загрузка аудитов в `data/clients.json` (до 15 записей)
2. **Режим презентации** — скрывает сайдбар, CSS-toggle, кнопка в сайдбаре и в хедере
3. **Валюта EUR/RUB/RSD** — конвертация отображения, радио-кнопка в сайдбаре
4. **Заметки в PDF** — поле `meeting_notes` в вкладке Паспорт, появляется в PDF
5. **Сравнение сценариев** — expander в ROI Breakdown: Сценарий A vs B (автоматизация, бюджет, цикл)
6. **Отраслевые бенчмарки** — delta на KPI-карточках (vs среднее по отрасли из `_BENCHMARKS`)
7. **QR-код в PDF** — `qrcode` lib, генерируется из `contact_url`, вставляется в футер PDF
8. **Имя аудитора** — text_input в сайдбаре, дефолт "Andrew | AI Product Advisor", передаётся в PDF

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
