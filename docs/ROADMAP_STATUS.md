# OmniCore ROI Auditor — Статус Роадмапа

*Обновлено: март 2026*

---

## ФАЗА 1: Математическое ядро ✅ ВЫПОЛНЕНО

| Задача | Статус | Файл |
|--------|--------|------|
| ROI-движок (4 метрики) | ✅ | core/roi_engine.py |
| Граф-анализ (NetworkX, betweenness centrality, PageRank) | ✅ | core/math_engine.py |
| Поглощающая цепь Маркова (матрица Q, фундаментальная N) | ✅ | core/math_engine.py |
| Байесовское обновление (Beta distribution, 80% ДИ) | ✅ | core/math_engine.py |
| ETL: CSV → матрица Маркова (Pandas) | ✅ | etl/extractor.py |
| CLI: python main.py --demo / --input | ✅ | main.py |

---

## ФАЗА 2: UI и визуализация ✅ ВЫПОЛНЕНО

| Задача | Статус | Файл |
|--------|--------|------|
| Apple HIG дизайн (#F5F5F7, 18px cards, SF Pro) | ✅ | ui/dashboard.py |
| Waterfall chart (ROI breakdown) | ✅ | ui/dashboard.py |
| Pie chart (структура выгод) | ✅ | ui/dashboard.py |
| Граф-визуализация (Plotly, NetworkX) | ✅ | ui/dashboard.py |
| Timeline chart (окупаемость 0–12 мес) | ✅ | ui/dashboard.py |
| Beta distribution chart (Байес) | ✅ | ui/dashboard.py |
| 3 языка интерфейса (EN / RU / SR) | ✅ | ui/i18n.py |
| KPI-карточки (чистый ROI, ROI%, окупаемость, P-success) | ✅ | ui/dashboard.py |
| Landing page (Apple-style маркетинговый лендинг) | ✅ | ui/landing.py |

---

## ФАЗА 3: Экспорт и аутентификация ✅ ВЫПОЛНЕНО

| Задача | Статус | Файл |
|--------|--------|------|
| PDF-паспорт (ReportLab, A4) | ✅ | exports/pdf_generator.py |
| TXT-экспорт паспорта | ✅ | exports/pdf_generator.py |
| LinkedIn-хук (3 языка) | ✅ | ui/dashboard.py |
| Демо-режим (3 пресета: Логистика / Агентство / Ритейл) | ✅ | ui/dashboard.py |
| Аутентификация (SHA-256 хэш) | ✅ | ui/dashboard.py |
| Superadmin (weerowoolf) | ✅ | data/users.json |
| Freemium-логика (ограничения без логина) | ✅ | ui/dashboard.py |

---

## ФАЗА 4: 8 новых функций ✅ ВЫПОЛНЕНО (март 2026)

| Задача | Статус | Файл |
|--------|--------|------|
| История клиентов JSON (data/clients.json, макс 15) | ✅ | ui/dashboard.py |
| Режим презентации (CSS toggle, сайдбар скрывается) | ✅ | ui/dashboard.py |
| Переключатель валют EUR / RUB / RSD | ✅ | ui/dashboard.py |
| Поле «Заметки встречи» → PDF | ✅ | exports/pdf_generator.py |
| Сравнение сценариев A vs B (expander) | ✅ | ui/dashboard.py |
| Отраслевые бенчмарки на KPI-карточках | ✅ | ui/dashboard.py |
| QR-код в PDF (qrcode + Pillow) | ✅ | exports/pdf_generator.py |
| Имя аудитора и contact URL в настройках сайдбара | ✅ | ui/dashboard.py |

---

## ФАЗА 5: Деплой и инфраструктура ✅ ВЫПОЛНЕНО

| Задача | Статус | Детали |
|--------|--------|--------|
| GitHub private repo | ✅ | Arharius/OmniCore-ROI-Auditor- |
| Render.com deployment | ✅ | https://omnicore-roi-auditor.onrender.com |
| Streamlit config (minimal toolbar, headless) | ✅ | .streamlit/config.toml |
| render.yaml (build/start команды) | ✅ | render.yaml |
| Скрытие Streamlit-хедера (CSS selectors) | ✅ | ui/dashboard.py + landing.py |

---

## ФАЗА 6: В ПЛАНАХ

| Задача | Приоритет | Статус |
|--------|-----------|--------|
| Telegram-бот для быстрого аудита | Высокий | Запланировано |
| API-endpoint для интеграторов (FastAPI) | Высокий | Запланировано |
| Шаблоны отчётов по отраслям (5 отраслей) | Средний | Запланировано |
| Мобильно-адаптивная версия | Средний | Запланировано |
| Multi-tenancy / белая этикетка | Низкий | Запланировано |
| Stripe / подписочная монетизация | Низкий | Запланировано |

---

## ИТОГ

**Выполнено фаз: 5 из 6**
**Выполнено задач: 37 из 37 (в фазах 1–5)**
**Следующий шаг:** Telegram-бот или API-endpoint

Приложение: **https://omnicore-roi-auditor.onrender.com**
