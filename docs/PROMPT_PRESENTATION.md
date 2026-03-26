# Промт для генерации презентации OmniCore ROI Auditor

## Промт на русском (для ChatGPT / Claude / Gemini)

```
Ты — профессиональный дизайнер презентаций и автор питч-деков уровня YCombinator / Series A.

Создай структуру слайдов для презентации продукта OmniCore ROI Auditor.

КОНТЕКСТ ПРОДУКТА:
OmniCore ROI Auditor — инструмент ROI-аудита для консультантов, которые продают автоматизацию бизнес-процессов. За 5 минут превращает параметры компании в полный финансовый разбор с PDF-паспортом. Использует три академические математические модели: граф-анализ (NetworkX), поглощающая цепь Маркова, байесовское обновление.

АУДИТОРИЯ ПРЕЗЕНТАЦИИ: [укажи — CFO, операционный директор, риск-офицер, другой консультант]
ЦЕЛЬ ПРЕЗЕНТАЦИИ: [питч / демо / обучение / партнёрство]
ДЛИНА: [5 минут / 15 минут / 30 минут]
ЯЗЫК: Русский

СТРУКТУРА СЛАЙДОВ (создай текст для каждого):

1. ТИТУЛЬНЫЙ СЛАЙД
   - Название, подзаголовок, имя презентующего
   
2. ПРОБЛЕМА (1 слайд)
   - Боль: «6 недель на матрицу, которая должна считаться за 60 секунд»
   - Статистика по риск-офицерам
   
3. ПОЧЕМУ СУЩЕСТВУЮЩИЕ РЕШЕНИЯ НЕ РАБОТАЮТ (1 слайд)
   - Excel: математически слабо
   - SaaS-калькуляторы: нет методологии
   - Big 4: дорого и долго
   
4. РЕШЕНИЕ (1 слайд)
   - 3 модели: Граф + Марков + Байес
   - Одна цифра. Один документ. 5 минут.
   
5. КАК ЭТО РАБОТАЕТ (1-2 слайда)
   - ETL: CSV → матрица → расчёт
   - 4 метрики ROI с формулами
   - PDF-паспорт на выходе
   
6. МАТЕМАТИКА (1 слайд)
   - Граф-анализ: betweenness centrality → узкое место
   - Марков: P(завершение) до/после, матрица Q, N = (I-Q)⁻¹
   - Байес: Beta(α,β) → posterior → 80% ДИ
   
7. КЕЙСЫ (1 слайд)
   - Логистика: 116 000 EUR, 2.4 мес.
   - Агентство: 95 000 EUR, 1.9 мес.
   - Ритейл: 42 000 EUR, 3.5 мес.
   
8. ДЕМО / LIVE AUDIT (1 слайд)
   - «Введём ваши данные прямо сейчас»
   - QR-код на приложение
   
9. КОНКУРЕНТНОЕ СРАВНЕНИЕ (1 слайд)
   - Таблица: Excel / SaaS / Big4 / OmniCore
   
10. БИЗНЕС-МОДЕЛЬ (1 слайд)
    - Форматы и цены
    
11. СЛЕДУЮЩИЙ ШАГ (1 слайд)
    - CTA: ДЕМО на ваших данных, 15 минут, без обязательств

СТИЛЬ: Минималистичный, тёмный фон (#0e1117), акцентный цвет #0071E3 (синий Apple), зелёный #34C759 для цифр ROI. Без клипарта. Только данные, математика, цифры.

Для каждого слайда дай:
- Заголовок (макс. 6 слов)
- 3-4 буллета или 1 ключевое число
- Заметки спикера (2-3 предложения)
```

---

## Prompt in English (for ChatGPT / Claude / Gemini)

```
You are a professional pitch deck designer at YCombinator / Series A level.

Create slide-by-slide content for a product presentation of OmniCore ROI Auditor.

PRODUCT CONTEXT:
OmniCore ROI Auditor is an ROI audit tool for consultants who sell business process automation. In under 5 minutes it turns company parameters into a complete financial breakdown with a PDF passport. Uses three academic mathematical models: graph analysis (NetworkX), absorbing Markov chain, Bayesian update.

AUDIENCE: [specify — CFO, COO, risk officer, fellow consultant]
GOAL: [pitch / demo / training / partnership]
LENGTH: [5 min / 15 min / 30 min]
LANGUAGE: English

SLIDE STRUCTURE (create content for each):

1. TITLE SLIDE
   - Product name, tagline, presenter name

2. THE PROBLEM (1 slide)
   - "6 weeks for a matrix that should compute in 60 seconds"
   - The 12 risk officers statistic

3. WHY EXISTING SOLUTIONS FAIL (1 slide)
   - Excel: mathematically weak
   - SaaS calculators: no methodology
   - Big 4: expensive and slow

4. THE SOLUTION (1 slide)
   - 3 models: Graph + Markov + Bayes
   - One number. One document. 5 minutes.

5. HOW IT WORKS (1-2 slides)
   - ETL: CSV → matrix → calculation
   - 4 ROI metrics with formulas
   - PDF passport output

6. THE MATH (1 slide)
   - Graph: betweenness centrality → bottleneck
   - Markov: P(complete) before/after, matrix Q, N = (I-Q)⁻¹
   - Bayes: Beta(α,β) → posterior → 80% CI

7. CASE STUDIES (1 slide)
   - Logistics: 116,000 EUR, 2.4 months payback
   - Agency: 95,000 EUR, 1.9 months payback
   - Retail: 42,000 EUR, 3.5 months payback

8. LIVE DEMO (1 slide)
   - "Let's enter your data right now"
   - QR code to the app

9. COMPETITIVE COMPARISON (1 slide)
   - Table: Excel / SaaS / Big4 / OmniCore

10. BUSINESS MODEL (1 slide)
    - Formats and pricing

11. NEXT STEP (1 slide)
    - CTA: DEMO on your data, 15 minutes, no commitment

DESIGN STYLE: Minimalist, dark background (#0e1117), accent color #0071E3 (Apple blue), green #34C759 for ROI numbers. No clipart. Data, math, numbers only.

For each slide provide:
- Headline (max 6 words)
- 3-4 bullets or 1 key number
- Speaker notes (2-3 sentences)
```

---

## Промт для создания ВИЗУАЛЬНОГО дизайна (Figma / Canva / Gamma.app)

```
Создай профессиональную презентацию в стиле Apple / Stripe / Linear.

ЦВЕТОВАЯ СХЕМА:
- Фон слайда: #0e1117 (тёмно-синий)
- Карточки/блоки: #1e2130
- Основной акцент: #0071E3 (синий Apple)
- ROI-цифры: #34C759 (зелёный Apple)
- Предупреждения / инвестиции: #FF3B30 (красный Apple)
- Вторичный текст: #AEAEB2
- Основной текст: #FFFFFF

ТИПОГРАФИКА:
- Заголовки: SF Pro Display / Inter, 700 weight, -0.04em letter-spacing
- Тело: SF Pro Text / Inter, 400 weight
- Цифры-герои: 56-72px, #34C759

ЭЛЕМЕНТЫ:
- border-radius: 18px на карточках
- Тонкие линии: rgba(0,113,227,0.3)
- Без градиентов (кроме hero-секции)
- Данные в таблицах с border rgba(255,255,255,0.08)

СЛАЙДЫ: [вставь контент из предыдущего промта]
```

---

## Промт для NotebookLM (Podcast / Audio Briefing)

```
Используй прикреплённый документ как источник для создания аудио-брифинга.

Формат: профессиональный подкаст, два ведущих (аналитик + скептик).
Длина: 8-12 минут.
Тон: деловой, конкретный, без воды.

Ключевые вопросы для раскрытия:
1. В чём реальная проблема с ROI-расчётами сегодня?
2. Что такое цепи Маркова и зачем они нужны для аудита?
3. Как Байесовское обновление защищает прогноз от скептицизма?
4. Почему граф-анализ даёт операционный приоритет?
5. Чем отличается от Excel и Big 4?
6. Какие реальные цифры показывают кейсы?
7. Кому это реально нужно?

Начни с хука: «Я задал 12 риск-офицерам один вопрос...»
```
