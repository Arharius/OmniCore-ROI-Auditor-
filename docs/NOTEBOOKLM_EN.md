# OmniCore ROI Auditor — NotebookLM Source Material (EN)
## Source document for presentation, podcast, or briefing preparation

---

## 1. PRODUCT CONTEXT

OmniCore ROI Auditor is a professional ROI audit tool for consultants who sell automation of business processes. It is designed for use during live client meetings: the auditor enters the client company's parameters in real time and within 3–5 minutes receives a complete financial breakdown with a PDF passport.

The core insight: most ROI calculators use simple multiplication (hours × rate = savings). This is mathematically weak and psychologically vulnerable — an experienced CFO or risk officer sees through it immediately. OmniCore uses three academic mathematical models that mutually confirm the result.

---

## 2. THE MATHEMATICAL ENGINE — WHAT EACH MODEL DOES

### Process Graph Analysis (NetworkX)
Business process stages are modeled as a directed weighted graph. The betweenness centrality algorithm identifies the node with the highest load — this is the bottleneck. The output is concrete: "Your bottleneck is the 'In Review' stage — 80% of transitions pass through it, creating the primary delay."

Practical application: the auditor gets an operational priority — which specific process to automate first for maximum ROI.

### Absorbing Markov Chain
The transition matrix between deal stages (Qualification → Proposal → Won/Lost) is built from the client's actual CSV data. The system calculates:
- Mathematically expected deal closing time (in hours)
- P(Won) before and after automation
- Fundamental matrix N = (I-Q)⁻¹ showing how many times on average a deal passes through each stage

If the client doesn't bring a CSV, a default matrix is used to illustrate the principle.

### Bayesian Update (Beta-distribution)
Answers the question: "How much can we trust this forecast?" The auditor enters historical signals (e.g., 8 of 10 past automation projects at this company delivered results). The system updates probability from a prior (34% — standard automation prior) to a posterior. Output: "80% credible interval: the forecast is correct with probability between 68% and 92%."

### 4-Metric ROI
Four components calculated independently:
1. Labor savings = manual hours × automation % × hourly rate × 12
2. Error reduction = (errors_before - errors_after) × error_cost × volume × 12
3. Revenue from cycle acceleration = (1 - cycle_after/cycle_before) × deals/mo × avg_deal × 12
4. Revenue from conversion uplift (Markov) = (P_after - P_before) × deals/mo × avg_deal × 12

Net ROI = sum of four components - implementation cost. Payback = implementation_cost / (total_benefit / 12).

---

## 3. FULL FEATURE LIST

### Core capabilities:
- Real-time 4-metric ROI calculation
- Process graph visualization (NetworkX + Plotly)
- Absorbing Markov chain with Q and N matrices
- Bayesian update with Beta-distribution visualization
- ETL pipeline: CSV upload → automatic Markov matrix construction

### Export:
- PDF Passport (ReportLab): full financial model + 3 math models + component table + QR code
- TXT passport export
- LinkedIn hook with real client numbers (3 languages)

### UI/UX:
- Apple HIG design (SF Pro fonts, #F5F5F7 background, 18px border-radius cards)
- 3 languages: English / Russian / Serbian
- Currency switcher: EUR / RUB / RSD
- Presentation mode (sidebar hides via CSS)
- Demo mode with 3 presets: Logistics / Agency / Retail

### Personalization:
- Auditor name and contact URL (appear in PDF)
- QR code in PDF (generated from contact_url)
- Meeting notes field in PDF
- Client history (save/load to JSON)
- Scenario A vs B comparison
- Industry benchmarks on KPI cards

### Authentication:
- Demo mode without login (limited: no CSV upload, no PDF download)
- Full mode after login (superadmin)
- SHA-256 password hashing

---

## 4. BUSINESS CASES (demo data)

### Case 1: Logistics — TransLogik MSC
- 3PL operator Moscow, 14 managers
- 520 manual hours/month, avg contract 12,000 €
- 75% automation, deal cycle: 18 days → 5 days
- Net ROI: ~116,000 EUR, payback: 2.4 months

### Case 2: Agency — MOKO Digital
- Performance agency, 18 clients
- 320 hrs/month on reporting, avg account 8,500 €
- 85% automation, conversion: 71% → 92%
- Net ROI: ~95,000 EUR, payback: 1.9 months

### Case 3: Retail — MegaMarket d.o.o.
- Retail chain Belgrade, 12 stores
- 380 hrs/month, 2,200 invoices/month
- 68% automation, errors: 9.2% → 1.5%
- Net ROI: ~42,000 EUR, payback: 3.5 months

---

## 5. TECH STACK

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI | Streamlit 1.29+ | Web interface |
| Math | NumPy, SciPy | ROI, Bayesian |
| Graph | NetworkX | Centrality, bottleneck |
| ML | SciPy.stats.beta | Beta-distribution |
| ETL | Pandas | CSV → Markov matrix |
| Charts | Plotly | Waterfall, Pie, Area, Beta |
| PDF | ReportLab | A4 passport with QR |
| QR | qrcode + Pillow | QR in PDF |
| Auth | SHA-256 + JSON | Users |
| Deploy | Render.com | Production |
| Repo | GitHub (private) | Version control |

---

## 6. MARKET POSITIONING

### Target audience
Primary: independent automation consultants, AI Product Advisors, fractional CTOs.
Secondary: CRM/ERP/RPA integrators, SaaS sales teams.
Future: risk management teams in banks and financial institutions.

### Unique value proposition
"This isn't Excel and it isn't slides. It's mathematics. Three independent models — graph, Markov, Bayes — compute the same thing with different methods. When all three converge, the forecast is defended against any skeptic in the room."

### Pricing
- One-time audit: 500–1,500 EUR (EU/Serbia) / 50,000–150,000 RUB (Russia)
- Audit + roadmap: 2,000–4,000 EUR
- Retainer (quarterly): 800–1,500 EUR/quarter
- Success fee: 8–12% of net ROI above 50,000 EUR threshold

---

## 7. COMPETITIVE ANALYSIS

**vs. Excel:**
Excel computes by multiplication — one formula, one error source, easy to challenge. OmniCore uses three independent models. If even one diverges from the others, the auditor sees it.

**vs. Standard SaaS ROI Calculators:**
Most tools (Salesforce ROI Calculator, HubSpot ROI Tool) give one number with no methodology. The client doesn't know where it came from. In OmniCore, every component is shown separately, every model is visualized, every conclusion is explained.

**vs. McKinsey / Big 4:**
A Big 4 engagement costs from 50,000 EUR and takes 3 months. OmniCore delivers 80% of the same insight in 5 minutes for 1,500 EUR.

---

## 8. ROADMAP STATUS

### Phase 1: Math Engine ✅ COMPLETE
- ROI engine (4 metrics) ✅
- Graph analysis (NetworkX betweenness centrality) ✅
- Absorbing Markov chain ✅
- Bayesian update (Beta distribution) ✅
- ETL: CSV → Markov matrix ✅

### Phase 2: UI & Visualization ✅ COMPLETE
- Apple HIG design ✅
- Waterfall chart (ROI breakdown) ✅
- Pie chart (benefit structure) ✅
- Graph visualization (Plotly) ✅
- Timeline chart (payback 0–12 months) ✅
- Beta distribution chart (Bayes) ✅
- 3 languages (EN/RU/SR) ✅

### Phase 3: Export & Auth ✅ COMPLETE
- PDF Passport (ReportLab) ✅
- TXT export ✅
- LinkedIn hook ✅
- Demo mode (3 presets) ✅
- Authentication (SHA-256) ✅
- Superadmin ✅

### Phase 4: 8 New Features ✅ COMPLETE (March 2026)
- Client history (JSON) ✅
- Presentation mode (CSS toggle) ✅
- Currency switcher EUR/RUB/RSD ✅
- Meeting notes in PDF ✅
- Scenario A vs B comparison ✅
- Industry benchmarks on KPIs ✅
- QR code in PDF ✅
- Auditor name in settings ✅

### Phase 5: Deployment ✅ COMPLETE
- Render.com deployment ✅
- GitHub private repo ✅
- Streamlit config (minimal toolbar) ✅

### Phase 6: Next Steps (Planned)
- Telegram bot for quick audit (planned)
- API for integrators (planned)
- Mobile-responsive version (planned)
- Industry-specific report templates (planned)
- Multi-tenancy / SaaS version (planned)

---

## 9. KEY SOUNDBITES

- "60 seconds instead of 3 weeks"
- "Three independent mathematical models"
- "CSV in → PDF passport out"
- "Your data, not my assumptions"
- "Defended against any skeptic in the room"
- "Graph, Markov, Bayes — not Excel"
- "Break-even: 1.9 months"
- "The math is solved. The pipeline is solved."
