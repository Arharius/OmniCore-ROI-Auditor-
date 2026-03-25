# OmniCore ROI Auditor

A Python data analytics and ROI auditing application built with Streamlit, FastAPI, and scientific computing libraries.

## Project Structure

```
omnicore-roi-auditor/
├── app.py              # Streamlit app entry point
├── requirements.txt    # Python dependencies
├── core/               # Core business logic
│   └── __init__.py
├── etl/                # ETL (Extract, Transform, Load) pipelines
│   └── __init__.py
├── ui/                 # UI components and helpers
│   └── __init__.py
├── exports/            # Export utilities (PDF, CSV, etc.)
│   └── __init__.py
├── api/                # FastAPI backend endpoints
│   └── __init__.py
└── data/               # Data storage directory
```

## Tech Stack

- **Frontend/UI**: Streamlit (port 5000)
- **Backend API**: FastAPI + Uvicorn
- **Data Processing**: Pandas, NumPy, SciPy
- **Graph Analysis**: NetworkX
- **Visualization**: Plotly
- **PDF Export**: ReportLab
- **Python**: 3.11

## Running the App

The Streamlit app runs on port 5000:

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

## Deployment

Configured for autoscale deployment on Replit, running the Streamlit app directly.
