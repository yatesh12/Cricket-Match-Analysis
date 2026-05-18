# CricketIQ — Enterprise Cricket Analytics Platform

**4 high-value modules · notebook-first · real business impact · Cognizant panel ready**

CricketIQ is a production-grade cricket analytics platform with four modules that solve genuinely hard problems the industry is actively working on — not textbook EDA or win probability.

---

## Architecture

```
CricketIQ/
├── src/                      # Core ML modules
│   ├── data_loader.py        # Shared data foundation (Cricsheet parser + synthetic)
│   ├── pressure_genome.py    # Module 1: Unsupervised ML + Feature Engineering
│   ├── impact_player.py      # Module 2: RL-inspired Decision AI
│   ├── broadcast_monetisation.py  # Module 3: Time-Series + Revenue Modelling
│   └── fantasy_clv.py        # Module 4: Survival Analysis + CLV Modelling
├── notebooks/                # Jupyter-compatible .py cell scripts
│   ├── module_1_pressure_genome.py
│   ├── module_2_impact_player_ai.py
│   ├── module_3_broadcast_monetisation.py
│   └── module_4_fantasy_churn_clv.py
├── app/                      # Production deployment layer
│   ├── streamlit_app.py      # Interactive War Room dashboard
│   ├── api.py                # FastAPI REST service
│   └── pdf_report.py         # Auto-generated PDF reports (ReportLab)
├── data/                     # Data directories
│   ├── raw/                  # Cricsheet JSON files
│   └── processed/            # Cleaned DataFrames
├── models/                   # Saved model artifacts
├── outputs/
│   ├── reports/              # Generated PDF reports
│   └── figures/              # Generated visualizations
├── tests/
├── requirements.txt
└── README.md
```

---

## Modules

### Module 1 — Pressure Genome 🧬

*Unsupervised ML to discover batsman psychological archetypes under match pressure.*

- 12 pressure-performance features per batsman
- PCA to 3 components + K-Means clustering
- UMAP projection for validation
- Plotly radar chart comparison (any 2 players)
- Selection recommendation engine for any match state
- **Pressure Mismatch Alert** — flags ill-suited lineups

**Stack:** pandas, scikit-learn (KMeans, PCA), UMAP, Plotly

### Module 2 — Impact Player AI 🤖

*RL-inspired decision engine for IPL Impact Player substitutions.*

- 14-dim match state vector (the RL "observation")
- XGBoost supervised baseline (ROC-AUC benchmark)
- Tabular Q-learning for substitution timing policy
- LightGBM-style candidate ranker (expected runs uplift)
- Counterfactual analysis on 5 famous IPL matches

**Stack:** XGBoost, Q-learning, cosine similarity, FastAPI

### Module 3 — Broadcast Monetisation 📺

*LSTM-powered peak engagement window predictor for ad revenue optimisation.*

- Excitement density per over (boundaries, wickets, momentum)
- Google Trends validation of proxy metric
- LSTM sequence model (predicts next 3 overs)
- Peak window detector (precision@1 metric)
- Revenue simulation: model vs random vs uniform (₹ crore)
- Auto-generated pre-match PDF Hot Zone Report

**Stack:** PyTorch (LSTM), time-series features, ReportLab, revenue simulation

### Module 4 — Fantasy Churn & CLV 💰

*Survival analysis + CLV modelling for fantasy cricket platforms.*

- Cox Proportional Hazards (handles censored data — correct approach)
- C-index evaluation (not just AUC — right metric for time-to-event)
- Kaplan-Meier curves by user segment
- XGBoost classifier with SHAP for comparison
- BG/NBD model for CLV (Buy-Till-You-Die framework)
- Intervention strategy matrix with ₹ revenue impact

**Stack:** lifelines (Cox PH), lifetimes (BG/NBD), XGBoost, SHAP, faker

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app/streamlit_app.py

# Run the FastAPI service
uvicorn app.api:app --reload

# Run a notebook as a Python script
python notebooks/module_1_pressure_genome.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/recommend-substitution` | POST | Match state → substitution timing + top-3 candidates |
| `/churn-score` | POST | User IDs → churn risk scores + CLV predictions |
| `/pressure-compatibility` | GET | Match situation → ranked batsmen by pressure fit |
| `/match-hot-zones` | GET | Match ID → predicted peak engagement windows |

## Data

- **Real data:** Place Cricsheet JSON files in `data/raw/`. The loader auto-detects and parses them.
- **Synthetic fallback:** If no Cricsheet data is found, the system generates realistic synthetic data automatically.

## Panel Presentation

Each module notebook ends with a **Summary** cell that lists key talking points. The 5 counterfactual IPL match analyses in Module 2 are the most compelling slide — real games, real decisions, model vs reality.

---

*Built for the Cognizant Cricket Analytics Panel — 4 genuinely unique approaches to problems no one has solved in production.*
