# Online Casino Analytics — Streamlit MVP

Purpose
-------
A Streamlit-based analytics app to analyze and optimize online casino operations:
- Player engagement & retention
- Race/leaderboard design & optimization
- Bonus/promotion experimentation
- Fraud detection & anti-abuse signals
- Segmentation, LTV, and marketing ROI analysis

Quick start (local)
-------------------
1. Create Python 3.11 virtual environment:
   python3.11 -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Download & prepare dataset (the app will call data_loader.download_dataset() automatically).
   Ensure internet is available and kagglehub can access Kaggle.

4. Run Streamlit:
   streamlit run app.py

Deploy to Streamlit Cloud
------------------------
- Push this repo to GitHub.
- From Streamlit Cloud, create a new app and set the main file to `app.py`.
- Add required secrets (if using DB or credentials) via Streamlit Cloud secrets manager.

Files
-----
- `app.py` : main Streamlit app
- `data_loader.py` : dataset download + load utilities (kagglehub)
- `src/analytics.py` : analytics functions
- `src/models.py` : modeling skeleton (churn, fraud)
- `src/utils.py` : helper functions

Notes & security
----------------
- Do NOT commit actual dataset files or credentials.
- Regulatory considerations: promotions and incentives might be regulated in certain jurisdictions — consult legal as needed before applying production promotion rules.

Author
------
Generated scaffold — extend per project needs.
