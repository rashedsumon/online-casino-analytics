"""
app.py
Main Streamlit app for Online Casino Analytics.

Entrypoint for local run or Streamlit Cloud:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import time

# Local imports
from data_loader import download_dataset, list_dataset_files, load_table
from src.analytics import (
    show_overview,
    show_races_dashboard,
    show_retention_dashboard,
    show_fraud_dashboard,
    show_segmentation_dashboard,
    show_experiments_dashboard
)

st.set_page_config(page_title="Online Casino Analytics", layout="wide")

st.title("🎰 Online Casino Analytics — Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Data & Controls")
    dataset_ref = st.text_input("Kaggle dataset ref", value="yogendras843/online-casino-dataset")
    if st.button("Download / Refresh dataset"):
        with st.spinner("Downloading dataset..."):
            try:
                data_dir = download_dataset(dataset_ref)
                st.success(f"Downloaded to {data_dir}")
            except Exception as e:
                st.error(f"Download failed: {e}")

    st.markdown("---")
    page = st.radio("Choose dashboard", (
        "Overview", "Races / Leaderboards", "Retention & Churn", "Fraud Detection",
        "Segmentation & LTV", "Experiments / A/B"
    ))

    st.markdown("---")
    st.caption("Technical requirements:\nPython 3.11.0; see README for setup.")

# Basic dataset inspection section (collapsible)
with st.expander("Inspect downloaded dataset files"):
    files = list_dataset_files()
    if files:
        for f in files:
            st.write(f"- {f}")
    else:
        st.info("No dataset found yet. Click 'Download / Refresh dataset' in the sidebar.")

# Load commonly used tables lazily
@st.cache_data(ttl=3600)
def _load_table_safe(name: str):
    try:
        df = load_table(name)
        return df
    except Exception as e:
        st.warning(f"Could not load {name}: {e}")
        return pd.DataFrame()

# Example naming — adapt to actual dataset file names
players = _load_table_safe("players.csv")
transactions = _load_table_safe("transactions.csv")
bets = _load_table_safe("bets.csv")
sessions = _load_table_safe("sessions.csv")

# Route to pages
if page == "Overview":
    show_overview(players=players, transactions=transactions, bets=bets, sessions=sessions)
elif page == "Races / Leaderboards":
    show_races_dashboard(bets=bets, transactions=transactions, players=players)
elif page == "Retention & Churn":
    show_retention_dashboard(sessions=sessions, players=players)
elif page == "Fraud Detection":
    show_fraud_dashboard(bets=bets, transactions=transactions, players=players, sessions=sessions)
elif page == "Segmentation & LTV":
    show_segmentation_dashboard(players=players, transactions=transactions, bets=bets)
elif page == "Experiments / A/B":
    show_experiments_dashboard(transactions=transactions, bets=bets, players=players)

# Footer and quick links
st.markdown("---")
st.caption("Extend analytics in src/analytics.py, src/models.py, and src/utils.py")
