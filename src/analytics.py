"""
src/analytics.py

Contains helper functions that power the Streamlit dashboards.
Keep this file focused on visualization and high-level analytics; heavy ML / modeling
should live in src/models.py.
"""

from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta

# ---------- Utility plotting helpers ----------
def _small_kpis(df: pd.DataFrame, label: str, column: str):
    if df is None or df.empty:
        return None
    val = df[column].sum() if column in df.columns else len(df)
    return val

# ---------- OVERVIEW ----------
def show_overview(players: pd.DataFrame, transactions: pd.DataFrame, bets: pd.DataFrame, sessions: pd.DataFrame):
    """
    High-level KPIs and time-series trends.
    """
    st.header("Overview")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Players (rows)", int(players.shape[0]) if players is not None else "N/A")
    with cols[1]:
        st.metric("Total Bets", int(bets.shape[0]) if bets is not None else "N/A")
    with cols[2]:
        st.metric("Total Transactions", int(transactions.shape[0]) if transactions is not None else "N/A")
    with cols[3]:
        st.metric("Active Sessions (rows)", int(sessions.shape[0]) if sessions is not None else "N/A")

    st.markdown("### Time-series (example)")
    # attempt to find a date column in bets/transactions
    if bets is not None and not bets.empty:
        date_col = next((c for c in bets.columns if "date" in c.lower() or "time" in c.lower()), None)
        if date_col:
            try:
                bets_copy = bets.copy()
                bets_copy[date_col] = pd.to_datetime(bets_copy[date_col], errors="coerce")
                ts = bets_copy.set_index(date_col).resample("D").size().rename("bets_count").reset_index()
                fig = px.line(ts, x=date_col, y="bets_count", title="Bets per day")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write("Could not render time-series:", e)
        else:
            st.info("No obvious date/time column found in bets table to render time-series.")
    else:
        st.info("Bets table not found or empty.")

    st.markdown("### Quick segmentation example: top players by total wager")
    if transactions is not None and not transactions.empty:
        # heuristic column names
        player_col = next((c for c in transactions.columns if "player" in c.lower() or "user" in c.lower()), None)
        amount_col = next((c for c in transactions.columns if "amount" in c.lower() or "wager" in c.lower() or "bet" in c.lower()), None)
        if player_col and amount_col:
            agg = transactions.groupby(player_col)[amount_col].sum().reset_index().sort_values(amount_col, ascending=False).head(20)
            st.dataframe(agg)
        else:
            st.info("transactions table needs columns like player/user and amount for this example.")


# ---------- RACES / LEADERBOARDS ----------
def show_races_dashboard(bets: pd.DataFrame, transactions: pd.DataFrame, players: pd.DataFrame):
    """
    Focused tooling to understand races/leaderboards:
    - simulate a leaderboard by total wager or net wins during a time window
    - explore prize allocation and ROI
    """
    st.header("Races & Leaderboards")

    if bets is None or bets.empty:
        st.info("No bets table available to analyze races.")
        return

    # find timestamp and player/wager columns heuristically
    ts_col = next((c for c in bets.columns if "date" in c.lower() or "time" in c.lower()), None)
    player_col = next((c for c in bets.columns if "player" in c.lower() or "user" in c.lower()), None)
    stake_col = next((c for c in bets.columns if "stake" in c.lower() or "amount" in c.lower() or "wager" in c.lower()), None)
    profit_col = next((c for c in bets.columns if "profit" in c.lower() or "win" in c.lower() or "pnl" in c.lower()), None)

    if not player_col or not stake_col:
        st.warning("Could not find player/wager columns in bets table. Please adjust column names or inspect the dataset.")
        st.write("Available columns:", bets.columns.tolist())
        return

    # Time window controls
    with st.form("race_controls"):
        st.write("Race configuration")
        window_days = st.number_input("Window length (days)", min_value=1, max_value=30, value=7)
        top_n = st.number_input("Top N players in leaderboard", min_value=5, max_value=200, value=20)
        scoring = st.selectbox("Scoring metric", ["Total Wager", "Net Profit"], index=0)
        submitted = st.form_submit_button("Compute leaderboard")

    if submitted:
        df = bets.copy()
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            end = df[ts_col].max()
            start = end - pd.Timedelta(days=int(window_days))
            df = df[(df[ts_col] >= start) & (df[ts_col] <= end)]
        # compute metrics
        if scoring == "Total Wager":
            leaderboard = df.groupby(player_col)[stake_col].sum().reset_index().rename(columns={stake_col: "total_wager"})
            leaderboard = leaderboard.sort_values("total_wager", ascending=False).head(int(top_n))
            st.subheader("Leaderboard — Total Wager")
            st.dataframe(leaderboard)
            fig = px.bar(leaderboard, x=player_col, y="total_wager", title="Top players by total wager")
            st.plotly_chart(fig, use_container_width=True)
        else:
            if profit_col:
                leaderboard = df.groupby(player_col)[profit_col].sum().reset_index().rename(columns={profit_col: "net_profit"})
                leaderboard = leaderboard.sort_values("net_profit", ascending=False).head(int(top_n))
                st.subheader("Leaderboard — Net Profit")
                st.dataframe(leaderboard)
                fig = px.bar(leaderboard, x=player_col, y="net_profit", title="Top players by net profit")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Net Profit metric not available in bets table. Choose Total Wager instead.")


# ---------- RETENTION & CHURN ----------
def show_retention_dashboard(sessions: pd.DataFrame, players: pd.DataFrame):
    st.header("Retention & Churn")
    if sessions is None or sessions.empty:
        st.info("Sessions table not available.")
        return
    # find session timestamp and player
    ts_col = next((c for c in sessions.columns if "date" in c.lower() or "time" in c.lower()), None)
    player_col = next((c for c in sessions.columns if "player" in c.lower() or "user" in c.lower()), None)
    if not ts_col or not player_col:
        st.warning("Sessions table is missing expected columns. Columns found: " + ", ".join(sessions.columns.tolist()))
        return

    sessions_copy = sessions.copy()
    sessions_copy[ts_col] = pd.to_datetime(sessions_copy[ts_col], errors="coerce")
    sessions_copy["date"] = sessions_copy[ts_col].dt.normalize()

    # simple daily active users (DAU) trend
    dau = sessions_copy.groupby("date")[player_col].nunique().reset_index().rename(columns={player_col: "dau"})
    fig = px.line(dau, x="date", y="dau", title="Daily Active Users (DAU)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Simple cohort retention (first session cohorts)")
    # Cohort example: compute 7-day retention by cohort creation date
    first = sessions_copy.groupby(player_col)["date"].min().reset_index().rename(columns={"date": "first_date"})
    merged = sessions_copy.merge(first, on=player_col, how="left")
    merged["days_since_first"] = (merged["date"] - merged["first_date"]).dt.days
    cohort = merged.groupby(["first_date", "days_since_first"])[player_col].nunique().reset_index()
    # pivot for simple heatmap
    pivot = cohort.pivot(index="first_date", columns="days_since_first", values=player_col).fillna(0)
    st.dataframe(pivot.head(20))


# ---------- FRAUD DETECTION ----------
def show_fraud_dashboard(bets: pd.DataFrame, transactions: pd.DataFrame, players: pd.DataFrame, sessions: pd.DataFrame):
    st.header("Fraud Detection & Risk Signals")
    st.markdown("This page shows simple heuristics for fraud / bot signals.")
    # Heuristic examples: extremely high wager rates, impossible RTPs, suspicious session patterns
    if bets is None or bets.empty:
        st.info("Bets table not available.")
        return

    player_col = next((c for c in bets.columns if "player" in c.lower() or "user" in c.lower()), None)
    stake_col = next((c for c in bets.columns if "stake" in c.lower() or "amount" in c.lower()), None)
    ts_col = next((c for c in bets.columns if "date" in c.lower() or "time" in c.lower()), None)

    if not player_col or not stake_col:
        st.warning("Bets table missing expected columns.")
        st.write("Columns:", bets.columns.tolist())
        return

    df = bets.copy()
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["hour"] = df[ts_col].dt.hour

    # 1) players with unusually high bet frequency per hour
    freq = df.groupby(player_col).size().reset_index(name="bet_count")
    suspicious_freq = freq[freq["bet_count"] > freq["bet_count"].quantile(0.99)]
    st.subheader("High-frequency bettors (99th percentile)")
    st.dataframe(suspicious_freq.head(50))

    # 2) players with extreme average stake (possible whales or abuse)
    avg_stake = df.groupby(player_col)[stake_col].mean().reset_index().rename(columns={stake_col: "avg_stake"})
    suspicious_stake = avg_stake[avg_stake["avg_stake"] > avg_stake["avg_stake"].quantile(0.995)]
    st.subheader("Extreme average stake (top 0.5%)")
    st.dataframe(suspicious_stake.head(50))


# ---------- SEGMENTATION & LTV ----------
def show_segmentation_dashboard(players: pd.DataFrame, transactions: pd.DataFrame, bets: pd.DataFrame):
    st.header("Segmentation & LTV")
    st.markdown("RFM style segmentation and simple LTV proxies.")

    if players is None or players.empty:
        st.info("Players table missing.")
        return

    # Quick RFM using transactions table
    if transactions is None or transactions.empty:
        st.info("Transactions table missing.")
        return

    # heuristics for column names
    player_col = next((c for c in transactions.columns if "player" in c.lower() or "user" in c.lower()), None)
    amount_col = next((c for c in transactions.columns if "amount" in c.lower() or "value" in c.lower() or "wager" in c.lower()), None)
    date_col = next((c for c in transactions.columns if "date" in c.lower() or "time" in c.lower()), None)

    if not player_col or not amount_col or not date_col:
        st.warning("Transactions table needs player/user, amount and date columns.")
        st.write("Columns available:", transactions.columns.tolist())
        return

    tx = transactions.copy()
    tx[date_col] = pd.to_datetime(tx[date_col], errors="coerce")
    snapshot_date = tx[date_col].max() + pd.Timedelta(days=1)
    rfm = tx.groupby(player_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        amount_col: ["sum", "count"]
    })
    rfm.columns = ["recency", "monetary", "frequency"]
    rfm = rfm.reset_index()

    # simple quantile binning
    rfm["r_score"] = pd.qcut(rfm["recency"], q=4, labels=[4,3,2,1])  # lower recency => higher score
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=4, labels=[1,2,3,4])
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=4, labels=[1,2,3,4])
    rfm["rfm_score"] = rfm["r_score"].astype(int) * 100 + rfm["f_score"].astype(int) * 10 + rfm["m_score"].astype(int)

    st.subheader("RFM sample (top 20)")
    st.dataframe(rfm.sort_values("monetary", ascending=False).head(20))


# ---------- EXPERIMENTS / A/B ----------
def show_experiments_dashboard(transactions: pd.DataFrame, bets: pd.DataFrame, players: pd.DataFrame):
    st.header("Experiments & A/B Testing")
    st.markdown("Tools to analyze experiment outcomes (promotions, races, prizes).")
    st.markdown(
        "This page should be extended with: experiment registry, sample size calculators, uplift/ITT estimates, and bootstrapped confidence intervals."
    )

    # sample: basic conversion table if experiment_id exists in players or transactions
    exp_col = next((c for c in transactions.columns if "experiment" in c.lower() or "exp_id" in c.lower()), None)
    player_col = next((c for c in transactions.columns if "player" in c.lower() or "user" in c.lower()), None)
    amount_col = next((c for c in transactions.columns if "amount" in c.lower()), None)

    if exp_col and player_col:
        conv = transactions.groupby([exp_col, player_col]).size().reset_index(name="events")
        summary = conv.groupby(exp_col)["events"].agg(['count', 'sum']).reset_index().rename(columns={'count': 'unique_players', 'sum': 'total_events'})
        st.dataframe(summary)
    else:
        st.info("No experiment column found in transactions table. If you run experiments, include an experiment ID on transactions.")
