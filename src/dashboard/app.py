"""Streamlit operations dashboard for denial-risk scoring."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("CLAIMS_API_URL", "http://localhost:8000")
PROCESSED_DATA_PATH = Path("data/processed_claims.csv")

st.set_page_config(
    page_title="Claims Denial Ops Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Healthcare Claims Operations & Risk Analyzer")
st.markdown("Real-time operational triage dashboard for medical billing teams.")

st.sidebar.header("Single Claim Scorer")
charge_amount = st.sidebar.number_input(
    "Charge Amount ($)", min_value=10.0, value=1250.0, step=50.0
)
code_count = st.sidebar.number_input(
    "Procedure Code Count", min_value=1, value=2, step=1
)
has_modifier_25 = st.sidebar.selectbox(
    "Has Modifier 25?",
    options=[0, 1],
    format_func=lambda value: "Yes" if value == 1 else "No",
)

if st.sidebar.button("Evaluate Claim Risk"):
    payload = {
        "charge_amount": charge_amount,
        "code_count": code_count,
        "has_modifier_25": has_modifier_25,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.ok:
            result = response.json()
            st.sidebar.metric(
                label="Denial Probability",
                value=f"{result['denial_probability'] * 100:.1f}%",
            )
            st.sidebar.info(
                f"**Risk Tier:** {result['risk_level']}\n\n"
                f"**Action:** {result['recommended_action']}"
            )
        else:
            detail = response.json().get("detail", "Scoring service returned an error.")
            st.sidebar.error(f"API error: {detail}")
    except requests.RequestException:
        st.sidebar.error(
            "Could not reach the FastAPI backend. Ensure it is running on port 8000."
        )

st.subheader("📊 Batch Claims Overview (Last 30 Days)")
try:
    claims = pd.read_csv(PROCESSED_DATA_PATH).head(100)
    columns = ["charge_amount", "code_count", "has_modifier_25", "denial_flag"]
    st.dataframe(claims[columns], use_container_width=True)
except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
    st.info(
        "Processed claims dataset not found. Run the preprocessing script to "
        "populate batch analytics."
    )
