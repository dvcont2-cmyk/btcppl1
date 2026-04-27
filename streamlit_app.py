import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Trading Dashboard",
    page_icon="📈",
)

st.sidebar.markdown("## 📊 Trading Dashboard")
st.sidebar.markdown("Use the navigation above to switch between pages.")

st.markdown("## Welcome to Trading Dashboard")
st.markdown("Select a page from the sidebar to get started:")
st.markdown("- **🪙 Crypto** — Crypto Signal Dashboard (CoinGecko)")
st.markdown("- **📈 ASX Stocks** — ASX Stock & ETF Dashboard (Yahoo Finance)")
