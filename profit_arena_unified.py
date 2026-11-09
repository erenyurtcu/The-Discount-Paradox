from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="The Discount Paradox | Unified Profit Arena",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "CAUSAL EXPERIMENTS": [
        st.Page(
            "profit_arena.py",
            title="Experiment / Policy / Models",
            icon=":material/science:",
            default=True
        )
    ],
    "RETAIL PRICING": [
        st.Page(
            "retail_pricing_lab.py",
            title="Pricing / Frontier / Uncertainty",
            icon=":material/monitoring:"
        )
    ]
}

navigation = st.navigation(pages, position="top")
navigation.run()
