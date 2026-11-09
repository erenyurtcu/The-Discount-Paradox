from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

CSV_NA_VALUES = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']


st.set_page_config(
    page_title="The Discount Paradox | Profit Arena",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Hillstrom_Dataset" / "Processed"

st.markdown("""
<style>
.stApp {
    background: #080D18;
    color: #E8EDF7;
}
[data-testid="stSidebar"] {
    background: #101827;
}
h1, h2, h3 {
    color: #F4F7FF !important;
}
div[data-testid="stMetric"] {
    background: #141F31;
    border: 1px solid #26364D;
    border-radius: 14px;
    padding: 18px;
}
div[data-testid="stMetricLabel"] {
    color: #A9B8CC;
}
div[data-testid="stMetricValue"] {
    color: #E8F2FF;
}
</style>
""", unsafe_allow_html=True)

COLORS = {
    "blue": "#38BDF8",
    "green": "#34D399",
    "orange": "#FBBF24",
    "red": "#FB7185",
    "purple": "#A78BFA"
}

def load_csv(name):
    path = DATA / name
    if not path.exists():
        st.error(f"Required data file missing: {path}")
        st.stop()
    return pd.read_csv(path, keep_default_na=False, na_values=CSV_NA_VALUES)

def load_parquet(name):
    path = DATA / name
    if not path.exists():
        st.error(f"Required data file missing: {path}")
        st.stop()
    return pd.read_parquet(path)

def plot_style(fig, height=440):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080D18",
        plot_bgcolor="#111A2A",
        font=dict(color="#DCE7F5"),
        height=height,
        margin=dict(l=25, r=25, t=45, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    return fig


st.title("THE DISCOUNT PARADOX")
st.caption(
    "INTERACTIVE CAUSAL DECISION LABORATORY  /  "
    "CSE 5007 MACHINE LEARNING AND REASONING"
)

if True and "Policy Arena":
    st.subheader("02 / Intervention Policy Simulator")

    policy = load_csv("hillstrom_policy_evaluation.csv")

    cost = st.slider(
        "Hypothetical cost per email",
        min_value=0.0,
        max_value=2.0,
        value=0.50,
        step=0.05
    )

    policy["net_value"] = (
        policy["incremental_spend_per_customer"]
        - cost * policy["target_fraction"]
    )

    policy["net_ci_low"] = (
        policy["ci95_low"]
        - cost * policy["target_fraction"]
    )

    policy["net_ci_high"] = (
        policy["ci95_high"]
        - cost * policy["target_fraction"]
    )

    winner = policy.loc[policy["net_value"].idxmax()]

    c1, c2, c3 = st.columns(3)

    c1.metric("Highest Estimated Value (Not Proven Best)", winner["policy"])
    c2.metric("Estimated Net Value", f"{winner['net_value']:+.4f}")
    c3.metric("Targeted Population", f"{winner['target_fraction']:.0%}")

    policy = policy.sort_values("net_value")

    colors = [
        COLORS["green"] if x == winner["policy"]
        else COLORS["blue"]
        for x in policy["policy"]
    ]

    fig = go.Figure(go.Bar(
        x=policy["net_value"],
        y=policy["policy"],
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            symmetric=False,
            array=policy["net_ci_high"] - policy["net_value"],
            arrayminus=policy["net_value"] - policy["net_ci_low"],
            color="#E2E8F0"
        )
    ))

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#94A3B8"
    )

    fig.update_layout(
        xaxis_title="Net Incremental Spend Minus Email Cost",
        yaxis_title=""
    )

    st.plotly_chart(
        plot_style(fig, 510),
        use_container_width=True
    )

    st.warning(
        "SCIENTIFIC INTERPRETATION: These are exploratory holdout "
        "policy estimates based on an earlier T-Learner ranking. "
        "The highest point estimate is NOT evidence of a "
        "statistically superior policy. Confidence intervals "
        "and paired policy differences must be considered. "
        "The email cost is hypothetical; these values represent "
        "incremental customer spending minus assumed sending "
        "cost, NOT observed accounting profit."
    )

    st.info(
        "SAMPLE DISTINCTION: The full Hillstrom randomized "
        "experiment gives a Men's Email spending ATE of "
        "approximately $0.770 per customer. The policy estimates "
        "shown here come from a separate evaluation subset, "
        "so their numerical values need not equal the "
        "full-experiment ATE."
    )

    st.dataframe(
        policy[[
            "policy",
            "target_fraction",
            "net_value",
            "net_ci_low",
            "net_ci_high"
        ]].round(4),
        use_container_width=True,
        hide_index=True
    )

