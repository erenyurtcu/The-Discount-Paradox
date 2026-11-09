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

if True:
    st.subheader("01 / Randomized Experiment")

    ate = load_csv("hillstrom_experimental_ate.csv")

    outcome = st.selectbox(
        "Outcome",
        ["spend", "conversion", "visit"],
        format_func=lambda x: {
            "spend": "Customer Spend",
            "conversion": "Purchase Conversion",
            "visit": "Website Visit"
        }[x]
    )

    selected = ate[
        (ate["outcome"] == outcome) &
        (ate["treatment"] == "Mens E-Mail")
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Treatment Mean", f"{selected['treated_mean']:.4f}")
    c2.metric("Control Mean", f"{selected['control_mean']:.4f}")
    c3.metric("Average Treatment Effect", f"{selected['ate']:+.4f}")
    c4.metric("P-value", f"{selected['p_value_normal']:.6f}")

    st.markdown("### Estimated treatment effects")

    chart_data = ate[ate["outcome"] == outcome].copy()
    chart_data["label"] = chart_data["treatment"]

    fig = go.Figure()

    for _, row in chart_data.iterrows():
        color = (
            COLORS["blue"]
            if row["treatment"] == "Mens E-Mail"
            else COLORS["purple"]
        )

        fig.add_trace(go.Scatter(
            x=[row["ate"]],
            y=[row["label"]],
            mode="markers",
            marker=dict(size=15, color=color),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[row["ci95_high"] - row["ate"]],
                arrayminus=[row["ate"] - row["ci95_low"]],
                color=color,
                thickness=2,
                width=7
            ),
            name=row["treatment"]
        ))

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#94A3B8"
    )

    fig.update_layout(
        xaxis_title="Average Treatment Effect",
        yaxis_title=""
    )

    st.plotly_chart(
        plot_style(fig),
        use_container_width=True
    )

    st.info(
        "Random assignment supports a causal interpretation "
        "of average email treatment effects. "
        "This experiment is not a randomized price-discount trial."
    )

