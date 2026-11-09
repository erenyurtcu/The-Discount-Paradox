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

if True and "Model Disagreement":
    st.subheader("03 / Causal Model Disagreement")

    dr = load_parquet(
        "DR_Verified_v1/hillstrom_dr_validation_predictions.parquet"
    )
    hurdle = load_parquet(
        "hillstrom_hurdle_validation_predictions.parquet"
    )
    agreement = load_csv("hillstrom_model_agreement.csv")

    if len(dr) != len(hurdle):
        st.error("Validation predictions have different lengths.")
        st.stop()

    if not np.array_equal(
        dr["treatment"].to_numpy(),
        hurdle["treatment"].to_numpy()
    ):
        st.error("Validation row alignment failed.")
        st.stop()

    if not np.allclose(
        dr["spend"].to_numpy(),
        hurdle["spend"].to_numpy()
    ):
        st.error("Validation outcome alignment failed.")
        st.stop()

    a = dr["predicted_cate"].to_numpy()
    b = hurdle["predicted_cate"].to_numpy()

    spearman = pd.Series(a).corr(
        pd.Series(b),
        method="spearman"
    )

    top_fraction = st.slider(
        "Targeting fraction",
        min_value=10,
        max_value=80,
        value=20,
        step=10
    ) / 100

    k = round(top_fraction * len(a))

    dr_top = set(np.argsort(-a)[:k])
    hurdle_top = set(np.argsort(-b)[:k])

    shared = len(dr_top & hurdle_top)
    overlap = shared / k
    jaccard = shared / len(dr_top | hurdle_top)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Spearman Correlation", f"{spearman:.3f}")
    c2.metric("Shared Customers", f"{shared:,}")
    c3.metric("Targeting Overlap", f"{overlap:.1%}")
    c4.metric("Jaccard Similarity", f"{jaccard:.3f}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=a,
        y=b,
        mode="markers",
        marker=dict(
            size=5,
            opacity=0.32,
            color=COLORS["blue"]
        ),
        name="Validation Customers"
    ))

    fig.update_layout(
        xaxis_title="DR-Learner Predicted CATE",
        yaxis_title="Hurdle T-Learner Predicted CATE"
    )

    st.plotly_chart(
        plot_style(fig, 490),
        use_container_width=True
    )

    fig2 = px.line(
        agreement,
        x="target_fraction",
        y="overlap_fraction",
        markers=True,
        title="Top-K Targeting Agreement"
    )

    fig2.update_traces(
        line_color=COLORS["green"],
        line_width=3
    )

    fig2.update_layout(
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        plot_style(fig2, 390),
        use_container_width=True
    )

    st.info(
        "Ranking agreement measures whether models prioritize "
        "the same customers. It does not measure causal accuracy "
        "or prove that one targeting policy is superior."
    )

st.divider()
st.caption(
    "THE DISCOUNT PARADOX | Research Prototype v0.1 | "
    "Randomized email effects and exploratory causal targeting"
)

# Provenance: DR predictions come from DR_Verified_v1; policy/agreement/hurdle files remain historical exploratory analyses.
