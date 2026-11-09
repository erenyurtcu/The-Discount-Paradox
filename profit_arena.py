from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="The Discount Paradox | Profit Arena",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
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
    return pd.read_csv(path)

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

st.sidebar.markdown("## THE DISCOUNT PARADOX")
st.sidebar.caption("Causal Machine Learning Research Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Experiment Lab",
        "Policy Arena",
        "Model Disagreement"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**Research context**")
st.sidebar.caption(
    "Hillstrom randomized email experiment. "
    "Intervention: Men's Email vs No Email."
)
st.sidebar.caption(
    "Campaign costs are hypothetical. "
    "Net values are not accounting profit."
)

st.title("THE DISCOUNT PARADOX")
st.caption(
    "INTERACTIVE CAUSAL DECISION LABORATORY  /  "
    "CSE 5007 MACHINE LEARNING AND REASONING"
)

if page == "Experiment Lab":
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

elif page == "Policy Arena":
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

    c1.metric("Best Point-Estimate Policy", winner["policy"])
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
        "These are exploratory holdout estimates for an earlier "
        "T-Learner ranking. Selecting the highest observed value "
        "does not establish statistical superiority. "
        "Campaign cost is hypothetical, and product margins "
        "are unavailable."
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

elif page == "Model Disagreement":
    st.subheader("03 / Causal Model Disagreement")

    dr = load_parquet(
        "hillstrom_dr_validation_predictions.parquet"
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
