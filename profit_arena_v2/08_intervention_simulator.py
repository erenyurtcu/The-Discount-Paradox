from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CSV_NA_VALUES = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Hillstrom_Dataset" / "Processed"

st.title("Intervention Simulator")
st.caption("Randomized Experiment | Causal ML Decision Laboratory")

@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA / name, keep_default_na=False, na_values=CSV_NA_VALUES)

@st.cache_data
def load_parquet(name):
    return pd.read_parquet(DATA / name)

ate = load_csv("hillstrom_experimental_ate.csv")
policy = load_csv("hillstrom_policy_evaluation.csv")
agreement = load_csv("hillstrom_model_agreement.csv")

dr = load_parquet("DR_Verified_v1/hillstrom_dr_validation_predictions.parquet")
hurdle = load_parquet("hillstrom_hurdle_validation_predictions.parquet")

st.warning(
    "This simulator studies randomized EMAIL interventions, "
    "not randomized price changes. Targeting superiority "
    "has not been established."
)

st.subheader("01 | Intervention Controls")

cost = st.slider(
    "Assumed cost per email ($)",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.05
)

fractions = sorted(agreement["target_fraction"].tolist())

fraction = st.select_slider(
    "Targeted customer fraction",
    options=fractions,
    value=0.2,
    format_func=lambda x: f"{x:.0%}"
)

st.divider()
st.subheader("02 | Randomized Experiment")

rct = ate[
    (ate["treatment"] == "Mens E-Mail")
    & (ate["control"] == "No E-Mail")
    & (ate["outcome"] == "spend")
].iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric("Experimental ATE", f'${rct["ate"]:.3f}')
c2.metric("Net ATE", f'${rct["ate"] - cost:.3f}')
c3.metric(
    "95% CI",
    f'[{rct["ci95_low"] - cost:.2f}, '
    f'{rct["ci95_high"] - cost:.2f}]'
)

st.caption(
    "Net ATE subtracts a hypothetical sending cost "
    "from incremental customer spend. It is not gross profit."
)

st.divider()
st.subheader("03 | Policy Comparison")

st.info(
    "POLICY SOURCE: The policy-value chart uses exploratory "
    "holdout estimates from the earlier T-Learner evaluation. "
    "The DR-Learner and Hurdle T-Learner distributions shown "
    "later are separate model diagnostics. They are NOT "
    "the source of the policy-value estimates in this chart. "
    "The full-experiment ATE and holdout policy estimates "
    "also come from different evaluation samples."
)

p = policy.copy()

p["net_value"] = (
    p["incremental_spend_per_customer"]
    - cost * p["target_fraction"]
)

p["net_low"] = p["ci95_low"] - cost * p["target_fraction"]
p["net_high"] = p["ci95_high"] - cost * p["target_fraction"]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=p["policy"],
    y=p["net_value"],
    marker_color="#38bdf8",
    error_y=dict(
        type="data",
        symmetric=False,
        array=p["net_high"] - p["net_value"],
        arrayminus=p["net_value"] - p["net_low"]
    ),
    name="Estimated net value"
))

fig.add_hline(y=0, line_dash="dash", line_color="gray")

fig.update_layout(
    template="plotly_dark",
    title="Estimated Policy Value and 95% Confidence Intervals",
    yaxis_title="Net incremental spend ($ per customer)",
    xaxis_tickangle=-25,
    height=450
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    p[["policy", "target_fraction", "net_value",
       "net_low", "net_high"]].round(4),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Policy estimates are read from existing evaluation results. "
    "They are exploratory and should not be interpreted "
    "as proof of optimal targeting."
)

st.divider()
st.subheader("04 | Model Disagreement")

row = agreement.iloc[
    (agreement["target_fraction"] - fraction).abs().argmin()
]

a, b, c = st.columns(3)

a.metric("Targeted Fraction", f'{row["target_fraction"]:.0%}')
b.metric("Shared Customers", f'{int(row["shared_customers"]):,}')
c.metric("Model Overlap", f'{row["overlap_fraction"]:.1%}')

st.progress(float(row["overlap_fraction"]))

st.caption(
    "Overlap is calculated from the existing model-agreement "
    "analysis. The prediction files have no customer IDs, "
    "so rows are not assumed to match."
)

st.divider()
st.subheader("05 | Predicted CATE Distributions")

fig2 = go.Figure()

fig2.add_trace(go.Histogram(
    x=dr["predicted_cate"],
    name="DR-Learner",
    opacity=0.6,
    nbinsx=50
))

fig2.add_trace(go.Histogram(
    x=hurdle["predicted_cate"],
    name="Hurdle T-Learner",
    opacity=0.6,
    nbinsx=50
))

fig2.update_layout(
    template="plotly_dark",
    barmode="overlay",
    title="Distribution of Predicted Treatment Effects",
    xaxis_title="Predicted incremental spend ($)",
    yaxis_title="Customers",
    height=430
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("06 | Scientific Interpretation")

st.markdown("""
**Experimental evidence**

- Men's Email has a positive average treatment effect
  on customer spending relative to No Email.
- Net value depends on the assumed cost per email.

**Model limitations**

- CATE predictions are estimates, not observed individual effects.
- Different learners can recommend different customers.
- Superior targeting performance has not been demonstrated.
- Historical validation results do not guarantee future outcomes.

**Decision principle**

Intervention decisions should consider randomized evidence,
policy value, uncertainty, and implementation cost.
""")

st.success("Intervention Simulator loaded successfully.")

# Provenance: DR predictions come from DR_Verified_v1; policy/agreement/hurdle files remain historical exploratory analyses.
