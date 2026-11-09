from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dominicks_Dataset" / "Analysis"

st.title("Model Evidence")
st.caption(
    "Dominick's Diet Coke | Observational pricing evidence "
    "| Not a causal effect estimate"
)

def load_csv(filename):
    path = DATA / filename
    if not path.exists():
        st.error(f"Missing analysis file: {path}")
        st.stop()
    return pd.read_csv(path)

week = load_csv("diet_coke_week_adjusted_results.csv")
promo = load_csv("diet_coke_promotion_adjusted_results.csv")
temporal = load_csv("diet_coke_temporal_stability.csv")
validation = load_csv("diet_coke_out_of_time_validation.csv")

st.warning(
    "These regressions estimate observational price-sales "
    "associations, not experimentally identified causal "
    "price elasticities. Do not interpret model coefficients "
    "as guaranteed intervention effects."
)

st.subheader("01 | Model Comparison")

raw = week.loc[
    week["model"] == "Raw first differences"
].iloc[0]

adjusted = week.loc[
    week["model"] == "Week-adjusted differences"
].iloc[0]

controlled = promo.loc[
    promo["model"] == "Week + promotion-adjusted"
].iloc[0]

comparison = pd.DataFrame([
    {
        "Model": "Raw first differences",
        "Coefficient": raw["coefficient"],
        "CI Lower": raw["ci_lower"],
        "CI Upper": raw["ci_upper"],
    },
    {
        "Model": "Week-adjusted",
        "Coefficient": adjusted["coefficient"],
        "CI Lower": adjusted["ci_lower"],
        "CI Upper": adjusted["ci_upper"],
    },
    {
        "Model": "Week + promotion-adjusted",
        "Coefficient": controlled["price_coefficient"],
        "CI Lower": controlled["ci_lower"],
        "CI Upper": controlled["ci_upper"],
    },
])

c1, c2, c3 = st.columns(3)

c1.metric(
    "Raw Association",
    f'{raw["coefficient"]:.3f}'
)
c2.metric(
    "Week-Adjusted",
    f'{adjusted["coefficient"]:.3f}'
)
c3.metric(
    "Promotion-Adjusted",
    f'{controlled["price_coefficient"]:.3f}'
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=comparison["Coefficient"],
    y=comparison["Model"],
    mode="markers",
    marker=dict(size=13, color="#38bdf8"),
    error_x=dict(
        type="data",
        symmetric=False,
        array=(
            comparison["CI Upper"] - comparison["Coefficient"]
        ),
        arrayminus=(
            comparison["Coefficient"] - comparison["CI Lower"]
        ),
        thickness=2,
        width=6,
    ),
    name="Coefficient and 95% CI",
))

fig.add_vline(x=0, line_dash="dash", line_color="gray")

fig.update_layout(
    title="Price-Sales Association | 95% Confidence Intervals",
    xaxis_title="Regression coefficient (not causal)",
    yaxis_title="",
    template="plotly_dark",
    height=380,
    margin=dict(l=20, r=20, t=60, b=30),
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    comparison.round(4),
    hide_index=True,
    use_container_width=True,
)

st.info(
    "The large shift from the raw to the week-adjusted "
    "coefficient indicates that synchronized weekly "
    "movements materially influence the observed relationship."
)

st.divider()
st.subheader("02 | Temporal Stability")

temporal_display = temporal[[
    "period",
    "observations",
    "stores",
    "coefficient",
    "ci_lower",
    "ci_upper",
    "r_squared",
]].copy()

st.dataframe(
    temporal_display.round(4),
    hide_index=True,
    use_container_width=True,
)

first = temporal.loc[
    temporal["period"] == "First half", "coefficient"
].iloc[0]

second = temporal.loc[
    temporal["period"] == "Second half", "coefficient"
].iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric("First Half", f"{first:.3f}")
c2.metric("Second Half", f"{second:.3f}")
c3.metric(
    "Coefficient Difference",
    f"{second - first:+.3f}"
)

st.caption(
    "The two periods have different store coverage. "
    "Similar coefficients do not establish causal identification "
    "or statistical equivalence."
)

st.divider()
st.subheader("03 | Out-of-Time Validation")

validation_display = validation.copy()

st.dataframe(
    validation_display.round(5),
    hide_index=True,
    use_container_width=True,
)

baseline = validation.loc[
    validation["model"] == "Zero-change baseline"
].iloc[0]

price_model = validation.loc[
    validation["model"] == "Price association model"
].iloc[0]

improvement = (
    (baseline["rmse"] - price_model["rmse"])
    / baseline["rmse"] * 100
)

c1, c2, c3 = st.columns(3)

c1.metric("Baseline RMSE", f'{baseline["rmse"]:.4f}')
c2.metric("Price Model RMSE", f'{price_model["rmse"]:.4f}')
c3.metric("RMSE Improvement", f"{improvement:.2f}%")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=validation["model"],
    y=validation["rmse"],
    name="RMSE",
))

fig2.add_trace(go.Bar(
    x=validation["model"],
    y=validation["mae"],
    name="MAE",
))

fig2.update_layout(
    title="Out-of-Time Error Comparison",
    barmode="group",
    template="plotly_dark",
    height=370,
    yaxis_title="Error in week-demeaned log-sales changes",
)

st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "The validation uses test-period weekly means for "
    "demeaning and observed test-period prices. "
    "It is not a deployable real-time forecast evaluation."
)

st.divider()
st.subheader("04 | Scientific Interpretation")

st.markdown("""
**Supported by the evidence**

- Price and sales changes show a strong negative association.
- Common weekly movements explain a substantial portion
  of the raw association.
- Week-adjusted coefficients are relatively stable
  across the two historical periods.
- The price association model reduces error on
  week-demeaned held-out observations.

**Not established by the evidence**

- Random assignment of price interventions.
- A causally identified price elasticity.
- A validated heterogeneous causal treatment effect.
- Guaranteed profit improvement from changing prices.

**Research positioning**

Dominick's supports observational pricing diagnostics
and assumption-based profit scenarios.

Hillstrom provides randomized email-treatment evidence
for causal intervention and policy evaluation.
The interventions are different and must not be conflated.
""")

st.success("Model Evidence | Analysis files loaded successfully")
