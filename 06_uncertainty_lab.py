from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Retail Pricing Lab | The Discount Paradox",
    page_icon="📊",
    layout="wide"
)

DATA = Path(__file__).resolve().parent.parent / "Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet"

@st.cache_data
def load_baseline():
    con = duckdb.connect()
    result = con.execute("""
        SELECT
            MEDIAN(price / qty) AS baseline_price,
            MEDIAN(move) AS baseline_units,
            MEDIAN(
                (price / qty) * (1 - margin_pct / 100.0)
            ) AS baseline_cost,
            COUNT(*) AS observations,
            COUNT(DISTINCT store) AS stores,
            COUNT(DISTINCT week) AS weeks
        FROM read_parquet(?)
        WHERE price > 0
          AND qty > 0
          AND move > 0
          AND margin_pct IS NOT NULL
          AND margin_pct <= 100
    """, [str(DATA)]).fetchdf().iloc[0]
    con.close()
    return result

if not DATA.exists():
    st.error(f"Dataset not found: {DATA}")
    st.stop()

baseline = load_baseline()

p0 = float(baseline["baseline_price"])
q0 = float(baseline["baseline_units"])
c0 = float(baseline["baseline_cost"])

st.title("The Discount Paradox")
st.subheader("Retail Pricing Lab")
st.caption(
    "Dominick's | Diet Coke 2 LT | "
    "Observational data + assumption-driven scenario simulation"
)

st.warning(
    "Research limitation: The demand response shown here is "
    "a scenario based on an assumed price elasticity. "
    "It is NOT an experimentally identified causal price effect."
)

st.info(
    "BASELINE CONSTRUCTION: Price, weekly unit sales, and "
    "implied unit cost are calculated as separate medians "
    "from valid positive-sales observations. The resulting "
    "baseline is a synthetic representative scenario, "
    "not an actual observed store-week. Zero-sales records "
    "are excluded from these descriptive medians. "
    "The demand response is assumed rather than causally identified."
)

with st.sidebar:
    st.header("Pricing controls")

    cost = st.slider(
        "Assumed unit cost ($)",
        min_value=0.0,
        max_value=2.0,
        value=float(np.clip(c0, 0, 2)),
        step=0.01
    )

    st.divider()
    st.write("**Data baseline**")
    st.write(f"Median unit price: ${p0:.3f}")
    st.write(f"Median weekly store sales: {q0:.1f} units")
    st.write(f"Median implied unit cost: ${c0:.3f}")
    st.write(f"Valid observations: {int(baseline['observations']):,}")
    st.write(f"Stores: {int(baseline['stores'])}")
    st.write(f"Weeks: {int(baseline['weeks'])}")

st.divider()
st.subheader("Uncertainty Lab")
st.caption(
    "Stress-test a pricing decision across plausible demand "
    "elasticities. This is a scenario envelope, not a "
    "statistical confidence interval."
)

low_e, high_e = st.slider(
    "Elasticity scenario range",
    min_value=-4.0,
    max_value=-0.5,
    value=(-3.0, -1.0),
    step=0.1
)

tested_change = st.slider(
    "Price change to stress-test (%)",
    min_value=-50,
    max_value=150,
    value=53,
    step=1,
    help="Negative values are discounts; positive values are increases."
)

scenario_elasticities = np.linspace(low_e, high_e, 101)
test_price = p0 * (1 + tested_change / 100)

scenario_units = (
    q0 * (test_price / p0) ** scenario_elasticities
)
scenario_profits = (test_price - cost) * scenario_units
baseline_profit = (p0 - cost) * q0
profit_changes = scenario_profits - baseline_profit

min_change = float(np.min(profit_changes))
max_change = float(np.max(profit_changes))
profitable_share = float(np.mean(profit_changes > 0))

u1, u2, u3 = st.columns(3)

u1.metric(
    "Worst scenario profit change",
    f"${min_change:+.2f}"
)
u2.metric(
    "Best scenario profit change",
    f"${max_change:+.2f}"
)
u3.metric(
    "Share of tested scenarios with profit gain",
    f"{profitable_share * 100:.1f}%"
)

uncertainty_fig = go.Figure()

uncertainty_fig.add_trace(go.Scatter(
    x=scenario_elasticities,
    y=profit_changes,
    mode="lines",
    name="Gross profit change",
    line=dict(width=4),
    hovertemplate=(
        "Elasticity: %{x:.2f}<br>"
        "Profit change: $%{y:+.2f}"
        "<extra></extra>"
    )
))

uncertainty_fig.add_hline(
    y=0,
    line_dash="dash",
    annotation_text="Break-even vs. current price"
)

uncertainty_fig.update_layout(
    title="How Robust Is Your Pricing Decision?",
    xaxis_title="Assumed price elasticity",
    yaxis_title="Change in weekly gross profit ($)",
    template="plotly_dark",
    height=470
)

st.plotly_chart(uncertainty_fig, width="stretch")

if min_change > 0:
    st.success(
        "This pricing decision increases gross profit "
        "in every tested elasticity scenario."
    )
elif max_change < 0:
    st.error(
        "This pricing decision reduces gross profit "
        "in every tested elasticity scenario."
    )
else:
    st.warning(
        "The decision is sensitive to demand elasticity: "
        "some tested scenarios gain profit, others lose profit."
    )

st.caption(
    "Scenario points are equally spaced elasticity assumptions, "
    "not equally likely outcomes. The scenario share is not a "
    "probability of success. Cost is held fixed, and the "
    "demand response is not causally identified."
)

