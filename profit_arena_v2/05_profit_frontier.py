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

    elasticity = st.slider(
        "Assumed price elasticity",
        min_value=-4.0,
        max_value=-0.5,
        value=-2.0,
        step=0.1
    )

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

def simulate(discount_fraction, elasticity, cost):
    price = p0 * (1 - discount_fraction)
    units = q0 * (price / p0) ** elasticity
    revenue = price * units
    profit = (price - cost) * units
    return price, units, revenue, profit


st.divider()
st.subheader("Profit Frontier - Price Optimization")
st.caption(
    "Explore price cuts and price increases across alternative "
    "demand elasticity assumptions."
)

price_change_grid = np.linspace(-50, 150, 401)
elasticity_grid = np.linspace(-4.0, -0.5, 71)

frontier_rows = []

for e in elasticity_grid:
    scenario_prices = p0 * (1 + price_change_grid / 100)
    scenario_units = q0 * (scenario_prices / p0) ** e
    scenario_profits = (scenario_prices - cost) * scenario_units

    best_idx = int(np.argmax(scenario_profits))

    frontier_rows.append({
        "Elasticity": e,
        "Optimal price change (%)": price_change_grid[best_idx],
        "Optimal price ($)": scenario_prices[best_idx],
        "Maximum gross profit ($)": scenario_profits[best_idx]
    })

frontier = pd.DataFrame(frontier_rows)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=frontier["Elasticity"],
    y=frontier["Optimal price change (%)"],
    mode="lines",
    name="Optimal price change",
    line=dict(width=4),
    customdata=frontier[
        ["Optimal price ($)", "Maximum gross profit ($)"]
    ].to_numpy(),
    hovertemplate=(
        "Elasticity: %{x:.2f}<br>"
        "Optimal price change: %{y:+.1f}%<br>"
        "Optimal price: $%{customdata[0]:.3f}<br>"
        "Gross profit: $%{customdata[1]:.2f}"
        "<extra></extra>"
    )
))

fig.add_hline(
    y=0,
    line_dash="dot",
    annotation_text="Current price"
)

fig.add_vline(
    x=elasticity,
    line_dash="dash",
    annotation_text="Selected elasticity"
)

fig.update_layout(
    title="Optimal Price Change Across Demand Scenarios",
    xaxis_title="Assumed price elasticity",
    yaxis_title="Profit-maximizing price change (%)",
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, width="stretch")

selected_prices = p0 * (1 + price_change_grid / 100)
selected_units = q0 * (selected_prices / p0) ** elasticity
selected_profits = (selected_prices - cost) * selected_units
selected_idx = int(np.argmax(selected_profits))

col1, col2, col3 = st.columns(3)

col1.metric(
    "Optimal price change",
    f"{price_change_grid[selected_idx]:+.1f}%"
)
col2.metric(
    "Optimal unit price",
    f"${selected_prices[selected_idx]:.3f}"
)
col3.metric(
    "Maximum scenario gross profit",
    f"${selected_profits[selected_idx]:.2f}"
)

if selected_idx in (0, len(price_change_grid) - 1):
    st.info(
        "The optimum lies at the search boundary. "
        "It may differ under a wider price range."
    )

st.warning(
    "These are assumption-driven pricing scenarios, not "
    "experimentally identified causal effects. "
    "Demand follows a constant-elasticity model; unit cost "
    "is held fixed and competitor responses are excluded."
)




