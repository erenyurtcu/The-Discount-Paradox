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

    discount = st.slider(
        "Discount (%)",
        min_value=0,
        max_value=50,
        value=15,
        step=1
    )

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


p1, q1, r1, g1 = simulate(
    discount / 100.0,
    elasticity,
    cost
)

r0 = p0 * q0
g0 = (p0 - cost) * q0

a, b, c, d = st.columns(4)

a.metric(
    "New unit price",
    f"${p1:.3f}",
    delta=f"{-discount}%"
)

b.metric(
    "Predicted weekly units",
    f"{q1:,.1f}",
    delta=f"{(q1 / q0 - 1) * 100:+.1f}%"
)

c.metric(
    "Scenario revenue",
    f"${r1:,.2f}",
    delta=f"{(r1-r0):+,.2f} $"
)

profit_change_pct = (
    (g1 / g0 - 1) * 100
    if abs(g0) > 1e-9
    else None
)

d.metric(
    "Scenario gross profit",
    f"${g1:,.2f}",
    delta=(
        f"{profit_change_pct:+.1f}%"
        if profit_change_pct is not None
        else "N/A"
    )
)

st.divider()

discount_grid = np.linspace(0, 0.50, 101)
rows = []

for disc in discount_grid:
    price, units, revenue, profit = simulate(
        disc, elasticity, cost
    )
    rows.append({
        "Discount (%)": disc * 100,
        "Unit price": price,
        "Units": units,
        "Revenue": revenue,
        "Gross profit": profit
    })

curve = pd.DataFrame(rows)

left, right = st.columns(2)

with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["Discount (%)"],
        y=curve["Revenue"],
        mode="lines",
        name="Revenue",
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=curve["Discount (%)"],
        y=curve["Gross profit"],
        mode="lines",
        name="Gross profit",
        line=dict(width=3)
    ))
    fig.add_vline(
        x=discount,
        line_dash="dash",
        annotation_text="Selected discount"
    )
    fig.update_layout(
        title="Revenue vs. Gross Profit",
        xaxis_title="Discount (%)",
        yaxis_title="Dollars per representative store-week",
        template="plotly_dark",
        height=440
    )
    st.plotly_chart(fig, width="stretch")

with right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["Discount (%)"],
        y=curve["Units"],
        mode="lines",
        name="Predicted units",
        line=dict(width=3)
    ))
    fig.add_vline(
        x=discount,
        line_dash="dash",
        annotation_text="Selected discount"
    )
    fig.update_layout(
        title="Assumed Demand Response",
        xaxis_title="Discount (%)",
        yaxis_title="Predicted weekly units",
        template="plotly_dark",
        height=440
    )
    st.plotly_chart(fig, width="stretch")

best_idx = curve["Gross profit"].idxmax()
best = curve.loc[best_idx]

st.subheader("Scenario Profit Optimization")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Profit-maximizing discount (within 0–50%)",
    f"{best['Discount (%)']:.1f}%"
)
col2.metric(
    "Price at scenario optimum",
    f"${best['Unit price']:.3f}"
)
col3.metric(
    "Maximum scenario gross profit",
    f"${best['Gross profit']:,.2f}"
)

if best["Discount (%)"] in [0, 50]:
    st.info(
        "The scenario optimum is at the boundary of the "
        "tested discount range. It is not necessarily "
        "the unrestricted optimum."
    )

st.caption(
    "Baseline price, sales and cost are separate medians "
    "from the observational dataset. Together they define "
    "a synthetic representative store-week, not an actual "
    "observed store-week. Gross profit excludes fixed costs "
    "and cross-product effects."
)

st.subheader("Sensitivity Analysis")

sensitivity = []

for e in [-1.0, -1.5, -2.0, -2.5, -3.0]:
    _, units, revenue, profit = simulate(
        discount / 100.0, e, cost
    )
    sensitivity.append({
        "Assumed elasticity": e,
        "Weekly units": round(units, 1),
        "Revenue ($)": round(revenue, 2),
        "Gross profit ($)": round(profit, 2),
        "Profit change ($)": round(profit - g0, 2)
    })

st.dataframe(
    pd.DataFrame(sensitivity),
    width="stretch",
    hide_index=True
)

st.info(
    "Explore Uncertainty Lab for elasticity sensitivity, "
    "Profit Frontier for assumption-based price optimization, "
    "and Model Evidence for observational diagnostics. "
    "Hillstrom provides separate randomized evidence for "
    "email interventions, not price changes."
)



