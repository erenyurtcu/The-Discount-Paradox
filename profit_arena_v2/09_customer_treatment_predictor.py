"""Customer Treatment Effect Predictor — verified, precomputed DR-Learner predictions.

Place in profit_arena_v2/09_customer_treatment_predictor.py.
No model retraining. No fabricated counterfactual outcomes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer Treatment Predictor", layout="wide")
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "Hillstrom_Dataset" / "Processed" / "DR_Verified_v1"
PRED_FILE = BASE / "hillstrom_dr_test_predictions.parquet"

st.title("Individual Treatment Effect Predictor")
st.caption("Verified DR-Learner predictions | Hillstrom randomized Mens E-Mail vs No E-Mail | Outcome: spend ($)")
st.info("This page displays previously generated out-of-sample predictions for recorded test customers. It does not retrain a model or predict for arbitrary new customers.")

if not PRED_FILE.exists():
    st.error(f"Verified prediction file not found: {PRED_FILE}")
    st.stop()

@st.cache_data(show_spinner=False)
def load_predictions(path: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(path)

pred = load_predictions(str(PRED_FILE), PRED_FILE.stat().st_mtime_ns)
required = {"predicted_cate", "treatment", "spend"}
missing = sorted(required.difference(pred.columns))
if missing:
    st.error(f"Required prediction columns missing: {', '.join(missing)}. Found: {list(pred.columns)}")
    st.stop()

pred = pred.reset_index(drop=True).copy()
pred["predicted_cate"] = pd.to_numeric(pred["predicted_cate"], errors="coerce")
if pred["predicted_cate"].isna().any():
    st.error("Prediction file contains invalid or missing CATE values.")
    st.stop()

# Index is a stable display identifier within the verified test prediction file,
# NOT an external customer ID or a row number from the original raw CSV.
pred.insert(0, "test_row", np.arange(len(pred), dtype=int))

with st.sidebar:
    st.header("Customer and decision controls")
    view = st.radio("Customer selection", ["Choose test customer", "Highest predicted uplift", "Lowest predicted uplift", "Near-zero predicted uplift"])
    if view == "Choose test customer":
        chosen = st.number_input("Test customer row (0-based)", min_value=0, max_value=len(pred)-1, value=0, step=1)
        index = int(chosen)
    elif view == "Highest predicted uplift":
        index = int(pred["predicted_cate"].idxmax())
    elif view == "Lowest predicted uplift":
        index = int(pred["predicted_cate"].idxmin())
    else:
        index = int(pred["predicted_cate"].abs().idxmin())
    email_cost = st.slider("Assumed cost per email ($)", 0.0, 5.0, 0.25, 0.05)

customer = pred.iloc[index]
cate = float(customer["predicted_cate"])
net = cate - email_cost
recommend = net > 0

st.subheader("1. Selected test customer")
a, b, c = st.columns(3)
a.metric("Test customer row", f"{index:,}")
b.metric("Observed treatment", "Mens E-Mail" if int(customer["treatment"]) == 1 else "No E-Mail")
c.metric("Observed spend ($)", f"${float(customer['spend']):,.2f}")
st.caption("Observed spend is a factual outcome under the assigned treatment; it is not a counterfactual prediction.")

st.subheader("2. Model prediction and economic decision")
a, b, c = st.columns(3)
a.metric("Predicted CATE ($)", f"${cate:+,.3f}")
b.metric("Assumed email cost ($)", f"${email_cost:,.2f}")
c.metric("Estimated net incremental value ($)", f"${net:+,.3f}")
if recommend:
    st.success("Decision under the stated assumptions: SEND EMAIL (predicted net value > 0).")
else:
    st.warning("Decision under the stated assumptions: DO NOT SEND EMAIL (predicted net value ≤ 0).")
st.caption("Decision rule: predicted CATE − assumed email cost > 0. This is a model-based decision, not a proven individual causal effect or guaranteed profit.")

fig = go.Figure()
fig.add_trace(go.Bar(x=["Predicted uplift", "Assumed email cost", "Estimated net value"], y=[cate, email_cost, net],
    text=[f"${v:+.3f}" for v in (cate, email_cost, net)], textposition="auto"))
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.update_layout(title="Predicted incremental value decomposition", yaxis_title="USD per customer", showlegend=False)
st.plotly_chart(fig, width="stretch")

st.subheader("3. Compare against the verified test population")
all_cate = pred["predicted_cate"].to_numpy(dtype=float)
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=all_cate, nbinsx=50, name="Verified test CATE predictions"))
fig2.add_vline(x=cate, line_dash="dash", line_color="red", annotation_text="Selected customer")
fig2.add_vline(x=email_cost, line_dash="dot", line_color="green", annotation_text="Cost threshold")
fig2.update_layout(xaxis_title="Predicted CATE ($)", yaxis_title="Number of test customers", showlegend=False)
st.plotly_chart(fig2, width="stretch")

share = float(np.mean(all_cate > email_cost))
st.metric("Share of test customers with predicted net value > 0", f"{share:.1%}")

with st.expander("Scientific interpretation and limitations", expanded=False):
    st.markdown("""
- **Treatment:** Mens E-Mail vs No E-Mail, in a randomized campaign; **not** a randomized discount or price change.
- **Target:** Incremental *spend*, not business profit. Subtracting an assumed email cost gives a simplified net incremental spend proxy, **not validated contribution margin or profit**.
- **Prediction:** The displayed CATE is an estimate from the already-trained DR-Learner, not an observed individual treatment effect.
- **No individual counterfactual ground truth:** Each customer was observed under only one assigned treatment.
- **Data provenance:** This page uses the versioned `DR_Verified_v1` test prediction file; it does not re-fit or modify models.
- **Inference:** Customer-level CATE predictions have no individual confidence intervals here; policy superiority was not statistically established.
- **Evaluation caution:** This test split had been inspected earlier, so it is not a pristine confirmatory holdout.
""")
