import streamlit as st

st.set_page_config(
    page_title="The Discount Paradox | Profit Arena V2",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "CAUSAL EXPERIMENTS": [
        st.Page("profit_arena_v2/01_experiment_lab.py",
                title="Experiment Lab"),
        st.Page("profit_arena_v2/02_policy_arena.py",
                title="Policy Arena"),
        st.Page("profit_arena_v2/03_model_disagreement.py",
                title="Model Disagreement"),
        st.Page("profit_arena_v2/09_customer_treatment_predictor.py", title="Customer Treatment Predictor"),
        st.Page("profit_arena_v2/08_intervention_simulator.py",
                title="Intervention Simulator")
    ],
    "RETAIL PRICING": [
        st.Page("profit_arena_v2/04_retail_pricing_lab.py",
                title="Retail Pricing Lab"),
        st.Page("profit_arena_v2/05_profit_frontier.py",
                title="Profit Frontier"),
        st.Page("profit_arena_v2/06_uncertainty_lab.py",
                title="Uncertainty Lab"),
        st.Page("profit_arena_v2/07_model_evidence.py",
                title="Model Evidence")
    ]
}

navigation = st.navigation(pages, position="top")
navigation.run()

