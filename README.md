# The Discount Paradox
### A Multi-Dataset Causal Machine Learning Framework for Retail Pricing and Profit Optimization

An academic research project investigating heterogeneous treatment effects, campaign targeting, observational price-demand relationships, and retail decision-making.

## Research objectives

- Estimate the average effect of randomized email marketing.
- Estimate heterogeneous treatment effects using causal machine learning.
- Compare campaign targeting policies and intervention costs.
- Analyze observational retail pricing and promotion patterns.
- Explore revenue and gross-profit trade-offs under explicit assumptions.

## Datasets

**Hillstrom E-Mail Marketing:** Randomized marketing experiment used to study the effect of Mens E-Mail versus No E-Mail on customer spending.

**Dominick's Finer Foods:** Retail pricing data used for observational price-demand analysis. This repository includes a compact prepared analysis dataset rather than the full raw archive.

**Dunnhumby:** Additional retail dataset included for supporting research and auditing.

## Application modules

1. Experiment Lab — Randomized campaign effects.
2. Policy Arena — Campaign targeting policies.
3. Model Disagreement — Comparison of treatment-effect estimates.
4. Retail Pricing Lab — Observational retail pricing analysis.
5. Profit Frontier — Revenue and gross-profit scenarios.
6. Uncertainty Lab — Sensitivity to assumptions.
7. Model Evidence — Pricing model diagnostics.
8. Intervention Simulator — Marketing policy and cost scenarios.
9. Customer Treatment Predictor — Previously generated out-of-sample customer treatment-effect predictions.

## Requirements

- Python 3.10 recommended
- Windows, macOS, or Linux
- Internet access for initial dependency installation

## Installation (Windows PowerShell)

Clone or download this repository, open PowerShell inside the project directory, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run profit_arena_v2.py
```

Alternatively, on Windows, double-click `run.bat`.

The application should become available at http://localhost:8501.

## Scientific interpretation

The Hillstrom email campaign is randomized, allowing causal estimation under the experimental design assumptions.

The Dominick's pricing analysis is observational. Its price-demand coefficients should not be interpreted as experimentally identified causal elasticities.

The DR-Learner estimates conditional average treatment effects (CATEs). Predictions do not establish individual causal effects.

The targeting policy was not demonstrated to be statistically superior to emailing everyone in the reported final test evaluation.

Predicted incremental customer spending minus an assumed email cost is not equivalent to verified accounting profit.

## Reproducibility and limitations

This is a compact research and application package. It contains the prepared inputs and outputs needed for the main application, but not the complete 15 GB raw Dominick's data archive.

Some historical audit scripts require additional raw data and are provided for methodological inspection rather than guaranteed execution.

The Customer Treatment Predictor displays precomputed out-of-sample estimates for recorded test observations. It is not a deployed inference service for arbitrary new customers.

## Academic context

Course: CSE 5007 — Machine Learning and Reasoning

Project: The Discount Paradox

## References

- Rubin (1974), Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies.
- Rosenbaum and Rubin (1983), The Central Role of the Propensity Score in Observational Studies for Causal Effects.
- Kunzel et al. (2019), Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning.
- Kennedy (2023), Towards Optimal Doubly Robust Estimation of Heterogeneous Causal Effects.
- Athey and Wager (2021), Policy Learning With Observational Data.

## Academic Reports

The complete academic reports are available in two languages:

- [English Academic Report](reports/The_Discount_Paradox_Academic_Report_EN.docx)
- [Turkish Academic Report](reports/The_Discount_Paradox_Academic_Report_TR.docx)

Both reports are provided in Microsoft Word format.
