# Hillstrom: verified DR-Learner subgroup audit

Treatment: Mens E-Mail versus No E-Mail; outcome: spend ($).
Subgroup definitions use only baseline variables. Median cutpoints derived from development sample.
Predicted CATE averages are **descriptive model predictions**, not demonstrated treatment effects.
Randomized subgroup ATE is an independent treatment-minus-control spend difference within each group.
95% CIs are exploratory normal approximations; multiple testing not adjusted.
Previously inspected test set is not an untouched confirmatory holdout.

## Subgroup results

### validation
| Feature | Group | N | Mean predicted CATE ($) | Randomized subgroup ATE ($) | Exploratory 95% CI ($) |
|---|---|---:|---:|---:|---|
| newbie_group | Existing customer | 3175 | +1.221 | -0.297 | [-1.137, +0.543] |
| newbie_group | New customer | 3217 | +0.573 | +0.495 | [-0.113, +1.103] |
| channel | Multichannel | 751 | +1.633 | +0.872 | [-0.024, +1.768] |
| channel | Phone | 2753 | +0.704 | +0.286 | [-0.510, +1.083] |
| channel | Web | 2888 | +0.885 | -0.273 | [-1.108, +0.563] |
| zip_code | Rural | 984 | +1.261 | -0.047 | [-1.056, +0.962] |
| zip_code | Surburban | 2877 | +0.807 | +0.278 | [-0.472, +1.028] |
| zip_code | Urban | 2531 | +0.852 | -0.049 | [-0.967, +0.869] |
| recency_group | Higher recency | 2738 | +0.515 | +0.255 | [-0.641, +1.151] |
| recency_group | Lower recency | 3654 | +1.179 | -0.012 | [-0.620, +0.595] |
| history_group | Higher history | 3143 | +1.262 | +0.656 | [+0.026, +1.287] |
| history_group | Lower history | 3249 | +0.540 | -0.433 | [-1.238, +0.373] |

### test
| Feature | Group | N | Mean predicted CATE ($) | Randomized subgroup ATE ($) | Exploratory 95% CI ($) |
|---|---|---:|---:|---:|---|
| newbie_group | Existing customer | 4287 | +1.246 | +0.654 | [-0.256, +1.564] |
| newbie_group | New customer | 4236 | +0.563 | +1.263 | [+0.283, +2.242] |
| channel | Multichannel | 1036 | +1.509 | +0.562 | [-0.743, +1.866] |
| channel | Phone | 3714 | +0.724 | +0.848 | [-0.130, +1.825] |
| channel | Web | 3773 | +0.921 | +1.152 | [+0.046, +2.259] |
| zip_code | Rural | 1197 | +1.339 | +1.545 | [-0.613, +3.702] |
| zip_code | Surburban | 3853 | +0.853 | +0.776 | [-0.291, +1.843] |
| zip_code | Urban | 3473 | +0.817 | +0.948 | [+0.101, +1.794] |
| recency_group | Higher recency | 3514 | +0.516 | +1.361 | [+0.325, +2.398] |
| recency_group | Lower recency | 5009 | +1.181 | +0.673 | [-0.200, +1.546] |
| history_group | Higher history | 4180 | +1.256 | +0.729 | [-0.326, +1.784] |
| history_group | Lower history | 4343 | +0.570 | +1.184 | [+0.356, +2.011] |

## Binary subgroup interaction contrasts (exploratory)

| Split | Feature | Group A - Group B | ATE difference ($) | 95% CI ($) | CI excludes 0? |
|---|---|---|---:|---|---|
| validation | newbie_group | Existing customer - New customer | -0.792 | [-1.829, +0.245] | No |
| validation | recency_group | Higher recency - Lower recency | +0.267 | [-0.815, +1.350] | No |
| validation | history_group | Higher history - Lower history | +1.089 | [+0.066, +2.112] | Yes |
| test | newbie_group | Existing customer - New customer | -0.609 | [-1.946, +0.728] | No |
| test | recency_group | Higher recency - Lower recency | +0.688 | [-0.667, +2.043] | No |
| test | history_group | Higher history - Lower history | -0.455 | [-1.795, +0.886] | No |

## Interpretation guide

A higher predicted CATE does not itself establish that the subgroup has a higher true treatment effect.
If a subgroup's randomized ATE CI includes zero, the data do not establish an effect at the nominal 95% level.
If an interaction contrast CI includes zero, the data do not establish a difference in treatment effects between the two groups.
Even if an exploratory CI excludes zero, account for multiple comparisons and the previously inspected test set.
These analyses concern randomized email treatment, not price elasticity or discount causality.
