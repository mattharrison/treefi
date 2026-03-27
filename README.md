# treefi

`treefi` is a dataframe-first library for inspecting fitted tree models.

It is a modern Python rewrite of `xgbfir`: instead of writing Excel workbooks, it returns `pandas.DataFrame` objects that you can sort, filter, join, plot, or export yourself.

`treefi` currently works with:

- scikit-learn trees and forests
- HistGradientBoosting via scikit-learn internals
- XGBoost
- CatBoost
- LightGBM

## Install

```bash
uv add treefi
```

If you are working on the repo itself, tests run with:

```bash
uv run pytest
```

## Quickstart

```pycon
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.ensemble import RandomForestRegressor
>>> import treefi
>>> diabetes = load_diabetes(as_frame=True)
>>> X = diabetes.frame[diabetes.feature_names]
>>> y = diabetes.frame[diabetes.target.name]
>>> model = RandomForestRegressor(
...     n_estimators=50,
...     max_depth=4,
...     random_state=0,
... ).fit(X, y)
>>> interactions = treefi.feature_interactions(
...     model,
...     max_interaction_depth=1,
...     sort_by="gain",
...     top_k=20,
... )
>>> interactions.head(5)
  interaction  interaction_order  ...  feature_count  path_probability_sum
0      bmi|s5                2.0  ...            2.0             42.135747
1         bmi                1.0  ...            1.0             66.757919
2      bmi|bp                2.0  ...            2.0             16.579186
3      bmi|s6                2.0  ...            2.0              8.889140
4      bmi|s1                2.0  ...            2.0              0.660633
<BLANKLINE>
[5 rows x 28 columns]

>>> interactions.columns
Index(['interaction', 'interaction_order', 'gain', 'cover', 'fscore',
       'weighted_fscore', 'average_weighted_fscore', 'average_gain',
       'expected_gain', 'average_tree_index', 'average_tree_depth',
       'tree_frequency', 'path_frequency', 'first_position_mean', 'min_depth',
       'max_depth', 'leaf_effect_mean', 'leaf_effect_var', 'rank_gain',
       'rank_fscore', 'rank_expected_gain', 'rank_consensus', 'backend',
       'model_type', 'occurrence_count', 'tree_count', 'feature_count',
       'path_probability_sum'],
      dtype='str')

>>> importance = treefi.feature_importance(
...     model,
...     sort_by="gain",
...     top_k=20,
... )
>>> importance.head()
  feature           gain  ...  occurrence_count  tree_count
0     bmi  218678.351847  ...               161        50.0
1      s5  211505.225081  ...               132        50.0
2      s6   66175.222795  ...                69        41.0
3      bp   65271.672281  ...               111        49.0
4      s4   59589.750490  ...                28        25.0
<BLANKLINE>
[5 rows x 15 columns]

>>> importance.columns
Index(['feature', 'gain', 'cover', 'weight', 'total_gain', 'total_cover',
       'weighted_fscore', 'average_weighted_fscore', 'expected_gain',
       'average_tree_index', 'average_tree_depth', 'backend', 'model_type',
       'occurrence_count', 'tree_count'],
      dtype='str')
      
>>> summary = treefi.summarize_model(
...     model,
...     max_interaction_depth=1,
...     top_k=20,
... )
>>> sorted(summary.metadata)
['backend', 'model_type']
```

Typical workflow:

1. Use `feature_importance(...)` to see which features show up most strongly overall.
2. Use `feature_interactions(...)` to see which features repeatedly appear together along tree paths.
3. Use normal pandas operations to filter, rank, export, or plot the result.

## Cross-Validated Stability

You can also validate whether interactions or importance rankings are stable across folds instead of trusting a single fitted model.

Use:

- `treefi.cross_validated_interactions(...)`
- `treefi.cross_validated_importance(...)`

Example:

```pycon
>>> cv_result = treefi.cross_validated_interactions(
...     model,
...     X,
...     y,
...     n_splits=5,
...     top_k=10,
... )

>>> cv_result.folds
    interaction  interaction_order          gain  ...  fold  train_size  test_size
0            s5                1.0  3.902246e+07  ...     0         353         89
1        bmi|s5                2.0  1.702226e+08  ...     0         353         89
2           bmi                1.0  4.378239e+07  ...     0         353         89
3     bmi|bp|s5                3.0  4.640013e+07  ...     0         353         89
4        bmi|bp                2.0  3.442126e+07  ...     0         353         89
..          ...                ...           ...  ...   ...         ...        ...
406   bp|s5|sex                3.0  8.947998e+05  ...     4         354         88
407      bp|sex                2.0  8.947998e+05  ...     4         354         88
408    s2|s4|s5                3.0  8.504072e+05  ...     4         354         88
409       s2|s4                2.0  8.504072e+05  ...     4         354         88
410   s4|s5|sex                3.0  8.646905e+05  ...     4         354         88
<BLANKLINE>
[411 rows x 31 columns]



>>> cv_result.importance_summary

>>> cv_result.interaction_folds
    interaction  interaction_order          gain  ...  fold  train_size  test_size
0            s5                1.0  3.902246e+07  ...     0         353         89
1        bmi|s5                2.0  1.702226e+08  ...     0         353         89
2           bmi                1.0  4.378239e+07  ...     0         353         89
3     bmi|bp|s5                3.0  4.640013e+07  ...     0         353         89
4        bmi|bp                2.0  3.442126e+07  ...     0         353         89
..          ...                ...           ...  ...   ...         ...        ...
406   bp|s5|sex                3.0  8.947998e+05  ...     4         354         88
407      bp|sex                2.0  8.947998e+05  ...     4         354         88
408    s2|s4|s5                3.0  8.504072e+05  ...     4         354         88
409       s2|s4                2.0  8.504072e+05  ...     4         354         88
410   s4|s5|sex                3.0  8.646905e+05  ...     4         354         88
<BLANKLINE>
[411 rows x 31 columns]

>>> cv_result.interaction_summary
    interaction     mean_gain  ...  cv_instability_flag  suspicious_feature_score
0            s5  3.662796e+07  ...                False                       0.0
1        bmi|s5  1.289095e+08  ...                False                       0.0
2           bmi  4.519529e+07  ...                False                       1.0
3     bmi|bp|s5  3.169041e+07  ...                False                       1.0
4        bmi|bp  2.544942e+07  ...                False                       1.0
..          ...           ...  ...                  ...                       ...
120       s1|s4  1.979224e+06  ...                False                       1.0
121   age|s1|s4  9.963473e+05  ...                False                       1.0
122   bp|s3|sex  8.689495e+05  ...                False                       1.0
123    bp|s1|s3  1.580895e+06  ...                False                       1.0
124   s4|s5|sex  8.646905e+05  ...                False                       1.0
<BLANKLINE>
[125 rows x 31 columns]

>>> cv_result.metadata
{'task': 'regression', 'splitter': 'KFold', 'n_splits': 5}

>>> cv_result.summary
    interaction     mean_gain  ...  cv_instability_flag  suspicious_feature_score
0            s5  3.662796e+07  ...                False                       0.0
1        bmi|s5  1.289095e+08  ...                False                       0.0
2           bmi  4.519529e+07  ...                False                       1.0
3     bmi|bp|s5  3.169041e+07  ...                False                       1.0
4        bmi|bp  2.544942e+07  ...                False                       1.0
..          ...           ...  ...                  ...                       ...
120       s1|s4  1.979224e+06  ...                False                       1.0
121   age|s1|s4  9.963473e+05  ...                False                       1.0
122   bp|s3|sex  8.689495e+05  ...                False                       1.0
123    bp|s1|s3  1.580895e+06  ...                False                       1.0
124   s4|s5|sex  8.646905e+05  ...                False                       1.0
<BLANKLINE>
[125 rows x 31 columns]

```

For feature importance:

```pycon
>>> cv_importance = treefi.cross_validated_importance(
...     model,
...     X,
...     y,
...     n_splits=5,
...     top_k=10,
... )

>>> cv_importance.summary.head()
  feature      mean_gain  ...  cv_instability_flag  suspicious_feature_score
0      s5  219332.121747  ...                False                       0.0
1     bmi  144604.201773  ...                False                       0.0
2      bp   57327.316224  ...                False                       0.0
3      s1   26106.343731  ...                False                       0.0
4     age   26647.991164  ...                False                       1.0
<BLANKLINE>
[5 rows x 35 columns]

>>> cv_importance.interaction_summary.head()
  feature      mean_gain  ...  cv_instability_flag  suspicious_feature_score
0      s5  219332.121747  ...                False                       0.0
1     bmi  144604.201773  ...                False                       0.0
2      bp   57327.316224  ...                False                       0.0
3      s1   26106.343731  ...                False                       0.0
4     age   26647.991164  ...                False                       1.0
<BLANKLINE>
[5 rows x 35 columns]

```

### Real-World Check With Added Random Columns

When you are trying to decide whether a suspicious feature is genuinely useful or
just picking up noise, it helps to inspect multiple importance views on the same
model.

The example below adds random columns to a realistic regression dataset, fits
XGBoost, and compares:

- `total_gain`: summed contribution across all splits
- `gain`: average gain per split
- `weight`: split count

```pycon
>>> import numpy as np
>>> import xgboost as xgb
>>> rng = np.random.default_rng(0)
>>> diabetes = load_diabetes(as_frame=True)
>>> X = diabetes.frame[diabetes.feature_names].copy()
>>> y = diabetes.frame[diabetes.target.name]
>>> for i in range(3):
...     X[f"rand_{i}"] = rng.normal(size=len(X))
>>> model = xgb.XGBRegressor(
...     objective="reg:squarederror",
...     max_depth=4,
...     n_estimators=40,
...     learning_rate=0.1,
...     random_state=0,
... )
>>> model.fit(X, y)
XGBRegressor(...)
>>> by_total = treefi.feature_importance(model, sort_by="total_gain")
>>> print(by_total.sort_values('total_gain', ascending=False)[['feature', 'gain', 'total_gain', 'cover', 'weight', 'occurrence_count']])
   feature          gain    total_gain       cover  weight  occurrence_count
0       s5  77670.656672  4.504898e+06  239.068966      58                58
1      bmi  34465.279722  2.653827e+06  162.155844      77                77
2       bp  13549.734904  8.129841e+05  125.466667      60                60
3   rand_1   9234.090478  5.448113e+05  158.762712      59                59
4       s6  14093.099014  4.227930e+05  121.700000      30                30
5       s1   8627.091134  4.054733e+05   51.191489      47                47
6   rand_0   8232.303106  3.539890e+05  142.581395      43                43
7       s2   9957.913378  3.485270e+05   47.657143      35                35
8       s3  10808.439566  3.350616e+05  145.774194      31                31
9      age   5474.074392  2.901259e+05   60.716981      53                53
10  rand_2   6445.974136  1.740413e+05   74.296296      27                27
11     sex   8337.355618  1.667471e+05  111.100000      20                20
12      s4  15119.451459  1.511945e+05  153.200000      10                10

>>> print(by_total.sort_values('gain', ascending=False)[['feature', 'gain', 'total_gain', 'cover', 'weight', 'occurrence_count']])
   feature          gain    total_gain       cover  weight  occurrence_count
0       s5  77670.656672  4.504898e+06  239.068966      58                58
1      bmi  34465.279722  2.653827e+06  162.155844      77                77
12      s4  15119.451459  1.511945e+05  153.200000      10                10
4       s6  14093.099014  4.227930e+05  121.700000      30                30
2       bp  13549.734904  8.129841e+05  125.466667      60                60
8       s3  10808.439566  3.350616e+05  145.774194      31                31
7       s2   9957.913378  3.485270e+05   47.657143      35                35
3   rand_1   9234.090478  5.448113e+05  158.762712      59                59
5       s1   8627.091134  4.054733e+05   51.191489      47                47
11     sex   8337.355618  1.667471e+05  111.100000      20                20
6   rand_0   8232.303106  3.539890e+05  142.581395      43                43
10  rand_2   6445.974136  1.740413e+05   74.296296      27                27
9      age   5474.074392  2.901259e+05   60.716981      53                53


```

This is the practical point:

- a noisy feature can rank surprisingly high by `total_gain` if the model uses it often
- `gain` helps separate "used often" from "strong each time"
- `weight` helps show whether a feature is just appearing everywhere

If a random-looking feature still seems suspicious, check its cross-validated
stability instead of trusting one fitted model:

```pycon
>>> cv_importance = treefi.cross_validated_importance(
...     xgb.XGBRegressor(
...         objective="reg:squarederror",
...         max_depth=4,
...         n_estimators=20,
...         learning_rate=0.1,
...         random_state=0,
...     ),
...     X,
...     y,
...     n_splits=3,
...     top_k=8,
... )
>>> print(
...     cv_importance.importance_summary[
...         [
...             "feature",
...             "mean_gain",
...             "fold_presence_rate",
...             "selection_rate_top_k",
...             "gain_cv",
...             "overfit_suspect_flag",
...             "high_weight_low_gain_flag",
...             "low_consensus_top_k_flag",
...             "suspicious_feature_score",
...         ]
...     ].to_string(index=False)
... )
feature    mean_gain  fold_presence_rate  selection_rate_top_k  gain_cv  overfit_suspect_flag  high_weight_low_gain_flag  low_consensus_top_k_flag  suspicious_feature_score
    bmi 52894.239138            1.000000              1.000000 0.435574                 False                      False                     False                       0.0
     s5 61718.260891            1.000000              1.000000 0.385884                 False                      False                     False                       0.0
     s1 10909.057178            1.000000              1.000000 0.271382                 False                      False                     False                       0.0
     s3 14398.340383            0.666667              0.666667 0.273421                 False                      False                      True                       1.0
     s6 20146.245117            1.000000              1.000000 0.551411                 False                      False                     False                       0.0
     bp 16654.948841            1.000000              1.000000 0.049289                 False                      False                     False                       0.0
 rand_1 10578.990241            0.666667              0.666667 0.276258                 False                       True                      True                       2.0
     s2 13392.392551            0.666667              0.666667 0.150914                 False                      False                      True                       1.0
 rand_0 10238.738508            0.333333              0.333333 0.000000                 False                      False                      True                       1.0
    sex  4059.374560            0.333333              0.333333 0.000000                 False                      False                      True                       1.0
    age  6981.577176            0.333333              0.333333 0.000000                 False                       True                      True                       2.0
```

These doctest examples are intentionally written as realistic usage examples.
They are executable, but the main goal is to show how you would inspect real
dataframes in practice rather than just assert one minimal condition.

Treat these suspicious-feature columns as heuristics rather than proof. A high
score is a prompt to inspect, regularize, or validate more carefully, not proof
that a feature is useless or leaked.

By default:

- regression uses sklearn `KFold`
- classification uses sklearn `StratifiedKFold`

You can override that with your own sklearn splitter:

```pycon
>>> from sklearn.model_selection import GroupKFold
>>> groups = (X.index % 5).to_numpy()
>>> grouped_cv = treefi.cross_validated_interactions(
...     model,
...     X,
...     y,
...     cv=GroupKFold(n_splits=5),
...     groups=groups,
... )
>>> grouped_cv.metadata["splitter"]
'GroupKFold'
```

### How To Read CV Stability Output

The CV summary tables are meant to help you decide whether a result is likely repeatable or just fold-specific noise.

Useful columns:

- `fold_presence_rate`: how often the feature or interaction appears across folds
- `mean_gain` and `std_gain`: average strength and variability
- `mean_expected_gain` and `std_expected_gain`: strength adjusted for prevalence
- `selection_rate_top_k`: how often the item lands in the fold-level top `k`
- `rank_stability_score`: higher means more stable rank across folds
- `rare_fold_flag`: appears in too few folds to trust much
- `overfit_suspect_flag`: heuristic warning for strong-but-unstable patterns
- `cv_instability_flag`: broader warning for fold-volatile patterns
- `high_total_gain_low_density_flag`: high total contribution driven more by repeated usage than by strong individual splits
- `high_weight_low_gain_flag`: used very often, but each use is weak on average
- `low_consensus_top_k_flag`: looks strong in one fit but rarely lands in the fold-level top `k`
- `weak_signal_density_flag`: weak prevalence-adjusted signal despite nontrivial total contribution
- `suspicious_feature_score`: simple combined heuristic that reduces reliance on any one boolean flag

Diagnostic taxonomy:

- unstable features:
  high fold-to-fold variance and weak repeatability; inspect `gain_cv`, `expected_gain_cv`, `cv_instability_flag`, and sometimes `overfit_suspect_flag`
- low-density features:
  features or interactions that accumulate `total_gain` mostly by being used a lot rather than by being strong each time; inspect `high_total_gain_low_density_flag`, `high_weight_low_gain_flag`, and `weak_signal_density_flag`
- weak-consensus features:
  features or interactions that can look good in one fitted model but do not repeatedly show up near the top across folds; inspect `selection_rate_top_k` and `low_consensus_top_k_flag`

Practical interpretation:

- high `mean_expected_gain` plus high `fold_presence_rate` is usually more trustworthy than one very high `gain`
- high `selection_rate_top_k` is a good sign that a result is not just a one-fold accident
- high `gain_cv` or `expected_gain_cv` means the result is unstable across folds
- `rare_fold_flag=True` is a warning to be skeptical
- `overfit_suspect_flag=True` means the pattern may be real, but it needs stronger validation before you engineer around it
- `suspicious_feature_score` is a triage aid, not proof that a feature is useless, noisy, or leaked

### Time-Series And Leakage-Sensitive Work

`treefi` does not guess a time-aware split for you.

If your problem has temporal order, grouped entities, or leakage constraints:

- pass your own splitter through `cv=`
- use `groups=` when your splitter requires it
- validate on the same split logic you would use for model selection

For time-series problems, prefer something like sklearn `TimeSeriesSplit` or a custom temporal splitter instead of the default `KFold` or `StratifiedKFold`.

## Main Functions

### `feature_importance(...)`

Returns one row per feature.

```pycon
>>> importance = treefi.feature_importance(model)
>>> importance.iloc[0]
feature                                s5
gain                         77670.656672
cover                          239.068966
weight                                 58
total_gain                  4504898.08698
total_cover                       13866.0
weighted_fscore                 31.371041
average_weighted_fscore           0.54088
expected_gain              4195112.317978
average_tree_index              15.327586
average_tree_depth                0.62069
backend                           xgboost
model_type                   XGBRegressor
occurrence_count                       58
tree_count                           33.0
Name: 0, dtype: object
```


Useful when you want:

- a dataframe replacement for tree-model feature importance summaries
- a ranked list of influential features
- an output that can be exported directly to CSV, Parquet, or Excel

For ensembles, repeated per-tree feature occurrences are aggregated into one
row per feature before ranking. That means `sort_by` and `top_k` operate on the
final feature-level totals or averages, not on raw per-tree rows.

### `feature_interactions(...)`

Returns one row per interaction.

```pycon
>>> interactions = treefi.feature_interactions(model)
>>> (interactions
...    .query('interaction_order>1')
...    .iloc[0])
interaction                        bmi|s5
interaction_order                     2.0
gain                       20656941.56573
cover                             39423.0
fscore                                 51
weighted_fscore                 20.058824
average_weighted_fscore          0.457025
average_gain                289595.156457
expected_gain              7284911.493426
average_tree_index              10.315789
average_tree_depth               1.315789
tree_frequency                        1.0
path_frequency                         51
first_position_mean              0.339474
min_depth                        1.210526
max_depth                        2.052632
leaf_effect_mean                -0.533484
leaf_effect_var                       0.0
rank_gain                        5.842105
rank_fscore                      6.263158
rank_expected_gain               2.315789
rank_consensus                   4.807018
backend                           xgboost
model_type                   XGBRegressor
occurrence_count                       51
tree_count                           19.0
feature_count                         2.0
path_probability_sum            20.058824
Name: 1, dtype: object
```

Useful when you want:

- pairwise or higher-order feature combinations
- repeated path structure across trees
- interaction rankings by gain, frequency, or expected gain

### `summarize_model(...)`

Returns an `AnalysisResult` bundle with:

- `interactions`
- `importance`
- `leaf_stats`
- `metadata`

Use this when you want one call that gives you the main analysis tables together.

```pycon
>>> summary = treefi.summarize_model(model)
>>> summary.interactions.iloc[0]
interaction                            s5
interaction_order                     1.0
gain                        4845136.24275
cover                             17146.0
fscore                                 62
weighted_fscore                   32.5181
average_weighted_fscore          0.590382
average_gain                 78306.963925
expected_gain              4299055.207119
average_tree_index                   17.0
average_tree_depth               0.757576
tree_frequency                        1.0
path_frequency                         62
first_position_mean              1.373232
min_depth                        0.757576
max_depth                        1.878788
leaf_effect_mean                -0.072035
leaf_effect_var                  1.272988
rank_gain                       18.212121
rank_fscore                      7.545455
rank_expected_gain               6.939394
rank_consensus                   10.89899
backend                           xgboost
model_type                   XGBRegressor
occurrence_count                       62
tree_count                           33.0
feature_count                         1.0
path_probability_sum              32.5181
Name: 0, dtype: object

>>> summary.importance.iloc[0]
feature                                s5
gain                         77670.656672
cover                          239.068966
weight                                 58
total_gain                  4504898.08698
total_cover                       13866.0
weighted_fscore                 31.371041
average_weighted_fscore           0.54088
expected_gain              4195112.317978
average_tree_index              15.327586
average_tree_depth                0.62069
backend                           xgboost
model_type                   XGBRegressor
occurrence_count                       58
tree_count                           33.0
Name: 0, dtype: object

>>> summary.leaf_stats.iloc[0]
interaction               s5
leaf_effect_mean   -0.072035
leaf_effect_var     1.272988
Name: 0, dtype: object

>>> summary.metadata
{'backend': 'xgboost', 'model_type': 'XGBRegressor'}
```

## What Interactions Mean

In `treefi`, an interaction means features that appear together along the same decision path in a tree.

Example:

- if a tree splits on `age`, then later on `fare`, that path contains an `age|fare` interaction
- if that same pattern appears across many trees, its interaction metrics will increase

This is a structural definition of interaction:

- it tells you which features work together inside the fitted tree logic
- it does not prove a causal relationship
- it does not mean the interaction would be significant in a linear-model sense

That makes treefi useful for:

- model interpretation
- feature engineering ideas
- debugging tree behavior
- comparing tree structure across libraries

### Ordered vs Unordered Interactions

`treefi.feature_interactions(...)` supports two interaction views:

- `interaction_mode="unordered"`:
  `age|fare` and `fare|age` collapse to the same key. This is the best default for ranking and summaries.
- `interaction_mode="ordered"`:
  path order is preserved. Use this when the sequence of splits matters.

Use unordered mode when you want simpler tables. Use ordered mode when you want to inspect tree logic more precisely.

### Choosing Interaction Depth

- `max_interaction_depth=0`: feature-only view
- `max_interaction_depth=1`: pairwise interactions
- `max_interaction_depth=2`: three-feature paths

For most end-user analysis, `1` is the best starting point.

### Using Interactions To Improve A Model

One practical use of `treefi` is to turn strong tree interactions into feature-engineering or modeling hypotheses.

Example workflow:

1. Rank interactions by `expected_gain` or `gain`.
2. Look for pairs that are both strong and repeated, not just one-off deep-path effects.
3. Ask whether the interaction suggests a feature transformation the model is currently learning the hard way.

Examples:

- `age|fare` might suggest trying:
  - a binned age feature
  - a fare-per-family or fare-per-class feature
  - an explicit crossed feature for linear or shallow models
- `income|debt_ratio` might suggest:
  - ratio features
  - thresholded risk buckets
  - monotonic or segmented business rules

This can help in a few situations:

- improving simpler models by giving them interaction features directly
- reducing depth needed in tree models
- creating more stable, interpretable features
- discovering domain-relevant thresholds or regimes

For XGBoost specifically, this is often useful because the model is already good at discovering interactions internally. `treefi` helps you inspect which ones are actually being used, then decide whether to:

- create explicit interaction features for a simpler downstream model
- restrict or regularize the model if it is relying on suspicious interactions
- design constraints, bins, or grouped features that make the structure easier to learn

### How To Validate An Interaction Hypothesis

Do not assume that a high-ranking interaction automatically deserves a new engineered feature. Treat it as a hypothesis and validate it.

Good validation workflow:

1. Create the proposed feature or feature set.
2. Refit the model under the same cross-validation scheme.
3. Compare against the baseline on the real selection metric.
4. Check whether the gain is stable across folds or seeds.
5. Inspect whether the new feature improves calibration, robustness, or simplicity, not just leaderboard score.

Useful checks:

- does validation performance improve consistently?
- does the simpler feature reduce required tree depth?
- do top interactions become easier to explain?
- does the feature still help on a time split or out-of-domain holdout?
- does it create leakage risk or encode target-like information?

Practical warning:

- interactions found in one fitted model can reflect noise, sample quirks, or overfitting
- always validate on held-out data
- prefer repeated evidence across folds, seeds, or model families before treating an interaction as “real”

### Using TreeFI To Improve Linear Or Logistic Regression

One of the best uses of `treefi` is to learn from a strong tree model, then transfer those insights into a simpler linear model such as linear regression or logistic regression.

Why this works:

- tree models naturally discover thresholds, nonlinear regions, and feature combinations
- linear models usually need those structures to be engineered explicitly
- `treefi` helps you see which structures the tree keeps using

What to look for in the report:

- high-ranking pairwise interactions
- features that repeatedly appear early in the trees
- combinations with high `expected_gain`
- evidence that a feature only matters after another feature splits first

How that can improve a linear model:

- add crossed features such as `age * fare`
- add ratio features such as `debt / income`
- add bucketed or thresholded versions of numeric features
- add piecewise terms such as `max(age - 50, 0)`
- add grouped regime indicators such as `high_income_and_high_balance`

Examples:

- if treefi shows `income|debt_ratio` repeatedly, try explicit interaction terms or segmented risk buckets in logistic regression
- if treefi shows `age` appearing early with multiple downstream splits, try splines, bins, or hinge features for `age`
- if `fare|pclass` is strong, try a crossed categorical/numeric representation instead of leaving the linear model to miss that structure

This is especially useful when you want:

- a model that is easier to explain
- coefficients and odds ratios
- simpler deployment
- fewer degrees of freedom than a large boosted tree model

### How To Validate Linear-Model Improvements

Use the treefi report to generate candidate features, then test them rigorously.

Recommended workflow:

1. Train a baseline linear or logistic regression model.
2. Add a small set of treefi-inspired features.
3. Refit using the same preprocessing and cross-validation.
4. Compare against the baseline on the same metric.
5. Keep only features that help consistently.

Things to check:

- does validation score improve?
- do coefficients remain stable across folds?
- does calibration improve for logistic regression?
- do the new terms make domain sense?
- do they still help on a stricter holdout set?

Practical guidance:

- add a few high-value features first instead of many at once
- prefer interactions that are both strong and common
- be careful with multicollinearity when adding many related transformed terms
- regularized linear models often work best once you start adding engineered interactions

## Metrics

The output dataframes include several ranking metrics. No single metric is always best, so in practice you usually want to compare more than one.

For XGBoost users, `treefi` exposes explicit compatibility columns on
feature-importance output:

- `weight`: split count
- `total_gain`: summed split gain
- `total_cover`: summed split cover
- `gain`: average gain per split
- `cover`: average cover per split

Backend caveat:

- XGBoost: these compatibility names are direct matches
- LightGBM: `total_gain` / `gain` are strong matches, while cover-style metrics remain approximate
- sklearn trees, forests, and HistGradientBoosting: gain/cover-style aliases are useful approximations, not literal XGBoost parity
- CatBoost: gain-style aliases are structural proxies derived from exported leaf values, and should be treated as synthetic approximations

### Reproducing XGBoost `get_score(...)`

For XGBoost models, treefi's canonical feature-importance names map directly to
`Booster.get_score(...)`:

- `get_score(importance_type="gain")` -> treefi `gain`
- `get_score(importance_type="cover")` -> treefi `cover`
- `get_score(importance_type="weight")` -> treefi `weight`
- `get_score(importance_type="total_gain")` -> treefi `total_gain`
- `get_score(importance_type="total_cover")` -> treefi `total_cover`

That makes it easy to reproduce XGBoost's native importance reports while still
working with a dataframe instead of a dict.

## Backend-Neutral Comparison

Use backend-specific metrics when you want parity with one library:

- for XGBoost, use `gain`, `cover`, `weight`, `total_gain`, and `total_cover`

Use backend-neutral metrics when you want to compare different tree libraries:

- `expected_gain` to balance strength and prevalence
- `weighted_fscore` to discount rare paths
- `occurrence_count` and `tree_count` for structural breadth
- CV outputs like `fold_presence_rate`, `selection_rate_top_k`, and `gain_cv`

This matters because XGBoost's importance semantics are exact for XGBoost, but
only approximate or synthetic for some other backends.

### Compatibility Names Vs treefi-native Names

`treefi` exposes two metric families:

- compatibility names:
  `gain`, `cover`, `weight`, `total_gain`, and `total_cover`
- treefi-native names:
  `expected_gain`, `weighted_fscore`, `average_weighted_fscore`,
  `tree_count`, `occurrence_count`, `tree_frequency`, and `path_frequency`

Use compatibility names when you want low-friction parity with XGBoost-style
reports or with `Booster.get_score(...)`.

Use treefi-native names when you want backend-neutral analysis that balances
strength, prevalence, and path structure.

The practical rule is:

- `feature_importance(...)` uses compatibility names as the primary surface
- interaction analysis and CV summaries are where treefi-native metrics add the most value



| Metric | What it means | Good for | Pros | Cons |
| --- | --- | --- | --- | --- |
| `total_gain` | Total split improvement summed across occurrences | First-pass ranking for total ensemble contribution | Aligns with XGBoost `total_gain` and surfaces features used often and effectively | Can overweight features that are used frequently but only moderately well |
| `gain` | Average split improvement per occurrence | Comparing per-split quality | Matches XGBoost `gain` semantics and helps separate frequent weak splits from stronger ones | A rare but strong split can look better than a broad, consistently useful feature |
| `weight` | Raw occurrence count | Seeing how often a feature or interaction appears | Matches XGBoost `weight` and is easy to explain | Frequency alone does not say whether the split mattered much |
| `weighted_fscore` | Path probability, following `xgbfir` semantics | Discounting rare paths and highlighting common structure | More informative than raw count when deep or low-mass paths exist | Depends on cover-like backend statistics and is not perfectly comparable across all libraries |
| `expected_gain` | `gain * weighted_fscore` | Balancing strength and prevalence | Good for prioritizing interactions that are both strong and common | Inherits the limitations of both `gain` and `weighted_fscore` |
| `total_cover` | Total node mass reaching the feature across occurrences | Seeing whether a pattern is broad or niche overall | Useful context for total gain and debugging model reach | Exact semantics vary by backend and totals can be large for frequently reused features |
| `cover` | Average node mass per occurrence | Comparing how broad a typical split is | Matches XGBoost `cover` semantics most closely | Still backend-dependent and approximate on some libraries |
| `tree_frequency` | How many trees contain the interaction | Breadth across the ensemble | Interpretable and useful next to gain | Structural frequency is not the same as predictive importance |
| `path_frequency` | How many distinct paths contain the interaction | Structural repetition within trees | Interpretable and useful next to gain | Structural frequency is not the same as predictive importance |
| `first_position_mean` | Where the feature or interaction tends to start in a path | Understanding whether it acts early or late | Useful for understanding how high up a feature acts | Not a direct importance score |
| `min_depth`, `max_depth` | Depth range where the feature or interaction appears | Understanding spread and path position | Helps show whether behavior is shallow or deep | Depth is informative, but not an importance metric by itself |
| `leaf_effect_mean`, `leaf_effect_var` | Summary of downstream leaf values for paths containing the interaction | Understanding downstream effect direction and variability | Helpful for exploring what happens after an interaction appears | Interpretation depends on model type and backend leaf semantics |

## How To Use The Metrics Together

A practical approach:

1. Sort by `expected_gain` to find interactions that are both strong and prevalent.
2. Check `total_gain` for total contribution and `gain` for per-split quality.
3. Check `weight`, `tree_frequency`, and `total_cover` to see whether the pattern is broad or niche.
4. Check `first_position_mean` and depth metrics to see whether it acts early or late in the trees.

## Backend Notes

Most columns are shared across backends, but some are exact and some are approximate.

### scikit-learn

- tree and forest `gain` is derived from weighted impurity decrease
- `cover` is derived from weighted node sample counts
- HistGradientBoosting uses sklearn's structured predictor-node `gain` and `count` fields

### XGBoost

- `gain` and `cover` come from structured tree statistics
- XGBoost sklearn-compatible wrappers like `XGBRegressor` and `XGBClassifier` are supported

### CatBoost

- support is based on CatBoost JSON model export
- feature indices are normalized back to names like `f0` when needed
- `cover` is derived from descendant leaf weights
- `gain` is approximated from weighted variance reduction over descendant leaf values
- this `gain` is a structural proxy, not CatBoost's internal training-time split score
- categorical split normalization is not implemented

### LightGBM

- `gain` comes from `split_gain`
- `cover` is approximate from exported internal and leaf counts

## Exporting Results

Because the outputs are ordinary dataframes, export is just pandas:

```python
interactions.to_csv("interactions.csv", index=False)
importance.to_parquet("importance.parquet", index=False)
interactions.to_excel("interactions.xlsx", index=False)
```

## Migration From xgbfir

Old pattern:

```python
# xgbfir
xgbfir.saveXgbFI(model, feature_names=names, OutputXlsxFile="out.xlsx")
```

New pattern:

```python
# treefi
df = treefi.feature_interactions(model, feature_names=names)
df.to_excel("out.xlsx", index=False)
```

The main difference is that `treefi` returns dataframes first. Export is optional and downstream.

## Example Notebook

See `nbs/sample.ipynb` for end-to-end examples using:

- scikit-learn regression
- scikit-learn classification
- XGBoost sklearn API models
- CatBoost sklearn API models
