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

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
import treefi

diabetes = load_diabetes(as_frame=True)
X = diabetes.frame[diabetes.feature_names]
y = diabetes.frame[diabetes.target.name]

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=4,
    random_state=0,
).fit(X, y)

interactions = treefi.feature_interactions(
    model,
    max_interaction_depth=1,
    sort_by="gain",
    top_k=20,
)

importance = treefi.feature_importance(
    model,
    sort_by="gain",
    top_k=20,
)

summary = treefi.summarize_model(
    model,
    max_interaction_depth=1,
    top_k=20,
)
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

```python
cv_result = treefi.cross_validated_interactions(
    model,
    X,
    y,
    n_splits=5,
    top_k=10,
)

fold_rows = cv_result.interaction_folds
summary = cv_result.interaction_summary
```

For feature importance:

```python
cv_importance = treefi.cross_validated_importance(
    model,
    X,
    y,
    n_splits=5,
    top_k=10,
)

importance_summary = cv_importance.importance_summary
```

By default:

- regression uses sklearn `KFold`
- classification uses sklearn `StratifiedKFold`

You can override that with your own sklearn splitter:

```python
from sklearn.model_selection import GroupKFold

cv_result = treefi.cross_validated_interactions(
    model,
    X,
    y,
    cv=GroupKFold(n_splits=5),
    groups=groups,
)
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

Practical interpretation:

- high `mean_expected_gain` plus high `fold_presence_rate` is usually more trustworthy than one very high `gain`
- high `selection_rate_top_k` is a good sign that a result is not just a one-fold accident
- high `gain_cv` or `expected_gain_cv` means the result is unstable across folds
- `rare_fold_flag=True` is a warning to be skeptical
- `overfit_suspect_flag=True` means the pattern may be real, but it needs stronger validation before you engineer around it

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

Useful when you want:

- a dataframe replacement for tree-model feature importance summaries
- a ranked list of influential features
- an output that can be exported directly to CSV, Parquet, or Excel

### `feature_interactions(...)`

Returns one row per interaction.

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

For most end-user analysis, `0` or `1` is the best starting point.

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

| Metric | What it means | Good for | Pros | Cons |
| --- | --- | --- | --- | --- |
| `gain` | Split improvement associated with the feature or interaction | First-pass ranking | Intuitive and often surfaces important model logic quickly | Backend semantics differ and it can overweight a few extreme splits |
| `fscore` | Raw occurrence count | Seeing how often a feature or interaction appears | Simple, stable, easy to explain | Frequency alone does not say whether the split mattered much |
| `weighted_fscore` | Path probability, following `xgbfir` semantics | Discounting rare paths and highlighting common structure | More informative than raw count when deep or low-mass paths exist | Depends on cover-like backend statistics and is not perfectly comparable across all libraries |
| `expected_gain` | `gain * weighted_fscore` | Balancing strength and prevalence | Good for prioritizing interactions that are both strong and common | Inherits the limitations of both `gain` and `weighted_fscore` |
| `cover` | Node mass: how much training weight reaches a split or path | Seeing whether a pattern is broad or niche | Useful context for gain and for debugging model reach | Exact semantics vary by backend |
| `average_gain` | `gain / fscore` | Separating repeated moderate effects from rare strong effects | Good secondary ranking metric | Can overrate rare events |
| `tree_frequency` | How many trees contain the interaction | Breadth across the ensemble | Interpretable and useful next to gain | Structural frequency is not the same as predictive importance |
| `path_frequency` | How many distinct paths contain the interaction | Structural repetition within trees | Interpretable and useful next to gain | Structural frequency is not the same as predictive importance |
| `first_position_mean` | Where the feature or interaction tends to start in a path | Understanding whether it acts early or late | Useful for understanding how high up a feature acts | Not a direct importance score |
| `min_depth`, `max_depth` | Depth range where the feature or interaction appears | Understanding spread and path position | Helps show whether behavior is shallow or deep | Depth is informative, but not an importance metric by itself |
| `leaf_effect_mean`, `leaf_effect_var` | Summary of downstream leaf values for paths containing the interaction | Understanding downstream effect direction and variability | Helpful for exploring what happens after an interaction appears | Interpretation depends on model type and backend leaf semantics |

## How To Use The Metrics Together

A practical approach:

1. Sort by `expected_gain` to find interactions that are both strong and prevalent.
2. Check `gain` to find rare but powerful splits.
3. Check `fscore`, `tree_frequency`, and `cover` to see whether the pattern is broad or niche.
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
