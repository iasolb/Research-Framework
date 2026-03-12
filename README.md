# ResearchHandler

A lightweight pandas-based data handling framework for research workflows. Manages full datasets and working subsets, tracks dependent/independent/control variables, and provides clean interfaces for transforming and attaching computed columns.

## Installation

No special installation required beyond standard dependencies:

```bash
pip install pandas numpy
```

For running the example workflows:

```bash
pip install statsmodels scikit-learn scipy
```

For running the tests:

```bash
pip install pytest
```

Or install everything at once from the requirements file:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
Research-Framework/
├── ResearchHandler.py     # Core data handling class
├── transforms.py          # Reusable single- and multi-column transforms
├── requirements.txt       # Dependencies (core + optional)
├── LICENSE                # MIT
├── .gitignore
├── README.md
├── tests/
│   └── test_handler.py    # pytest suite with synthetic data
└── examples/
    ├── ols_mincer.py              # OLS Mincer wage equation
    ├── random_forest_churn.py     # Random forest churn prediction
    └── heckman_selection.py       # Heckman two-step selection model
```

## Quick Start

```python
import numpy as np
from research_handler import ResearchHandler
from transforms import mean_center, log_transform, z_score

# Define a cleaning function
def clean(df):
    df.columns = df.columns.str.lower().str.strip()
    df = df.dropna(subset=["income", "age", "education"])
    df["female"] = (df["gender"] == "F").astype(int)
    return df

# Initialize — input file must be CSV
rh = ResearchHandler("survey_data.csv", clean)

# Transform with named functions from transforms.py
rh.normalize_and_attach("income", log_transform, "log_income")
rh.normalize_and_attach("age", mean_center, "age_centered")

# Create a working subset
rh.create_subset(lambda df: (df["age"] >= 18) & (df["employed"] == 1))

# Set up variables from the subset
rh.set_dependent("log_income", full=False)
rh.add_independents("age_centered", "education", full=False)
rh.add_controls("female", full=False)

# Retrieve design matrix and outcome vector
X = rh.get_X()
y = rh.get_y()
```

## API Reference

### `ResearchHandler(filepath, handling_function)`

Constructor. Reads a CSV and passes the raw DataFrame through your cleaning function.

```python
def clean(df):
    df.columns = df.columns.str.lower()
    df["married"] = (df["marital_status"] == "married").astype(int)
    df = df.drop_duplicates()
    return df.dropna()

rh = ResearchHandler("data.csv", clean)
```

The cleaning function receives the raw `pd.DataFrame` from `read_csv` and must return a cleaned `pd.DataFrame`. If reading or cleaning fails, `rh.data` will be `None` and all downstream methods will print a warning and return early.

### `create_subset(condition)`

Creates a working subset of the full dataset based on a boolean condition.

```python
rh.create_subset(lambda df: df["age"] > 30)
rh.create_subset(lambda df: (df["income"] > 20000) & (df["employed"] == 1))
rh.create_subset(lambda df: df["country"].isin(["US", "UK", "CA"]))
```

### `reset_subset()`

Clears the working subset back to `None`.

```python
rh.reset_subset()
```

### `set_dependent(col, full=True)`

Sets the dependent (outcome) variable.

```python
rh.set_dependent("log_income")
rh.set_dependent("log_income", full=False)
```

### `add_independents(*cols, full=True)`

Adds one or more independent (predictor) variables.

```python
rh.add_independents("education", "experience", "tenure")
rh.add_independents("education", "experience", full=False)
```

### `add_controls(*cols, full=True)`

Adds one or more control variables.

```python
rh.add_controls("female", "married", "region_code")
rh.add_controls("female", "married", full=False)
```

### `get_X()`

Returns the design matrix as a `pd.DataFrame` by concatenating all independents and controls.

```python
X = rh.get_X()
```

### `get_y()`

Returns the dependent variable as a `pd.Series`.

```python
y = rh.get_y()
```

### `attach(col_name, series, to_full=True, quiet=False)`

Attaches a precomputed Series to the full dataset or subset.

```python
from transforms import square

rh.attach("age_sq", square(rh.data["age"]))
rh.attach("age_sq", square(rh.subset["age"]), to_full=False)
```

### `normalize_and_attach(source_col, normalizing_function, new_colname, full=True)`

Applies a single-column transformation and attaches the result.

```python
from transforms import log_transform, z_score, mean_center, min_max_scale

rh.normalize_and_attach("income", log_transform, "log_income")
rh.normalize_and_attach("gpa", z_score, "gpa_z")
rh.normalize_and_attach("age", mean_center, "age_centered")
rh.normalize_and_attach("score", min_max_scale, "score_scaled")
rh.normalize_and_attach("wage", log_transform, "log_wage", full=False)
```

### `apply_and_attach(source_cols, func, new_colname, full=True)`

Applies a multi-column transformation and attaches the result. The function receives a DataFrame subset of the specified columns.

```python
from transforms import interaction, row_mean, row_sum, safe_ratio

rh.apply_and_attach(["education", "experience"], interaction, "edu_x_exp")
rh.apply_and_attach(["math", "reading", "science"], row_mean, "avg_score")
rh.apply_and_attach(["q1", "q2", "q3", "q4"], row_sum, "annual_total")
rh.apply_and_attach(
    ["revenue", "visits"],
    safe_ratio("revenue", "visits"),
    "rev_per_visit",
    full=False
)
```

### `clear_caches()`

Clears the dependent, independents, and controls so you can set up a new specification without reinitializing.

```python
rh.clear_caches()
```

## Transforms Reference

`transforms.py` provides reusable functions so you don't have to write lambdas inline every time.

### Single-column transforms (Series → Series)

For use with `normalize_and_attach`:

| Function | Description | Example |
|----------|-------------|---------|
| `mean_center` | `x - mean(x)` | `rh.normalize_and_attach("age", mean_center, "age_c")` |
| `z_score` | `(x - mean) / std` | `rh.normalize_and_attach("gpa", z_score, "gpa_z")` |
| `min_max_scale` | Scale to [0, 1] | `rh.normalize_and_attach("score", min_max_scale, "score_01")` |
| `log_transform` | `ln(x)` | `rh.normalize_and_attach("income", log_transform, "log_inc")` |
| `log1p_transform` | `ln(1 + x)`, safe for zeros | `rh.normalize_and_attach("tickets", log1p_transform, "log_tix")` |
| `square` | `x²` | `rh.normalize_and_attach("exp", square, "exp_sq")` |
| `rank_transform` | Replace with rank | `rh.normalize_and_attach("score", rank_transform, "score_rank")` |

### Factory transforms (return a callable)

| Function | Description | Example |
|----------|-------------|---------|
| `winsorize(lower, upper)` | Clip at quantiles | `rh.normalize_and_attach("income", winsorize(0.01, 0.99), "inc_wins")` |
| `demean_by_group(group_col)` | Subtract group means | `rh.normalize_and_attach("income", demean_by_group(rh.data["industry"]), "inc_dm")` |

### Multi-column transforms (DataFrame → Series)

For use with `apply_and_attach`:

| Function | Description | Example |
|----------|-------------|---------|
| `interaction` | Product of first two columns | `rh.apply_and_attach(["edu", "exp"], interaction, "edu_x_exp")` |
| `row_mean` | Row-wise average | `rh.apply_and_attach(["m", "r", "s"], row_mean, "avg")` |
| `row_sum` | Row-wise sum | `rh.apply_and_attach(["q1", "q2"], row_sum, "total")` |
| `safe_ratio(num, denom)` | Division, 0 → NaN | `rh.apply_and_attach(["rev", "vis"], safe_ratio("rev", "vis"), "rpv")` |

## Example Workflows

All examples in `examples/` generate their own synthetic data so you can clone and run immediately:

```bash
python examples/ols_mincer.py
python examples/random_forest_churn.py
python examples/heckman_selection.py
```

### OLS Regression with statsmodels

A standard Mincer wage equation with log wages, centered experience, and a squared term.

```python
import numpy as np
import statsmodels.api as sm
from ResearchHandler import ResearchHandler
from transforms import log_transform, mean_center, square

def clean(df):
    df.columns = df.columns.str.lower()
    df["female"] = (df["gender"] == "F").astype(int)
    return df.dropna(subset=["wage", "education", "experience", "age", "gender"])

rh = ResearchHandler("labor_data.csv", clean)

# Transform variables
rh.normalize_and_attach("wage", log_transform, "log_wage")
rh.normalize_and_attach("experience", mean_center, "exp_centered")
rh.attach("exp_centered_sq", square(rh.data["exp_centered"]))

# Specification 1: Full sample
rh.set_dependent("log_wage")
rh.add_independents("education", "exp_centered", "exp_centered_sq")
rh.add_controls("female")

X = sm.add_constant(rh.get_X())
y = rh.get_y()

model1 = sm.OLS(y, X).fit()
print(model1.summary())

# Specification 2: Women only
rh.clear_caches()
rh.create_subset(lambda df: df["female"] == 1)

rh.set_dependent("log_wage", full=False)
rh.add_independents("education", "exp_centered", "exp_centered_sq", full=False)

X2 = sm.add_constant(rh.get_X())
y2 = rh.get_y()

model2 = sm.OLS(y2, X2).fit()
print(model2.summary())
```

### Random Forest with scikit-learn

Predicting customer churn with engineered features and standardized inputs.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ResearchHandler import ResearchHandler
from transforms import z_score, log1p_transform, safe_ratio

def clean(df):
    df.columns = df.columns.str.lower()
    df = df.dropna()
    df["gender_code"] = df["gender"].map({"M": 0, "F": 1})
    df["region_code"] = df["region"].astype("category").cat.codes
    return df

rh = ResearchHandler("customer_data.csv", clean)

# Feature engineering
rh.apply_and_attach(
    ["revenue", "visits"],
    safe_ratio("revenue", "visits"),
    "rev_per_visit"
)
rh.normalize_and_attach("tenure", z_score, "tenure_z")
rh.normalize_and_attach("support_tickets", log1p_transform, "log_tickets")

rh.set_dependent("churned")
rh.add_independents("rev_per_visit", "tenure_z", "log_tickets")
rh.add_controls("gender_code", "region_code")

X = rh.get_X()
assert X is not None, "No independent variables set"
X = X.fillna(0)
y = rh.get_y()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

print(classification_report(y_test, rf.predict(X_test)))
```

### Heckman Selection Model (Two-Step)

Correct for selection bias in observed wages using the inverse Mills ratio.

```python
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from ResearchHandler import ResearchHandler
from transforms import mean_center, log_transform

def clean(df):
    df.columns = df.columns.str.lower()
    df["married"] = (df["marital_status"] == "married").astype(int)
    return df.dropna(subset=["employed", "age", "education", "wage", "children"])

rh = ResearchHandler("labor_survey.csv", clean)

# Center age on the full sample
rh.normalize_and_attach("age", mean_center, "age_centered")

# Step 1: Selection equation (probit) on full sample
rh.set_dependent("employed")
rh.add_independents("age_centered", "education")
rh.add_controls("married", "children")

X_select = sm.add_constant(rh.get_X())
y_select = rh.get_y()

probit = sm.Probit(y_select, X_select).fit(disp=0)

# Compute and attach inverse Mills ratio
imr = norm.pdf(probit.fittedvalues) / norm.cdf(probit.fittedvalues)
rh.attach("imr", imr)

# Step 2: Outcome equation on workers only, with IMR as control
rh.clear_caches()
rh.create_subset(lambda df: df["employed"] == 1)

rh.normalize_and_attach("wage", log_transform, "log_wage", full=False)

rh.set_dependent("log_wage", full=False)
rh.add_independents("age_centered", "education", full=False)
rh.add_controls("imr", full=False)

X_outcome = sm.add_constant(rh.get_X())
y_outcome = rh.get_y()

ols = sm.OLS(y_outcome, X_outcome).fit()
print(ols.summary())
```

## Running Tests

From the repo root:

```bash
pytest tests/test_handler.py -v
```

The test suite covers the full `ResearchHandler` class (init, subsetting, variable setting, attach/transform, guard clauses) and every function in `transforms.py`, all using synthetic data with no external dependencies.

Run a specific test class or method:

```bash
pytest tests/test_handler.py::TestSubset -v
pytest tests/test_handler.py::TestTransforms::test_z_score -v
```

## Design Notes

Every method that accesses data follows the same guard pattern: check `is not None` (not bare truthiness, which raises `ValueError` on DataFrames), handle both `full=True` and `full=False` branches explicitly, and bail early with a printed message when the needed dataset isn't available.

The `independents` and `controls` caches store references to Series pulled from either the full dataset or the subset. Call `clear_caches()` before setting up a new model specification to avoid mixing columns from different sources.