"""Price model, baseline and evaluation shared by the model notebooks.

Everything here works on pandas frames with the columns of ga2_gold.features_<city>.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

CATEGORICAL = ["room_type", "city"]
NUMERIC = ["accommodates", "bedrooms", "bathrooms", "log_amenities", "log_distance_to_centre",
           "log_minimum_nights", "log_host_listings", "log_reviews", "review_scores_rating"]
BOOLEAN = ["bathroom_shared", "long_stay", "has_reviews", "host_is_superhost"]
FEATURES = CATEGORICAL + NUMERIC + BOOLEAN

# Direction of an effect we are sure about, all else equal. Trees may only learn effects in these directions,
# so a noisy group of listings cannot teach the model that an extra bedroom makes a flat cheaper.
MONOTONIC = {
    "accommodates": 1, "bedrooms": 1, "bathrooms": 1, "log_amenities": 1,
    "log_distance_to_centre": -1, "log_minimum_nights": -1, "bathroom_shared": -1, "long_stay": -1,
}

SEED = 42


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Booleans as floats (nulls stay missing), so both models accept them."""
    return df.assign(**{c: df[c].astype(float) for c in BOOLEAN})


def build_linear(categorical: list[str] = CATEGORICAL) -> Pipeline:
    """Linear regression on log price: the first model of week 1, kept as a reference."""
    return Pipeline([
        ("prepare", ColumnTransformer([
            # handle_unknown="ignore": an unseen category gets no adjustment instead of failing.
            ("categories", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
            # review_scores_rating is missing when there are no reviews; has_reviews carries that signal.
            ("numbers", SimpleImputer(strategy="median"), NUMERIC),
            ("flags", SimpleImputer(strategy="most_frequent"), BOOLEAN),
        ], verbose_feature_names_out=False)),
        ("regression", LinearRegression()),
    ])


def build_boosting(categorical: list[str] = CATEGORICAL) -> Pipeline:
    """Gradient boosting of small trees on log price with monotonic constraints.

    Absolute-error loss predicts the median log price, and exp(median of log) is the median price: the
    prediction that minimises MAE, the metric we are judged on. Missing values are handled by the trees.
    """
    encode = ColumnTransformer(
        # An unseen category becomes missing; the trees send it down the branch most training rows took.
        [("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), categorical)],
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    boosting = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=300, max_depth=5, min_samples_leaf=30,
        categorical_features=categorical, monotonic_cst=MONOTONIC, random_state=SEED,
    )
    return Pipeline([("encode", encode), ("regression", boosting)])


CANDIDATES: dict[str, Callable[[], RegressorMixin]] = {
    "linear": build_linear,
    "boosting": build_boosting,
}


def fit_baseline(df: pd.DataFrame) -> dict:
    """Median training price per room_type x accommodates, with room_type and overall medians as fallbacks."""
    return {
        "by_room_guests": df.groupby(["room_type", "accommodates"])["price"].median(),
        "by_room": df.groupby("room_type")["price"].median(),
        "overall": df["price"].median(),
    }


def predict_baseline(baseline: dict, df: pd.DataFrame) -> pd.Series:
    keys = list(zip(df["room_type"], df["accommodates"]))
    pred = pd.Series([baseline["by_room_guests"].get(k, np.nan) for k in keys], index=df.index, dtype=float)
    pred = pred.fillna(df["room_type"].map(baseline["by_room"]))
    return pred.fillna(baseline["overall"])


def mae(predicted, actual) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def cross_val_mae(make_model: Callable[[], RegressorMixin], df: pd.DataFrame, features: list[str],
                  folds: int = 5) -> float:
    """MAE in CHF over k folds of the training rows; the test split is never touched."""
    errors = []
    for fit_idx, val_idx in KFold(folds, shuffle=True, random_state=SEED).split(df):
        fit, val = df.iloc[fit_idx], df.iloc[val_idx]
        model = make_model().fit(fit[features], fit["log_price"])
        errors.append(mae(np.exp(model.predict(val[features])), val["price"]))
    return float(np.mean(errors))


def cross_val_baseline_mae(df: pd.DataFrame, folds: int = 5) -> float:
    errors = []
    for fit_idx, val_idx in KFold(folds, shuffle=True, random_state=SEED).split(df):
        fit, val = df.iloc[fit_idx], df.iloc[val_idx]
        errors.append(mae(predict_baseline(fit_baseline(fit), val), val["price"]))
    return float(np.mean(errors))


def price_level_offset(model: RegressorMixin, df: pd.DataFrame, features: list[str]) -> float:
    """Median log residual of the model on a city's known prices: how much pricier the city is than the model
    expects. exp(offset) - 1 is the city's price level in %."""
    return float(np.median(df["log_price"] - model.predict(df[features])))


def area_offsets(residuals: pd.Series, areas: pd.Series, prior_listings: int = 20) -> pd.Series:
    """Price level per area (e.g. municipality) on top of the city level, from known log residuals.

    The mean residual is shrunk towards 0 as if each area had prior_listings extra listings at the city level, so
    an area with 3 known prices barely moves and one with 300 gets its own level.
    """
    grouped = residuals.groupby(areas)
    return grouped.sum() / (grouped.count() + prior_listings)


def mae_table(df: pd.DataFrame, combined_label: str | None = "A + B") -> pd.DataFrame:
    """MAE in CHF of baseline_price and model_price against price, per city and (optionally) combined."""
    parts = list(df.groupby("city"))
    if combined_label:
        parts.append((combined_label, df))
    rows = [{
        "city": name,
        "test_listings": len(part),
        "baseline_mae": mae(part["baseline_price"], part["price"]),
        "model_mae": mae(part["model_price"], part["price"]),
        "median_price": part["price"].median(),
    } for name, part in parts]
    out = pd.DataFrame(rows)
    out["improvement_pct"] = 100 * (1 - out["model_mae"] / out["baseline_mae"])
    return out.round(1)
