"""How well the price model moves between cities, measured with the training cities only."""

import numpy as np
import pandas as pd

from bda.modeling import (
    BOOLEAN, CATEGORICAL, FEATURES, NUMERIC, build_boosting, build_linear, fit_baseline, mae, predict_baseline,
    price_level_offset,
)

CITY_FREE = [c for c in CATEGORICAL if c != "city"]
FEATURES_CITY_FREE = [c for c in FEATURES if c != "city"]


def leave_one_city_out(data: pd.DataFrame) -> pd.DataFrame:
    """Train on one city, test on another city's test rows, as if it were a new client.

    raw: the model as is. calibrated: shifted by the target city's price level, measured on the known prices of
    its train rows. own_city: a model trained on the target city itself, the best we could hope for.
    """
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]
    rows = []
    for source in sorted(data["city"].unique()):
        for target in sorted(data["city"].unique()):
            if source == target:
                continue
            fit = train[train["city"] == source]
            known = train[train["city"] == target]
            evaluate = test[test["city"] == target]

            model = build_boosting(CITY_FREE).fit(fit[FEATURES_CITY_FREE], fit["log_price"])
            raw = np.exp(model.predict(evaluate[FEATURES_CITY_FREE]))
            offset = price_level_offset(model, known, FEATURES_CITY_FREE)
            own = build_boosting(CITY_FREE).fit(known[FEATURES_CITY_FREE], known["log_price"])
            rows.append({
                "trained_on": source,
                "tested_on": target,
                "test_listings": len(evaluate),
                "baseline_mae": mae(predict_baseline(fit_baseline(fit), evaluate), evaluate["price"]),
                "raw_mae": mae(raw, evaluate["price"]),
                "calibrated_mae": mae(raw * np.exp(offset), evaluate["price"]),
                "own_city_mae": mae(np.exp(own.predict(evaluate[FEATURES_CITY_FREE])), evaluate["price"]),
                "price_level_pct": 100 * (np.exp(offset) - 1),
            })
    out = pd.DataFrame(rows)
    out["transfer_penalty_pct"] = 100 * (out["raw_mae"] / out["own_city_mae"] - 1)
    return out.round(1)


def calibration_curve(data: pd.DataFrame, sizes=(25, 50, 100, 200, 400), repeats: int = 30,
                      seed: int = 0) -> pd.DataFrame:
    """How many known prices a new city must give us before the price-level shift helps.

    For each pair of cities and each size, draws that many known prices from the target's train rows, shifts the
    transferred model by their median residual and measures MAE on the target's test rows.
    """
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]
    rows = []
    for source in sorted(data["city"].unique()):
        fit = train[train["city"] == source]
        model = build_boosting(CITY_FREE).fit(fit[FEATURES_CITY_FREE], fit["log_price"])
        for target in sorted(set(data["city"]) - {source}):
            pool, evaluate = train[train["city"] == target], test[test["city"] == target]
            raw = np.exp(model.predict(evaluate[FEATURES_CITY_FREE]))
            for size in sizes:
                errors = [mae(raw * np.exp(price_level_offset(model, pool.sample(size, random_state=seed + i),
                                                              FEATURES_CITY_FREE)), evaluate["price"])
                          for i in range(repeats)]
                rows.append({"trained_on": source, "tested_on": target, "known_prices": size,
                             "raw_mae": mae(raw, evaluate["price"]), "calibrated_mae": np.mean(errors),
                             "calibrated_mae_worst": np.max(errors)})
    return pd.DataFrame(rows).round(1)


def effect_stability(data: pd.DataFrame, repeats: int = 300, seed: int = 0) -> pd.DataFrame:
    """Per-city linear effects with 90% bootstrap intervals, and whether the two cities agree.

    An effect is `shared` when the 95% interval of the difference between cities contains zero: it looks like a
    Swiss-wide rule. Otherwise it is `city-specific` and the riskiest part of a transfer.
    """
    rng = np.random.default_rng(seed)
    train = data[data["split"] == "train"]
    cities = sorted(train["city"].unique())
    if len(cities) != 2:
        raise ValueError(f"effect_stability compares two cities, got {cities}")

    draws = {}
    for city in cities:
        rows = train[train["city"] == city]
        samples = []
        for _ in range(repeats):
            sample = rows.sample(len(rows), replace=True, random_state=int(rng.integers(2**31)))
            model = build_linear(CITY_FREE).fit(sample[FEATURES_CITY_FREE], sample["log_price"])
            names = model.named_steps["prepare"].get_feature_names_out()
            samples.append(pd.Series(model.named_steps["regression"].coef_, index=names))
        draws[city] = pd.concat(samples, axis=1)

    # Hotel and shared rooms are a handful of listings per city; their effects are noise, not a finding.
    effects = [f for f in NUMERIC + BOOLEAN + ["room_type_Private room"] if f in draws[cities[0]].index]
    first, second = (draws[c].loc[effects] for c in cities)
    difference = first - second.values

    def pct(x):
        return 100 * (np.exp(x) - 1)

    out = pd.DataFrame({
        "feature": effects,
        f"{cities[0].lower()}_effect_pct": pct(first.median(axis=1)).values,
        f"{cities[0].lower()}_low": pct(first.quantile(0.05, axis=1)).values,
        f"{cities[0].lower()}_high": pct(first.quantile(0.95, axis=1)).values,
        f"{cities[1].lower()}_effect_pct": pct(second.median(axis=1)).values,
        f"{cities[1].lower()}_low": pct(second.quantile(0.05, axis=1)).values,
        f"{cities[1].lower()}_high": pct(second.quantile(0.95, axis=1)).values,
    })
    agree = (difference.quantile(0.025, axis=1) < 0) & (difference.quantile(0.975, axis=1) > 0)
    out["verdict"] = np.where(agree.values, "shared", "city-specific")
    return out.round(1)
