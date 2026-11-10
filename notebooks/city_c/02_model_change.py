# Databricks notebook source
# MAGIC %md
# MAGIC # City C · 02 · Model changes: price levels from the new city's own known prices
# MAGIC
# MAGIC The first run showed two things the week 1 model cannot know about city C:
# MAGIC
# MAGIC - **its price level**: `city` is a label the model has never seen for C, so C gets a training city's level;
# MAGIC - **its local markets**: C is a canton. Distance to one centre means "suburb" in Geneva or Zurich but "Alpine
# MAGIC   resort" 30+ km from Lausanne, and chalets barely exist in the training cities.
# MAGIC
# MAGIC Both are fixed with prices the client already has: the **train rows of C are its known prices**, the **test
# MAGIC rows** are scored. On top of one **city-free** model trained on A + B:
# MAGIC
# MAGIC 1. **city level**: one number per city, the median log residual on its known prices;
# MAGIC 2. **area level**: one more number per municipality (`neighbourhood`), the mean residual left after step 1,
# MAGIC    shrunk towards 0 for municipalities with few known prices (`area_offsets`).
# MAGIC
# MAGIC The same rule is applied to A and B (their train rows are their known prices) to check it does not hurt them.
# MAGIC Writes `ga2_gold.city_c_changes`, the `after_change` row of `ga2_gold.city_c_report` and
# MAGIC `ga2_gold.predictions_after_change_<city>` for city C.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import CATALOG, GOLD, SILVER, cities_by_role, get_city, table_name  # noqa: E402
from bda.modeling import area_offsets, build_boosting, mae, price_level_offset, prepare  # noqa: E402
from bda.transfer import CITY_FREE, FEATURES_CITY_FREE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRAIN_CITIES = cities_by_role("train")
CITY_KEY = cities_by_role("new_client")[0]
REPORT = f"{CATALOG}.{GOLD}.city_c_report"
CHANGES = f"{CATALOG}.{GOLD}.city_c_changes"


def load(key: str) -> pd.DataFrame:
    features = spark.table(table_name(GOLD, "features", key)).filter("label_ok")
    areas = spark.table(table_name(SILVER, "listings", key)).selectExpr("id", "neighbourhood_cleansed AS neighbourhood")
    return prepare(features.join(areas, "id", "left").toPandas())


cities = {key: load(key) for key in TRAIN_CITIES + [CITY_KEY]}
ab = pd.concat([cities[k] for k in TRAIN_CITIES], ignore_index=True)
ab_train = ab[ab["split"] == "train"]

# COMMAND ----------

# MAGIC %md ## Fit the city-free model, then each city's price levels

# COMMAND ----------

model = build_boosting(CITY_FREE).fit(ab_train[FEATURES_CITY_FREE], ab_train["log_price"])


def score(frame: pd.DataFrame) -> dict:
    known, test = frame[frame["split"] == "train"], frame[frame["split"] == "test"]
    city_offset = price_level_offset(model, known, FEATURES_CITY_FREE)
    residual = known["log_price"] - model.predict(known[FEATURES_CITY_FREE]) - city_offset
    by_area = area_offsets(residual, known["neighbourhood"])

    log_pred = model.predict(test[FEATURES_CITY_FREE])
    with_city = np.exp(log_pred + city_offset)
    with_area = np.exp(log_pred + city_offset + test["neighbourhood"].map(by_area).fillna(0).to_numpy())
    return {
        "test": test.assign(price_city_level=with_city, price_area_level=with_area),
        "city_level_pct": 100 * (np.exp(city_offset) - 1),
        "areas_with_level": int((residual.groupby(known["neighbourhood"]).count() >= 5).sum()),
        "mae_city_free": mae(np.exp(log_pred), test["price"]),
        "mae_city_level": mae(with_city, test["price"]),
        "mae_area_level": mae(with_area, test["price"]),
    }


scores = {key: score(frame) for key, frame in cities.items()}
summary = pd.DataFrame([{
    "city": get_city(key)["name"], "role": get_city(key)["role"], "test_listings": len(s["test"]),
    "city_level_pct": s["city_level_pct"], "areas_with_5_known": s["areas_with_level"],
    "mae_city_free": s["mae_city_free"], "mae_city_level": s["mae_city_level"], "mae_area_level": s["mae_area_level"],
} for key, s in scores.items()]).round(1)
summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where the area level helps in city C
# MAGIC `median_pct_error` above 0 means over-priced. Compare the first run (`predicted_price` from `06_predict`) with
# MAGIC the city and area levels on the same test listings.

# COMMAND ----------

first_run = spark.table(table_name(GOLD, "predictions", CITY_KEY)).select("id", "predicted_price").toPandas()
c_test = scores[CITY_KEY]["test"].merge(first_run, on="id")


def pct_error(predicted: pd.Series) -> pd.Series:
    return 100 * (predicted - c_test["price"]) / c_test["price"]


top_areas = c_test["neighbourhood"].value_counts().head(10).index
(c_test[c_test["neighbourhood"].isin(top_areas)]
 .assign(first_run=pct_error(c_test["predicted_price"]), city_level=pct_error(c_test["price_city_level"]),
         area_level=pct_error(c_test["price_area_level"]))
 .groupby("neighbourhood")
 .agg(listings=("id", "size"), median_price=("price", "median"), first_run=("first_run", "median"),
      city_level=("city_level", "median"), area_level=("area_level", "median"))
 .round(1).sort_values("listings", ascending=False))

# COMMAND ----------

# MAGIC %md ## Save

# COMMAND ----------

c = scores[CITY_KEY]
first = spark.table(REPORT).filter("run = 'first_run' AND rows = 'test'").first()
ab_rows = summary[summary["role"] == "train"]
ab_mae = float(np.average(ab_rows["mae_area_level"], weights=ab_rows["test_listings"]))

changes = pd.DataFrame([
    {"step": 0, "change": "first run: week 1 model, unchanged", "model_mae_c": first["model_mae_c"]},
    {"step": 1, "change": "city-free model + city price level", "model_mae_c": c["mae_city_level"]},
    {"step": 2, "change": "+ municipality price level", "model_mae_c": c["mae_area_level"]},
]).round(1)
changes["baseline_mae_c"] = first["baseline_mae_c"]
spark.createDataFrame(changes).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(CHANGES)

after_change = pd.DataFrame([{
    "run": "after_change", "rows": "test", "listings": len(c["test"]), "median_price": first["median_price"],
    "baseline_mae_c": first["baseline_mae_c"], "model_mae_ab": ab_mae, "model_mae_c": c["mae_area_level"],
    "median_pct_error_c": float(np.median(pct_error(c_test["price_area_level"]))),
}]).round(1)
spark.sql(f"DELETE FROM {REPORT} WHERE run = 'after_change'")  # safe to re-run
(spark.createDataFrame(after_change.assign(city=get_city(CITY_KEY)["name"]), schema=spark.table(REPORT).schema)
 .write.mode("append").saveAsTable(REPORT))

(spark.createDataFrame(c_test[["id", "city", "neighbourhood", "room_type", "accommodates", "price", "predicted_price",
                               "price_city_level", "price_area_level"]]
                       .rename(columns={"predicted_price": "price_first_run"}).round(1))
 .write.mode("overwrite").option("overwriteSchema", True)
 .saveAsTable(table_name(GOLD, "predictions_after_change", CITY_KEY)))

display(spark.table(CHANGES).orderBy("step"))
display(spark.table(REPORT).orderBy("run", "rows"))
