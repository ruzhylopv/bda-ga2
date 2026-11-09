# Databricks notebook source
# MAGIC %md
# MAGIC # City C · 02 · Model change: city price level instead of a city label
# MAGIC
# MAGIC The week 1 model has `city` as a feature. A new city is a category it has never seen, so city C is priced as
# MAGIC if it were one of the training cities. The transfer rehearsal (`model/07_transfer_rehearsal`) showed the fix:
# MAGIC
# MAGIC - train one **city-free** model on A + B: it learns what makes a listing pricier than another (size,
# MAGIC   bathrooms, distance from the centre, ...);
# MAGIC - give each city **one number, its price level**: the median log residual of that model on the city's known
# MAGIC   prices. For A and B these are their train rows; for city C, its train rows, which a client could hand us.
# MAGIC
# MAGIC Evaluated on the **test** rows of every city, so it compares directly with the `test` row of `01_first_run`.
# MAGIC Appends the `after_change` row to `ga2_gold.city_c_report`.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import CATALOG, GOLD, cities_by_role, get_city, table_name  # noqa: E402
from bda.modeling import build_boosting, mae, price_level_offset, prepare  # noqa: E402
from bda.transfer import CITY_FREE, FEATURES_CITY_FREE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRAIN_CITIES = cities_by_role("train")
CITY_KEY = cities_by_role("new_client")[0]
REPORT = f"{CATALOG}.{GOLD}.city_c_report"


def load(key: str) -> pd.DataFrame:
    return prepare(spark.table(table_name(GOLD, "features", key)).filter("label_ok").toPandas())


cities = {key: load(key) for key in TRAIN_CITIES + [CITY_KEY]}
ab = pd.concat([cities[k] for k in TRAIN_CITIES], ignore_index=True)
ab_train = ab[ab["split"] == "train"]

# COMMAND ----------

# MAGIC %md ## Fit the city-free model and each city's price level

# COMMAND ----------

model = build_boosting(CITY_FREE).fit(ab_train[FEATURES_CITY_FREE], ab_train["log_price"])

rows = []
for key, frame in cities.items():
    known, test = frame[frame["split"] == "train"], frame[frame["split"] == "test"]
    offset = price_level_offset(model, known, FEATURES_CITY_FREE)
    predicted = np.exp(model.predict(test[FEATURES_CITY_FREE]) + offset)
    rows.append({"city": get_city(key)["name"], "role": get_city(key)["role"], "known_prices": len(known),
                 "price_level_pct": 100 * (np.exp(offset) - 1), "test_listings": len(test),
                 "model_mae": mae(predicted, test["price"]),
                 "median_pct_error": np.median(100 * (predicted - test["price"]) / test["price"])})
levels = pd.DataFrame(rows).round(1)
levels

# COMMAND ----------

# MAGIC %md ## Report: the three numbers side by side

# COMMAND ----------

ab_rows = levels[levels["role"] == "train"]
model_mae_ab = np.average(ab_rows["model_mae"], weights=ab_rows["test_listings"])
c_row = levels[levels["role"] == "new_client"].iloc[0]

first_run = spark.table(REPORT).filter("run = 'first_run' AND rows = 'test'").toPandas().iloc[0]
after_change = pd.DataFrame([{
    "run": "after_change", "rows": "test", "listings": int(c_row["test_listings"]),
    "median_price": first_run["median_price"], "baseline_mae_c": first_run["baseline_mae_c"],
    "model_mae_ab": model_mae_ab, "model_mae_c": c_row["model_mae"], "median_pct_error_c": c_row["median_pct_error"],
}]).round(1)

spark.sql(f"DELETE FROM {REPORT} WHERE run = 'after_change'")  # safe to re-run
(spark.createDataFrame(after_change.assign(city=get_city(CITY_KEY)["name"]), schema=spark.table(REPORT).schema)
 .write.mode("append").saveAsTable(REPORT))
display(spark.table(REPORT).orderBy("run", "rows"))
