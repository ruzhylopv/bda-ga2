# Databricks notebook source
# MAGIC %md
# MAGIC # City C · 01 · First run
# MAGIC
# MAGIC Measures what the **unchanged** week 1 pipeline and model did on the new-client city. Run it after:
# MAGIC
# MAGIC 1. `ga2_etl` with `city = C_Vaud` (the pipeline as tagged `week1-final`),
# MAGIC 2. `ga2_predict` with `city = C_Vaud` (the `champion` model, not retrained).
# MAGIC
# MAGIC If a step failed, write the error into `CHANGES.md` before fixing anything; this notebook only reads tables.
# MAGIC
# MAGIC Writes the `first_run` rows of `ga2_gold.city_c_report`.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import BRONZE, CATALOG, GOLD, SILVER, cities_by_role, get_city, table_name  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CITY_KEY = cities_by_role("new_client")[0]
city = get_city(CITY_KEY)
REPORT = f"{CATALOG}.{GOLD}.city_c_report"
print(f"New client: {CITY_KEY}")

# COMMAND ----------

# MAGIC %md ## 1. Did every stage produce its table?

# COMMAND ----------

stages = [("bronze", table_name(BRONZE, "listings_raw", CITY_KEY)),
          ("silver", table_name(SILVER, "listings", CITY_KEY)),
          ("gold metrics", table_name(GOLD, "city_metrics", CITY_KEY)),
          ("features", table_name(GOLD, "features", CITY_KEY)),
          ("predictions", table_name(GOLD, "predictions", CITY_KEY))]
status = pd.DataFrame([{"stage": stage, "table": t, "exists": spark.catalog.tableExists(t),
                        "rows": spark.table(t).count() if spark.catalog.tableExists(t) else None}
                       for stage, t in stages])
status

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Error on city C
# MAGIC All of city C is unseen, so every priced listing counts (`all`). The `test` rows are reported too: the model
# MAGIC change in `02_model_change` uses the `train` rows as known prices, so `test` is the fair comparison.

# COMMAND ----------

predictions = spark.table(table_name(GOLD, "predictions", CITY_KEY)).filter("label_ok").toPandas()
evaluation = spark.table(f"{CATALOG}.{GOLD}.model_evaluation").toPandas()
expectation = spark.table(f"{CATALOG}.{GOLD}.city_c_expectation").first()

in_city_mae = evaluation.loc[evaluation["city"] == "A + B", "model_mae"].iloc[0]
report = pd.DataFrame([{
    "run": "first_run",
    "rows": rows,
    "listings": len(part),
    "median_price": part["price"].median(),
    "baseline_mae_c": part["baseline_abs_error"].mean(),
    "model_mae_ab": in_city_mae,
    "model_mae_c": part["abs_error"].mean(),
    "median_pct_error_c": part["pct_error"].median(),
} for rows, part in [("all", predictions), ("test", predictions[predictions["split"] == "test"])]]).round(1)

print(f"Expected before the run: MAE CHF {expectation['expected_mae_low']} to {expectation['expected_mae_high']}")
report

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Where the error comes from
# MAGIC `median_pct_error` above 0 means the model over-prices that group; a shift shared by every group points to a
# MAGIC price level the model never saw, a shift in one group to a feature that means something else in city C.

# COMMAND ----------

def by(column: str, frame: pd.DataFrame = predictions) -> pd.DataFrame:
    return (frame.groupby(column, observed=True)
            .agg(listings=("id", "size"), median_price=("price", "median"),
                 model_mae=("abs_error", "mean"), baseline_mae=("baseline_abs_error", "mean"),
                 median_pct_error=("pct_error", "median"))
            .round(1).sort_values("listings", ascending=False))


display(by("room_type"))

# COMMAND ----------

features = spark.table(table_name(GOLD, "features", CITY_KEY)).select("id", "log_distance_to_centre").toPandas()
located = predictions.merge(features, on="id")
located["distance_km"] = pd.cut(np.expm1(located["log_distance_to_centre"]), [0, 1, 3, 10, 30, np.inf],
                                labels=["<1 km", "1-3 km", "3-10 km", "10-30 km", "30+ km"], include_lowest=True)
display(by("distance_km", located))

# COMMAND ----------

display(by("neighbourhood").head(15))

# COMMAND ----------

# MAGIC %md ## Save

# COMMAND ----------

(spark.createDataFrame(report.assign(city=city["name"]))
 .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(REPORT))
spark.sql(f"COMMENT ON TABLE {REPORT} IS 'City C report: baseline MAE on C, model MAE on A + B and on C (CHF), "
          f"first run of the unchanged model and after the model change.'")
display(spark.table(REPORT))
