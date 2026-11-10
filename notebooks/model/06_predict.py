# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Predict
# MAGIC
# MAGIC Prices **every listing** of one city with the `champion` model and writes `ga2_gold.predictions_<city>`, the
# MAGIC table the dashboard and Genie use for the model page.
# MAGIC
# MAGIC - The model is never retrained here. For city C this is the unchanged week 1 model.
# MAGIC - The baseline is refitted from the train rows of the training cities, exactly as in `05_train_evaluate`.
# MAGIC - `error = predicted_price - price`: positive means the model **over-prices** the listing.
# MAGIC - Listings without a usable price still get a prediction (a suggested price); their error is null.
# MAGIC
# MAGIC **Parameter `city`**: a key from `config/cities.json`, e.g. `A_Geneva`.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import CATALOG, GOLD, SILVER, cities_by_role, get_city, load_config, table_name  # noqa: E402
from bda.modeling import FEATURES, fit_baseline, predict_baseline, prepare  # noqa: E402

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

dbutils.widgets.text("city", "A_Geneva", "City key")
CITY_KEY = dbutils.widgets.get("city").strip()

city = get_city(CITY_KEY)
MODEL_URI = f"models:/{CATALOG}.{GOLD}.price_model@champion"
FEATURES_TABLE = table_name(GOLD, "features", CITY_KEY)
TARGET = table_name(GOLD, "predictions", CITY_KEY)
print(f"{CITY_KEY}: {FEATURES_TABLE} + {MODEL_URI} -> {TARGET}")

# COMMAND ----------

# MAGIC %md ## Model and baseline

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model = mlflow.sklearn.load_model(MODEL_URI)
model_version = mlflow.MlflowClient().get_model_version_by_alias(f"{CATALOG}.{GOLD}.price_model", "champion").version

train_rows = pd.concat(
    [spark.table(table_name(GOLD, "features", k)).filter("label_ok AND split = 'train'").toPandas()
     for k in cities_by_role("train")],
    ignore_index=True)
baseline = fit_baseline(train_rows)
print(f"Model version {model_version}; baseline from {len(train_rows)} train listings")

# COMMAND ----------

# MAGIC %md ## Predict

# COMMAND ----------

listings = prepare(spark.table(FEATURES_TABLE).toPandas())

predictions = listings[["city_key", "city", "snapshot_date", "id", "room_type", "accommodates", "price",
                        "label_ok", "split"]].assign(
    predicted_price=np.exp(model.predict(listings[FEATURES])).round(1),
    baseline_price=predict_baseline(baseline, listings).round(1),
)
labelled = predictions["label_ok"]
error = (predictions["predicted_price"] - predictions["price"]).where(labelled)
predictions["error"] = error.round(1)
predictions["abs_error"] = error.abs().round(1)
baseline_error = (predictions["baseline_price"] - predictions["price"]).where(labelled)
predictions["baseline_abs_error"] = baseline_error.abs().round(1)
predictions["pct_error"] = (100 * error / predictions["price"]).round(1)
predictions["model_version"] = str(model_version)

# COMMAND ----------

# MAGIC %md ## Checks

# COMMAND ----------

if predictions["predicted_price"].isna().any() or not np.isfinite(predictions["predicted_price"]).all():
    raise ValueError(f"{CITY_KEY}: the model returned missing or infinite prices")

scored = predictions[labelled]
summary = pd.DataFrame([{
    "split": split,
    "listings": len(part),
    "model_mae": part["abs_error"].mean(),
    "baseline_mae": part["baseline_abs_error"].mean(),
    "median_price": part["price"].median(),
} for split, part in [*scored.groupby("split"), ("all", scored)]]).round(1)
summary

# COMMAND ----------

# MAGIC %md ## Write
# MAGIC Neighbourhood and listing name come from silver, so the dashboard can show errors by neighbourhood.

# COMMAND ----------

listing_info = spark.table(table_name(SILVER, "listings", CITY_KEY)).select(
    "id", "name", "listing_url",
    F.col("neighbourhood_cleansed").alias("neighbourhood"),
    F.col("neighbourhood_group_cleansed").alias("neighbourhood_group"),
)
(spark.createDataFrame(predictions)
 .join(listing_info, "id", "left")
 .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(TARGET))

spark.sql(f"COMMENT ON TABLE {TARGET} IS 'Model and baseline price (CHF) for every {city['name']} listing. "
          f"error = predicted_price - price, positive means over-priced. Use label_ok = true for accuracy; "
          f"split = test are the listings the model never saw (all of a new-client city is unseen).'")

# COMMAND ----------

# MAGIC %md ## Cross-city view

# COMMAND ----------

tables = [table_name(GOLD, "predictions", k) for k in load_config()["cities"]]
tables = [t for t in tables if spark.catalog.tableExists(t)]
union = "\nUNION ALL\n".join(f"SELECT * FROM {t}" for t in tables)
spark.sql(f"CREATE OR REPLACE VIEW {CATALOG}.{GOLD}.predictions_all AS\n{union}")
spark.sql(f"COMMENT ON VIEW {CATALOG}.{GOLD}.predictions_all IS "
          "'Model and baseline price per listing, all cities. error = predicted_price - price "
          "(positive = over-priced). Filter label_ok = true for accuracy.'")
print(f"predictions_all: {len(tables)} cities")
