# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC
# MAGIC Prepares the Unity Catalog objects the lakehouse needs and lands the raw snapshot files.
# MAGIC Safe to re-run: every step is `IF NOT EXISTS` or an overwrite.
# MAGIC
# MAGIC | Object | Purpose |
# MAGIC |---|---|
# MAGIC | `workspace.ga2_bronze` | raw files (volume) and raw tables |
# MAGIC | `workspace.ga2_silver` | cleaned, typed listings |
# MAGIC | `workspace.ga2_gold` | metrics, model features, predictions |
# MAGIC | `/Volumes/workspace/ga2_bronze/raw/<city>/listings.csv` | copy of `data/<city>/listings.csv` from this repo |
# MAGIC
# MAGIC Files are copied from the Git folder rather than downloaded from Inside Airbnb, because Free Edition
# MAGIC restricts outbound internet and the repo copy pins the exact snapshot we analysed.
# MAGIC
# MAGIC **Parameter `cities`**: comma-separated city keys from `config/cities.json`, `train` (A and B — the default),
# MAGIC or `all`. City C stays out of the volume until week 2.

# COMMAND ----------

import os
import shutil
import sys

# The notebook's working directory is notebooks/ inside the Git folder; src/ holds the shared package.
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import (  # noqa: E402
    BRONZE, CATALOG, GOLD, RAW_VOLUME, SILVER,
    cities_by_role, get_city, load_config, volume_csv_path,
)

# COMMAND ----------

dbutils.widgets.text("cities", "train", "Cities to land (keys, 'train' or 'all')")
cities_param = dbutils.widgets.get("cities").strip()

if cities_param == "train":
    city_keys = cities_by_role("train")
elif cities_param == "all":
    city_keys = list(load_config()["cities"])
else:
    city_keys = [c.strip() for c in cities_param.split(",") if c.strip()]

print("Repo root:", REPO_ROOT)
print("Cities:", city_keys)

# COMMAND ----------

# MAGIC %md ## 1. Schemas and raw volume

# COMMAND ----------

for schema in (BRONZE, SILVER, GOLD):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{BRONZE}.{RAW_VOLUME}")

display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))

# COMMAND ----------

# MAGIC %md ## 2. Land raw files in the volume

# COMMAND ----------

landed = []
for key in city_keys:
    city = get_city(key)
    src = os.path.join(REPO_ROOT, city["raw_path"])
    dst = volume_csv_path(key)
    if not os.path.exists(src):
        raise FileNotFoundError(f"{key}: {src} not found in the Git folder")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

    src_size, dst_size = os.path.getsize(src), os.path.getsize(dst)
    if src_size != dst_size:
        raise IOError(f"{key}: copy incomplete ({dst_size} of {src_size} bytes)")
    landed.append((key, city["name"], city["letter"], city["snapshot_date"], dst, dst_size))

display(spark.createDataFrame(
    landed, "city_key string, city string, letter string, snapshot_date string, path string, bytes long"
))

# COMMAND ----------

# MAGIC %md ## 3. Sanity check: Spark can parse the landed files
# MAGIC
# MAGIC Free-text fields (`description`, `host_about`, …) contain line breaks and quotes, so the CSV must be read
# MAGIC with `multiLine` and `escape='"'`. Without them one listing spills over several rows and the counts are wrong.

# COMMAND ----------

checks = []
for key in city_keys:
    df = (spark.read
          .option("header", True)
          .option("multiLine", True)
          .option("escape", '"')
          .csv(volume_csv_path(key)))
    checks.append((key, df.count(), df.select("id").distinct().count(), len(df.columns)))

display(spark.createDataFrame(checks, "city_key string, rows long, distinct_ids long, columns int"))
