# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze ingest
# MAGIC
# MAGIC Loads one city's raw `listings.csv` from the volume into `ga2_bronze.listings_raw_<city>`, **as published**:
# MAGIC every source column stays a string, nothing is cleaned or dropped. Cleaning happens in silver, so any later
# MAGIC number can be traced back to the exact source value.
# MAGIC
# MAGIC Added columns: `city_key`, `city`, `snapshot_date` (from `config/cities.json`), `source_file`, `ingested_at`.
# MAGIC
# MAGIC **Parameter `city`**: a key from `config/cities.json`, e.g. `A_Geneva`. Requires `00_setup` to have landed the file.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import BRONZE, get_city, table_name, volume_csv_path  # noqa: E402

dbutils.widgets.text("city", "A_Geneva", "City key")
CITY_KEY = dbutils.widgets.get("city").strip()

city = get_city(CITY_KEY)
SOURCE = volume_csv_path(CITY_KEY)
TARGET = table_name(BRONZE, "listings_raw", CITY_KEY)
print(f"{CITY_KEY}: {SOURCE} -> {TARGET}")

# COMMAND ----------

# MAGIC %md ## Read the CSV
# MAGIC
# MAGIC No `inferSchema`: types are decided in silver, so a new city with an odd value cannot break the load.
# MAGIC `multiLine` + `escape='"'` because free-text fields contain line breaks and quotes (see EDA).

# COMMAND ----------

from pyspark.sql import functions as F  # noqa: E402

if not os.path.exists(SOURCE):
    raise FileNotFoundError(f"{SOURCE} not found. Run notebooks/00_setup with cities={CITY_KEY} first.")

raw = (spark.read
       .option("header", True)
       .option("multiLine", True)
       .option("escape", '"')
       .option("mode", "FAILFAST")
       .csv(SOURCE))

bronze = raw.select(
    F.lit(CITY_KEY).alias("city_key"),
    F.lit(city["name"]).alias("city"),
    F.lit(city["snapshot_date"]).cast("date").alias("snapshot_date"),
    *raw.columns,
    F.col("_metadata.file_path").alias("source_file"),
    F.current_timestamp().alias("ingested_at"),
)

# COMMAND ----------

# MAGIC %md ## Checks
# MAGIC
# MAGIC Bronze does not clean, but it refuses to load a file that is clearly not a listings snapshot:
# MAGIC missing key columns, no rows, or more rows than listings (a sign the CSV was parsed wrongly).

# COMMAND ----------

REQUIRED = ["id", "price", "room_type", "accommodates", "neighbourhood_cleansed", "last_scraped"]
missing_cols = [c for c in REQUIRED if c not in raw.columns]
if missing_cols:
    raise ValueError(f"{CITY_KEY}: required columns missing from source: {missing_cols}")

stats = raw.agg(F.count("*").alias("rows"), F.countDistinct("id").alias("distinct_ids"),
                F.sum(F.col("id").isNull().cast("int")).alias("null_ids")).first()
if stats["rows"] == 0:
    raise ValueError(f"{CITY_KEY}: source file has no rows")
if stats["null_ids"] or stats["rows"] != stats["distinct_ids"]:
    raise ValueError(f"{CITY_KEY}: {stats['rows']} rows but {stats['distinct_ids']} distinct ids "
                     f"({stats['null_ids']} null) — CSV likely mis-parsed")

print(f"{CITY_KEY}: {stats['rows']} rows, {len(raw.columns)} source columns — OK")

# COMMAND ----------

# MAGIC %md ## Write

# COMMAND ----------

(bronze.write
 .mode("overwrite")
 .option("overwriteSchema", True)
 .saveAsTable(TARGET))

spark.sql(f"COMMENT ON TABLE {TARGET} IS 'Raw Inside Airbnb listings for {city['name']} "
          f"(snapshot {city['snapshot_date']}), all source columns as strings.'")

display(spark.table(TARGET).groupBy("city_key", "snapshot_date").count())
