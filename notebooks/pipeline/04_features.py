# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Features
# MAGIC
# MAGIC Builds `ga2_gold.features_<city>` from `ga2_silver.listings_<city>`: one row per listing with the model inputs,
# MAGIC the target and the train/test split.
# MAGIC
# MAGIC - **Only leak-free columns** (see the leakage review in the EDA): what a host decides when setting up the listing,
# MAGIC   plus long-run review history. No price quotes, revenue, occupancy or availability.
# MAGIC - **Location without neighbourhood names**: `log_distance_to_centre`, where the centre is the densest
# MAGIC   ~1 km cell of listings. Neighbourhood names do not exist in a new city; a distance does.
# MAGIC - **Every listing is kept**, so `06_predict` can price all of them. `label_ok` marks the rows the model may learn
# MAGIC   from and be evaluated on: a price exists and it is not `implausible`.
# MAGIC - **Split before any modelling**: 20% of listings per city go to `test`, chosen by a hash of the listing id —
# MAGIC   reproducible and independent of how often the pipeline runs.
# MAGIC
# MAGIC **Parameter `city`**: a key from `config/cities.json`, e.g. `A_Geneva`.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import GOLD, SILVER, get_city, table_name  # noqa: E402

dbutils.widgets.text("city", "A_Geneva", "City key")
CITY_KEY = dbutils.widgets.get("city").strip()

city = get_city(CITY_KEY)
SOURCE = table_name(SILVER, "listings", CITY_KEY)
TARGET = table_name(GOLD, "features", CITY_KEY)
print(f"{CITY_KEY}: {SOURCE} -> {TARGET}")

# COMMAND ----------

# MAGIC %md ## Features

# COMMAND ----------

from pyspark.sql import functions as F  # noqa: E402

LONG_STAY_NIGHTS = 28   # monthly rentals: ~30–40% lower nightly price in the EDA
TEST_SHARE_BUCKETS = 5  # 1 bucket of 5 -> 20% test

FEATURES = ["room_type", "city", "accommodates", "bedrooms", "bathrooms", "bathroom_shared", "log_amenities",
            "log_distance_to_centre", "log_minimum_nights", "log_host_listings", "long_stay", "has_reviews",
            "log_reviews", "review_scores_rating", "host_is_superhost"]
LEAKY = ["price_quote_price_per_night", "price_quote_total_price", "estimated_revenue_l365d",
         "estimated_occupancy_l365d", "availability_30", "availability_60", "availability_90", "availability_365",
         "number_of_reviews_ltm", "reviews_per_month", "review_scores_value"]
assert not set(FEATURES) & set(LEAKY), "a leaky column is listed as a feature"

silver = spark.table(SOURCE)

# The centre is the densest ~1 km grid cell of listings, found from coordinates alone (no prices), so it works
# for any new city without a hand-picked landmark.
CENTRE_GRID_DEGREES = 2
centre = (silver
    .groupBy(F.round("latitude", CENTRE_GRID_DEGREES).alias("lat"),
             F.round("longitude", CENTRE_GRID_DEGREES).alias("lon"))
    .count()
    .orderBy(F.desc("count"), "lat", "lon")
    .first())
print(f"{CITY_KEY}: centre at {centre['lat']}, {centre['lon']} ({centre['count']} listings in the cell)")

KM_PER_DEGREE = 111.2
distance_km = F.sqrt(
    F.pow((F.col("latitude") - centre["lat"]) * KM_PER_DEGREE, 2)
    + F.pow((F.col("longitude") - centre["lon"]) * KM_PER_DEGREE * F.cos(F.radians(F.lit(centre["lat"]))), 2))

features = silver.select(
    "city_key", "city", "snapshot_date", "id",
    "room_type",
    "accommodates",
    # For rooms, bedrooms describe the whole property (a hotel lists 28); a guest cannot use more than one per person.
    F.least("bedrooms", "accommodates").alias("bedrooms"),
    "bathrooms",
    "bathroom_shared",
    F.log1p("amenities_count").alias("log_amenities"),
    F.log1p(distance_km).alias("log_distance_to_centre"),
    F.log1p("minimum_nights").alias("log_minimum_nights"),
    (F.col("minimum_nights") >= LONG_STAY_NIGHTS).alias("long_stay"),
    # Professional hosts (serviced apartments, agencies) price differently from someone renting out their flat.
    F.log1p("calculated_host_listings_count").alias("log_host_listings"),
    (F.col("number_of_reviews") > 0).alias("has_reviews"),
    F.log1p("number_of_reviews").alias("log_reviews"),
    "review_scores_rating",  # null when there are no reviews; imputed in training, has_reviews carries the signal
    F.coalesce("host_is_superhost", F.lit(False)).alias("host_is_superhost"),
    "price",
    F.log("price").alias("log_price"),
    (F.col("has_price") & (F.col("outlier_class") != "implausible")).alias("label_ok"),
    F.when(F.abs(F.xxhash64(F.col("id").cast("string"))) % TEST_SHARE_BUCKETS == 0, "test")
     .otherwise("train").alias("split"),
)

# COMMAND ----------

# MAGIC %md ## Checks

# COMMAND ----------

missing = [c for c in FEATURES if c not in features.columns]
if missing:
    raise ValueError(f"{CITY_KEY}: feature columns missing: {missing}")

counts = {r["split"]: r["count"] for r in features.filter("label_ok").groupBy("split").count().collect()}
n_ok = sum(counts.values())
if not n_ok:
    raise ValueError(f"{CITY_KEY}: no listing has a usable price label")
print(f"{CITY_KEY}: {features.count()} listings, {n_ok} with a usable label "
      f"(train {counts.get('train', 0)}, test {counts.get('test', 0)} = {counts.get('test', 0) / n_ok:.0%})")

# COMMAND ----------

# MAGIC %md ## Write

# COMMAND ----------

features.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(TARGET)
spark.sql(f"COMMENT ON TABLE {TARGET} IS 'Leak-free model features for {city['name']}, one row per listing. "
          f"Train/evaluate only on label_ok = true; split = test is held out.'")
