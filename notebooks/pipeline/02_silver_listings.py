# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver listings
# MAGIC
# MAGIC Turns `ga2_bronze.listings_raw_<city>` into `ga2_silver.listings_<city>`: one clean, typed row per listing.
# MAGIC Every step comes from the cleaning table in the
# MAGIC [exploratory analysis](../eda/01_exploratory_analysis_A_B.ipynb):
# MAGIC
# MAGIC | Step | Reason |
# MAGIC |---|---|
# MAGIC | `price`: strip `$` and `,`, cast to double | stored as text like `"$1,025.00"`; currency is CHF |
# MAGIC | `bathrooms` from `bathrooms_text` (`Half-bath` → 0.5), flag `bathroom_shared` | the `bathrooms` column is empty |
# MAGIC | cast numbers, `t`/`f` booleans, dates | everything arrives as text |
# MAGIC | keep listings without price, flag `has_price` | missing price = inactive listing; never impute the target |
# MAGIC | flag `outlier_class` (`implausible` / `high_end` / `normal`) | placeholder prices must not reach training |
# MAGIC | drop `price_quote_*` | duplicates of the price (leakage review); nothing downstream needs them |
# MAGIC
# MAGIC Availability, occupancy and revenue are kept: the dashboard needs them. Leaky columns are removed when
# MAGIC building model features (`04_features`), not here.
# MAGIC
# MAGIC **Parameter `city`**: a key from `config/cities.json`, e.g. `A_Geneva`.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import BRONZE, SILVER, get_city, table_name  # noqa: E402

dbutils.widgets.text("city", "A_Geneva", "City key")
CITY_KEY = dbutils.widgets.get("city").strip()

city = get_city(CITY_KEY)
SOURCE = table_name(BRONZE, "listings_raw", CITY_KEY)
TARGET = table_name(SILVER, "listings", CITY_KEY)
print(f"{CITY_KEY}: {SOURCE} -> {TARGET}")

# COMMAND ----------

# MAGIC %md ## Rules
# MAGIC
# MAGIC Outlier thresholds are **fixed numbers from the A + B analysis**, not recomputed per city, so a new city is
# MAGIC judged by the same rules the model was trained under.

# COMMAND ----------

from pyspark.sql import functions as F  # noqa: E402

IMPLAUSIBLE_PRICE_PER_GUEST = 500   # CHF/guest/night, and fewer than 10 reviews ever
IMPLAUSIBLE_MAX_REVIEWS = 10
HIGH_END_PRICE_PER_GUEST = 311      # 99th percentile of price per guest in A + B

INTEGER_COLS = ["accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights",
                "availability_30", "availability_60", "availability_90", "availability_365",
                "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
                "host_listings_count", "calculated_host_listings_count", "hosts_time_as_host_years",
                "estimated_occupancy_l365d"]
DOUBLE_COLS = ["latitude", "longitude", "reviews_per_month", "estimated_revenue_l365d",
               "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
               "review_scores_checkin", "review_scores_communication", "review_scores_location",
               "review_scores_value"]
BOOLEAN_COLS = ["host_is_superhost", "host_identity_verified", "host_has_profile_pic", "has_availability"]
DATE_COLS = ["last_scraped", "first_review", "last_review"]
TEXT_COLS = ["listing_url", "name", "property_type", "room_type",
             "neighbourhood_cleansed", "neighbourhood_group_cleansed"]


def to_number(col, kind):
    # try_cast: a malformed value becomes null instead of failing the job; nulls are counted in the checks below.
    return F.expr(f"try_cast(`{col}` AS {kind})")


def to_bool(col):
    return F.when(F.col(col) == "t", True).when(F.col(col) == "f", False)

# COMMAND ----------

# MAGIC %md ## Transform

# COMMAND ----------

bronze = spark.table(SOURCE)

bath_text = F.lower(F.col("bathrooms_text"))

silver = bronze.select(
    "city_key", "city", "snapshot_date",
    F.col("id").alias("_source_id"),  # raw id, only for the checks below; dropped before writing
    F.expr("try_cast(id AS BIGINT)").alias("id"),
    F.expr("try_cast(host_id AS BIGINT)").alias("host_id"),
    *TEXT_COLS,
    F.expr("try_cast(regexp_replace(price, '[$,]', '') AS DOUBLE)").alias("price"),
    F.when(bath_text.contains("half"), F.lit(0.5))
     .otherwise(F.expr("try_cast(regexp_extract(bathrooms_text, '([0-9.]+)', 1) AS DOUBLE)")).alias("bathrooms"),
    bath_text.contains("shared").alias("bathroom_shared"),
    F.size(F.from_json("amenities", "array<string>")).alias("amenities_count"),
    *[to_number(c, "INT").alias(c) for c in INTEGER_COLS],
    *[to_number(c, "DOUBLE").alias(c) for c in DOUBLE_COLS],
    *[to_bool(c).alias(c) for c in BOOLEAN_COLS],
    *[F.expr(f"try_cast({c} AS DATE)").alias(c) for c in DATE_COLS],
)

price_per_guest = F.col("price") / F.col("accommodates")
silver = (silver
    .withColumn("has_price", F.col("price").isNotNull())
    .withColumn("outlier_class",
        F.when(~F.col("has_price"), F.lit(None))
         .when((price_per_guest > IMPLAUSIBLE_PRICE_PER_GUEST) &
               (F.col("number_of_reviews") < IMPLAUSIBLE_MAX_REVIEWS), "implausible")
         .when(price_per_guest > HIGH_END_PRICE_PER_GUEST, "high_end")
         .otherwise("normal")))

# COMMAND ----------

# MAGIC %md ## Checks
# MAGIC
# MAGIC A value that was present in bronze but is null after typing means the city's data has a format we did not
# MAGIC expect. For `price` (the target) and `id` this stops the run; for other columns it is reported.

# COMMAND ----------

typed = ["id", "price", "bathrooms"] + INTEGER_COLS + DOUBLE_COLS + BOOLEAN_COLS + DATE_COLS
source_of = {"bathrooms": "bathrooms_text"}

joined = bronze.select(F.col("id").alias("_source_id"), *[F.col(source_of.get(c, c)).alias(f"_raw_{c}") for c in typed]) \
    .join(silver.select("_source_id", *typed), "_source_id")
failures = joined.agg(*[F.sum((F.col(f"_raw_{c}").isNotNull() & F.col(c).isNull()).cast("int")).alias(c)
                        for c in typed]).first().asDict()
failures = {c: n for c, n in failures.items() if n}

print(f"Unparseable values per column: {failures or 'none'}")
critical = {c: failures[c] for c in ("id", "price") if c in failures}
if critical:
    raise ValueError(f"{CITY_KEY}: unparseable values in critical columns {critical}; inspect {SOURCE}")

summary = silver.groupBy("outlier_class").count().orderBy("outlier_class")
display(summary)

# COMMAND ----------

# MAGIC %md ## Write

# COMMAND ----------

(silver.drop("_source_id").write
 .mode("overwrite")
 .option("overwriteSchema", True)
 .saveAsTable(TARGET))

spark.sql(f"COMMENT ON TABLE {TARGET} IS 'Cleaned, typed Airbnb listings for {city['name']} "
          f"(snapshot {city['snapshot_date']}); price in CHF; flags has_price and outlier_class.'")

display(spark.table(TARGET).agg(
    F.count("*").alias("listings"),
    F.sum(F.col("has_price").cast("int")).alias("with_price"),
    F.expr("percentile_approx(price, 0.5)").alias("median_price"),
))
