# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold metrics
# MAGIC
# MAGIC Builds the business-ready tables for the dashboard and Genie from `ga2_silver.listings_<city>`:
# MAGIC
# MAGIC | Table | Grain | Answers |
# MAGIC |---|---|---|
# MAGIC | `city_metrics_<city>` | one row per room type **plus an `All` row** (the city total) | listings, activity, hosts, price level, availability, reviews, room-type mix |
# MAGIC | `neighbourhood_metrics_<city>` | one row per neighbourhood | neighbourhoods by size and price, map position |
# MAGIC
# MAGIC and refreshes the cross-city views `city_metrics_all`, `neighbourhood_metrics_all`, `listings_all`
# MAGIC over every city whose tables exist (city C joins automatically after its first run).
# MAGIC
# MAGIC **Two populations.** Counts, hosts and availability use *all* listings, so inactive listings stay visible.
# MAGIC Price figures use only listings with a price that are not `implausible` — the population the model learns from.
# MAGIC
# MAGIC **Parameter `city`**: a key from `config/cities.json`, e.g. `A_Geneva`.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import CATALOG, GOLD, SILVER, get_city, load_config, table_name  # noqa: E402

dbutils.widgets.text("city", "A_Geneva", "City key")
CITY_KEY = dbutils.widgets.get("city").strip()

city = get_city(CITY_KEY)
SOURCE = table_name(SILVER, "listings", CITY_KEY)
CITY_METRICS = table_name(GOLD, "city_metrics", CITY_KEY)
NEIGHBOURHOOD_METRICS = table_name(GOLD, "neighbourhood_metrics", CITY_KEY)
print(f"{CITY_KEY}: {SOURCE} -> {CITY_METRICS}, {NEIGHBOURHOOD_METRICS}")

# COMMAND ----------

# MAGIC %md ## City metrics: room-type rows + `All` row
# MAGIC
# MAGIC `rollup` computes every metric once per room type and once for the whole city, each directly from listings.
# MAGIC Medians and distinct host counts cannot be combined across room types, so the `All` row is never a sum of
# MAGIC the others — and must not be summed with them.

# COMMAND ----------

from pyspark.sql import functions as F  # noqa: E402

listings = (spark.table(SOURCE)
            .withColumn("price_ok", F.col("has_price") & (F.col("outlier_class") != "implausible")))
total_listings = listings.count()

ok_price = F.when(F.col("price_ok"), F.col("price"))


def pct(condition):
    return F.round(100 * F.avg(condition.cast("double")), 1)


city_metrics = (listings
    .rollup("room_type")
    .agg(
        F.count("*").alias("listings"),
        F.sum(F.col("has_price").cast("int")).alias("active_listings"),
        pct(F.col("has_price")).alias("pct_active"),
        F.countDistinct("host_id").alias("hosts"),
        F.round(100 * F.countDistinct(F.when(F.col("calculated_host_listings_count") > 1, F.col("host_id")))
                / F.countDistinct("host_id"), 1).alias("pct_multi_listing_hosts"),
        F.round(F.percentile(ok_price, 0.5), 1).alias("median_price"),
        F.round(F.percentile(ok_price, 0.25), 1).alias("p25_price"),
        F.round(F.percentile(ok_price, 0.75), 1).alias("p75_price"),
        F.percentile("availability_365", 0.5).alias("median_availability_365"),
        pct(F.col("availability_365") == 0).alias("pct_closed_all_year"),
        F.round(F.avg("review_scores_rating"), 2).alias("avg_review_rating"),
        pct(F.col("number_of_reviews") > 0).alias("pct_with_reviews"),
        pct(F.coalesce(F.col("host_is_superhost"), F.lit(False))).alias("pct_superhost"),
    )
    .withColumn("room_type", F.coalesce("room_type", F.lit("All")))
    .withColumn("pct_of_listings", F.round(100 * F.col("listings") / F.lit(total_listings), 1))
    .select(F.lit(CITY_KEY).alias("city_key"), F.lit(city["name"]).alias("city"),
            F.lit(city["snapshot_date"]).cast("date").alias("snapshot_date"), "*")
    .orderBy(F.col("room_type") != "All", F.desc("listings")))

display(city_metrics)

# COMMAND ----------

# MAGIC %md ## Neighbourhood metrics

# COMMAND ----------

neighbourhood_metrics = (listings
    .groupBy(F.col("neighbourhood_cleansed").alias("neighbourhood"))
    .agg(
        F.first("neighbourhood_group_cleansed", ignorenulls=True).alias("neighbourhood_group"),
        F.count("*").alias("listings"),
        F.sum(F.col("has_price").cast("int")).alias("active_listings"),
        F.round(F.percentile(ok_price, 0.5), 1).alias("median_price"),
        F.sum(F.col("price_ok").cast("int")).alias("priced_listings"),
        pct(F.col("room_type") == "Entire home/apt").alias("pct_entire_home"),
        F.round(F.avg("review_scores_rating"), 2).alias("avg_review_rating"),
        F.round(F.avg("latitude"), 5).alias("latitude"),
        F.round(F.avg("longitude"), 5).alias("longitude"),
    )
    .select(F.lit(CITY_KEY).alias("city_key"), F.lit(city["name"]).alias("city"),
            F.lit(city["snapshot_date"]).cast("date").alias("snapshot_date"), "*")
    .orderBy(F.desc("listings")))

display(neighbourhood_metrics)

# COMMAND ----------

# MAGIC %md ## Checks

# COMMAND ----------

all_row = city_metrics.filter(F.col("room_type") == "All").first()
if all_row is None or all_row["listings"] != total_listings:
    raise ValueError(f"{CITY_KEY}: 'All' row does not match silver row count {total_listings}")
if not all_row["active_listings"]:
    raise ValueError(f"{CITY_KEY}: no listing has a price — nothing for the dashboard or the model")
if neighbourhood_metrics.agg(F.sum("listings")).first()[0] != total_listings:
    raise ValueError(f"{CITY_KEY}: neighbourhood listings do not add up to {total_listings}")
null_neighbourhoods = neighbourhood_metrics.filter(F.col("neighbourhood").isNull()).count()

print(f"{CITY_KEY}: {total_listings} listings, {all_row['active_listings']} active, "
      f"median price CHF {all_row['median_price']}, "
      f"{neighbourhood_metrics.count()} neighbourhoods ({null_neighbourhoods} unnamed) — OK")

# COMMAND ----------

# MAGIC %md ## Write

# COMMAND ----------

for df, target, comment in [
    (city_metrics, CITY_METRICS,
     f"Metrics for {city['name']} per room_type. room_type = 'All' is the city total: filter on it, "
     f"never sum across room types. Prices in CHF exclude listings without price and implausible prices."),
    (neighbourhood_metrics, NEIGHBOURHOOD_METRICS,
     f"Metrics for {city['name']} per neighbourhood. Prices in CHF exclude listings without price and implausible prices."),
]:
    df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(target)
    spark.sql(f"COMMENT ON TABLE {target} IS '{comment}'")

# COMMAND ----------

# MAGIC %md ## Cross-city views
# MAGIC
# MAGIC Rebuilt over every city in `config/cities.json` whose tables exist, so they include each city as soon as
# MAGIC its pipeline has run. `listings_all` is the curated listing-level view for Genie and the dashboard: no leak
# MAGIC columns, readable names.

# COMMAND ----------

LISTING_COLUMNS = """city_key, city, snapshot_date, id, name, listing_url,
    neighbourhood_cleansed AS neighbourhood, neighbourhood_group_cleansed AS neighbourhood_group,
    room_type, property_type, accommodates, bedrooms, beds, bathrooms, bathroom_shared,
    price, has_price, outlier_class, minimum_nights, availability_365,
    number_of_reviews, review_scores_rating, host_id, host_is_superhost, latitude, longitude"""

VIEWS = {
    "city_metrics_all": (GOLD, "city_metrics", "*"),
    "neighbourhood_metrics_all": (GOLD, "neighbourhood_metrics", "*"),
    "listings_all": (SILVER, "listings", LISTING_COLUMNS),
}

for view, (schema, base, columns) in VIEWS.items():
    tables = [table_name(schema, base, k) for k in load_config()["cities"]]
    tables = [t for t in tables if spark.catalog.tableExists(t)]
    union = "\nUNION ALL\n".join(f"SELECT {columns} FROM {t}" for t in tables)
    spark.sql(f"CREATE OR REPLACE VIEW {CATALOG}.{GOLD}.{view} AS\n{union}")
    print(f"{view}: {len(tables)} cities")

spark.sql(f"COMMENT ON VIEW {CATALOG}.{GOLD}.city_metrics_all IS "
          "'Metrics per city and room_type. room_type = All is the city total: filter on it, never sum across room types.'")
spark.sql(f"COMMENT ON VIEW {CATALOG}.{GOLD}.listings_all IS "
          "'One row per listing, all cities. Price in CHF; has_price = false means inactive; "
          "exclude outlier_class = implausible from price statistics.'")

display(spark.table(f"{CATALOG}.{GOLD}.city_metrics_all").filter("room_type = 'All'"))
