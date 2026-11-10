# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Transfer rehearsal (A and B only)
# MAGIC
# MAGIC Before city C is opened, we pretend one training city is the new client: train on Geneva and price Zurich,
# MAGIC then the other way round. This tells us what to expect on city C and why, **before** we see it.
# MAGIC
# MAGIC 1. **Leave one city out**: how much worse is a model trained elsewhere than one trained on the city itself?
# MAGIC 2. **Price-level calibration**: if the client gives us some known prices, does shifting the model to the
# MAGIC    city's price level help, and how many prices are enough?
# MAGIC 3. **Effect stability**: which effects are the same in both cities (Swiss-wide rules) and which are local?
# MAGIC 4. **Expectation for city C**, saved to `ga2_gold.city_c_expectation` and compared with the real result in
# MAGIC    `city_c/01_first_run`.
# MAGIC
# MAGIC Uses only the train rows of A and B for fitting and their test rows for scoring. City C is not read.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import CATALOG, GOLD, cities_by_role, table_name  # noqa: E402
from bda.modeling import prepare  # noqa: E402
from bda.transfer import calibration_curve, effect_stability, leave_one_city_out  # noqa: E402

import pandas as pd  # noqa: E402

EXPECTATION = f"{CATALOG}.{GOLD}.city_c_expectation"

data = pd.concat([spark.table(table_name(GOLD, "features", k)).filter("label_ok").toPandas()
                  for k in cities_by_role("train")], ignore_index=True)
data = prepare(data)
data.groupby(["city", "split"]).size().unstack()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leave one city out
# MAGIC `raw` is the model as it would run on a new city. `own_city` is a model trained on the target city: the gap
# MAGIC between them, `transfer_penalty_pct`, is the cost of never having seen the city.

# COMMAND ----------

loco = leave_one_city_out(data)
loco

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. How many known prices does a new client need to give us?
# MAGIC The calibrated model is the transferred model times one number per city, its price level, estimated from
# MAGIC N known prices. Repeated 30 times per N; `calibrated_mae_worst` is the unluckiest draw.

# COMMAND ----------

curve = calibration_curve(data)
curve

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Which effects are Swiss-wide?
# MAGIC Per-city linear model on log price, 300 bootstrap resamples. Effects in % on the price, all else equal, with
# MAGIC a 90% interval. `shared` means the cities agree within noise.

# COMMAND ----------

stability = effect_stability(data)
stability

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Expectation for city C
# MAGIC The champion model's A + B test MAE, scaled by the transfer penalties seen above: the lower penalty gives the
# MAGIC optimistic end, the higher one the pessimistic end. City C may still fall outside this range: it is a canton
# MAGIC with several towns, not one city, and the rehearsal cannot show that.

# COMMAND ----------

from pyspark.sql import functions as F  # noqa: E402

in_city_mae = spark.table(f"{CATALOG}.{GOLD}.model_evaluation").filter("city = 'A + B'").first()["model_mae"]
expectation = pd.DataFrame([{
    "in_city_mae": in_city_mae,
    "expected_mae_low": in_city_mae * (1 + loco["transfer_penalty_pct"].min() / 100),
    "expected_mae_high": in_city_mae * (1 + loco["transfer_penalty_pct"].max() / 100),
    "shared_effects": ", ".join(stability.loc[stability["verdict"] == "shared", "feature"]),
    "city_specific_effects": ", ".join(stability.loc[stability["verdict"] == "city-specific", "feature"]),
}]).round(1)

(spark.createDataFrame(expectation).withColumn("recorded_at", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(EXPECTATION))
spark.sql(f"COMMENT ON TABLE {EXPECTATION} IS 'Expected MAE (CHF) of the champion model on the new-client city, "
          f"estimated from A and B only, before city C was loaded.'")
display(spark.table(EXPECTATION))
