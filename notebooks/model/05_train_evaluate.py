# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Train and evaluate the price model
# MAGIC
# MAGIC Chooses and trains **one price model on log price** for the training cities (A and B) and compares it with the
# MAGIC required baseline on the held-out test set.
# MAGIC
# MAGIC - **Data**: `ga2_gold.features_<city>` for every city with role `train`, rows with `label_ok`.
# MAGIC - **Split**: fixed in `04_features` (20% of listings per city, by id hash). Model choice uses 5-fold
# MAGIC   cross-validation on the train rows only; the test set is used once, for the final numbers.
# MAGIC - **Baseline**: median training price for the same `room_type` × `accommodates`; if that group is empty, the
# MAGIC   median for the `room_type`; else the overall median.
# MAGIC - **Candidates** (`src/bda/modeling.py`):
# MAGIC   - `linear`: linear regression, the first week 1 model. A coefficient *b* means ×e^b on the price.
# MAGIC   - `boosting`: shallow gradient boosting (depth 3) with absolute-error loss and **monotonic constraints**:
# MAGIC     more guests, bedrooms, bathrooms or amenities can only raise the price; distance from the centre, a
# MAGIC     shared bathroom or a long-stay minimum can only lower it.
# MAGIC - **Metric**: MAE in CHF on the test set: city A, city B and both together.
# MAGIC
# MAGIC Outputs: the model in MLflow / Unity Catalog (`ga2_gold.price_model`, alias `champion`) and
# MAGIC `ga2_gold.model_evaluation`.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import CATALOG, GOLD, cities_by_role, table_name  # noqa: E402
from bda.modeling import (  # noqa: E402
    CANDIDATES, FEATURES, cross_val_baseline_mae, cross_val_mae, fit_baseline, mae_table, predict_baseline, prepare,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRAIN_CITIES = cities_by_role("train")
MODEL_NAME = f"{CATALOG}.{GOLD}.price_model"
EVALUATION = f"{CATALOG}.{GOLD}.model_evaluation"

data = pd.concat([spark.table(table_name(GOLD, "features", k)).filter("label_ok").toPandas() for k in TRAIN_CITIES],
                 ignore_index=True)
data = prepare(data)
train, test = data[data["split"] == "train"], data[data["split"] == "test"]

print(f"Training cities: {TRAIN_CITIES}")
data.groupby(["city", "split"]).size().unstack()

# COMMAND ----------

# MAGIC %md ## Model choice: cross-validation on the train rows

# COMMAND ----------

selection = pd.DataFrame(
    [{"candidate": "baseline", "cv_mae": cross_val_baseline_mae(train)}]
    + [{"candidate": name, "cv_mae": cross_val_mae(make, train, FEATURES)} for name, make in CANDIDATES.items()]
).round(1)
CHOSEN = selection[selection["candidate"] != "baseline"].sort_values("cv_mae").iloc[0]["candidate"]
print(f"Chosen: {CHOSEN}")
selection

# COMMAND ----------

# MAGIC %md ## Fit the chosen model and the baseline on all train rows

# COMMAND ----------

model = CANDIDATES[CHOSEN]().fit(train[FEATURES], train["log_price"])
baseline = fit_baseline(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the model learned
# MAGIC Permutation importance on the train rows: how much the error on log price grows when one feature is shuffled.

# COMMAND ----------

from sklearn.inspection import permutation_importance  # noqa: E402

importance = permutation_importance(model, train[FEATURES], train["log_price"], n_repeats=5, random_state=0,
                                    scoring="neg_mean_absolute_error")
pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean}) \
    .sort_values("importance", ascending=False).round(4)

# COMMAND ----------

# MAGIC %md ## Test-set evaluation

# COMMAND ----------

test = test.assign(
    baseline_price=predict_baseline(baseline, test),
    model_price=np.exp(model.predict(test[FEATURES])),
)
evaluation = mae_table(test)
evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model
# MAGIC
# MAGIC Registered in Unity Catalog with alias `champion`; `06_predict` loads `models:/…@champion`. For city C the job
# MAGIC only runs `06_predict`, so city C is priced with exactly this model.

# COMMAND ----------

import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name=f"{CHOSEN}_log_price") as run:
    mlflow.log_params({"model": CHOSEN, "features": ", ".join(FEATURES),
                       "train_cities": ", ".join(TRAIN_CITIES), "train_rows": len(train)})
    mlflow.log_metrics({f"cv_mae_{r['candidate']}": r["cv_mae"] for _, r in selection.iterrows()})
    for _, r in evaluation.iterrows():
        tag = r["city"].replace(" + ", "_").replace(" ", "_")
        mlflow.log_metrics({f"test_mae_{tag}": r["model_mae"], f"baseline_mae_{tag}": r["baseline_mae"]})
    info = mlflow.sklearn.log_model(
        model, name="model", registered_model_name=MODEL_NAME,
        input_example=train[FEATURES].head(5),
        signature=infer_signature(train[FEATURES], model.predict(train[FEATURES])),
    )

version = info.registered_model_version
mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
print(f"{MODEL_NAME} version {version} -> alias 'champion' (run {run.info.run_id})")

# COMMAND ----------

# MAGIC %md ## Save the evaluation for the dashboard

# COMMAND ----------

(spark.createDataFrame(evaluation.assign(model=CHOSEN, model_version=str(version), evaluated_on="test"))
 .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(EVALUATION))
spark.sql(f"COMMENT ON TABLE {EVALUATION} IS 'Test-set MAE (CHF) of the price model vs the baseline "
          f"(median price per room_type x accommodates), per training city and combined.'")
display(spark.table(EVALUATION))
