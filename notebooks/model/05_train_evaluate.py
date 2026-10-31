# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Train and evaluate the price model
# MAGIC
# MAGIC Trains **one linear regression on log price** for the training cities (A and B) and compares it with the
# MAGIC required baseline on the held-out test set.
# MAGIC
# MAGIC - **Data**: `ga2_gold.features_<city>` for every city with role `train`, rows with `label_ok`.
# MAGIC - **Split**: fixed in `04_features` (20% of listings per city, by id hash). The test set is only used for the
# MAGIC   final numbers below.
# MAGIC - **Baseline**: median training price for the same `room_type` × `accommodates`; if that group is empty, the
# MAGIC   median for the `room_type`; else the overall median.
# MAGIC - **Model**: linear regression on `log(price)`, predictions converted back with `exp`. A coefficient *b* means
# MAGIC   "×e^b on the price", i.e. roughly +100·b %.
# MAGIC - **Metric**: MAE in CHF on the test set — city A, city B and both together.
# MAGIC
# MAGIC Outputs: the model in MLflow / Unity Catalog (`ga2_gold.price_model`, alias `champion`) and
# MAGIC `ga2_gold.model_evaluation`.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from bda.config import CATALOG, GOLD, cities_by_role, table_name  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRAIN_CITIES = cities_by_role("train")
MODEL_NAME = f"{CATALOG}.{GOLD}.price_model"
EVALUATION = f"{CATALOG}.{GOLD}.model_evaluation"

CATEGORICAL = ["room_type", "city"]
NUMERIC = ["accommodates", "bedrooms", "bathrooms", "log_reviews", "review_scores_rating"]
BOOLEAN = ["bathroom_shared", "long_stay", "has_reviews", "host_is_superhost"]
FEATURES = CATEGORICAL + NUMERIC + BOOLEAN

data = pd.concat([spark.table(table_name(GOLD, "features", k)).filter("label_ok").toPandas() for k in TRAIN_CITIES],
                 ignore_index=True)
data[BOOLEAN] = data[BOOLEAN].astype(float)
train, test = data[data["split"] == "train"], data[data["split"] == "test"]

print(f"Training cities: {TRAIN_CITIES}")
data.groupby(["city", "split"]).size().unstack()

# COMMAND ----------

# MAGIC %md ## Baseline: median price by room type × guests

# COMMAND ----------

def fit_baseline(df):
    return {
        "by_room_guests": df.groupby(["room_type", "accommodates"])["price"].median(),
        "by_room": df.groupby("room_type")["price"].median(),
        "overall": df["price"].median(),
    }


def predict_baseline(b, df):
    keys = list(zip(df["room_type"], df["accommodates"]))
    pred = pd.Series([b["by_room_guests"].get(k, np.nan) for k in keys], index=df.index)
    pred = pred.fillna(df["room_type"].map(b["by_room"]))
    return pred.fillna(b["overall"])


baseline = fit_baseline(train)

# COMMAND ----------

# MAGIC %md ## Model: linear regression on log price

# COMMAND ----------

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

model = Pipeline([
    ("prepare", ColumnTransformer([
        # handle_unknown="ignore": an unseen room type or city gets no adjustment instead of failing.
        ("categories", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        # review_scores_rating is missing when there are no reviews; has_reviews carries that signal.
        ("numbers", SimpleImputer(strategy="median"), NUMERIC),
        # flags are null when their source is (e.g. no bathrooms_text); fill with the most common value.
        ("flags", SimpleImputer(strategy="most_frequent"), BOOLEAN),
    ], verbose_feature_names_out=False)),
    ("regression", LinearRegression()),
])
model.fit(train[FEATURES], train["log_price"])

coefficients = pd.DataFrame({
    "feature": model.named_steps["prepare"].get_feature_names_out(),
    "coefficient": model.named_steps["regression"].coef_,
})
coefficients["price_effect_pct"] = (100 * (np.exp(coefficients["coefficient"]) - 1)).round(1)
coefficients.sort_values("price_effect_pct", key=abs, ascending=False)

# COMMAND ----------

# MAGIC %md ## Test-set evaluation

# COMMAND ----------

test = test.assign(
    baseline_price=predict_baseline(baseline, test),
    model_price=np.exp(model.predict(test[FEATURES])),
)


def mae_table(df):
    rows = []
    for name, part in [*df.groupby("city"), ("A + B", df)]:
        rows.append({
            "city": name,
            "test_listings": len(part),
            "baseline_mae": (part["baseline_price"] - part["price"]).abs().mean(),
            "model_mae": (part["model_price"] - part["price"]).abs().mean(),
            "median_price": part["price"].median(),
        })
    out = pd.DataFrame(rows)
    out["improvement_pct"] = 100 * (1 - out["model_mae"] / out["baseline_mae"])
    return out.round(1)


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

with mlflow.start_run(run_name="linear_regression_log_price") as run:
    mlflow.log_params({"model": "LinearRegression on log(price)", "features": ", ".join(FEATURES),
                       "train_cities": ", ".join(TRAIN_CITIES), "train_rows": len(train)})
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

(spark.createDataFrame(evaluation.assign(model_version=str(version), evaluated_on="test"))
 .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(EVALUATION))
spark.sql(f"COMMENT ON TABLE {EVALUATION} IS 'Test-set MAE (CHF) of the price model vs the baseline "
          f"(median price per room_type x accommodates), per training city and combined.'")
display(spark.table(EVALUATION))
