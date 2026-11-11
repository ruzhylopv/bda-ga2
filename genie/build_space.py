"""Generates genie/space.json, the Genie Agent definition (deliverable 3.6).

Usage: python3 genie/build_space.py ga2_gold genie/space.json
Create:  databricks genie create-space <warehouse_id> "$(cat genie/space.json)" --title "GA2 Airbnb Switzerland"
Update:  databricks genie update-space <space_id> --serialized-space "$(cat genie/space.json)"
"""
import hashlib
import json
import sys

SCHEMA = sys.argv[1]
OUT = sys.argv[2]
G = f"workspace.{SCHEMA}"


def sid(text: str) -> str:
    """Stable 32-hex id, so regenerating the file does not change ids."""
    return hashlib.md5(text.encode()).hexdigest()


def lines(text: str) -> list[str]:
    parts = text.strip().split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


TABLES = {
    "city_metrics_all": "One row per city and room_type. room_type = 'All' is the city total.",
    "neighbourhood_metrics_all": "One row per city and neighbourhood (in Vaud: municipality).",
    "listings_all": "One row per listing, all cities. Price in CHF per night.",
    "predictions_all": "Model and baseline price per listing. error = predicted_price - price.",
    "model_evaluation": "Test-set MAE of the model and the baseline for Geneva, Zurich and both together ('A + B').",
    "city_c_report": "City C (Vaud): baseline and model MAE, first run and after the model changes.",
    "city_c_changes": "City C (Vaud): model MAE after each change step.",
}

INSTRUCTIONS = """
Data: Inside Airbnb listings for three Swiss areas. City A = Geneva, city B = Zurich, city C = Vaud.
Vaud is a canton, not a city: its neighbourhoods are municipalities (Lausanne, Montreux, Alpine resorts).
All prices are in CHF per night, as advertised on the snapshot date. Always name the city and its letter in answers.

City-level numbers (listings, hosts, median price, availability, review scores, room-type mix):
- use city_metrics_all. room_type = 'All' is the city total. Never sum or average the room-type rows: medians and
  distinct host counts do not add up.
- For questions about room types (most common, cheapest, mix), use the room-type rows: room_type <> 'All'.
  pct_of_listings is the share of the city's listings.
- Room type values are exactly 'Entire home/apt', 'Private room', 'Hotel room', 'Shared room'.
  "Entire homes", "whole apartments" or "flats" mean 'Entire home/apt'.

Price statistics computed from listings_all:
- only listings with has_price = true AND outlier_class <> 'implausible'.
- "median" means percentile(price, 0.5). Report medians, not averages, unless the user asks for the average.
- Counts of listings include inactive listings (has_price = false) unless the user asks for active listings.

Neighbourhoods: use the column neighbourhood. For "most listings" use neighbourhood_metrics_all.listings.
When ranking neighbourhoods by price, only include those with priced_listings >= 10: a median of 1-5 listings is
noise. Say that small neighbourhoods were left out.

Model accuracy (predictions_all):
- only rows with label_ok = true AND split = 'test' (listings the model never saw). MAE = avg(abs_error).
- error = predicted_price - price. Under-priced: error < 0 (the model suggests less than the host asks).
  Over-priced: error > 0.
- "Where does the model under-price most" means: by neighbourhood, the most negative avg(error), only neighbourhoods
  with at least 15 test listings. Show the city, the number of test listings and avg(error) in CHF.
- The Vaud rows in predictions_all come from the first run (the unchanged week 1 model).

Model against baseline:
- Geneva, Zurich and A + B: model_evaluation (city = 'A + B' is the combined row; do not average the cities).
- Vaud: city_c_report rows = 'test'; run = 'first_run' or 'after_change'. For any question about the model's error
  or accuracy on Vaud / city C, return both runs and the baseline: baseline 84.0, first run (unchanged model) and
  after the changes (current result).
- Improvement in % = 100 * (baseline_mae - model_mae) / baseline_mae.
- The baseline is the median training price for the same room type and number of guests.

Round CHF to 1 decimal and percentages to 1 decimal.
"""

EXAMPLES = [
    ("What is the median price of entire homes in City A?", f"""
SELECT city, room_type, median_price AS median_price_chf, active_listings
FROM {G}.city_metrics_all
WHERE city = 'Geneva' AND room_type = 'Entire home/apt'"""),
    ("Which neighbourhoods in City B have the most listings?", f"""
SELECT neighbourhood, listings, active_listings, median_price AS median_price_chf
FROM {G}.neighbourhood_metrics_all
WHERE city = 'Zurich'
ORDER BY listings DESC
LIMIT 10"""),
    ("Where does the model under-price listings the most?", f"""
SELECT city, neighbourhood, count(*) AS test_listings, round(avg(error), 1) AS avg_error_chf,
       round(percentile(pct_error, 0.5), 1) AS median_pct_error
FROM {G}.predictions_all
WHERE label_ok AND split = 'test'
GROUP BY city, neighbourhood
HAVING count(*) >= 15
ORDER BY avg_error_chf ASC
LIMIT 10"""),
    ("How much better than the baseline is the model in each city?", f"""
SELECT city, baseline_mae, model_mae, round(100 * (baseline_mae - model_mae) / baseline_mae, 1) AS improvement_pct,
       'test set' AS evaluated_on
FROM {G}.model_evaluation
UNION ALL
SELECT concat('Vaud (', CASE run WHEN 'first_run' THEN 'first run' ELSE 'after changes' END, ')'),
       baseline_mae_c, model_mae_c, round(100 * (baseline_mae_c - model_mae_c) / baseline_mae_c, 1), 'test set'
FROM {G}.city_c_report
WHERE rows = 'test'"""),
    ("Compare the median price per guest in each city", f"""
SELECT city, round(percentile(price / accommodates, 0.5), 1) AS median_price_per_guest_chf,
       round(avg(accommodates), 2) AS avg_guests, count(*) AS priced_listings
FROM {G}.listings_all
WHERE has_price AND outlier_class <> 'implausible'
GROUP BY city"""),
]

SAMPLE_QUESTIONS = [
    "What is the median price of entire homes in City A?",
    "Which neighbourhoods in City B have the most listings?",
    "Where does the model under-price listings the most?",
    "How much better than the baseline is the model in each city?",
    "Compare listings, hosts and median price across the three cities",
]

space = {
    "version": 1,
    "config": {"sample_questions": sorted(({"id": sid("q" + q), "question": [q]} for q in SAMPLE_QUESTIONS),
                                          key=lambda x: x["id"])},
    "data_sources": {"tables": [{"identifier": f"{G}.{t}", "description": [d]} for t, d in sorted(TABLES.items())]},
    "instructions": {
        "text_instructions": [{"id": sid("instructions"), "content": lines(INSTRUCTIONS)}],
        "example_question_sqls": sorted(({"id": sid("e" + q), "question": [q], "sql": lines(sql)}
                                         for q, sql in EXAMPLES), key=lambda x: x["id"]),
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(space, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"wrote {OUT}: {len(TABLES)} tables, {len(EXAMPLES)} example SQLs, {len(SAMPLE_QUESTIONS)} sample questions")
