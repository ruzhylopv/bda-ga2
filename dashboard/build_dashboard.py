"""Generates dashboard/ga2_dashboard.lvdash.json (deliverable 3.5). Usage: python3 dashboard/build_dashboard.py ga2_gold dashboard/ga2_dashboard.lvdash.json"""
import json
import sys

SCHEMA = sys.argv[1]
OUT = sys.argv[2]
G = f"workspace.{SCHEMA}"

PRICE_OK = "has_price AND outlier_class <> 'implausible'"

DATASETS = {
    # ---- page 1
    "city_totals": f"""
SELECT city, listings, active_listings, pct_active, hosts, round(listings / hosts, 2) AS listings_per_host,
       pct_multi_listing_hosts, median_price, p25_price, p75_price, median_availability_365, pct_closed_all_year,
       avg_review_rating, pct_with_reviews, pct_superhost
FROM {G}.city_metrics_all
WHERE room_type = 'All'
ORDER BY CASE city WHEN 'Geneva' THEN 1 WHEN 'Zurich' THEN 2 ELSE 3 END""",
    "room_mix": f"""
SELECT city, room_type, listings, pct_of_listings, median_price
FROM {G}.city_metrics_all
WHERE room_type <> 'All'""",
    "neighbourhoods": f"""
SELECT city, neighbourhood, listings, priced_listings, median_price, pct_entire_home,
       row_number() OVER (PARTITION BY city ORDER BY listings DESC) AS size_rank
FROM {G}.neighbourhood_metrics_all
QUALIFY size_rank <= 15""",
    "price_bands": f"""
WITH b AS (
  SELECT city, CASE WHEN price < 50 THEN '0-49' WHEN price < 100 THEN '50-99' WHEN price < 150 THEN '100-149'
                    WHEN price < 200 THEN '150-199' WHEN price < 300 THEN '200-299' WHEN price < 500 THEN '300-499'
                    ELSE '500+' END AS price_band,
         CASE WHEN price < 50 THEN 1 WHEN price < 100 THEN 2 WHEN price < 150 THEN 3 WHEN price < 200 THEN 4
              WHEN price < 300 THEN 5 WHEN price < 500 THEN 6 ELSE 7 END AS band_order
  FROM {G}.listings_all WHERE {PRICE_OK}
)
SELECT city, concat(band_order, ') ', price_band) AS price_band,
       round(100 * count(*) / sum(count(*)) OVER (PARTITION BY city), 1) AS pct_of_listings
FROM b GROUP BY city, band_order, price_band""",
    # ---- page 2
    "ab_compare": f"""
WITH m AS (
  SELECT * FROM {G}.city_metrics_all WHERE room_type = 'All' AND city IN ('Geneva', 'Zurich')
), long AS (
  SELECT city, stack(10,
    1, 'Listings', CAST(listings AS DOUBLE),
    2, 'Active listings (%)', pct_active,
    3, 'Hosts', CAST(hosts AS DOUBLE),
    4, 'Listings per host', listings / hosts,
    5, 'Median price (CHF)', median_price,
    6, 'Median availability (days / year)', CAST(median_availability_365 AS DOUBLE),
    7, 'Closed all year (%)', pct_closed_all_year,
    8, 'Average review score', avg_review_rating,
    9, 'With reviews (%)', pct_with_reviews,
    10, 'Superhosts (%)', pct_superhost
  ) AS (ord, metric, value)
  FROM m
)
SELECT ord, metric,
       round(max(CASE WHEN city = 'Geneva' THEN value END), 2) AS geneva,
       round(max(CASE WHEN city = 'Zurich' THEN value END), 2) AS zurich,
       round(100 * (max(CASE WHEN city = 'Zurich' THEN value END) / max(CASE WHEN city = 'Geneva' THEN value END) - 1), 1)
         AS zurich_vs_geneva_pct
FROM long GROUP BY ord, metric ORDER BY ord""",
    "ab_room_price": f"""
SELECT city, room_type, median_price, listings
FROM {G}.city_metrics_all
WHERE city IN ('Geneva', 'Zurich') AND room_type IN ('Entire home/apt', 'Private room')""",
    "ab_like_for_like": f"""
SELECT city, room_type, CAST(accommodates AS STRING) AS guests, count(*) AS listings,
       round(percentile(price, 0.5), 1) AS median_price
FROM {G}.listings_all
WHERE {PRICE_OK} AND city IN ('Geneva', 'Zurich') AND room_type = 'Entire home/apt' AND accommodates BETWEEN 1 AND 6
GROUP BY ALL""",
    "ab_size": f"""
SELECT city, CASE WHEN accommodates >= 6 THEN '6+' ELSE CAST(accommodates AS STRING) END AS guests,
       round(100 * count(*) / sum(count(*)) OVER (PARTITION BY city), 1) AS pct_of_listings
FROM {G}.listings_all
WHERE {PRICE_OK} AND city IN ('Geneva', 'Zurich')
GROUP BY ALL""",
    # ---- page 3
    "mae_by_city": f"""
WITH ev AS (
  SELECT CASE city WHEN 'Geneva' THEN '1 Geneva (A)' WHEN 'Zurich' THEN '2 Zurich (B)' ELSE '3 A + B' END AS city_label,
         baseline_mae, model_mae
  FROM {G}.model_evaluation
  UNION ALL
  SELECT CASE run WHEN 'first_run' THEN '4 Vaud (C), first run' ELSE '5 Vaud (C), after changes' END,
         baseline_mae_c, model_mae_c
  FROM {G}.city_c_report WHERE rows = 'test'
)
SELECT city_label, 'Baseline' AS method, baseline_mae AS mae FROM ev
UNION ALL
SELECT city_label, 'Model', model_mae FROM ev""",
    "test_predictions": f"""
SELECT city, room_type, neighbourhood, name, listing_url, price, predicted_price, baseline_price,
       error, abs_error, baseline_abs_error, pct_error
FROM {G}.predictions_all
WHERE label_ok AND split = 'test'""",
    "mae_by_room_type": f"""
SELECT city, room_type, count(*) AS test_listings, round(avg(abs_error), 1) AS model_mae,
       round(avg(baseline_abs_error), 1) AS baseline_mae
FROM {G}.predictions_all
WHERE label_ok AND split = 'test'
GROUP BY ALL
HAVING count(*) >= 20""",
    "mae_by_neighbourhood": f"""
SELECT concat(neighbourhood, ' (', city, ')') AS area, city, count(*) AS test_listings,
       round(avg(abs_error), 1) AS model_mae, round(avg(baseline_abs_error), 1) AS baseline_mae,
       round(percentile(pct_error, 0.5), 1) AS median_pct_error
FROM {G}.predictions_all
WHERE label_ok AND split = 'test'
GROUP BY ALL
HAVING count(*) >= 15""",
    "extremes": f"""
SELECT city, CASE WHEN error > 0 THEN 'over-priced' ELSE 'under-priced' END AS direction,
       name, neighbourhood, room_type, accommodates, price, predicted_price, error, pct_error, listing_url,
       row_number() OVER (PARTITION BY city, error > 0 ORDER BY abs(error) DESC) AS rank
FROM {G}.predictions_all
WHERE label_ok AND split = 'test' AND error <> 0
QUALIFY rank <= 10
ORDER BY city, direction, rank""",
    "vaud_steps": f"""
WITH c AS (
  SELECT CASE step WHEN 0 THEN '1 first run' WHEN 1 THEN '2 + city level' ELSE '3 + municipality level' END AS step,
         model_mae_c, baseline_mae_c
  FROM {G}.city_c_changes
)
SELECT step, 'Model' AS method, model_mae_c AS mae FROM c
UNION ALL
SELECT step, 'Baseline', baseline_mae_c FROM c""",
    "vaud_areas": f"""
WITH top AS (
  SELECT neighbourhood FROM {G}.predictions_after_change_vaud GROUP BY neighbourhood ORDER BY count(*) DESC LIMIT 8
), p AS (
  SELECT neighbourhood, price, price_first_run, price_city_level, price_area_level
  FROM {G}.predictions_after_change_vaud WHERE neighbourhood IN (SELECT neighbourhood FROM top)
)
SELECT neighbourhood, '1 first run' AS step, round(percentile(100 * (price_first_run - price) / price, 0.5), 1) AS median_pct_error, count(*) AS test_listings FROM p GROUP BY neighbourhood
UNION ALL
SELECT neighbourhood, '2 + city level', round(percentile(100 * (price_city_level - price) / price, 0.5), 1), count(*) FROM p GROUP BY neighbourhood
UNION ALL
SELECT neighbourhood, '3 + municipality level', round(percentile(100 * (price_area_level - price) / price, 0.5), 1), count(*) FROM p GROUP BY neighbourhood""",
}


# ---------------------------------------------------------------- widget helpers

# Fixed colours so an entity keeps its colour on every page (dark-surface steps of the reference palette).
COLOURS = {
    "city": {"Geneva": "#3987e5", "Zurich": "#d95926", "Vaud": "#199e70"},
    "method": {"Baseline": "#8a8f98", "Model": "#d55181"},
    "step": {"1 first run": "#8a8f98", "2 + city level": "#c98500", "3 + municipality level": "#199e70"},
}

def field(name, expr=None):
    return {"name": name, "expression": expr or f"`{name}`"}


def query(dataset, fields, disaggregated=True, name="main_query"):
    return {"name": name, "query": {"datasetName": dataset, "fields": fields, "disaggregated": disaggregated}}


def pos(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


_n = 0


def wname(prefix):
    global _n
    _n += 1
    return f"{prefix}_{_n}"


def text(lines, x, y, w, h):
    lines = [line + "\n" for line in "\n\n".join(lines).split("\n")]
    return {"widget": {"name": wname("text"), "multilineTextboxSpec": {"lines": lines}}, "position": pos(x, y, w, h)}


def frame(title, description=None):
    f = {"showTitle": True, "title": title}
    if description:
        f["showDescription"] = True
        f["description"] = description
    return f


def enc(fieldname, kind, display, **extra):
    e = {"fieldName": fieldname, "scale": {"type": kind}, "displayName": display}
    e["scale"].update(extra.pop("scale", {}))
    e.update(extra)
    return e


def colour_enc(color):
    mapping = COLOURS.get(color[0])
    extra = {"scale": {"mappings": [{"value": v, "color": c} for v, c in mapping.items()]}} if mapping else {}
    return enc(color[0], "categorical", color[1], **extra)


def bar(dataset, x, y, title, color=None, layout="group", description=None, horizontal=False, labels=True,
        x_kind=None, y_kind=None, sort=None):
    """x, y, color: (column, display name)."""
    cols = [x, y] + ([color] if color else [])
    fields = [field(c[0]) for c in cols]
    x_kind = x_kind or ("quantitative" if horizontal else "categorical")
    y_kind = y_kind or ("categorical" if horizontal else "quantitative")
    encodings = {
        "x": enc(x[0], x_kind, x[1], scale={"sort": sort} if sort and x_kind == "categorical" else {}),
        "y": enc(y[0], y_kind, y[1], scale={"sort": sort} if sort and y_kind == "categorical" else {}),
    }
    if color:
        encodings["color"] = colour_enc(color)
    if labels:
        encodings["label"] = {"show": True}
    spec = {"version": 3, "widgetType": "bar", "encodings": encodings, "frame": frame(title, description)}
    if color:
        spec["mark"] = {"layout": layout}
    return {"name": wname("bar"), "queries": [query(dataset, fields)], "spec": spec}


def scatter(dataset, x, y, color, title, description=None):
    fields = [field(c[0]) for c in (x, y, color)]
    spec = {"version": 3, "widgetType": "scatter",
            "encodings": {"x": enc(x[0], "quantitative", x[1]), "y": enc(y[0], "quantitative", y[1]),
                          "color": colour_enc(color)},
            "frame": frame(title, description)}
    return {"name": wname("scatter"), "queries": [query(dataset, fields)], "spec": spec}


def table(dataset, columns, title, description=None):
    """columns: (column, display name, type) with type string | integer | float | link."""
    fields = [field(c[0]) for c in columns]
    cols = []
    for i, (name, display, kind) in enumerate(columns):
        col = {"fieldName": name, "displayName": display, "title": display, "visible": True, "order": i,
               "type": "string" if kind == "link" else kind,
               "displayAs": "link" if kind == "link" else ("number" if kind in ("integer", "float") else "string"),
               "alignContent": "right" if kind in ("integer", "float") else "left"}
        if kind == "link":
            col.update({"linkUrlTemplate": "{{ @ }}", "linkTextTemplate": "open", "linkOpenInNewTab": True})
        if kind == "float":
            col["numberFormat"] = "0.0"
        cols.append(col)
    spec = {"version": 2, "widgetType": "table", "encodings": {"columns": cols}, "frame": frame(title, description)}
    return {"name": wname("table"), "queries": [query(dataset, fields)], "spec": spec}


def city_filter(datasets, title="City", multi=False, default=None):
    queries = [query(ds, [field("city"), field("city_associativity", "COUNT_IF(`associative_filter_predicate_group`)")],
                     disaggregated=False, name=f"filter_{ds}") for ds in datasets]
    spec = {"version": 2, "widgetType": "filter-multi-select" if multi else "filter-single-select",
            "encodings": {"fields": [{"fieldName": "city", "displayName": "City", "queryName": q["name"]}
                                     for q in queries]},
            "frame": frame(title)}
    if default:
        spec["selection"] = {"defaultSelection": {"values": {"dataType": "STRING",
                                                             "values": [{"value": v} for v in default]}}}
    return {"name": wname("filter"), "queries": queries, "spec": spec}


def place(widget, x, y, w, h):
    return {"widget": widget, "position": pos(x, y, w, h)}


# ---------------------------------------------------------------- page 1: city overview

p1 = [
    text(["## Entire homes make up 70-76% of listings in every city, and about a third of all listings are inactive",
          "Geneva (A), Zurich (B) and Vaud (C), Inside Airbnb snapshots of Sep-Oct 2026. Prices in CHF per night; "
          "price figures use only listings with a usable price (inactive listings and 14 implausible prices excluded)."],
         0, 0, 6, 2),
    place(table("city_totals", [
        ("city", "City", "string"), ("listings", "Listings", "integer"), ("pct_active", "Active %", "float"),
        ("hosts", "Hosts", "integer"), ("listings_per_host", "Listings / host", "float"),
        ("median_price", "Median price", "float"), ("p25_price", "P25 price", "float"),
        ("p75_price", "P75 price", "float"), ("median_availability_365", "Median availability (days)", "float"),
        ("pct_closed_all_year", "Closed all year %", "float"), ("avg_review_rating", "Avg review score", "float"),
        ("pct_superhost", "Superhost %", "float"),
    ], "Vaud has the most listings; Zurich is the most expensive and the least available",
        "One row per city, from city_metrics_all where room_type = 'All' (never a sum of room types)."), 0, 2, 6, 4),
    place(bar("room_mix", ("city", "City"), ("pct_of_listings", "% of listings"),
              "Entire homes dominate everywhere; private rooms are 24-30%", color=("room_type", "Room type"),
              layout="stack"), 0, 6, 3, 6),
    place(bar("room_mix", ("city", "City"), ("median_price", "Median price (CHF)"),
              "An entire home costs 1.7-2.2 times a private room", color=("room_type", "Room type")),
          3, 6, 3, 6),
    place(bar("price_bands", ("price_band", "Price per night (CHF)"), ("pct_of_listings", "% of priced listings"),
              "Vaud has the longest expensive tail: 20% of its listings ask CHF 300 or more (Geneva 12%, Zurich 18%)",
              color=("city", "City"), description="Share of each city's listings with a usable price, per price band."),
          0, 12, 6, 6),
    place(city_filter(["neighbourhoods"], "City for the neighbourhood chart", default=["Geneva"]), 0, 18, 2, 2),
    text(["Vaud's neighbourhoods are municipalities: Lausanne is cheaper than the resorts (Ollon, Gryon, Montreux)."],
         2, 18, 4, 2),
    place(bar("neighbourhoods", ("median_price", "Median price (CHF)"), ("neighbourhood", "Neighbourhood"),
              "Neighbourhood prices: the 15 largest neighbourhoods of the selected city",
              horizontal=True, sort={"by": "x-reversed"}), 0, 20, 6, 8),
]

# ---------------------------------------------------------------- page 2: Geneva vs Zurich

p2 = [
    text(["## Zurich is 20% pricier per night than Geneva, but only 8-11% for a same-size flat: Zurich listings are bigger",
          "City A = Geneva, city B = Zurich. Median prices in CHF per night, listings with a usable price."],
         0, 0, 6, 2),
    place(table("ab_compare", [
        ("metric", "Metric", "string"), ("geneva", "Geneva (A)", "float"), ("zurich", "Zurich (B)", "float"),
        ("zurich_vs_geneva_pct", "Zurich vs Geneva %", "float"),
    ], "Zurich: higher prices, 28% fewer open days, more listings per host and superhosts",
        "City totals (room_type = 'All'). Last column: how much higher (+) or lower (-) Zurich is."), 0, 2, 3, 7),
    place(bar("ab_room_price", ("room_type", "Room type"), ("median_price", "Median price (CHF)"),
              "The gap is in entire homes (+15%); private rooms differ by 6%", color=("city", "City")), 3, 2, 3, 7),
    place(bar("ab_like_for_like", ("guests", "Guests (accommodates)"), ("median_price", "Median price (CHF)"),
              "Same number of guests, much smaller gap: entire homes for 2 guests +8%, for 4 guests +11%",
              color=("city", "City"), description="Entire homes, 1-6 guests."), 0, 9, 3, 7),
    place(bar("ab_size", ("guests", "Guests (accommodates)"), ("pct_of_listings", "% of listings"),
              "Zurich has more listings for 4+ guests, which lifts its city-wide median",
              color=("city", "City")), 3, 9, 3, 7),
]

# ---------------------------------------------------------------- page 3: model

p3 = [
    text(["## The model beats the baseline by 20% on Geneva and Zurich and by 19% on Vaud after the changes; "
          "its largest errors are in the Vaud Alps",
          "Every number is on the test set (20% of listings per city, never used for training or model choice). "
          "Baseline = median training price for the same room type and number of guests. "
          "error = predicted - actual: positive = the model over-prices. Vaud rows in the charts below are the "
          "**first run** (unchanged week 1 model); the changes are shown at the bottom."],
         0, 0, 6, 3),
    place(bar("mae_by_city", ("city_label", "City"), ("mae", "MAE on test set (CHF)"),
              "Model vs baseline: lower is better", color=("method", "Method"),
              description="Vaud is harder for any method: its baseline error is 19% higher than on A + B."),
          0, 3, 6, 6),
    place(city_filter(["test_predictions", "mae_by_room_type", "extremes"], "City", multi=True), 0, 9, 2, 2),
    place(scatter("test_predictions", ("price", "Actual price (CHF)"), ("predicted_price", "Predicted price (CHF)"),
                  ("city", "City"), "Predicted vs actual price: points above the diagonal are over-priced",
                  "Test listings. The model is pulled towards the middle: cheap listings are over-priced, "
                  "expensive ones under-priced."), 0, 11, 4, 8),
    place(bar("mae_by_room_type", ("room_type", "Room type"), ("model_mae", "Model MAE (CHF)"),
              "Entire homes carry the largest errors in CHF", color=("city", "City"),
              description="Room types with at least 20 test listings."),
          4, 11, 2, 8),
    place(bar("mae_by_neighbourhood", ("model_mae", "Model MAE (CHF)"), ("area", "Neighbourhood"),
              "Alpine municipalities (Ormont-Dessous, Gryon, Château-d'Oex) have the largest errors; "
              "Lausanne and Zurich's residential districts the smallest",
              color=("city", "City"), layout="stack", horizontal=True, sort={"by": "x-reversed"},
              description="Neighbourhoods with at least 15 test listings."), 0, 19, 6, 9),
    place(table("extremes", [
        ("city", "City", "string"), ("direction", "Direction", "string"), ("name", "Listing", "string"),
        ("neighbourhood", "Neighbourhood", "string"), ("room_type", "Room type", "string"),
        ("accommodates", "Guests", "integer"), ("price", "Actual", "float"),
        ("predicted_price", "Predicted", "float"), ("error", "Error (CHF)", "float"),
        ("pct_error", "Error %", "float"), ("listing_url", "Link", "link"),
    ], "The 10 most over- and under-priced test listings per city",
        "Under-priced (error < 0): the model suggests less than the host asks. Over-priced: more."), 0, 28, 6, 9),
    text(["## Vaud (city C): the municipality price level removed the bias the first run had",
          "Change 1: one city-free model plus a price level for the city, learnt from Vaud's known (train) prices. "
          "Change 2: plus a price level per municipality, shrunk to 0 where few prices are known."],
         0, 37, 6, 2),
    place(bar("vaud_steps", ("step", "Step"), ("mae", "MAE on Vaud test set (CHF)"),
              "Vaud MAE: 71.9 first run, 70.6 with a city level, 68.1 with municipality levels (baseline 84.0)",
              color=("method", "Method")), 0, 39, 3, 7),
    place(bar("vaud_areas", ("neighbourhood", "Municipality"), ("median_pct_error", "Median error (%)"),
              "Lausanne went from 21% over-priced to -4%; Montreux from -12% to 0%",
              color=("step", "Step"), description="The 8 Vaud municipalities with the most test listings."),
          3, 39, 3, 7),
]

dashboard = {
    "datasets": [{"name": k, "displayName": k, "queryLines": [line + "\n" for line in v.strip().splitlines()]}
                 for k, v in DATASETS.items()],
    "pages": [
        {"name": "city_overview", "displayName": "1. City overview", "pageType": "PAGE_TYPE_CANVAS", "layout": p1},
        {"name": "geneva_vs_zurich", "displayName": "2. Geneva vs Zurich", "pageType": "PAGE_TYPE_CANVAS",
         "layout": p2},
        {"name": "model", "displayName": "3. Price model", "pageType": "PAGE_TYPE_CANVAS", "layout": p3},
    ],
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(dashboard, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"wrote {OUT}: {len(DATASETS)} datasets, {sum(len(p['layout']) for p in dashboard['pages'])} widgets")
