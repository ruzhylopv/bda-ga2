"""City configuration and Unity Catalog names shared by all notebooks."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "cities.json"

# Databricks Free Edition ships a default catalog called `workspace`.
CATALOG = "workspace"
# Schemas are prefixed so they do not clash with other projects in the same workspace.
BRONZE, SILVER, GOLD = "ga2_bronze", "ga2_silver", "ga2_gold"
RAW_VOLUME = "raw"
RAW_VOLUME_PATH = f"/Volumes/{CATALOG}/{BRONZE}/{RAW_VOLUME}"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_city(city_key: str) -> dict:
    cities = load_config()["cities"]
    if city_key not in cities:
        raise KeyError(f"Unknown city '{city_key}'. Known: {sorted(cities)}")
    return {"key": city_key, **cities[city_key]}


def cities_by_role(role: str) -> list[str]:
    """`train` -> cities A and B; `new_client` -> city C."""
    return [k for k, c in load_config()["cities"].items() if c["role"] == role]


def volume_csv_path(city_key: str) -> str:
    """Where 00_setup copies the raw CSV; Spark reads from here."""
    return f"{RAW_VOLUME_PATH}/{city_key}/listings.csv"
