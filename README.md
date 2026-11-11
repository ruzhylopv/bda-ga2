# Big Data Processing. Group assignment 2 repository.

#### Repository created at: Saturday, 31 Oct 2026.

<h2 align="center">Team: BIG</h2>

<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ruzhylopv">
        <img src="https://avatars.githubusercontent.com/u/187111006?v=4" width="100px;" alt="Ruzhylo Pavlo"/><br />
        <sub><b>Ruzhylo Pavlo</b></sub>
      </a><br />
      <sub>@ruzhylopv</sub>
    </td>
    <td align="center">
      <a href="https://github.com/dhvgop-i">
        <img src="https://avatars.githubusercontent.com/u/106429184?v=4" width="100px;" alt="Ivan Maksymchuk"/><br />
        <sub><b>Ivan Maksymchuk</b></sub>
      </a><br />
      <sub>@dhvgop-i</sub>
    </td>
    <td align="center">
      <a href="https://github.com/Oleksii-Lasiichuk">
        <img src="https://avatars.githubusercontent.com/u/172442553?v=4" width="100px;" alt="Oleksii Lasiichuk"/><br />
        <sub><b>Oleksii Lasiichuk</b></sub>
      </a><br />
      <sub>@Oleksii-Lasiichuk</sub>
    </td>
  </tr>
</table>
</div>

## Assignment

A short-term-rental analytics start-up builds a lakehouse over Airbnb data for two client cities, then gets a third
client in the same region a week later and runs the unchanged pipeline and model on it.

**Region: Switzerland. Currency: CHF** (all prices are CHF per night).

| Letter | City | Role |
|---|---|---|
| A | Geneva | training city |
| B | Zurich | training city |
| C | Vaud | new client, kept unseen until the A + B pipeline and model were finished |

**Why Vaud as city C:** Vaud is a canton, not a city: Lausanne, the Lake Geneva towns and Alpine ski resorts. We chose
it knowing the transfer would be harder than to another city, because its prices and geography differ from Geneva and
Zurich in ways the model never saw.

## Data snapshots

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/), `listings.csv.gz` (detailed listings), CC BY 4.0.

| City | Country | Snapshot date | Listings | Source data | Repo copy |
|---|---|---|---|---|---|
| A · Geneva | Switzerland | 23 September 2026 | 2,538 | [listings.csv.gz](https://data.insideairbnb.com/switzerland/geneva/geneva/2026-09-23/data/listings.csv.gz) | [data/A_Geneva/listings.csv](data/A_Geneva/listings.csv) |
| B · Zurich | Switzerland | 25 September 2026 | 3,174 | [listings.csv.gz](https://data.insideairbnb.com/switzerland/z%C3%BCrich/zurich/2026-09-25/data/listings.csv.gz) | [data/B_Zurich/listings.csv](data/B_Zurich/listings.csv) |
| C · Vaud | Switzerland | 05 October 2026 | 5,235 | [listings.csv.gz](https://data.insideairbnb.com/switzerland/vd/vaud/2026-10-05/data/listings.csv.gz) | [data/C_Vaud/listings.csv](data/C_Vaud/listings.csv) |

## Deliverables

| # | Deliverable | Where |
|---|---|---|
| 3.1 | Exploratory analysis of A and B, leakage review | [notebooks/eda/01_exploratory_analysis_A_B.ipynb](notebooks/eda/01_exploratory_analysis_A_B.ipynb) |
| 3.2 | Solution architecture | [docs/architecture.md](docs/architecture.md) |
| 3.3 | ETL pipeline, one job with the city as a parameter | [resources/ga2_etl.job.yml](resources/ga2_etl.job.yml), [notebooks/pipeline/](notebooks/pipeline/) |
| 3.4 | Price model against the baseline | [notebooks/model/](notebooks/model/), [src/bda/modeling.py](src/bda/modeling.py) |
| 3.5 | AI/BI dashboard (3 pages) | [dashboard/ga2_dashboard.lvdash.json](dashboard/ga2_dashboard.lvdash.json) |
| 3.6 | Genie Agent: instructions, example questions, test log | [genie/README.md](genie/README.md) |
| 3.7 | City C: first run, changes, the three numbers | [notebooks/city_c/CHANGES.md](notebooks/city_c/CHANGES.md) |
| 3.8 | Presentation | [presentation/presentation.pdf](presentation/presentation.pdf) |

## Results

MAE in CHF per night on the held-out test set (20% of listings per city). Baseline: median training price for the same
room type and number of guests.

| | Geneva (A) | Zurich (B) | A + B | Vaud (C), first run | Vaud (C), after changes |
|---|---|---|---|---|---|
| Baseline | 64.6 | 75.3 | 70.6 | 84.0 | 84.0 |
| Model | 53.5 | 58.9 | 56.5 | 71.9 | 68.1 |
| Better than baseline | 17% | 22% | 20% | 14% | 19% |

The pipeline ran on Vaud unchanged; the model needed two changes (a price level per city and per municipality instead
of the `city` feature). Details and the explanation of the gap: [notebooks/city_c/CHANGES.md](notebooks/city_c/CHANGES.md).

## Repository structure

```
bda-ga2/
├── config/
│   └── cities.json          # the only place cities are defined: letter, role, snapshot date, source URL, raw path
├── data/                    # raw Inside Airbnb snapshots, one folder per city
│   ├── A_Geneva/            # city A, training
│   ├── B_Zurich/            # city B, training
│   └── C_Vaud/              # city C, new client: kept unseen until week 2
├── notebooks/
│   ├── 00_setup.py          # creates ga2_bronze / ga2_silver / ga2_gold schemas, lands raw CSVs in a volume
│   ├── eda/                 # 3.1 exploratory analysis of cities A and B
│   ├── pipeline/            # 3.3 job ga2_etl (city as a parameter): 01 bronze -> 02 silver -> 03 gold metrics -> 04 features
│   ├── model/               # 3.4 job ga2_model: 05 train + evaluate vs baseline (A + B), 06 predictions per city;
│   │                        #     07 transfer rehearsal: Geneva <-> Zurich as a stand-in for city C
│   └── city_c/              # 3.7 01 first run on city C, 02 model change, CHANGES.md fixes log
├── src/bda/                 # shared Python code imported by the notebooks
│   ├── config.py            # loads cities.json; catalog, schema and volume names
│   ├── modeling.py          # price model candidates, baseline, MAE
│   └── transfer.py          # leave-one-city-out, price-level calibration, effect stability
├── resources/               # Databricks Asset Bundle resources: jobs ga2_etl, ga2_etl_all_cities, ga2_model, ga2_predict; the dashboard
├── dashboard/               # 3.5 exported AI/BI dashboard (.lvdash.json) and build_dashboard.py that generates it
├── genie/                   # 3.6 Genie Agent definition (space.json), build_space.py, instructions and test log
├── docs/                    # 3.2 solution architecture diagram and description
└── presentation/            # 3.8 slides (PDF or link)
```

## How to run

Everything runs on Databricks Free Edition (serverless, no cost). Tables live in the `workspace` catalog:
`ga2_bronze` (raw), `ga2_silver` (cleaned, typed listings), `ga2_gold` (metrics, features, predictions).

With the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) logged in to the workspace:

```bash
databricks bundle deploy                                     # notebooks, data, jobs and the dashboard
databricks bundle run ga2_etl_all_cities                     # 1. ETL for Geneva and Zurich
databricks bundle run ga2_model                              # 2. train on A + B, predict A and B
```

3. Run `notebooks/model/07_transfer_rehearsal` (writes the expected error for city C).
4. City C: `databricks bundle run ga2_etl --params city=C_Vaud`, then `databricks bundle run ga2_predict --params city=C_Vaud`.
5. Run `notebooks/city_c/01_first_run`, then `notebooks/city_c/02_model_change`.
6. Genie: `python3 genie/build_space.py ga2_gold genie/space.json`, then create the space as in
   [genie/README.md](genie/README.md).

Without the CLI: clone the repo as a Git folder in the workspace and run the notebooks in the same order, starting with
`notebooks/00_setup.py` (widget `cities`) and the pipeline notebooks with the widget `city`.

Adding a city: one entry in [config/cities.json](config/cities.json) and its `listings.csv` under `data/`; no code changes.
