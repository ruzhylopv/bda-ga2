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

## Data snapshots date

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/), `listings.csv.gz` (detailed listings).

| City | Region | Country | Snapshot date | Source data | Repo copy |
|---|---|---|---|---|---|
| Geneva | Geneva | Switzerland | 23 September 2026 | [listings.csv.gz](https://data.insideairbnb.com/switzerland/geneva/geneva/2026-09-23/data/listings.csv.gz) | [data/A_Geneva/listings.csv](data/A_Geneva/listings.csv) |
| Zurich | Zürich | Switzerland | 25 September 2026 | [listings.csv.gz](https://data.insideairbnb.com/switzerland/z%C3%BCrich/zurich/2026-09-25/data/listings.csv.gz) | [data/B_Zurich/listings.csv](data/B_Zurich/listings.csv) |
| Vaud | Vaud | Switzerland | 05 October 2026 | [listings.csv.gz](https://data.insideairbnb.com/switzerland/vd/vaud/2026-10-05/data/listings.csv.gz) | [data/C_Vaud/listings.csv](data/C_Vaud/listings.csv) |

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
│   ├── model/               # 3.4 job ga2_model: 05 train + evaluate vs baseline (A + B), 06 predictions per city
│   └── city_c/              # 3.7 first run on city C, fixes log, error report
├── src/bda/                 # shared Python code imported by the notebooks
│   └── config.py            # loads cities.json; catalog, schema and volume names
├── resources/               # Databricks Job definitions: ga2_etl and ga2_model
├── dashboard/               # 3.5 exported AI/BI dashboard
├── genie/                   # 3.6 Genie Agent instructions and example questions
├── docs/                    # 3.2 solution architecture diagram and description
└── presentation/            # 3.8 slides (PDF or link)
```

All notebooks run on Databricks Free Edition from a Git folder clone of this repo. Run `notebooks/00_setup.py` first.
Tables live in the `workspace` catalog: `ga2_bronze` (raw), `ga2_silver` (cleaned, typed listings), `ga2_gold` (metrics, features, predictions).
