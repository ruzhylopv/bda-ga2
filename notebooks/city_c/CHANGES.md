# City C (Vaud): first run and changes

Vaud is a **canton**, not a city: Lausanne, the Lake Geneva towns and the Alpine resorts (Villars, Leysin, Les
Diablerets, Château-d'Oex). We chose it knowing the transfer would be harder.

All runs: Oleksii's Free Edition workspace, 10 Nov 2026. Week 1 code: tag `week1-final` (`97548ee`).

## Expected before the run

From `model/07_transfer_rehearsal` (Geneva <-> Zurich only, saved in `ga2_gold.city_c_expectation` before Vaud was
loaded): moving to an unseen city costs 22-35% MAE, so **expected MAE on C: 69-76 CHF**.

## First run (unchanged code)

| Step | Result |
|---|---|
| `ga2_etl city=C_Vaud` | ran, all 5 tasks. 5,235 listings, 3,630 with a usable price, no unparseable values, no duplicate ids |
| `ga2_predict city=C_Vaud` | ran. Every listing priced |
| Model MAE on C | **77.5 CHF** on all priced listings, 71.9 on the test rows. Baseline: 89.2 / 84.0 |

Nothing failed. What came out was wrong in two systematic ways (`city_c/01_first_run`):

- **Lausanne over-priced by 24%** (median error). Vaud is an unseen `city` value, so it gets a training city's price
  level, and Lausanne is cheaper than central Geneva or Zurich (median CHF 131).
- **43% of priced listings are 30+ km from the centre**, which the model never saw (max 16 km in A and B). Those are
  Alpine resorts and chalets (median CHF 214-386); MAE there is ~100 CHF vs ~50 in Lausanne.

## Changes

**pipeline**: should have been handled from the start. **model**: the model did not transfer.

| # | Change | Kind | Reason | MAE on C test |
|---|---|---|---|---|
| 0 | none (first run) | | | 71.9 |
| 1 | Drop `city`; one city-free model + a per-city price level from the city's known prices | model | `city` is a label that means nothing for a new city | 70.6 |
| 2 | + a price level per municipality, shrunk to 0 when few prices are known | model | "distance to one centre" means suburb in Geneva and ski resort in Vaud; local price levels replace it where they are known | **68.1** |

Pipeline fixes for city C: **none needed**. Two pipeline bugs were found and fixed earlier, while running A and B for
the first time on Databricks (both in `week1-final`): the city list parameter of `ga2_etl_all_cities` overrode the
`cities` field of `00_setup`, and a quote inside a table comment broke the `COMMENT` statement in `03_gold_metrics`.

Effect on A and B of the same rule: Geneva 53.5 -> 52.6, Zurich 58.9 -> 60.3 (`ga2_gold.city_c_changes`).

## Three numbers

| Baseline MAE on C | Model MAE on A + B | Model MAE on C, first run | Model MAE on C, after changes |
|---|---|---|---|
| 84.0 | 56.5 | 71.9 | 68.1 |

C numbers are on its test rows (717 listings), because the changes use the train rows as known prices.

## Why the gap

- **Vaud is harder for any method.** The baseline itself is 19% worse on Vaud (84.0) than on A + B (70.6): chalets and
  resort flats spread prices far more than city flats do.
- **Relative to the baseline, the model transfers.** It beats the baseline by 20% on A + B, 14% on Vaud in the first
  run and 19% after the changes.
- **What is left is the mountains.** Without seasonality, altitude or a "ski resort" signal, a 30 km-away chalet looks
  like a big suburban house. The municipality level fixes the median bias (Lausanne +21% -> -4%, Montreux -12% -> 0%),
  not the spread inside a resort.
- **The rehearsal was right about the size.** Expected 69-76, got 71.9 on the test rows.
