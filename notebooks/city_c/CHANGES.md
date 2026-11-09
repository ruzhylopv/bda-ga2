# City C (Vaud): first run and changes

Week 1 code is tagged `week1-final`. The first run uses that tag, unchanged.

## First run

| Step | Result |
|---|---|
| `ga2_etl city=C_Vaud` | |
| `ga2_predict city=C_Vaud` | |
| Expected MAE (from `07_transfer_rehearsal`) | |
| Model MAE on C | |

## Changes

Kind: **pipeline** = the pipeline should have handled it from the start; **model** = the model did not transfer.

| # | Change | Kind | Reason | Effect |
|---|---|---|---|---|
| 1 | Replace the `city` feature with a per-city price level (`02_model_change`) | model | `city` is a label the model cannot use for an unseen city: C is priced as one of the training cities. The rehearsal on A and B showed a price-level shift recovers most of the transfer gap. | |

## Three numbers (test rows of C)

| Baseline MAE on C | Model MAE on A + B | Model MAE on C, first run | Model MAE on C, after change |
|---|---|---|---|
| | | | |

## Why the gap
