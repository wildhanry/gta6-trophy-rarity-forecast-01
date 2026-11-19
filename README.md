# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wildhanry/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wildhanry/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic, offline forecaster for the earn rate of every Grand Theft
Auto VI PlayStation 5 trophy on Exophase at a case's target time. For each
case file in the input directory it writes five quantiles (5%, 25%, 50%,
75%, 95%) per trophy, scored by pinball loss on the logit scale.

The model works per trophy on the logit of the observed rates: a
recency-weighted linear fit estimates each trophy's drift, that drift is
shrunk toward the cross-trophy median drift according to how precisely it
is measured, and the extrapolation over the forecast horizon is damped so
a short window cannot project an extreme move a week out. A second,
count-based view (earners versus players tracked, each grown over the
same damped horizon) never moves the median - backtests rejected that -
but disagreement between the two views widens the predictive spread,
which also combines a random-walk term, the fit's residual scale, the
uncertainty of the shrunk drift, and a refresh-jump allowance; both that
allowance and the spread floor grow with the horizon, so minutes-ahead
forecasts stay tight while week-ahead forecasts stay honest. Histories
that are too short, rates that are missing in the last read, and rates
pinned at 0% or 100% all fall back to guarded behaviour instead of
failing.

Python 3.13, standard library only, no network access, no randomness:
the same case always produces the same output bytes.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`backtest.py` replays the practice cases with holdout cuts inside their
histories and on synthetic moving data, and reports pinball loss against
the persistence baseline:

```sh
python3 backtest.py <kit-input-dir>
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=29b3jrvf&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
