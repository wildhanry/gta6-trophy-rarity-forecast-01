"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file (<id>.in, JSON) in the input directory and writes one
<id>.out per case holding five quantiles of each trophy's Exophase earn rate
(percent of tracked players) at the case's target time. Scoring is pinball
loss on the logit scale, so all modelling happens in logit space.

Model, per trophy
-----------------
1. Collect the non-null reads of the trophy's rate, convert each to a logit,
   and fit a straight line over time with recency weights (an exponential
   half-life of 1.25 days): recent reads describe the current regime, older
   reads mostly pin down the level.
2. Shrink the fitted slope toward the median slope across all trophies of
   the case. The shrinkage weight is precision based: it grows with the
   cross-trophy spread of slopes and shrinks with the slope's own standard
   error. A series with very few distinct values (a page Exophase had not
   refreshed, or a rate pinned at a bound) is treated as low information
   and shrunk harder, because a flat stale page says little about drift.
3. Extrapolate with damping: the effective horizon is
   tau_d * (1 - exp(-H / tau_d)) with tau_d = 2.5 days, so a drift measured
   over a short window is never projected linearly across the whole week,
   and the total logit move is capped at 3.
   The anchor is a blend of the last observed logit (65%) and the fitted
   line's value at the last read (35%), which trims rounding noise from
   rates printed to two decimals without drifting away from the last read.
4. A count view, used as an uncertainty signal - not as a second median.
   A rate is earners / players, and the two sides move for different
   reasons: new players dilute a rate, existing players earning trophies
   raise it. The same weighted fit is run on log(players tracked) once
   per case and on log(earners + 0.75) per trophy (earners reconstructed
   as rate * players / 100), each slope shrunk toward its cross-trophy
   median and extrapolated over the same damped horizon, giving an
   implied rate at the target. An earlier version of this entry blended
   that implied rate into the median; backtests rejected that (it helped
   a count-generated scenario slightly but cost more on static data,
   where the count view is mostly rounding noise). What survives is the
   disagreement: when the count view and the logit view point apart,
   that gap is genuine model uncertainty, so it widens the spread.
5. Predictive spread (the sigma multiplied by each level's normal z):
   a random-walk term 0.5 * sqrt(H days), as in the house baseline, plus
   the fit's residual scale, plus the shrunk slope's uncertainty projected
   over the damped horizon, plus an allowance for the jump an Exophase
   refresh can cause between reads. Both the allowance and the spread
   floor grow with the horizon - a refresh is unlikely inside half an
   hour and near-certain inside a week - from near zero for very short
   horizons to their full values for the final's multi-day horizon, and
   the total spread is capped so no trophy gets an absurd band.
6. Regime gate on those extra spread terms. The terms beyond the
   random-walk one are uncertainty about movement, and on a page that
   has not observably moved they mostly price rounding noise twice:
   backtests found the centre already equals persistence on flat data
   while the extra terms alone cost several percent against it. So each
   trophy's extra terms are scaled by a gate built from the observed
   increments of its own series: the mean squared logit step per day is
   an empirical diffusion rate for the page process (stale stretches of
   repeated reads count, because the target is the page's own value),
   pooled toward the case median rate with half a day of prior weight
   so a sub-day history cannot set its own gate, and the gate is that
   rate over itself plus the random-walk rate - near zero on a still
   page, near one when the page moves faster than the baseline assumes.
   The random-walk term itself is never gated: a still page can still
   refresh once before the target.

Fallbacks: a trophy with fewer than two usable reads keeps its last logit
(or the case-wide median logit if it was never observed) and takes the
cross-trophy median drift with a wide uncertainty; a case with no history
at all still produces a valid, wide forecast instead of crashing.

Deterministic, offline, Python standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# With -P the interpreter leaves this file's directory off sys.path; restore
# it first so sibling modules would import the same way locally and graded.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

RATE_FLOOR = 0.01        # the scorer clamps percents into this range
RATE_CEIL = 99.99
HALF_LIFE_DAYS = 1.25    # recency weighting of the drift fit
WEIGHT_FLOOR = 0.04      # oldest reads keep a little weight
DAMP_TAU_DAYS = 2.5      # horizon damping scale for drift extrapolation
MAX_LOGIT_MOVE = 3.0     # hard cap on the extrapolated logit change
RW_SIGMA = 0.5           # random-walk logit spread per sqrt(day)
MIN_SLOPE_SE = 0.08      # per-day floor on slope uncertainty (rounding)
SPREAD_FLOOR_SHORT = 0.06  # predictive sigma floor as the horizon -> 0
SPREAD_FLOOR_LONG = 0.22   # predictive sigma floor for multi-day horizons
FLOOR_TAU_DAYS = 1.0       # how fast the floor grows with the horizon
REFRESH_MAX = 0.22         # logit allowance for refresh jumps, long horizon
REFRESH_TAU_DAYS = 1.5     # how fast that allowance grows with the horizon
MAX_SPREAD = 2.75        # predictive sigma cap, logit units
COUNT_PSEUDO = 0.75        # pseudocount (earners) inside the log count fit
COUNT_MAX_DEVIATION = 2.0  # cap on the count view's logit distance used
DISAGREE_COEF = 0.35       # share of the model disagreement added to spread
DISAGREE_CAP = 0.6         # cap on that spread contribution, logit units
MAX_COUNT_SLOPE = 1.5      # per-day cap on log count growth slopes
ACTIVITY_PRIOR_DAYS = 0.5  # pooling weight (days) for the regime gate

_NORMAL = NormalDist()


def clamp_rate(pct):
    return min(RATE_CEIL, max(RATE_FLOOR, pct))


def logit_of_pct(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(value):
    # Numerically stable inverse logit, returned as a percent.
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    exp_v = math.exp(value)
    return 100.0 * exp_v / (1.0 + exp_v)


def round4(value):
    """Round to 4 decimals, half up; whole numbers come back as int so the
    JSON bytes match what a JavaScript writer would emit for them."""
    scaled = math.floor(value * 10000.0 + 0.5)
    result = scaled / 10000.0
    return int(result) if result == int(result) else result


def parse_days(text):
    """ISO-8601 timestamp -> days since the epoch (float, UTC)."""
    if text is None:
        return None
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp() / 86400.0


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def weighted_line_fit(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, se_b, resid_sd, n_eff), or None when the points cannot
    support a slope (fewer than two points or no time spread).
    """
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    sx = sum(w * x for x, w in zip(xs, ws))
    sy = sum(w * y for y, w in zip(ys, ws))
    sxx = sum(w * x * x for x, w in zip(xs, ws))
    sxy = sum(w * x * y for x, y, w in zip(xs, ys, ws))
    denom = sw * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (sw * sxy - sx * sy) / denom
    anchor = (sy - slope * sx) / sw
    resid_sq = 0.0
    for x, y, w in zip(xs, ys, ws):
        resid = y - (anchor + slope * x)
        resid_sq += w * resid * resid
    n_eff = sw * sw / sum(w * w for w in ws)
    dof = max(1.0, n_eff - 2.0)
    resid_var = (resid_sq / sw) * (n_eff / dof)
    resid_sd = math.sqrt(max(resid_var, 0.0))
    var_slope = resid_var * sw / denom
    se_slope = math.sqrt(max(var_slope, 0.0))
    return anchor, slope, se_slope, resid_sd, n_eff


def spread_floor(horizon_days):
    """Sigma floor that is small for minutes-ahead forecasts and reaches
    its full value once the horizon is measured in days."""
    growth = 1.0 - math.exp(-horizon_days / FLOOR_TAU_DAYS) if horizon_days > 0 else 0.0
    return SPREAD_FLOOR_SHORT + (SPREAD_FLOOR_LONG - SPREAD_FLOOR_SHORT) * growth


def refresh_allowance(horizon_days):
    """Logit allowance for an Exophase refresh jump between the last read
    and the target; the chance of at least one refresh grows with time."""
    if horizon_days <= 0:
        return 0.0
    return REFRESH_MAX * (1.0 - math.exp(-horizon_days / REFRESH_TAU_DAYS))


def collect_series(case):
    """Per trophy index: list of (days, rate, players) for non-null reads,
    plus the list of read times and the (days, players) series. Times
    prefer the read's slot over its timestamp."""
    history = case.get("history") or []
    read_times = []
    for read in history:
        stamp = parse_days(read.get("slot") or read.get("at"))
        read_times.append(stamp)
    trophies = case.get("trophies") or []
    series = [[] for _ in trophies]
    players_series = []
    for read, stamp in zip(history, read_times):
        if stamp is None:
            continue
        players = read.get("players_tracked")
        if players is not None and float(players) > 0:
            players_series.append((stamp, float(players)))
        rates = read.get("rates") or []
        for idx in range(min(len(trophies), len(rates))):
            rate = rates[idx]
            if rate is not None:
                series[idx].append((stamp, float(rate), players))
    return series, read_times, players_series


def fit_players_target(players_series, last_time, damped_horizon):
    """Forecast players tracked at the target from a recency-weighted fit
    of log(players); returns None when the series cannot support one."""
    if len(players_series) < 2 or last_time is None:
        if players_series:
            return players_series[-1][1]
        return None
    xs = []
    ys = []
    ws = []
    for stamp, players in players_series:
        age = last_time - stamp
        xs.append(-age)
        ys.append(math.log(players))
        ws.append(max(WEIGHT_FLOOR, 0.5 ** (age / HALF_LIFE_DAYS)))
    fit = weighted_line_fit(xs, ys, ws)
    last_players = players_series[-1][1]
    if fit is None:
        return last_players
    slope = min(MAX_COUNT_SLOPE, max(-MAX_COUNT_SLOPE, fit[1]))
    target = last_players * math.exp(slope * damped_horizon)
    return max(1.0, target)


def increment_sums(points):
    """Squared logit steps between consecutive reads and the days they
    span. Repeated reads of an unchanged page add time but no step,
    which is the point: the pair estimates how fast the page itself
    moves, stale stretches included."""
    sq = 0.0
    days = 0.0
    for earlier, later in zip(points, points[1:]):
        gap = later[0] - earlier[0]
        if gap <= 0:
            continue
        step = logit_of_pct(later[1]) - logit_of_pct(earlier[1])
        sq += step * step
        days += gap
    return sq, days


def fit_trophy(points, last_time):
    """First-pass per-trophy fit summary (no shrinkage yet): a logit fit
    for the rate itself and a log fit for the earner count behind it."""
    if not points:
        return None
    last_logit = logit_of_pct(points[-1][1])
    incr_sq, incr_days = increment_sums(points)
    earners_last = None
    if points[-1][2] is not None:
        earners_last = points[-1][1] * float(points[-1][2]) / 100.0
    if len(points) < 3:
        return {
            "last_logit": last_logit,
            "fit": None,
            "distinct": len({rate for _, rate, _ in points}),
            "span_days": 0.0,
            "count_fit": None,
            "earners_last": earners_last,
            "distinct_earners": 0,
            "incr_sq": incr_sq,
            "incr_days": incr_days,
        }
    xs = []
    ys = []
    ws = []
    cxs = []
    cys = []
    cws = []
    earners_seen = []
    for stamp, rate, players in points:
        age = last_time - stamp
        weight = max(WEIGHT_FLOOR, 0.5 ** (age / HALF_LIFE_DAYS))
        xs.append(-age)  # x = days relative to the last read, so a = level there
        ys.append(logit_of_pct(rate))
        ws.append(weight)
        if players is not None and float(players) > 0:
            earners = rate * float(players) / 100.0
            earners_seen.append(round(earners, 3))
            cxs.append(-age)
            cys.append(math.log(earners + COUNT_PSEUDO))
            cws.append(weight)
    fit = weighted_line_fit(xs, ys, ws)
    count_fit = weighted_line_fit(cxs, cys, cws) if len(cxs) >= 3 else None
    span = points[-1][0] - points[0][0]
    return {
        "last_logit": last_logit,
        "fit": fit,
        "distinct": len({rate for _, rate, _ in points}),
        "span_days": span,
        "count_fit": count_fit,
        "earners_last": earners_last,
        "distinct_earners": len(set(earners_seen)),
        "incr_sq": incr_sq,
        "incr_days": incr_days,
    }


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    z_scores = [_NORMAL.inv_cdf(level) for level in levels]
    trophies = case.get("trophies") or []
    series, read_times, players_series = collect_series(case)

    known_times = [t for t in read_times if t is not None]
    last_time = known_times[-1] if known_times else None

    # Target time: prefer the case's own target, else checkpoint + horizon.
    target_time = parse_days(case.get("target"))
    if target_time is None:
        checkpoint = parse_days(case.get("at"))
        horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
        if checkpoint is not None:
            target_time = checkpoint + horizon_days
    if last_time is not None and target_time is not None:
        horizon = max(0.0, target_time - last_time)
    elif last_time is not None:
        horizon = float(case.get("horizon_minutes") or 0) / 1440.0
    else:
        horizon = float(case.get("horizon_minutes") or 0) / 1440.0

    summaries = []
    for points in series:
        if last_time is None or not points:
            summaries.append(None if not points else {
                "last_logit": logit_of_pct(points[-1][1]),
                "fit": None, "distinct": 1, "span_days": 0.0,
                "count_fit": None, "earners_last": None, "distinct_earners": 0,
                "incr_sq": 0.0, "incr_days": 0.0,
            })
        else:
            summaries.append(fit_trophy(points, last_time))

    # Cross-trophy drift centre and scale, from the trophies whose own fit
    # supports a slope. Median and MAD keep a few wild trophies from
    # dragging the centre.
    slopes = [s["fit"][1] for s in summaries if s and s["fit"] is not None]
    if slopes:
        drift_centre = median(slopes)
        mad = median([abs(b - drift_centre) for b in slopes])
        drift_scale = min(0.75, max(0.05, 1.4826 * mad))
    else:
        drift_centre = 0.0
        drift_scale = 0.2
    resid_scales = [s["fit"][3] for s in summaries if s and s["fit"] is not None]
    resid_fallback = median(resid_scales) if resid_scales else 0.15
    last_logits = [s["last_logit"] for s in summaries if s]
    level_fallback = median(last_logits) if last_logits else logit_of_pct(1.0)

    # Cross-trophy centre and scale for the earner-count growth slopes.
    count_slopes = [s["count_fit"][1] for s in summaries
                    if s and s.get("count_fit") is not None]
    if count_slopes:
        count_centre = median(count_slopes)
        count_mad = median([abs(g - count_centre) for g in count_slopes])
        count_scale = min(0.75, max(0.05, 1.4826 * count_mad))
    else:
        count_centre = 0.0
        count_scale = 0.2

    # Case-wide activity centre for the regime gate (point 6 of the
    # header): the median per-trophy diffusion rate among the trophies
    # whose reads span any time at all.
    rates_seen = [s["incr_sq"] / s["incr_days"] for s in summaries
                  if s and s.get("incr_days", 0.0) > 0]
    activity_centre = median(rates_seen) if rates_seen else 0.0

    damped_horizon = DAMP_TAU_DAYS * (1.0 - math.exp(-horizon / DAMP_TAU_DAYS)) if horizon > 0 else 0.0
    players_target = fit_players_target(players_series, last_time, damped_horizon)

    rows = []
    for trophy, summary in zip(trophies, summaries):
        if summary is None:
            anchor = level_fallback
            drift = drift_centre
            se_adj = 0.4
            resid_sd = max(resid_fallback, 0.2)
        else:
            anchor_last = summary["last_logit"]
            fit = summary["fit"]
            if fit is None:
                anchor = anchor_last
                drift = drift_centre
                se_adj = 0.4
                resid_sd = max(resid_fallback, 0.15)
            else:
                fit_anchor, slope, se_slope, resid_sd, _n_eff = fit
                # Keep the fitted level near the last read; a line through
                # noisy rounded rates can sit far from it by the last point.
                fit_anchor = min(anchor_last + 1.0, max(anchor_last - 1.0, fit_anchor))
                anchor = 0.65 * anchor_last + 0.35 * fit_anchor
                se_adj = math.sqrt(se_slope * se_slope + MIN_SLOPE_SE ** 2)
                precision_weight = (drift_scale ** 2) / (drift_scale ** 2 + se_adj ** 2)
                info = min(1.0, summary["distinct"] / 4.0)
                if summary["span_days"] < 0.05:
                    info *= 0.5
                weight = precision_weight * info
                drift = drift_centre + weight * (slope - drift_centre)
                resid_sd = min(1.2, max(0.04, resid_sd))
        move = drift * damped_horizon
        move = min(MAX_LOGIT_MOVE, max(-MAX_LOGIT_MOVE, move))
        centre_logit = anchor + move
        # Count view: shrink the earner growth slope like the drift, grow
        # the last earner count over the damped horizon, and divide by the
        # forecast player count. The implied rate's distance from the
        # logit median is treated as model disagreement (see the header):
        # it widens the spread below and never moves the median.
        disagreement = 0.0
        if (summary is not None and summary.get("count_fit") is not None
                and summary.get("earners_last") is not None
                and players_target and horizon > 0):
            _c_anchor, c_slope, c_se, _c_resid, _c_n = summary["count_fit"]
            c_se_adj = math.sqrt(c_se * c_se + MIN_SLOPE_SE ** 2)
            c_precision = (count_scale ** 2) / (count_scale ** 2 + c_se_adj ** 2)
            c_info = min(1.0, summary.get("distinct_earners", 0) / 4.0)
            if summary["span_days"] < 0.05:
                c_info *= 0.5
            growth = count_centre + c_precision * c_info * (c_slope - count_centre)
            growth = min(MAX_COUNT_SLOPE, max(-MAX_COUNT_SLOPE, growth))
            earners_target = (summary["earners_last"] + COUNT_PSEUDO) * math.exp(
                growth * damped_horizon) - COUNT_PSEUDO
            earners_target = max(0.0, earners_target)
            count_logit = logit_of_pct(100.0 * earners_target / players_target)
            gap = abs(count_logit - centre_logit)
            disagreement = min(DISAGREE_CAP,
                               DISAGREE_COEF * min(gap, COUNT_MAX_DEVIATION))
        # Regime gate: pool this trophy's diffusion rate toward the
        # case centre with half a day of prior weight, then weigh the
        # observed rate against the random-walk rate it would take for
        # the extra spread terms to be the main story.
        if summary is not None:
            pooled = (
                (summary.get("incr_sq", 0.0)
                 + ACTIVITY_PRIOR_DAYS * activity_centre)
                / (summary.get("incr_days", 0.0) + ACTIVITY_PRIOR_DAYS)
            )
            gate = pooled / (pooled + RW_SIGMA ** 2) if pooled > 0 else 0.0
        else:
            # Never observed at all: the uncertainty is about the level
            # itself, not about movement, so the gate does not apply.
            gate = 1.0
        var = RW_SIGMA ** 2 * horizon + gate * (
            0.35 * resid_sd ** 2
            + (se_adj * damped_horizon) ** 2
            + refresh_allowance(horizon) ** 2
            + disagreement ** 2
        )
        spread = min(MAX_SPREAD, max(spread_floor(horizon), math.sqrt(var)))
        quantiles = []
        previous = 0.0
        for z in z_scores:
            value = round4(pct_of_logit(centre_logit + z * spread))
            value = min(100.0, max(0.0, float(value)))
            value = max(value, previous)  # quantiles never decrease
            quantiles.append(int(value) if float(value) == int(value) else value)
            previous = float(value)
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        payload = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
