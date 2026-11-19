"""Backtests for the GTA VI trophy rarity forecaster.

Usage: python3 backtest.py <dir-with-practice-.in-files>

Three checks, all deterministic:

1. Holdout inside each practice case's history: cut the history at every
   usable point, forecast the next read, and compare the mean pinball loss
   on the logit scale against the persistence baseline (last rate, spread
   0.5 * sqrt(horizon in days)). The practice histories are a static page,
   so this mostly proves the model does not invent drift where none moved.

2. Synthetic moving data: histories generated with known per-trophy logit
   drifts (and a static control set), plus a count-generated set where a
   growing player count and per-trophy earner counts produce the rates.
   This is the test the practice cases cannot give: an adaptive model
   must beat persistence when rates really move, and must not lose badly
   when they do not.

3. Format validation of the .out files main.py writes for the practice
   cases: exact key shape, one row per trophy, five non-decreasing
   quantiles in [0, 100], at most 65,536 bytes.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import forecast_case, logit_of_pct, parse_days  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(level) for level in LEVELS]
EPOCH_DAY_ZERO = 20597.0  # arbitrary fixed day count for synthetic cases


def pinball_mean(forecast_rows, actual_rates):
    """Mean pinball loss on the logit scale over trophies x levels."""
    total = 0.0
    count = 0
    for row, actual in zip(forecast_rows, actual_rates):
        if actual is None:
            continue
        y = logit_of_pct(actual)
        for tau, q in zip(LEVELS, row["q"]):
            d = y - logit_of_pct(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count if count else float("nan")


def baseline_rows(case):
    """The house persistence baseline, written from its stated formula."""
    history = case["history"]
    last = history[-1]
    spread = 0.5 * math.sqrt(float(case["horizon_minutes"]) / 1440.0)
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        rate = last["rates"][idx]
        centre = logit_of_pct(rate if rate is not None else 0.0)
        qs = []
        for z in Z:
            pct = 100.0 / (1.0 + math.exp(-(centre + z * spread)))
            qs.append(min(100.0, max(0.0, pct)))
        rows.append({"id": trophy["id"], "q": qs})
    return rows


def minutes_between(start_text, end_text):
    return (parse_days(end_text) - parse_days(start_text)) * 1440.0


def holdout_report(case):
    history = case["history"]
    ours_losses = []
    base_losses = []
    for cut in range(3, len(history)):
        truncated = dict(case)
        truncated["history"] = history[:cut]
        truncated["at"] = history[cut - 1]["slot"]
        truncated["target"] = history[cut]["slot"]
        truncated["horizon_minutes"] = minutes_between(
            history[cut - 1]["slot"], history[cut]["slot"])
        actual = history[cut]["rates"]
        ours = forecast_case(truncated)["trophies"]
        ours_losses.append(pinball_mean(ours, actual))
        base_losses.append(pinball_mean(baseline_rows(truncated), actual))
    return {
        "cuts": len(ours_losses),
        "ours": sum(ours_losses) / len(ours_losses),
        "baseline": sum(base_losses) / len(base_losses),
    }


def iso_from_days(days):
    from datetime import datetime, timezone
    moment = datetime.fromtimestamp(days * 86400.0, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def synthetic_case(base_rates, drifts, span_days=3.0, step_days=0.25,
                   horizon_days=1.0, wiggle=0.03):
    """One synthetic case: logit rates move linearly at `drifts` per day."""
    trophies = [{"id": i + 1} for i in range(len(base_rates))]
    history = []
    steps = int(span_days / step_days)
    for k in range(steps + 1):
        t = k * step_days
        rates = []
        for j, base in enumerate(base_rates):
            centre = logit_of_pct(base) + drifts[j] * t
            centre += wiggle * math.sin(3.0 * t + 1.3 * j)
            pct = 100.0 / (1.0 + math.exp(-centre))
            rates.append(round(min(99.99, max(0.01, pct)), 2))
        history.append({
            "slot": iso_from_days(EPOCH_DAY_ZERO + t),
            "at": iso_from_days(EPOCH_DAY_ZERO + t),
            "players_tracked": 5000 + 700 * k,
            "trophies_earned": 9000,
            "rates": rates,
        })
    target_t = span_days + horizon_days
    actual = []
    for j, base in enumerate(base_rates):
        centre = logit_of_pct(base) + drifts[j] * target_t
        pct = 100.0 / (1.0 + math.exp(-centre))
        actual.append(round(min(99.99, max(0.01, pct)), 2))
    case = {
        "id": "synthetic",
        "at": iso_from_days(EPOCH_DAY_ZERO + span_days),
        "target": iso_from_days(EPOCH_DAY_ZERO + target_t),
        "horizon_minutes": horizon_days * 1440.0,
        "levels": LEVELS,
        "trophies": trophies,
        "history": history,
    }
    return case, actual


def count_synthetic_case(base_rates):
    """A case generated the way the real page is built: a player count
    that grows fast and saturates, and per-trophy earner counts that grow
    at their own rates, so each rate is earners / players at each read.
    Trophies whose earners grow slower than the player count dilute."""
    start_players = 2500.0

    def players_at(t):
        return start_players * math.exp((0.55 / 0.35) * (1.0 - math.exp(-0.35 * t)))

    def earner_growth(j, base):
        growth = 0.35 + 0.5 * math.sin(1.3 * j + 0.2)
        if base < 5.0:
            growth += 0.30
        if base > 50.0:
            growth -= 0.25
        return growth

    trophies = [{"id": i + 1} for i in range(len(base_rates))]
    history = []
    for k in range(13):
        t = k * 0.25
        players = players_at(t)
        rates = []
        for j, base in enumerate(base_rates):
            earners = (base * start_players / 100.0) * math.exp(earner_growth(j, base) * t)
            rates.append(round(min(99.99, max(0.01, 100.0 * earners / players)), 2))
        history.append({
            "slot": iso_from_days(EPOCH_DAY_ZERO + t),
            "at": iso_from_days(EPOCH_DAY_ZERO + t),
            "players_tracked": int(round(players)),
            "trophies_earned": 9000,
            "rates": rates,
        })
    players = players_at(4.0)
    actual = []
    for j, base in enumerate(base_rates):
        earners = (base * start_players / 100.0) * math.exp(earner_growth(j, base) * 4.0)
        actual.append(round(min(99.99, max(0.01, 100.0 * earners / players)), 2))
    case = {
        "id": "synthetic-counts",
        "at": iso_from_days(EPOCH_DAY_ZERO + 3.0),
        "target": iso_from_days(EPOCH_DAY_ZERO + 4.0),
        "horizon_minutes": 1440.0,
        "levels": LEVELS,
        "trophies": trophies,
        "history": history,
    }
    return case, actual


def synthetic_report(base_rates):
    drifts = []
    for j, base in enumerate(base_rates):
        drift = 0.55 * math.sin(1.7 * j + 0.4)
        if base < 5.0:
            drift += 0.25   # rare trophies get earned as players progress
        if base > 50.0:
            drift -= 0.20   # common early trophies dilute as players join
        drifts.append(drift)
    moving_case, moving_actual = synthetic_case(base_rates, drifts)
    moving = {
        "ours": pinball_mean(forecast_case(moving_case)["trophies"], moving_actual),
        "baseline": pinball_mean(baseline_rows(moving_case), moving_actual),
    }
    still_case, still_actual = synthetic_case(
        base_rates, [0.0] * len(base_rates), wiggle=0.0)
    still = {
        "ours": pinball_mean(forecast_case(still_case)["trophies"], still_actual),
        "baseline": pinball_mean(baseline_rows(still_case), still_actual),
    }
    count_case, count_actual = count_synthetic_case(base_rates)
    counts = {
        "ours": pinball_mean(forecast_case(count_case)["trophies"], count_actual),
        "baseline": pinball_mean(baseline_rows(count_case), count_actual),
    }
    return {"moving": moving, "static": still, "count_moving": counts}


def validate_outputs(input_dir):
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["MAKEFUN_INPUT_DIR"] = str(input_dir)
        env["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
        subprocess.run([sys.executable, str(Path(__file__).parent / "main.py")],
                       env=env, check=True)
        for in_path in sorted(Path(input_dir).glob("*.in")):
            case = json.loads(in_path.read_text(encoding="utf-8"))
            out_path = out_dir / (in_path.stem + ".out")
            raw = out_path.read_bytes()
            if len(raw) > 65536:
                problems.append(f"{out_path.name}: {len(raw)} bytes > 65536")
                continue
            data = json.loads(raw.decode("utf-8"))
            if set(data.keys()) != {"trophies"}:
                problems.append(f"{out_path.name}: top-level keys {sorted(data.keys())}")
                continue
            rows = data["trophies"]
            want_ids = [t["id"] for t in case["trophies"]]
            got_ids = [r["id"] for r in rows]
            if sorted(got_ids) != sorted(want_ids) or len(got_ids) != len(set(got_ids)):
                problems.append(f"{out_path.name}: trophy ids mismatch")
                continue
            for row in rows:
                if set(row.keys()) != {"id", "q"} or len(row["q"]) != 5:
                    problems.append(f"{out_path.name}: bad row shape for id {row.get('id')}")
                    break
                q = row["q"]
                if any((not isinstance(v, (int, float))) or not math.isfinite(v)
                       or v < 0 or v > 100 for v in q):
                    problems.append(f"{out_path.name}: q out of range for id {row['id']}")
                    break
                if any(q[i] > q[i + 1] for i in range(4)):
                    problems.append(f"{out_path.name}: q decreasing for id {row['id']}")
                    break
    return problems


def main_cli():
    input_dir = Path(sys.argv[1])
    cases = []
    for path in sorted(input_dir.glob("*.in")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    print("holdout (practice histories, next-read forecasts):")
    all_ours, all_base = [], []
    for case in cases:
        report = holdout_report(case)
        all_ours.append(report["ours"])
        all_base.append(report["baseline"])
        print(f"  {case['id']}: cuts={report['cuts']} "
              f"ours={report['ours']:.6f} baseline={report['baseline']:.6f}")
    print(f"  mean: ours={sum(all_ours) / len(all_ours):.6f} "
          f"baseline={sum(all_base) / len(all_base):.6f}")
    base_rates = cases[0]["history"][-1]["rates"]
    base_rates = [r if r is not None else 1.0 for r in base_rates]
    synth = synthetic_report(base_rates)
    print("synthetic (1-day horizon after 3 days of reads):")
    for name, pair in synth.items():
        print(f"  {name}: ours={pair['ours']:.6f} baseline={pair['baseline']:.6f}")
    problems = validate_outputs(input_dir)
    print("format validation:", "OK" if not problems else problems)


if __name__ == "__main__":
    main_cli()
