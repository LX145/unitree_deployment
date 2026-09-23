#!/usr/bin/env python3
"""Analyze DepthVelocity timing logs produced by State_RLBase.

The controller writes one row per policy step (50 Hz) with:

    policy_step, t_policy_start, t_policy_end, policy_step_ms,
    depth_valid, depth_seq, depth_frame_number, depth_source_stamp,
    depth_rx_time, depth_age_ms, depth_seq_delta,
    lowstate_tick, lowstate_monitor_tick, lowstate_rx_time,
    lowstate_age_ms, lowstate_tick_delta, lowstate_rx_seq

This script reports the timing health of the depth and proprioception
streams and flags the failure modes that matter for sim2real debugging:

  * stale depth   -> depth_seq_delta == 0 for long stretches
  * depth starvation -> depth_valid == 0
  * policy jitter -> irregular t_policy_start spacing
  * slow steps    -> policy_step_ms inflation (inference stalls)
  * proprio lag   -> lowstate_age_ms growth or tick_delta drift

Usage:
    python3 tools/analyze_timing_log.py log/depth_velocity_timing.csv
    python3 tools/analyze_timing_log.py a.csv b.csv          # compare logs
    python3 tools/analyze_timing_log.py log/*.csv --plot     # write PNGs
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("numpy is required: pip install numpy\n")
    raise

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

FLOAT_FIELDS = (
    "t_policy_start",
    "t_policy_end",
    "policy_step_ms",
    "depth_source_stamp",
    "depth_capture_time",
    "depth_rx_time",
    "depth_latency_ms",
    "depth_age_ms",
    "depth_interval_ms",
    "depth_wait_ms",
    "depth_process_ms",
    "depth_filter_ms",
    "lowstate_rx_time",
    "lowstate_age_ms",
)
INT_FIELDS = (
    "policy_step",
    "depth_valid",
    "depth_seq",
    "depth_frame_number",
    "depth_frame_gap",
    "depth_seq_delta",
    "lowstate_tick",
    "lowstate_monitor_tick",
    "lowstate_tick_delta",
    "lowstate_rx_seq",
)

# int64 cannot represent NaN, so missing values are tracked separately.
MISSING_INT = -1


def _to_float(text: str) -> float:
    text = text.strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return math.nan
    return float(text)


def _to_int(text: str) -> int:
    text = text.strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return MISSING_INT
    return int(float(text))


class TimingLog:
    """Parsed timing CSV with numpy columns for fast statistics."""

    def __init__(self, path: Path):
        self.path = path
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: empty file")
            self.fieldnames = [name.strip() for name in reader.fieldnames]
            rows = list(reader)

        self.n = len(rows)
        self.col: dict[str, np.ndarray] = {}
        for name in FLOAT_FIELDS:
            if name in self.fieldnames:
                self.col[name] = np.array([_to_float(r.get(name, "")) for r in rows],
                                          dtype=np.float64)
        for name in INT_FIELDS:
            if name in self.fieldnames:
                self.col[name] = np.array([_to_int(r.get(name, "")) for r in rows],
                                          dtype=np.int64)

        # Convenience derived series.
        self.t = self.col.get("t_policy_start", np.arange(self.n, dtype=np.float64))
        self.t_rel = self.t - self.t[0] if self.n else self.t

    # -- small helpers ------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.col

    def finite(self, name: str) -> np.ndarray:
        """Return the finite (non-NaN) values of a column."""
        if not self.has(name):
            return np.array([], dtype=np.float64)
        values = self.col[name]
        return values[np.isfinite(values)]

    def valid_mask(self, name: str) -> np.ndarray:
        if not self.has(name):
            return np.zeros(self.n, dtype=bool)
        return np.isfinite(self.col[name]) & (self.col[name] != MISSING_INT)


# ---------------------------------------------------------------------------
# Statistics helpers (avoid a hard scipy/pandas dependency)
# ---------------------------------------------------------------------------

def describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def fmt_stat(stats: dict, unit: str = "") -> str:
    if stats.get("n", 0) == 0:
        return "no data"
    return ("mean {:8.3f}  std {:7.3f}  min {:8.3f}  p50 {:8.3f}  "
            "p95 {:8.3f}  p99 {:8.3f}  max {:8.3f} {}").format(
        stats["mean"], stats["std"], stats["min"], stats["p50"],
        stats["p95"], stats["p99"], stats["max"], unit).rstrip()


def mode_of(values: np.ndarray):
    if values.size == 0:
        return None, 0
    counter = Counter(int(v) for v in values if v != MISSING_INT)
    if not counter:
        return None, 0
    value, count = counter.most_common(1)[0]
    return value, count


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def detect_runs(mask: np.ndarray):
    """Return (start, length) for each consecutive True run in mask."""
    runs = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def analyze(log: TimingLog, label: str, stale_threshold: int = 5) -> dict:
    out: dict = {"label": label, "path": str(log.path), "rows": log.n, "warnings": []}
    print("=" * 78)
    print(f"{label}")
    print(f"  file : {log.path}")
    print(f"  rows : {log.n}")
    if log.n < 2:
        print("  not enough rows to analyze")
        return out

    duration = float(log.t[-1] - log.t[0])
    out["duration_s"] = duration
    print(f"  span : {duration:.3f} s")

    # ---- policy loop ----------------------------------------------------
    period_ms = np.diff(log.t) * 1000.0
    step_ms = log.finite("policy_step_ms")
    print("\n[policy loop]")
    print(f"  step period (ms) : {fmt_stat(describe(period_ms))}")
    print(f"  step cost   (ms) : {fmt_stat(describe(step_ms))}")
    out["period_ms"] = describe(period_ms)
    out["step_cost_ms"] = describe(step_ms)

    expected_dt = float(np.median(period_ms)) if period_ms.size else 0.0
    out["policy_hz"] = 1000.0 / expected_dt if expected_dt > 0 else float("nan")
    print(f"  effective rate   : {out['policy_hz']:.3f} Hz (median period {expected_dt:.3f} ms)")

    late = period_ms > expected_dt * 1.5 if period_ms.size else np.zeros(0, dtype=bool)
    if late.any():
        worst = float(np.max(period_ms))
        out["warnings"].append(f"{int(late.sum())} late policy step(s), worst {worst:.1f} ms")
        print(f"  !! {int(late.sum())} late step(s); worst gap {worst:.3f} ms")

    # ---- depth ----------------------------------------------------------
    print("\n[depth]")
    if log.has("depth_seq"):
        seq = log.col["depth_seq"]
        updates = int(np.sum(np.diff(seq) > 0))
        print(f"  buffer updates   : {updates} over {duration:.3f} s "
              f"-> {updates / duration:.2f} Hz")
        out["depth_hz"] = updates / duration if duration > 0 else float("nan")

    if log.has("depth_valid"):
        invalid = int(np.sum(log.col["depth_valid"] == 0))
        print(f"  invalid frames   : {invalid} / {log.n}")
        if invalid:
            out["warnings"].append(f"{invalid} policy step(s) with invalid depth")
        out["depth_invalid"] = invalid

    age = log.finite("depth_age_ms")
    if age.size:
        stats = describe(age)
        out["depth_age_ms"] = stats
        print(f"  depth age   (ms) : {fmt_stat(stats)}")
        # A clock-base mismatch (for example a camera timestamp taken from a
        # process-relative epoch) shows up as an absurd constant offset.
        if abs(stats["p50"]) > 1.0e5:
            out["warnings"].append(
                "depth age is on a different clock base than t_policy_start; "
                "latency is NOT measurable in this log")
        elif stats["max"] > 100.0:
            out["warnings"].append(f"depth age peaked at {stats['max']:.1f} ms")

        # Phase relationship between the depth stream and the policy loop.
        # A tightly locked producer (sim2sim, sim steps on lowcmd) keeps the
        # age nearly constant. Independent clocks (real camera at its own
        # rate) spread the age roughly uniformly over one frame period.
        spread = stats["std"] / stats["mean"] if stats["mean"] > 1e-9 else 0.0
        out["depth_age_cv"] = spread
        if spread < 0.25:
            print("  phase            : locked to the policy loop "
                  "(age ~constant; typical of sim2sim)")
        else:
            print("  phase            : independent of the policy loop "
                  "(age spread over the frame period)")

    if log.has("depth_seq_delta"):
        deltas = log.col["depth_seq_delta"][1:]  # first row has no predecessor
        deltas = deltas[deltas != MISSING_INT]
        zero = int(np.sum(deltas == 0))
        print(f"  reuse rate       : {zero} / {deltas.size} steps reused the previous frame "
              f"({100.0 * zero / max(deltas.size, 1):.1f}%)")
        out["depth_reuse"] = zero
        runs = detect_runs(deltas == 0)
        long_runs = [r for r in runs if r[1] >= stale_threshold]
        if long_runs:
            worst = max(long_runs, key=lambda r: r[1])
            out["warnings"].append(
                f"depth stalled for up to {worst[1]} consecutive steps "
                f"(~{worst[1] * expected_dt:.0f} ms)")
            print(f"  !! {len(long_runs)} stall(s) >= {stale_threshold} steps; "
                  f"longest {worst[1]} steps")

    # ---- depth pipeline stage breakdown ---------------------------------
    latency = log.finite("depth_latency_ms")
    if latency.size:
        stats = describe(latency)
        out["depth_latency_ms"] = stats
        print(f"  capture->write   : {fmt_stat(stats, 'ms')}")
        if stats["p95"] > 50.0:
            out["warnings"].append(
                f"depth capture->write latency p95 {stats['p95']:.1f} ms")

    interval = log.finite("depth_interval_ms")
    interval = interval[interval > 0.0]
    if interval.size:
        stats = describe(interval)
        out["depth_interval_ms"] = stats
        print(f"  update interval  : {fmt_stat(stats, 'ms')}")
        print(f"  -> {1000.0 / stats['mean']:.2f} Hz depth update rate")
        out["depth_update_hz"] = 1000.0 / stats["mean"]
        policy_hz = float(out.get("policy_hz") or 0.0)
        if policy_hz > 0.0 and out["depth_update_hz"] < 0.8 * policy_hz:
            out["warnings"].append(
                f"depth updates at only {out['depth_update_hz']:.1f} Hz vs "
                f"{policy_hz:.1f} Hz policy rate")

    for key, label in (("depth_wait_ms", "wait in pipeline"),
                       ("depth_process_ms", "preprocess total"),
                       ("depth_filter_ms", "  SDK filter part")):
        values = log.finite(key)
        values = values[values > 0.0]
        if values.size:
            out[key] = describe(values)
            print(f"  {label:<17}: {fmt_stat(out[key], 'ms')}")

    if log.has("depth_frame_gap"):
        gap = log.col["depth_frame_gap"]
        gap = gap[gap > 0]
        if gap.size:
            value, count = mode_of(gap)
            print(f"  sensor frames/write: mean {float(np.mean(gap)):.2f}, "
                  f"mode {value} ({count}/{gap.size}), max {int(gap.max())}")
            out["depth_frame_gap"] = float(np.mean(gap))

    if log.has("depth_source_stamp"):
        stamps = log.finite("depth_source_stamp")
        nonzero = int(np.sum(stamps != 0.0)) if stamps.size else 0
        if nonzero == 0:
            print("  source stamp     : all zero (publisher does not set stamp)")

    # ---- proprioception -------------------------------------------------
    print("\n[proprioception]")
    ls_age = log.finite("lowstate_age_ms")
    if ls_age.size:
        stats = describe(ls_age)
        out["lowstate_age_ms"] = stats
        print(f"  lowstate age (ms): {fmt_stat(stats)}")
        if stats["p99"] > 20.0:
            out["warnings"].append(f"proprio age p99 {stats['p99']:.1f} ms")

    if log.has("lowstate_tick_delta"):
        deltas = log.col["lowstate_tick_delta"][1:]
        deltas = deltas[deltas != MISSING_INT]
        value, count = mode_of(deltas)
        if value is not None:
            print(f"  tick delta       : mode {value} ({count}/{deltas.size} steps)")
        if deltas.size:
            rate = float(np.mean(deltas)) / expected_dt * 1000.0 if expected_dt > 0 else 0.0
            out["lowstate_hz"] = rate
            print(f"  implied rate     : {rate:.2f} Hz")
        zeros = int(np.sum(deltas == 0))
        if zeros:
            out["warnings"].append(f"{zeros} step(s) with no new proprioception")

    # The timing monitor and the controller own separate DDS subscriptions to
    # rt/lowstate, so their delivery order can differ by a packet or two. A
    # negative gap means the monitor is ahead, i.e. the reported age is a
    # slight underestimate of the proprioception actually consumed.
    if log.has("lowstate_tick") and log.has("lowstate_monitor_tick"):
        used = log.col["lowstate_tick"]
        seen = log.col["lowstate_monitor_tick"]
        gap = used - seen
        gap = gap[gap != MISSING_INT]
        if gap.size:
            ahead = int(np.sum(gap < 0))
            behind = int(np.sum(gap > 0))
            print(f"  monitor vs used  : exact {int(np.sum(gap == 0))}/{gap.size}, "
                  f"monitor ahead {ahead}/{gap.size}, used ahead {behind}/{gap.size} "
                  f"(range {int(gap.min())}..{int(gap.max())} tick)")
            if ahead:
                out["monitor_ahead"] = ahead
                print(f"    note: monitor ahead by up to {abs(int(gap.min()))} tick; "
                      "the age above is an underestimate by that much")

    # ---- verdict --------------------------------------------------------
    print("\n[verdict]")
    if out["warnings"]:
        for warning in out["warnings"]:
            print(f"  WARN  {warning}")
    else:
        print("  OK    no timing anomalies detected")
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_logs(logs, out_dir: Path) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return []

    written = []
    for log in logs:
        if log.n < 2:
            continue
        stem = log.path.stem
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

        axes[0].plot(log.t_rel, np.diff(log.t, prepend=log.t[0]) * 1000.0, lw=0.8)
        axes[0].set_ylabel("step period [ms]")
        axes[0].set_title(f"{stem}: policy loop")
        axes[0].grid(alpha=0.3)

        if log.has("depth_age_ms"):
            axes[1].plot(log.t_rel, log.col["depth_age_ms"], lw=0.8, color="tab:orange")
        axes[1].set_ylabel("depth age [ms]")
        axes[1].grid(alpha=0.3)

        if log.has("lowstate_age_ms"):
            axes[2].plot(log.t_rel, log.col["lowstate_age_ms"], lw=0.8, color="tab:green")
        axes[2].set_ylabel("proprio age [ms]")
        axes[2].grid(alpha=0.3)

        if log.has("depth_seq_delta"):
            axes[3].step(log.t_rel, log.col["depth_seq_delta"], lw=0.8, color="tab:red")
        axes[3].set_ylabel("depth_seq delta")
        axes[3].set_xlabel("time [s]")
        axes[3].grid(alpha=0.3)

        fig.tight_layout()
        target = out_dir / f"{stem}_timing.png"
        fig.savefig(target, dpi=110)
        plt.close(fig)
        written.append(target)
        print(f"plot written: {target}")
    return written


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze DepthVelocity timing logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("csv", nargs="+", help="timing CSV file(s)")
    parser.add_argument("--plot", action="store_true",
                        help="write PNG plots next to each CSV")
    parser.add_argument("--plot-dir", default=None,
                        help="directory for plots (default: alongside the CSV)")
    parser.add_argument("--stale-threshold", type=int, default=5,
                        help="consecutive reused depth frames that count as a stall")
    args = parser.parse_args()

    logs = []
    for pattern in args.csv:
        for path in sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") \
                else [Path(pattern)]:
            if not path.is_file():
                print(f"skip missing file: {path}", file=sys.stderr)
                continue
            try:
                logs.append(TimingLog(path))
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"failed to parse {path}: {exc}", file=sys.stderr)

    if not logs:
        print("no logs to analyze", file=sys.stderr)
        return 1

    labels = []
    seen_stems: Counter = Counter()
    for log in logs:
        seen_stems[log.path.stem] += 1
        suffix = "" if seen_stems[log.path.stem] == 1 else f"#{seen_stems[log.path.stem]}"
        labels.append(f"{log.path.stem}{suffix}")

    results = [analyze(log, label, args.stale_threshold)
               for log, label in zip(logs, labels)]

    if len(results) > 1:
        print("\n" + "=" * 78)
        print("comparison")
        width = max(14, max(len(r["label"]) for r in results) + 2)
        print(f"{'metric':<22}" + "".join(f"{r['label']:>{width}}" for r in results))
        for key, name in (("policy_hz", "policy Hz"),
                          ("depth_hz", "depth Hz"),
                          ("lowstate_hz", "proprio Hz")):
            row = f"{name:<22}"
            for r in results:
                value = r.get(key)
                row += f"{value:>{width}.2f}" if value is not None else f"{'-':>{width}}"
            print(row)
        for key, name in (("depth_latency_ms", "depth latency mean"),
                          ("depth_age_ms", "depth age mean"),
                          ("lowstate_age_ms", "proprio age mean"),
                          ("step_cost_ms", "step cost mean")):
            row = f"{name:<22}"
            for r in results:
                stats = r.get(key)
                if not stats:
                    row += f"{'-':>{width}}"
                    continue
                row += (f"{stats['mean']:>{width}.3f}" if name.endswith("mean")
                        else f"{stats['p99']:>{width}.3f}")
            print(row)
        print("\n(note: 'depth age' is only meaningful when both timestamps share a clock)")

    if args.plot:
        out_dir = Path(args.plot_dir) if args.plot_dir else None
        for log in logs:
            target_dir = out_dir if out_dir else log.path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            plot_logs([log], target_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
