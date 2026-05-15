#!/usr/bin/env python3
"""Plot scalability sweep results into publication-ready PNG figures.

Reads the gpu_metrics_*gpu_<jobid>.csv files and the corresponding
scalability_<jobid>.out training logs, emits four figures into
results/plots/:

  1. scaling_efficiency.png   — throughput vs GPU count + ideal line
  2. gpu_util_timeseries.png  — util % over training time per GPU count
  3. memory_profile.png       — peak / avg memory per GPU count
  4. power_profile.png        — power draw time series per GPU count

Usage:
  python plot_scaling_results.py [--results-dir DIR] [--logs-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_jobid_from_filename(path: Path) -> str | None:
    """gpu_metrics_2gpu_40047373.csv -> '40047373'."""
    m = re.search(r"_(\d{8})\.csv$", path.name)
    return m.group(1) if m else None


def parse_gpu_count_from_filename(path: Path) -> int | None:
    """gpu_metrics_2gpu_40047373.csv -> 2."""
    m = re.search(r"_(\d+)gpu_", path.name)
    return int(m.group(1)) if m else None


def extract_throughput_from_log(log_path: Path) -> float | None:
    """Parse 'Step N: loss=...' lines from training log and infer steps/sec.

    Uses Step 60 → Step 150 (skip first 50 warmup) interval for steady-state.
    Returns seconds-per-step, or None if log not parseable.
    """
    if not log_path.is_file():
        return None
    times: dict[int, float] = {}
    pat = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] .* Step (\d+):")
    for line in log_path.read_text().splitlines():
        m = pat.match(line)
        if not m:
            continue
        ts = pd.to_datetime(m.group(1))
        step = int(m.group(2))
        times[step] = ts.timestamp()
    if 60 not in times or 150 not in times:
        return None
    return (times[150] - times[60]) / (150 - 60)


def parse_rank_bs_from_generated(jobid: str, gpus: int, gen_dir: Path) -> tuple[int | None, int | None]:
    """Look up rank and batch_size from _generated/scal_lora_r{R}_{N}gpu_{JID}.yaml."""
    for p in gen_dir.glob(f"scal_lora_r*_{gpus}gpu_{jobid}.yaml"):
        m = re.search(r"scal_lora_r(\d+)_", p.name)
        if not m:
            continue
        rank = int(m.group(1))
        bs = None
        for line in p.read_text().splitlines():
            mm = re.match(r"\s*batch_size:\s*(\d+)", line)
            if mm:
                bs = int(mm.group(1))
                break
        return rank, bs
    return None, None


def load_runs(results_dir: Path, logs_dir: Path) -> list[dict]:
    """Discover gpu_metrics_*.csv files and pair with logs + config metadata."""
    runs = []
    gen_dir = results_dir.parent / "_generated"
    for csv_path in sorted(results_dir.glob("gpu_metrics_*.csv")):
        jobid = parse_jobid_from_filename(csv_path)
        gpus = parse_gpu_count_from_filename(csv_path)
        if jobid is None or gpus is None:
            continue
        log_path = logs_dir / f"scalability_{jobid}.out"
        sec_per_step = extract_throughput_from_log(log_path)
        rank, bs = parse_rank_bs_from_generated(jobid, gpus, gen_dir)
        df = pd.read_csv(csv_path)
        for col in ("gpu_util_pct", "mem_used_mib", "power_w"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        runs.append({
            "jobid": jobid,
            "gpus": gpus,
            "rank": rank,
            "bs": bs,
            "csv": csv_path,
            "log": log_path,
            "df": df,
            "sec_per_step": sec_per_step,
            "samples_per_sec": (gpus * bs / sec_per_step) if (sec_per_step and bs) else None,
        })
    return runs


def plot_scaling_efficiency(runs: list[dict], out_dir: Path) -> None:
    """One curve per LoRA rank. Speedup = samples/sec vs 1-GPU baseline of same rank.

    DDP with fixed per-GPU batch size: effective batch scales linearly with
    GPU count, so the correct throughput metric is samples/sec, not steps/sec.
    """
    valid = [r for r in runs if r.get("samples_per_sec") and r.get("rank")]
    if not valid:
        print("scaling_efficiency: no runs with parseable throughput + rank; skipping")
        return
    by_rank: dict[int, list[dict]] = {}
    for r in valid:
        by_rank.setdefault(r["rank"], []).append(r)
    for r_list in by_rank.values():
        r_list.sort(key=lambda x: x["gpus"])

    # Two-panel: left = speedup curves (clean, legend-only labels);
    #            right = efficiency bars (per-config % of ideal).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})
    colors = {8: "#2E86C1", 16: "#28B463", 32: "#CA6F1E", 64: "#7D3C98"}
    max_gpu = max((r["gpus"] for r in valid), default=4)

    # ---- left: speedup curves ----
    for rank in sorted(by_rank):
        runs_r = by_rank[rank]
        base = next((r["samples_per_sec"] for r in runs_r if r["gpus"] == 1), None)
        if not base:
            continue
        xs = [r["gpus"] for r in runs_r]
        speedup = [r["samples_per_sec"] / base for r in runs_r]
        bs = runs_r[0]["bs"]
        ax1.plot(xs, speedup, "o-", lw=2.2, ms=10,
                 color=colors.get(rank, "#555"),
                 label=f"rank={rank}, bs={bs}")
    ax1.plot([1, max_gpu], [1, max_gpu], "k--", lw=1.2, alpha=0.5,
             label="Ideal (linear)")
    ax1.set_xlabel("GPU count")
    ax1.set_ylabel("Throughput speedup\n(samples/sec, vs 1-GPU of same rank)")
    ax1.set_title("DDP speedup curves")
    ax1.set_xticks(sorted({r["gpus"] for r in valid}))
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ---- right: efficiency bars ----
    bar_rows = []
    for rank in sorted(by_rank):
        runs_r = by_rank[rank]
        base = next((r["samples_per_sec"] for r in runs_r if r["gpus"] == 1), None)
        if not base:
            continue
        for r in runs_r:
            bar_rows.append({
                "label": f"{r['gpus']}G r={rank} bs={r['bs']}",
                "rank": rank,
                "gpus": r["gpus"],
                "efficiency": (r["samples_per_sec"] / base) / r["gpus"] * 100,
            })
    bar_rows.sort(key=lambda x: (x["rank"], x["gpus"]))
    positions = list(range(len(bar_rows)))
    bar_colors = [colors.get(b["rank"], "#555") for b in bar_rows]
    bars = ax2.bar(positions, [b["efficiency"] for b in bar_rows],
                   color=bar_colors, alpha=0.85)
    for bar, b in zip(bars, bar_rows):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{b['efficiency']:.0f}%", ha="center", fontsize=9)
    ax2.axhline(100, color="k", ls=":", lw=1, alpha=0.6, label="Ideal (100 %)")
    ax2.axhline(80, color="k", ls=":", lw=0.8, alpha=0.35,
                label="80 % target")
    ax2.set_xticks(positions)
    ax2.set_xticklabels([b["label"] for b in bar_rows],
                        rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Scaling efficiency (% of ideal)")
    ax2.set_title("Per-config DDP scaling efficiency")
    ax2.set_ylim(0, max(120, max(b["efficiency"] for b in bar_rows) + 8))
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("FLUX.1-Kontext LoRA — DDP scaling on A100 80GB\n"
                 "(per-GPU batch fixed → effective batch scales linearly with GPU count)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out = out_dir / "scaling_efficiency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def _config_label(r: dict) -> str:
    """Human label `{N}G r={rank} bs={bs}` for legend / xtick use."""
    rank = r.get("rank")
    bs = r.get("bs")
    rank_s = f"r={rank}" if rank is not None else "r=?"
    bs_s = f"bs={bs}" if bs is not None else "bs=?"
    return f"{r['gpus']}G {rank_s} {bs_s}"


def plot_util_timeseries(runs: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    # Sort by (rank, gpus, bs) so legend order is predictable; same colour per rank.
    rank_color = {8: "#2E86C1", 16: "#28B463", 32: "#CA6F1E", 64: "#7D3C98"}
    gpu_dash = {1: "-", 2: "--", 4: ":"}
    runs_sorted = sorted(
        runs,
        key=lambda r: (r.get("rank") or 99, r.get("bs") or 9, r["gpus"]),
    )
    for r in runs_sorted:
        df = r["df"]
        per_t = df.groupby("timestamp")["gpu_util_pct"].mean().reset_index()
        per_t["elapsed_s"] = (per_t["timestamp"] - per_t["timestamp"].iloc[0]).dt.total_seconds()
        per_t["util_smooth"] = per_t["gpu_util_pct"].rolling(15, min_periods=1).mean()
        ax.plot(per_t["elapsed_s"], per_t["util_smooth"],
                lw=1.6, alpha=0.9,
                color=rank_color.get(r.get("rank"), "#555"),
                linestyle=gpu_dash.get(r["gpus"], "-"),
                label=_config_label(r))
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("GPU utilisation (%, 15-s rolling mean)")
    ax.set_title("GPU utilisation over training run — FLUX.1-Kontext LoRA\n"
                 "(colour = LoRA rank; line style = GPU count: solid=1G, dashed=2G, dotted=4G)")
    ax.axhline(80, color="k", ls=":", alpha=0.4, lw=1)
    ax.text(0.99, 80.5, "80 % target", ha="right", fontsize=8, alpha=0.6,
            transform=ax.get_yaxis_transform())
    ax.set_ylim(0, 105)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, framealpha=0.9, borderaxespad=0.)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "gpu_util_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_memory_profile(runs: list[dict], out_dir: Path) -> None:
    rows = []
    runs_sorted = sorted(
        runs,
        key=lambda r: (r.get("rank") or 99, r.get("bs") or 9, r["gpus"]),
    )
    for r in runs_sorted:
        df = r["df"]
        steady = df.iloc[60 * r["gpus"]:]   # 1Hz × N GPUs ≈ 60 s warmup
        mem_peak = steady.groupby("gpu_id")["mem_used_mib"].max().mean() / 1024
        mem_avg = steady.groupby("gpu_id")["mem_used_mib"].mean().mean() / 1024
        rows.append({
            "label": _config_label(r),
            "rank": r.get("rank"),
            "peak_gb": mem_peak,
            "avg_gb": mem_avg,
        })
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(rows)))
    width = 0.36
    ax.bar([i - width/2 for i in x], [r["peak_gb"] for r in rows], width,
           label="Peak", color="#cf3030")
    ax.bar([i + width/2 for i in x], [r["avg_gb"] for r in rows], width,
           label="Avg",  color="#3070cf")
    ax.axhline(80, color="k", ls=":", alpha=0.4, lw=1, label="A100 80GB cap")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=30, ha="right",
                       fontsize=9)
    # Light vertical separators between rank groups for readability
    seen_ranks = []
    for i, r in enumerate(rows):
        if r["rank"] not in seen_ranks and seen_ranks:
            ax.axvline(i - 0.5, color="k", alpha=0.1, lw=1)
        if r["rank"] not in seen_ranks:
            seen_ranks.append(r["rank"])
    ax.set_ylabel("GPU memory per GPU (GiB)")
    ax.set_title("Per-GPU memory profile (steady-state, post-warmup) — "
                 "FLUX.1-Kontext LoRA on A100 80 GB")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "memory_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_power_profile(runs: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    rank_color = {8: "#2E86C1", 16: "#28B463", 32: "#CA6F1E", 64: "#7D3C98"}
    gpu_dash = {1: "-", 2: "--", 4: ":"}
    runs_sorted = sorted(
        runs,
        key=lambda r: (r.get("rank") or 99, r.get("bs") or 9, r["gpus"]),
    )
    for r in runs_sorted:
        df = r["df"]
        # Per-GPU mean rather than sum, to make ranks comparable independent of N
        per_t = df.groupby("timestamp")["power_w"].mean().reset_index()
        per_t["elapsed_s"] = (per_t["timestamp"] - per_t["timestamp"].iloc[0]).dt.total_seconds()
        per_t["pwr_smooth"] = per_t["power_w"].rolling(15, min_periods=1).mean()
        ax.plot(per_t["elapsed_s"], per_t["pwr_smooth"],
                lw=1.6, alpha=0.9,
                color=rank_color.get(r.get("rank"), "#555"),
                linestyle=gpu_dash.get(r["gpus"], "-"),
                label=_config_label(r))
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Power draw per GPU (W, 15-s rolling mean)")
    ax.set_title("Per-GPU power consumption over training run — FLUX.1-Kontext LoRA\n"
                 "(colour = LoRA rank; line style = GPU count: solid=1G, dashed=2G, dotted=4G)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, framealpha=0.9, borderaxespad=0.)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "power_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    ap.add_argument("--logs-dir", type=Path, default=SCRIPT_DIR / "logs")
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results" / "plots")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results_dir, args.logs_dir)
    if not runs:
        print(f"no gpu_metrics_*.csv found in {args.results_dir}")
        return 1
    print(f"found {len(runs)} runs: " + ", ".join(f"{r['gpus']}gpu({r['jobid']})" for r in runs))

    plot_scaling_efficiency(runs, args.out_dir)
    plot_util_timeseries(runs, args.out_dir)
    plot_memory_profile(runs, args.out_dir)
    plot_power_profile(runs, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
