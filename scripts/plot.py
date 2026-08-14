#!/usr/bin/env python3
"""Generate the Phase-4 plots.

  Plot 1  throughput_vs_batch.png   (from results/results.csv)
          x = max_batch (log2), y = throughput. One line per max_wait_ms.
          -> RISES then PLATEAUS.

  Plot 2  latency_vs_throughput.png (the latency/throughput tradeoff)
          x = throughput, y = p99 latency.
          * If results/results_load.csv exists (a LOAD sweep: fixed batch,
            varying concurrency) it is used -> the classic UP-slope: pushing
            offered load raises throughput AND tail latency. Points labelled
            by concurrency.
          * Otherwise it falls back to results/results.csv (the BATCH sweep at
            fixed concurrency) -> a Pareto curve where batching lowers latency
            while raising throughput. Points labelled by batch size.

Dependency-light: matplotlib + standard-library csv (no pandas needed).
Run:  python3 scripts/plot.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Optional RESULTS_SUBDIR (e.g. "onnx", "dummy") selects results/<sub>/.
SUB = os.environ.get("RESULTS_SUBDIR", "")
RES = os.path.join(ROOT, "results", SUB) if SUB else os.path.join(ROOT, "results")
CSV_BATCH = os.path.join(RES, "results.csv")
CSV_LOAD = os.path.join(RES, "results_load.csv")


def read(path):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["throughput_inf_s"] not in ("", "0")]


def plot_throughput_vs_batch():
    if not os.path.exists(CSV_BATCH):
        print(f"skip Plot 1: {CSV_BATCH} not found (run the batch sweep).")
        return
    rows = read(CSV_BATCH)
    by_wait = defaultdict(list)
    for r in rows:
        by_wait[int(r["max_wait_ms"])].append(
            (int(r["max_batch"]), float(r["throughput_inf_s"])))
    for w in by_wait:
        by_wait[w].sort()

    plt.figure(figsize=(7, 4.5))
    for w in sorted(by_wait):
        xs = [b for b, _ in by_wait[w]]
        ys = [t for _, t in by_wait[w]]
        plt.plot(xs, ys, marker="o", label=f"max_wait={w}ms")
    plt.xscale("log", base=2)
    batches = sorted({b for pts in by_wait.values() for b, _ in pts})
    plt.xticks(batches, [str(b) for b in batches])
    plt.xlabel("max_batch_size")
    plt.ylabel("Throughput (inferences/sec)")
    plt.title("Throughput vs. batch size")
    plt.ylim(bottom=0)  # absolute scale: a flat line looks flat, a rise looks like a rise
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(RES, "throughput_vs_batch.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print("wrote", out)


def plot_latency_vs_throughput():
    if os.path.exists(CSV_LOAD):
        rows = read(CSV_LOAD)
        label_key, label_fmt = "concurrency", lambda v: f"c={v}"
        title = "Latency vs. throughput tradeoff (load sweep)"
    elif os.path.exists(CSV_BATCH):
        rows = read(CSV_BATCH)
        label_key, label_fmt = "max_batch", lambda v: f"b={v}"
        title = "Latency vs. throughput (batch sweep, fixed load)"
    else:
        print("skip Plot 2: no results CSV found.")
        return

    # Connect points in SWEEP order (by the varied knob: concurrency, or batch
    # in the fallback), NOT by throughput. When throughput is saturated (e.g.
    # compute-bound CPU) the x-values barely move, so ordering by throughput
    # would scramble the line; sweep order makes it trace the real experiment.
    pts = sorted(((float(r["throughput_inf_s"]),
                   float(r["p99_ms"]) if r["p99_ms"] else float("nan"),
                   r[label_key]) for r in rows), key=lambda t: int(t[2]))

    plt.figure(figsize=(7, 4.5))
    plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
    for thr, p99, lab in pts:
        plt.annotate(label_fmt(lab), (thr, p99),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Throughput (inferences/sec)")
    plt.ylabel("p99 latency (ms)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RES, "latency_vs_throughput.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print("wrote", out)


def main():
    if not os.path.exists(CSV_BATCH) and not os.path.exists(CSV_LOAD):
        raise SystemExit("No results found -- run scripts/sweep.sh first.")
    plot_throughput_vs_batch()
    plot_latency_vs_throughput()


if __name__ == "__main__":
    main()
