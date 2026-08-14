#!/usr/bin/env python3
"""Turn results/results.csv into a clean summary table.

Prints the table to the console AND writes results/results_table.md.
Dependency-light: standard-library csv only (no pandas/numpy needed).

Run:  python3 scripts/make_table.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Optional RESULTS_SUBDIR (e.g. "onnx", "dummy") selects results/<sub>/.
SUB = os.environ.get("RESULTS_SUBDIR", "")
RESULTS_DIR = os.path.join(ROOT, "results", SUB) if SUB else os.path.join(ROOT, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")
MD_PATH = os.path.join(RESULTS_DIR, "results_table.md")


def fmt(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"{CSV_PATH} not found -- run scripts/sweep.sh first.")

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("results.csv has no data rows.")

    # Baseline = the batch=1 rows (batching effectively off). Average their
    # throughput so the speedup column has a stable reference.
    base_vals = [float(r["throughput_inf_s"]) for r in rows if r["max_batch"] == "1"
                 and r["throughput_inf_s"] not in ("", "0")]
    baseline = sum(base_vals) / len(base_vals) if base_vals else None

    headers = ["Config", "Throughput (inf/s)", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Speedup"]
    table = []
    for r in sorted(rows, key=lambda r: (int(r["max_wait_ms"]), int(r["max_batch"]))):
        b, w = r["max_batch"], r["max_wait_ms"]
        cfg = f"batch={b}, wait={w}ms" + ("  (baseline)" if b == "1" else "")
        thr = r["throughput_inf_s"]
        speedup = "-"
        if baseline and thr not in ("", "0"):
            speedup = f"{float(thr)/baseline:.2f}x"
        table.append([cfg, fmt(thr), fmt(r["p50_ms"]), fmt(r["p95_ms"]),
                      fmt(r["p99_ms"]), speedup])

    # ---- render markdown ----
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    def line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    md = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    md += [line(row) for row in table]

    engine = rows[0]["engine"]
    conc = rows[0]["concurrency"]
    dur = rows[0]["duration_s"]
    caption = f"Engine: **{engine}** | concurrency: {conc} clients | {dur}s per config"

    best = max((float(r["throughput_inf_s"]) for r in rows
                if r["throughput_inf_s"] not in ("", "0")), default=0.0)
    summary = ""
    if baseline and best:
        summary = (f"\nBaseline (batch=1): **{baseline:.1f} inf/s**  |  "
                   f"Best: **{best:.1f} inf/s**  |  "
                   f"Peak gain: **{best/baseline:.2f}x**\n")

    out = caption + "\n\n" + "\n".join(md) + "\n" + summary
    print(out)
    with open(MD_PATH, "w") as f:
        f.write(out)
    print(f"\nwrote {MD_PATH}")


if __name__ == "__main__":
    main()
