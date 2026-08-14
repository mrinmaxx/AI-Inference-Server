#!/usr/bin/env bash
#
# Phase 4 sweep: measure how dynamic batching affects throughput and latency.
#
# TWO modes (set MODE=...):
#
#   MODE=batch  (default)  -> results/results.csv
#       Sweep max_batch x max_wait_ms at a FIXED client concurrency. Shows the
#       batching throughput win. batch=1 is the explicit BASELINE. Because the
#       client count is fixed, batching here improves BOTH throughput AND
#       latency (Little's Law: latency ~= clients / throughput).
#
#   MODE=load              -> results/results_load.csv
#       FIX the batch (LOAD_BATCH) and wait (LOAD_WAIT), sweep the client
#       CONCURRENCY. This is what produces the classic latency/throughput
#       TRADEOFF: as offered load pushes throughput up, tail latency rises.
#
# Run:   ./scripts/sweep.sh                 # batch sweep
#        MODE=load ./scripts/sweep.sh       # load sweep (for the tradeoff plot)
# Then:  python3 scripts/make_table.py  &&  python3 scripts/plot.py
#
set -uo pipefail

# =============================== CONFIG (edit me) ============================
# Every knob is overridable from the environment, e.g.:
#   DURATION=30 CONCURRENCY=128 ./scripts/sweep.sh
MODE="${MODE:-batch}"                       # batch | load
ENGINE="${ENGINE:-onnx}"                    # onnx | dummy
MODEL_PATH="${MODEL_PATH:-resnet50.onnx}"   # used only when ENGINE=onnx
PROVIDER="${PROVIDER:-cpu}"                 # cpu | cuda  (onnx execution provider)
DURATION="${DURATION:-20}"                  # measured seconds per config
WARMUP="${WARMUP:-3}"                        # discarded warm-up seconds per config
INPUT_SIZE="${INPUT_SIZE:-150528}"          # 3*224*224 floats per request
INTRA_OP_THREADS="${INTRA_OP_THREADS:-0}"   # ONNX Runtime CPU threads (0=auto)
HOST="${HOST:-localhost}"
BASE_PORT="${BASE_PORT:-50060}"             # each run uses a fresh port (no TIME_WAIT clashes)
RESULTS_SUBDIR="${RESULTS_SUBDIR:-}"        # optional subdir under results/ (e.g. onnx, dummy)

# --- MODE=batch knobs ---
BATCH_SIZES=(${BATCH_SIZES:-1 2 4 8 16 32}) # first entry (1) is the baseline
WAIT_MS=(${WAIT_MS:-1 5 10})                # max_wait_ms values to sweep
CONCURRENCY="${CONCURRENCY:-64}"            # fixed concurrent clients

# --- MODE=load knobs ---
LOAD_BATCH="${LOAD_BATCH:-16}"              # fixed batch while sweeping load
LOAD_WAIT="${LOAD_WAIT:-5}"                 # fixed max_wait_ms while sweeping load
CONCURRENCY_LIST=(${CONCURRENCY_LIST:-1 2 4 8 16 32 64 128})
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER="$ROOT_DIR/build/inference_server"
CLIENT="$ROOT_DIR/build/bench_client"
RESULTS_DIR="$ROOT_DIR/results${RESULTS_SUBDIR:+/$RESULTS_SUBDIR}"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

# ---- sanity checks --------------------------------------------------------
[[ -x "$SERVER" ]] || { echo "ERROR: $SERVER not found. Build first: cmake --build build -j"; exit 1; }
[[ -x "$CLIENT" ]] || { echo "ERROR: $CLIENT not found. Build first: cmake --build build -j"; exit 1; }
if [[ "$ENGINE" == "onnx" ]]; then
  [[ -f "$MODEL_PATH" ]] || MODEL_PATH="$ROOT_DIR/$MODEL_PATH"
  [[ -f "$MODEL_PATH" ]] || { echo "ERROR: ONNX model '$MODEL_PATH' not found. Export it: python3 scripts/export_resnet50.py"; exit 1; }
fi

# ---- robust cleanup: never leave an orphan server behind ------------------
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

PORT="$BASE_PORT"

# run_one <batch> <wait_ms> <concurrency> <csv>  -- start server with the given
# batch/wait, warm it up, run the benchmark at the given concurrency, append one
# CSV row, then stop the server cleanly.
run_one() {
  local batch="$1" wait="$2" conc="$3" csv="$4"
  local addr="0.0.0.0:$PORT" target="$HOST:$PORT"
  local log="$LOG_DIR/server_b${batch}_w${wait}_c${conc}.log"
  local tag="batch=$batch wait=${wait}ms conc=$conc"
  [[ "$batch" == "1" && "$MODE" == "batch" ]] && tag="$tag  [BASELINE]"
  echo ">>> $tag  (port $PORT)"

  "$SERVER" --address="$addr" --engine="$ENGINE" --model-path="$MODEL_PATH" \
    --provider="$PROVIDER" \
    --max-batch="$batch" --max-wait-ms="$wait" --intra-op-threads="$INTRA_OP_THREADS" \
    > "$log" 2>&1 &
  SERVER_PID=$!

  local up=0
  for _ in $(seq 1 240); do                 # up to 60s (model load can be slow)
    if grep -q "listening" "$log" 2>/dev/null; then up=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  server exited during startup:"; sed 's/^/    /' "$log"; break; }
    sleep 0.25
  done
  if [[ "$up" -ne 1 ]]; then
    echo "  ERROR: server did not come up; skipping this config"
    cleanup; PORT=$((PORT+1)); return
  fi

  if [[ "$WARMUP" -gt 0 ]]; then
    "$CLIENT" "$target" "$conc" "$WARMUP" "$INPUT_SIZE" >/dev/null 2>&1
    sleep 0.3   # let in-flight warm-up requests drain (no overlap with the measured run)
  fi

  local out thr ok err p50 p95 p99
  out="$("$CLIENT" "$target" "$conc" "$DURATION" "$INPUT_SIZE" 2>&1)"
  thr=$(awk -F': *' '/Throughput/{print $2}' <<<"$out" | awk '{print $1}')
  ok=$( awk -F': *' '/Successful/{print $2}' <<<"$out")
  err=$(awk -F': *' '/^Errors/{print $2}'   <<<"$out")
  p50=$(awk -F': *' '/p50/{print $2}' <<<"$out" | awk '{print $1}')
  p95=$(awk -F': *' '/p95/{print $2}' <<<"$out" | awk '{print $1}')
  p99=$(awk -F': *' '/p99/{print $2}' <<<"$out" | awk '{print $1}')
  : "${thr:=0}" "${ok:=0}" "${err:=0}"

  echo "$ENGINE,$batch,$wait,$conc,$DURATION,$thr,$p50,$p95,$p99,$ok,$err" >> "$csv"
  printf "    throughput=%-9s p50=%-8s p95=%-8s p99=%-8s (ok=%s err=%s)\n" \
         "$thr" "$p50" "$p95" "$p99" "$ok" "$err"

  cleanup
  wait 2>/dev/null
  PORT=$((PORT+1))
  sleep 0.4
}

HEADER="engine,max_batch,max_wait_ms,concurrency,duration_s,throughput_inf_s,p50_ms,p95_ms,p99_ms,ok,errors"

if [[ "$MODE" == "load" ]]; then
  CSV="$RESULTS_DIR/results_load.csv"
  echo "MODE=load  engine=$ENGINE  batch=$LOAD_BATCH  wait=${LOAD_WAIT}ms  concurrency=[${CONCURRENCY_LIST[*]}]  duration=${DURATION}s"
  echo "writing $CSV"; echo
  echo "$HEADER" > "$CSV"
  for conc in "${CONCURRENCY_LIST[@]}"; do
    run_one "$LOAD_BATCH" "$LOAD_WAIT" "$conc" "$CSV"
  done
else
  CSV="$RESULTS_DIR/results.csv"
  echo "MODE=batch  engine=$ENGINE  batches=[${BATCH_SIZES[*]}]  waits=[${WAIT_MS[*]}]ms  concurrency=$CONCURRENCY  duration=${DURATION}s (+${WARMUP}s warmup)"
  echo "writing $CSV"; echo
  echo "$HEADER" > "$CSV"
  for wait in "${WAIT_MS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
      run_one "$batch" "$wait" "$CONCURRENCY" "$CSV"
    done
  done
fi

echo
echo "Sweep complete -> $CSV"
echo "Next:  python3 scripts/make_table.py    # -> console + results/results_table.md"
echo "       python3 scripts/plot.py          # -> results/*.png"
