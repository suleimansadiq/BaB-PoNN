#!/usr/bin/env bash
set -euo pipefail

###############################################
# Local Robustness Sweep for MNIST LeNet-5 (Posit)
# ---------------------------------------------
# - Compiles posit_bab_verify_lenet5_replay_nolace once
# - Runs idx 0..NUM_IMAGES-1
# - Up to MAX_JOBS in parallel
# - Saves per-idx logs
# - Aggregates STATUS lines into one CSV
###############################################

# ---- Config (edit + reuse) ------------------

# Verifier source and binary
SRC=posit_bab_verify_lenet5_replay_nolace.c
BIN=./posit_bab_verify_lenet5_replay_nolace

# Number of images (0..N-1)
NUM_IMAGES=10000

# Max parallel jobs
MAX_JOBS=35

# Output structure
OUTDIR=local_robustness_lenet555
LOGDIR="$OUTDIR/logs"
SUMMARY="$OUTDIR/local_robustness_lenet5.csv"

# Common verifier args (local robustness)
COMMON_ARGS="--topx 2 \
             --kmax 2 \
	     --roi-heur 2 \
             --bmax 4 \
             --widen 1.0 \
             --verbose 2 \
             --depth 500000000 \
             --timelimit 25000 \
             --rank-fast"

# ---------------------------------------------
#  Environment prep
# ---------------------------------------------

# Large stack for BaB recursion
ulimit -s unlimited || true

# Compile binary (recompile if missing or src newer)
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
  echo "Compiling $SRC -> $BIN"
  gcc -O3 -std=c11 "$SRC" -o "$BIN" \
    -I/usr/local/include -L/usr/local/lib \
    -march=native -flto -fomit-frame-pointer -DNDEBUG \
    -lgmp -lm -pthread -l:softposit.a
  echo "Compile done."
fi

mkdir -p "$LOGDIR"

echo "==============================================="
echo " Local Robustness Sweep (LeNet-5)"
echo " Binary:       $BIN"
echo " Images:       0..$((NUM_IMAGES-1))"
echo " Parallelism:  $MAX_JOBS"
echo " Logs:         $LOGDIR"
echo " Summary CSV:  $SUMMARY"
echo " Args:         $COMMON_ARGS"
echo "==============================================="

# Start global timer
START_TIME=$(date +%s)

# ---------------------------------------------
#  Launch jobs
# ---------------------------------------------
for idx in $(seq 0 $((NUM_IMAGES-1))); do
  # Throttle to MAX_JOBS background jobs
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 0.5
  done

  (
    ulimit -s unlimited || true
    "$BIN" --idx "$idx" $COMMON_ARGS > "$LOGDIR/run_${idx}.log" 2>&1
  ) &
done

wait
echo "All verification runs finished. Parsing logs…"

# ---------------------------------------------
#  Parse STATUS → CSV (including preds + patch)
# ---------------------------------------------
NUM_IMAGES_ENV="$NUM_IMAGES"

python3 << 'EOF'
import csv, re, pathlib, os

outdir = pathlib.Path("local_robustness_lenet5")
logdir = outdir / "logs"
summary_path = outdir / "local_robustness_lenet5.csv"

num_images = int(os.environ.get("NUM_IMAGES_ENV", "10000"))

clean_re   = re.compile(r'clean prediction:\s*(\d+)\s+\(true label\s+(\d+)\)')
newpred_re = re.compile(r'new prediction:\s*(\d+)\s*\(true')
status_tag = "STATUS:"

def parse_log(idx, lines):
    row = {
        'idx': idx,
        'clean_pred': '',
        'true_label': '',
        'new_pred': '',
        'outcome': '',
        'best_upper_margin': '',
        'elapsed_s': '',
        'pixels_changed': '',
        'total_bit_flips': '',
        'avg_hamming_per_pixel': '',
        'patch_rows': '',
        'patch_cols': '',
        'note': '',
    }

    status_line = None

    for line in lines:
        # clean prediction + true label
        m = clean_re.search(line)
        if m:
            row['clean_pred']  = m.group(1)
            row['true_label']  = m.group(2)

        # new prediction (when a counterexample is found)
        m = newpred_re.search(line)
        if m:
            row['new_pred'] = m.group(1)

        # STATUS line
        if line.startswith(status_tag):
            status_line = line.strip()

    if status_line is None:
        return row

    parts = [p.strip() for p in status_line.split('|')]

    # First token: "STATUS: X"
    first = parts[0]
    if first.startswith(status_tag):
        first = first[len(status_tag):].strip()
    row['outcome'] = first

    for p in parts[1:]:
        if '=' not in p:
            continue
        key, val = [x.strip() for x in p.split('=', 1)]
        if key == 'best_upper_margin':
            row['best_upper_margin'] = val
        elif key == 'elapsed':
            row['elapsed_s'] = val.rstrip('s')
        elif key == 'pixels_changed':
            row['pixels_changed'] = val
        elif key == 'total_bit_flips':
            row['total_bit_flips'] = val
        elif key == 'avg_hamming_per_pixel':
            row['avg_hamming_per_pixel'] = val
        elif key == 'patch_rows':
            row['patch_rows'] = val
        elif key == 'patch_cols':
            row['patch_cols'] = val
        elif key == 'note':
            row['note'] = val

    return row

rows = []
for idx in range(num_images):
    log_file = logdir / f"run_{idx}.log"
    if not log_file.exists():
        rows.append({'idx': idx})
        continue
    lines = log_file.read_text().splitlines()
    rows.append(parse_log(idx, lines))

fieldnames = [
    'idx',
    'clean_pred',
    'true_label',
    'new_pred',
    'outcome',
    'best_upper_margin',
    'elapsed_s',
    'pixels_changed',
    'total_bit_flips',
    'avg_hamming_per_pixel',
    'patch_rows',
    'patch_cols',
    'note'
]

with summary_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, '')
        w.writerow(r)

EOF

echo "Done. Summary CSV: $SUMMARY"

# ---------------------------------------------
#  Total runtime logging
# ---------------------------------------------
END_TIME=$(date +%s)
TOTAL_SEC=$((END_TIME - START_TIME))

# Format as H:M:S
H=$((TOTAL_SEC/3600))
M=$(( (TOTAL_SEC%3600) / 60 ))
S=$((TOTAL_SEC%60))

TOTAL_FMT=$(printf "%02d:%02d:%02d" "$H" "$M" "$S")

echo "==============================================="
echo " Local Robustness Sweep Complete (LeNet-5)"
echo " Total Runtime: ${TOTAL_SEC}s (${TOTAL_FMT})"
echo "==============================================="

RUNTIME_LOG="$OUTDIR/total_runtime.log"
{
  echo "Local Robustness Sweep Complete (LeNet-5)"
  echo "Total Runtime Seconds: $TOTAL_SEC"
  echo "Total Runtime (H:M:S): $TOTAL_FMT"
  echo "Finished: $(date)"
} > "$RUNTIME_LOG"

echo "Runtime log saved to: $RUNTIME_LOG"

