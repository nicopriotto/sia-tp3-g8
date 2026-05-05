#!/usr/bin/env bash
# Run one optimizer slice of the EJ2 Phase-3 matrix sweep with seeds in parallel.
#
# Usage from repo root:
#     experiments/ej2/experiments/run_matrix_slice.sh adam
#     experiments/ej2/experiments/run_matrix_slice.sh sgd
#     experiments/ej2/experiments/run_matrix_slice.sh momentum
#
# What it does:
#   - Spawns 3 background processes, one per seed, each restricted to 2 BLAS
#     threads (OMP_NUM_THREADS=2) so they share the 8 CPU cores without
#     thrashing.
#   - Each process runs ONE optimizer's 4x4 sub-matrix (LR x architecture =
#     16 cells) for ONE seed, sequentially. Three processes => 16 runs in
#     parallel wall time.
#   - Waits for all 3 to finish before returning. Logs go to /tmp/matrix_<opt>_seed<N>.log.
#
# After running each slice, the run dirs are persisted in
# results/ej2/comparasion/matrix_lr_opt_arch/runs/. To rebuild the consolidated
# summary + all 3 heatmaps once every slice is done, run:
#     python3 -m experiments.ej2.experiments.matrix_lr_opt_arch --seeds 42 123 2026

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {sgd|momentum|adam}" >&2
    exit 1
fi

OPT="$1"
case "$OPT" in
    sgd|momentum|adam) ;;
    *) echo "Invalid optimizer: $OPT (must be sgd/momentum/adam)" >&2; exit 1 ;;
esac

SEEDS=(42 123 2026)
echo "=== Matrix slice: optimizer=$OPT, seeds=${SEEDS[*]} ==="
echo "=== Start: $(date +%H:%M:%S) ==="

PIDS=()
for SEED in "${SEEDS[@]}"; do
    LOG="/tmp/matrix_${OPT}_seed${SEED}.log"
    echo "  -> seed=$SEED  log=$LOG"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        python3 -m experiments.ej2.experiments.matrix_lr_opt_arch \
            --optimizers "$OPT" --seeds "$SEED" \
            > "$LOG" 2>&1 &
    PIDS+=($!)
done

EXIT=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        EXIT=1
    fi
done

echo "=== End: $(date +%H:%M:%S) ==="

if [ "$EXIT" -ne 0 ]; then
    echo "ERROR: at least one seed run failed. Check /tmp/matrix_${OPT}_seed*.log" >&2
    exit "$EXIT"
fi

echo "OK. Slice $OPT complete. Run dirs in results/ej2/comparasion/matrix_lr_opt_arch/runs/"
echo "When all 3 slices are done, regenerate consolidated outputs with:"
echo "  python3 -m experiments.ej2.experiments.matrix_lr_opt_arch --seeds 42 123 2026"
