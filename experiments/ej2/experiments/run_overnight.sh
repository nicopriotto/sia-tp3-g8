#!/usr/bin/env bash
# Overnight runner for EJ2 Phase-3 matrix sweep.
#
# Encadena las 3 slices (adam, momentum, sgd) + consolidación final + resumen
# de los mejores cells. Es seguro lanzarlo MIENTRAS la slice de adam ya está
# corriendo: el script primero espera a que cualquier proceso de matrix
# preexistente termine, después arranca lo que falte.
#
# Cada llamada interna a run_matrix_slice.sh saltea automáticamente los runs
# que ya estén completos, así que es resumable: si por algún motivo se
# interrumpe, podés relanzarlo y continúa donde quedó.
#
# Usage from anywhere:
#     bash experiments/ej2/experiments/run_overnight.sh
#
# Salida principal:
#     /tmp/overnight_ej2.log
#     results/ej2/comparasion/matrix_lr_opt_arch/{summary.csv, aggregate_summary.csv, report.md}
#     results/ej2/comparasion/matrix_lr_opt_arch/heatmaps/{adam,momentum,sgd}__{val_macro_f1,best_val_macro_f1,peak_epoch}.png
#     results/ej2/comparasion/matrix_lr_opt_arch/curves/{adam,momentum,sgd}__curves_by_arch.png

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

LOG="/tmp/overnight_ej2.log"
: > "$LOG"

step() {
    local msg="$*"
    echo "" | tee -a "$LOG"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $msg ===" | tee -a "$LOG"
}

step "overnight ej2 START"
echo "Working dir: $(pwd)" | tee -a "$LOG"
echo "Log file:    $LOG" | tee -a "$LOG"

# 1) Wait for any currently-running matrix process to finish (e.g. adam slice ya corriendo).
if pgrep -f "experiments\.ej2\.experiments\.matrix_lr_opt_arch.*--optimizers" >/dev/null; then
    step "Detected matrix process already running — waiting..."
    n_iter=0
    while pgrep -f "experiments\.ej2\.experiments\.matrix_lr_opt_arch.*--optimizers" >/dev/null; do
        n_running=$(pgrep -fc "experiments\.ej2\.experiments\.matrix_lr_opt_arch.*--optimizers" || true)
        if (( n_iter % 5 == 0 )); then
            echo "[$(date '+%H:%M:%S')] still $n_running process(es) running, sleeping 60s..." | tee -a "$LOG"
        fi
        n_iter=$((n_iter + 1))
        sleep 60
    done
    step "Pre-existing matrix processes done"
fi

# 2) Slice runner por optimizer (cada uno usa run_matrix_slice.sh que ya hace skip-if-complete)
for opt in adam momentum sgd; do
    step "running slice: $opt"
    t0=$(date +%s)
    if experiments/ej2/experiments/run_matrix_slice.sh "$opt" 2>&1 | tee -a "$LOG"; then
        elapsed=$(( $(date +%s) - t0 ))
        step "slice $opt OK · ${elapsed}s"
    else
        elapsed=$(( $(date +%s) - t0 ))
        step "slice $opt FAILED · ${elapsed}s · continuando con la siguiente"
    fi
done

# 3) Consolidación final (regenera summary.csv, aggregate, heatmaps, curves, report.md
#    leyendo todos los run dirs ya completos)
step "consolidation"
if python3 -m experiments.ej2.experiments.matrix_lr_opt_arch --seeds 42 123 2026 2>&1 | tee -a "$LOG"; then
    step "consolidation OK"
else
    step "consolidation FAILED — revisar log"
fi

# 4) Resumen al final del log: top-10 cells y conteo de archivos generados
step "SUMMARY"
{
    OUT_DIR=results/ej2/comparasion/matrix_lr_opt_arch
    echo "Output dir: $OUT_DIR"
    echo ""
    echo "Heatmaps generados:"
    ls -1 "$OUT_DIR/heatmaps/" 2>/dev/null | sed 's/^/  /'
    echo ""
    echo "Curves generadas:"
    ls -1 "$OUT_DIR/curves/" 2>/dev/null | sed 's/^/  /'
    echo ""
    echo "Run dirs completos: $(ls -d $OUT_DIR/runs/*__seed* 2>/dev/null | wc -l) / 270 esperados"
    echo ""
    echo "Top 10 cells por val_macro_f1_mean:"
    python3 -c "
import csv
agg = list(csv.DictReader(open('$OUT_DIR/aggregate_summary.csv')))
agg.sort(key=lambda r: -float(r.get('val_macro_f1_mean') or 0))
print(f\"  {'variant':55s} {'f1_mean':>9s} {'f1_std':>8s} {'best_mean':>10s}\")
print('  ' + '-' * 86)
for r in agg[:10]:
    print(f\"  {r['variant']:55s} {float(r['val_macro_f1_mean']):>9.4f} {float(r['val_macro_f1_std']):>8.4f} {float(r['best_val_macro_f1_mean']):>10.4f}\")
" 2>&1
} | tee -a "$LOG"

step "overnight ej2 END"
