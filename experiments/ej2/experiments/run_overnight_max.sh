#!/usr/bin/env bash
# Maximum-throughput overnight runner for EJ2 Phase-3 matrix sweep.
#
# Diferencia vs run_overnight.sh:
#   - Lanza los 9 jobs (3 opts × 3 seeds) EN PARALELO con OMP_NUM_THREADS=1
#     en lugar de slice-by-slice. Aprovecha los 8 cores con 1 hilo BLAS por
#     proceso (oversub 9/8 = 12% que el scheduler absorbe sin problema).
#   - Permite override del nro de epochs vía 1er argumento (default 800).
#   - Genera un base-config temporal con epochs override; el original
#     experiments/ej2/config_base.json no se toca.
#
# Presupuesto observado: con E=800, el adam-seed job (bottleneck) tarda
# ~6-7h wall en este equipo (i7-7700, 8 cores). Subir a E=1000 cae cerca
# del límite de 8h. Bajar a E=500 deja todo en ~4-5h.
#
# Resumable: si el script se interrumpe, los runs ya completados se
# saltean al relanzar (matrix_lr_opt_arch chequea history.json/evaluation.json).
#
# Usage from anywhere:
#     bash experiments/ej2/experiments/run_overnight_max.sh             # E=800, N=8
#     bash experiments/ej2/experiments/run_overnight_max.sh 600         # E=600, N=8
#     bash experiments/ej2/experiments/run_overnight_max.sh 800 6       # E=800, N=6 parallel
#
# Salida principal:
#     /tmp/overnight_ej2_max.log                              (log agregado)
#     /tmp/matrix_max_<opt>_seed<N>.log                       (1 por job)
#     results/ej2/comparasion/matrix_lr_opt_arch/             (artefactos finales)
#
# Limpieza al final:
#     /tmp/ej2_base_overnight.json se borra al cerrar.

# Note: NO usamos `set -u` aca porque en bash, ${#assoc_array[@]} sobre un
# associative array recien declarado dispara "unbound variable" antes del
# primer assignment, lo cual rompe nuestro patron de semaforo. pipefail si
# nos sirve para detectar fallos en pipes.
set -o pipefail

EPOCHS="${1:-800}"
N_PARALLEL="${2:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

LOG="/tmp/overnight_ej2_max.log"
: > "$LOG"

step() {
    local msg="$*"
    echo "" | tee -a "$LOG"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $msg ===" | tee -a "$LOG"
}

step "overnight ej2 MAX START · EPOCHS=$EPOCHS · N_PARALLEL=$N_PARALLEL"
echo "Working dir: $(pwd)"      | tee -a "$LOG"
echo "Log file:    $LOG"        | tee -a "$LOG"
echo "Per-job logs: /tmp/matrix_max_<opt>_seed<N>.log" | tee -a "$LOG"

# 0) Sanity: el config_base no debe tener marcadores de merge ni epochs raro.
if grep -q '<<<<<<<\|=======\|>>>>>>>' experiments/ej2/config_base.json; then
    echo "ERROR: experiments/ej2/config_base.json todavia tiene marcadores de merge." | tee -a "$LOG"
    exit 1
fi

# 1) Generar base-config temporal con el epochs override.
TMP_BASE="/tmp/ej2_base_overnight.json"
trap 'rm -f "$TMP_BASE"' EXIT
python3 - <<EOF
import json, sys
p = "experiments/ej2/config_base.json"
with open(p) as f:
    cfg = json.load(f)
cfg["epochs"] = int($EPOCHS)
with open("$TMP_BASE", "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote $TMP_BASE with epochs={cfg['epochs']}")
EOF
echo "base-config temporal: $TMP_BASE (epochs=$EPOCHS)" | tee -a "$LOG"

# 2) Esperar a que termine cualquier matrix process pre-existente.
if pgrep -f "experiments\.ej2\.experiments\.matrix_lr_opt_arch.*--optimizers" >/dev/null; then
    step "Detected matrix process already running — waiting..."
    n_iter=0
    while pgrep -f "experiments\.ej2\.experiments\.matrix_lr_opt_arch.*--optimizers" >/dev/null; do
        if (( n_iter % 5 == 0 )); then
            n=$(pgrep -fc "experiments\.ej2\.experiments\.matrix_lr_opt_arch.*--optimizers" || true)
            echo "[$(date '+%H:%M:%S')] still $n process(es), sleeping 60s..." | tee -a "$LOG"
        fi
        n_iter=$((n_iter + 1))
        sleep 60
    done
fi

# 3) Lanzar los 9 jobs (opt × seed) en paralelo, max N_PARALLEL a la vez,
#    pinneando cada worker a un core fisico distinto via taskset.
#
# Razon: el scheduler de Linux por defecto puede tirar 2 workers en HT
# siblings del mismo core fisico (vimos casos donde CPU 3 y CPU 7 — ambos
# del core 3 — corrian un worker cada uno mientras core 0 quedaba idle).
# Eso degrada throughput ~3x para nuestro workload compute-bound.
# Solucion: detectar la primary CPU de cada core fisico y pinear con
# taskset, una primary distinta por worker.
SEEDS=(42 123 2026)
OPTS=(adam momentum sgd)

# Detectar primary CPU de cada core fisico (1 CPU per physical core).
PRIMARY_CPUS=()
declare -A SEEN_CORE
while IFS=, read -r cpu core socket rest; do
    [[ "$cpu" == "#"* ]] && continue
    [[ -z "$cpu" ]] && continue
    if [[ -z "${SEEN_CORE[$core]:-}" ]]; then
        PRIMARY_CPUS+=("$cpu")
        SEEN_CORE[$core]=1
    fi
done < <(lscpu -p=CPU,CORE,SOCKET 2>/dev/null)
N_PHYSICAL=${#PRIMARY_CPUS[@]}

# Construir el pool de CPUs para pinear. Si N_PARALLEL <= cores fisicos,
# usamos solo los primaries (un worker por core, sin colisión HT).
# Si N_PARALLEL > cores fisicos, agregamos los HT siblings al final.
PIN_POOL=("${PRIMARY_CPUS[@]}")
if (( N_PARALLEL > N_PHYSICAL )); then
    # Agregar todos los CPUs que no esten ya en el pool (HT siblings).
    while IFS=, read -r cpu core socket rest; do
        [[ "$cpu" == "#"* ]] && continue
        [[ -z "$cpu" ]] && continue
        already=0
        for p in "${PIN_POOL[@]}"; do (( p == cpu )) && already=1 && break; done
        (( already == 0 )) && PIN_POOL+=("$cpu")
    done < <(lscpu -p=CPU,CORE,SOCKET 2>/dev/null)
fi
# Truncar a N_PARALLEL.
PIN_POOL=("${PIN_POOL[@]:0:$N_PARALLEL}")

echo "Detected physical cores: $N_PHYSICAL · primary CPUs: ${PRIMARY_CPUS[*]}" | tee -a "$LOG"
echo "Worker pin pool (size $N_PARALLEL): ${PIN_POOL[*]}" | tee -a "$LOG"
if (( N_PARALLEL > N_PHYSICAL )); then
    echo "WARN: N_PARALLEL ($N_PARALLEL) > cores fisicos ($N_PHYSICAL): algunos workers competirán via HT" | tee -a "$LOG"
fi

# Pool de CPUs libres (queue). Pop al lanzar, push al recolectar.
FREE_CPUS=("${PIN_POOL[@]}")

step "spawning 9 jobs (3 opts × 3 seeds), max $N_PARALLEL parallel · OMP_NUM_THREADS=1 · pinned to CPUs ${PIN_POOL[*]}"
T0_ALL=$(date +%s)

declare -A JOB_PIDS    # pid -> "opt seed"
declare -A JOB_T0      # pid -> start epoch
declare -A JOB_CPU     # pid -> cpu pinned
declare -A JOB_RC      # "opt seed" -> rc
declare -A JOB_DT      # "opt seed" -> elapsed

launched=0
total=$(( ${#SEEDS[@]} * ${#OPTS[@]} ))

# Helper: wait for any child to finish, record its result, free a slot.
# El rc de wait -n no necesariamente corresponde al PID que encontramos
# muerto en el scan, asi que para ser correctos hacemos un `wait $pid`
# explicito (que con un proceso ya muerto devuelve su rc cacheado).
collect_one() {
    # bash 4.3+: wait -n bloquea hasta que CUALQUIER background termina.
    # Si no quedan hijos vuelve con rc=127 — no abortamos por eso.
    wait -n 2>/dev/null || true
    for pid in "${!JOB_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local key="${JOB_PIDS[$pid]}"
            local t0="${JOB_T0[$pid]}"
            local cpu="${JOB_CPU[$pid]}"
            local elapsed=$(( $(date +%s) - t0 ))
            # `wait $pid` sobre un proceso ya muerto devuelve su exit code real.
            wait "$pid" 2>/dev/null
            local rc=$?
            JOB_RC["$key"]=$rc
            JOB_DT["$key"]=$elapsed
            # Devolver el CPU al pool de libres.
            FREE_CPUS+=("$cpu")
            unset "JOB_PIDS[$pid]" "JOB_T0[$pid]" "JOB_CPU[$pid]"
            local mark="OK"
            (( rc != 0 )) && mark="FAIL(rc=$rc)"
            echo "[$(date '+%H:%M:%S')] finished: $key  cpu=$cpu  $mark  elapsed=${elapsed}s" | tee -a "$LOG"
            return 0
        fi
    done
    return 0
}

for opt in "${OPTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Si ya alcanzamos el cap, esperar a que algun proceso termine.
        while (( ${#JOB_PIDS[@]} >= N_PARALLEL )); do
            collect_one
        done

        # Pop un CPU del pool de libres.
        cpu="${FREE_CPUS[0]}"
        FREE_CPUS=("${FREE_CPUS[@]:1}")

        launched=$((launched + 1))
        JOB_LOG="/tmp/matrix_max_${opt}_seed${seed}.log"
        : > "$JOB_LOG"
        echo "[$(date '+%H:%M:%S')] launch [$launched/$total]: opt=$opt seed=$seed cpu=$cpu -> $JOB_LOG" | tee -a "$LOG"

        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            taskset -c "$cpu" \
            python3 -m experiments.ej2.experiments.matrix_lr_opt_arch \
                --optimizers "$opt" --seeds "$seed" \
                --base-config "$TMP_BASE" \
                > "$JOB_LOG" 2>&1 &
        pid=$!
        JOB_PIDS[$pid]="$opt $seed"
        JOB_T0[$pid]=$(date +%s)
        JOB_CPU[$pid]="$cpu"
    done
done

# Esperar al resto.
while (( ${#JOB_PIDS[@]} > 0 )); do
    collect_one
done

T_TRAIN=$(( $(date +%s) - T0_ALL ))
step "all jobs done · train wall=${T_TRAIN}s ($(printf '%.2f' $(echo "$T_TRAIN/3600" | bc -l))h)"

# 4) Reporte por job.
{
    echo "Per-job summary:"
    printf "  %-10s %-8s %-8s %s\n" "opt" "seed" "rc" "elapsed_sec"
    for opt in "${OPTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            key="$opt $seed"
            printf "  %-10s %-8s %-8s %s\n" "$opt" "$seed" "${JOB_RC[$key]:-?}" "${JOB_DT[$key]:-?}"
        done
    done
} | tee -a "$LOG"

n_failed=0
for k in "${!JOB_RC[@]}"; do
    (( JOB_RC[$k] != 0 )) && n_failed=$((n_failed + 1))
done

# 5) Consolidacion final (regenera summary, heatmaps, curves, report.md sobre
#    todo lo que este completo). IMPORTANTE: si TODOS los jobs fallaron,
#    NO ejecutamos esto porque matrix_lr_opt_arch no tiene flag "consolidate-only"
#    y empezaria a entrenar las 270 corridas SECUENCIALMENTE en este proceso.
RUN_DIR_COUNT=$(ls -d results/ej2/comparasion/matrix_lr_opt_arch/runs/*__seed* 2>/dev/null | wc -l)
if (( n_failed == total )) || (( RUN_DIR_COUNT == 0 )); then
    step "consolidation SKIPPED (no successful runs: failed=$n_failed/$total, run_dirs=$RUN_DIR_COUNT)"
else
    step "consolidation"
    if python3 -m experiments.ej2.experiments.matrix_lr_opt_arch \
            --seeds 42 123 2026 \
            --base-config "$TMP_BASE" 2>&1 | tee -a "$LOG"; then
        step "consolidation OK"
    else
        step "consolidation FAILED — revisar log"
    fi
fi

# 6) Resumen final.
step "SUMMARY"
{
    OUT_DIR=results/ej2/comparasion/matrix_lr_opt_arch
    echo "Output dir: $OUT_DIR"
    echo "Epochs configurados: $EPOCHS"
    echo "Workers paralelos:   $N_PARALLEL"
    echo "Failed jobs:         $n_failed / $total"
    echo "Wall train time:     ${T_TRAIN}s"
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
import csv, os
agg_path = '$OUT_DIR/aggregate_summary.csv'
if not os.path.exists(agg_path):
    print('  (aggregate_summary.csv no existe — consolidacion fallo)')
else:
    agg = list(csv.DictReader(open(agg_path)))
    agg.sort(key=lambda r: -float(r.get('val_macro_f1_mean') or 0))
    print(f\"  {'variant':55s} {'f1_mean':>9s} {'f1_std':>8s} {'best_mean':>10s}\")
    print('  ' + '-' * 86)
    for r in agg[:10]:
        print(f\"  {r['variant']:55s} {float(r['val_macro_f1_mean']):>9.4f} {float(r['val_macro_f1_std']):>8.4f} {float(r['best_val_macro_f1_mean']):>10.4f}\")
" 2>&1
} | tee -a "$LOG"

step "overnight ej2 MAX END"

exit "$n_failed"
