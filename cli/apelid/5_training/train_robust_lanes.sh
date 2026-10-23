#!/usr/bin/env bash
# Multi-lane parallel robust retraining.
#
# Lanes (each lane = its own independent queue, run in parallel with the others):
#   gpu   : GPU models, sequential (GPU_PARALLEL=1 -> at most 1 training on GPU)
#   cpu   : CPU models, sequential (CPU_PARALLEL=1 -> at most 1 on CPU lane)
#   solo  : independent lane (gbm by default). sklearn GBMs are single-threaded
#           (~1 core each), so this lane may run several jobs in parallel.
#
# Usage:
#   ./cli/apelid/5_training/train_robust_lanes.sh <resource> [resource ...]
#
# Examples:
#   ./cli/apelid/5_training/train_robust_lanes.sh cic2018 nslkdd            # default: solo x3
#   SOLO_PARALLEL=2 ./cli/apelid/5_training/train_robust_lanes.sh cic2018
#   LANES=solo SOLO_PARALLEL=3 ./cli/apelid/5_training/train_robust_lanes.sh cic2018 nslkdd
#
# Each job trains one (resource, model, seed) robust model from
#   resources/<RES>/adv_samples/robust/seed_<seed>/<res>_<model>_robust_train_post.csv
# producing outputs tagged with _seed<seed> (see retrain_robust.sh).
#
# Env overrides:
#   SEEDS="1338 1339 1340"     sampling seeds to train
#   LANES="gpu cpu solo"       which lanes to run
#   GPU_MODELS / CPU_MODELS / SOLO_MODELS
#   GPU_PARALLEL=1 CPU_PARALLEL=1 SOLO_PARALLEL=3
#   GPU_DEVICE=auto  CPU_DEVICE=cpu  SOLO_DEVICE=cpu
#   PYTHON / PARAMS_DIR        forwarded to retrain_robust.sh
#   SKIP_EXISTING=1            skip jobs whose _seed<seed> checkpoint already exists (resume)
#   LANE_LOG_DIR=...           lane logs (default: /tmp/retrain_lane_logs)
#   RETRAIN_LOG_DIR=...        per-model logs (default: /tmp/retrain_logs)
set -uo pipefail

SEEDS="${SEEDS:-1338 1339 1340}"
LANES="${LANES:-gpu cpu solo}"
GPU_MODELS="${GPU_MODELS:-dnn tabnet xgb catb}"
CPU_MODELS="${CPU_MODELS:-bagging}"
SOLO_MODELS="${SOLO_MODELS:-gbm}"
GPU_PARALLEL="${GPU_PARALLEL:-1}"
CPU_PARALLEL="${CPU_PARALLEL:-1}"
SOLO_PARALLEL="${SOLO_PARALLEL:-3}"
GPU_DEVICE="${GPU_DEVICE:-auto}"
CPU_DEVICE="${CPU_DEVICE:-cpu}"
SOLO_DEVICE="${SOLO_DEVICE:-cpu}"
LANE_LOG_DIR="${LANE_LOG_DIR:-/tmp/retrain_lane_logs}"
export LOG_DIR="${RETRAIN_LOG_DIR:-${LOG_DIR:-/tmp/retrain_logs}}"

KNOWN_RESOURCES=(cic2018 nslkdd)

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

[ $# -ge 1 ] || usage
RESOURCES=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || die "cannot cd to repo root"

for r in "${RESOURCES[@]}"; do
    case " ${KNOWN_RESOURCES[*]} " in *" $r "*) ;; *) die "unknown resource '$r' (expected: ${KNOWN_RESOURCES[*]})";; esac
done

all_models="$GPU_MODELS $CPU_MODELS $SOLO_MODELS"
dupes="$(echo $all_models | tr ' ' '\n' | sort | uniq -d | tr '\n' ' ')"
[ -z "$dupes" ] || die "model(s) assigned to more than one lane: $dupes"

for n in $GPU_PARALLEL $CPU_PARALLEL $SOLO_PARALLEL; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "parallelism must be a positive integer, got '$n'"
done

mkdir -p "$LANE_LOG_DIR" "$LOG_DIR"

# ---- build per-lane job queues (resource:model:seed) ------------------------
build_queue() {
    local models="$1" res seed m adv resdir ext
    for res in "${RESOURCES[@]}"; do
        resdir="$([ "$res" = cic2018 ] && echo CIC-2018 || echo NSLKDD)"
        for seed in $SEEDS; do
            for m in $models; do
                ext=".pkl"; case "$m" in dnn|tabnet) ext=".pth";; esac
                if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "resources/${resdir}/models/${res}_${m}_robust_seed${seed}${ext}" ]; then
                    echo "SKIP (checkpoint exists): ${res} ${m} seed=${seed}" >&2
                    continue
                fi
                adv="resources/${resdir}/adv_samples/robust/seed_${seed}/${res}_${m}_robust_train_post.csv"
                if [ -f "$adv" ]; then
                    printf '%s:%s:%s\n' "$res" "$m" "$seed"
                else
                    echo "MISSING pool: $adv" >&2
                fi
            done
        done
    done
}

# ---- workers ----------------------------------------------------------------
run_job() {  # <lane> <device> <res:model:seed> <log>
    local lane="$1" device="$2" job="$3" log="$4" res m seed
    IFS=: read -r res m seed <<< "$job"
    if DEVICE="$device" bash cli/apelid/5_training/retrain_robust.sh "$res" "$m" --seed "$seed" >> "$log" 2>&1; then
        echo "[$lane] $(date '+%T') OK   $res $m seed=$seed" >> "$log"
        return 0
    fi
    echo "[$lane] $(date '+%T') FAIL $res $m seed=$seed" >> "$log"
    return 1
}

run_lane() {  # <lane> <parallel> <device> <jobs...>
    local lane="$1" par="$2" device="$3"; shift 3
    local jobs=("$@")
    local njobs=${#jobs[@]}
    [ "$njobs" -gt 0 ] || { echo "[$lane] no jobs"; return 0; }
    [ "$par" -le "$njobs" ] || par="$njobs"

    echo "[$lane] device=$device jobs=$njobs parallel=$par started $(date '+%F %T')" > "$LANE_LOG_DIR/lane_${lane}.log"
    local pids=() fails=0 w i
    for ((w=0; w<par; w++)); do
        : > "$LANE_LOG_DIR/lane_${lane}_w${w}.log"
        (
            local wf=0
            for ((i=w; i<njobs; i+=par)); do
                run_job "$lane" "$device" "${jobs[$i]}" "$LANE_LOG_DIR/lane_${lane}_w${w}.log" || wf=$((wf+1))
            done
            exit $wf
        ) >> "$LANE_LOG_DIR/lane_${lane}_w${w}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fails=$((fails+1)); done
    echo "[$lane] finished $(date '+%F %T') workers_with_failures=$fails" >> "$LANE_LOG_DIR/lane_${lane}.log"
    return $fails
}

lane_enabled() { case " $LANES " in *" $1 "*) return 0;; *) return 1;; esac; }

: > "$LANE_LOG_DIR/../lanes_note.tmp" 2>/dev/null || true
echo "[lanes] repo=$REPO_ROOT resources=${RESOURCES[*]} seeds=$SEEDS lanes=[$LANES]"

declare -a PIDS=() LANE_NAMES=()
for lane in gpu cpu solo; do
    lane_enabled "$lane" || continue
    case "$lane" in
        gpu)  LANE_MODELS="$GPU_MODELS";  LANE_PAR="$GPU_PARALLEL";  LANE_DEV="$GPU_DEVICE";;
        cpu)  LANE_MODELS="$CPU_MODELS";  LANE_PAR="$CPU_PARALLEL";  LANE_DEV="$CPU_DEVICE";;
        solo) LANE_MODELS="$SOLO_MODELS"; LANE_PAR="$SOLO_PARALLEL"; LANE_DEV="$SOLO_DEVICE";;
    esac
    mapfile -t Q < <(build_queue "$LANE_MODELS")
    echo "[lanes] $lane: device=$LANE_DEV parallel=$LANE_PAR models=[$LANE_MODELS] jobs=${#Q[@]}"
    run_lane "$lane" "$LANE_PAR" "$LANE_DEV" ${Q[@]+"${Q[@]}"} &
    PIDS+=($!); LANE_NAMES+=("$lane")
done

rc_all=0
for idx in "${!PIDS[@]}"; do
    wait "${PIDS[$idx]}"
    rc=$?
    echo "[lanes] ${LANE_NAMES[$idx]} lane rc=$rc"
    [ "$rc" -eq 0 ] || rc_all=1
done

echo "[lanes] lane logs -> $LANE_LOG_DIR (lane_<name>.log summary, lane_<name>_w<N>.log workers)"
if [ "$rc_all" -ne 0 ]; then echo "[lanes] some jobs failed"; exit 1; fi
echo "[lanes] all lanes completed OK"
