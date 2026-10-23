#!/usr/bin/env bash
# Retrain robust (adversarial-trained) models, with optional per-seed sampling pools.
#
# Usage:
#   ./cli/apelid/5_training/retrain_robust.sh <resource> <model> [model ...] [--seed <seed>]
#
# Seeds:
#   --seed S            use the S-seed pool for every model:
#                         resources/<RES>/adv_samples/robust/seed_<S>/<res>_<model>_robust_train_post.csv
#   SEED=S              same as --seed S (env)
#   SEED_<model>=S      per-model override (wins over --seed), e.g. SEED_xgb=1338 SEED_catb=1339
#   ADV_IN_<model>=path explicit adversarial CSV (wins over everything)
#   (no seed)           legacy behaviour: root pool
#                         resources/<RES>/adv_samples/robust/<res>_<model>_robust_train_post.csv
#                       and output names without a seed suffix.
#
# Outputs (seed mode adds the suffix "_seed<S>" so runs never overwrite each other):
#   model checkpoint : resources/<RES>/models/<res>_<model>_robust[_seed<S>].{pkl,pth}
#   metrics json     : reports/<res>/<model>/<model>_robust[_seed<S>]_metrics.json
#   confusion matrix : reports/<res>/<model>/<model>_robust[_seed<S>]_cm.png
#
# Examples:
#   ./cli/apelid/5_training/retrain_robust.sh cic2018 xgb catb bagging
#   ./cli/apelid/5_training/retrain_robust.sh nslkdd xgb catb bagging dnn tabnet gbm --seed 1338
#   SEED_xgb=1338 SEED_catb=1339 SEED_bagging=1340 \
#       ./cli/apelid/5_training/retrain_robust.sh cic2018 xgb catb bagging
#
# Env overrides:
#   PYTHON=...        interpreter (default: /home/soc/minhtq/workspace/.conda/bin/python)
#   PARAMS_DIR=...    (default: params/training)
#   LOG_DIR=...       (default: /tmp/retrain_logs)
#   DEVICE=...        passed to model_training.py (default: auto)
set -uo pipefail

PYTHON="${PYTHON:-/home/soc/minhtq/workspace/.conda/bin/python}"
PARAMS_DIR="${PARAMS_DIR:-params/training}"
LOG_DIR="${LOG_DIR:-/tmp/retrain_logs}"
DEVICE="${DEVICE:-auto}"

KNOWN_RESOURCES=(cic2018 nslkdd)
KNOWN_MODELS=(xgb dnn tabnet catb bagging gbm)

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[ $# -ge 1 ] || usage
RESOURCE="$1"; shift
SEED_ARG=""
MODELS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --seed)    [ $# -ge 2 ] || die "--seed needs a value"; SEED_ARG="$2"; shift 2;;
        --seed=*)  SEED_ARG="${1#*=}"; shift;;
        -h|--help) usage;;
        -*)        die "unknown flag '$1'";;
        *)         MODELS+=("$1"); shift;;
    esac
done
[ ${#MODELS[@]} -ge 1 ] || usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || die "cannot cd to repo root"

case " ${KNOWN_RESOURCES[*]} " in *" $RESOURCE "*) ;; *) die "unknown resource '$RESOURCE' (expected: ${KNOWN_RESOURCES[*]})";; esac
for m in "${MODELS[@]}"; do
    case " ${KNOWN_MODELS[*]} " in *" $m "*) ;; *) die "unknown model '$m' (expected: ${KNOWN_MODELS[*]})";; esac
done
[ -z "$SEED_ARG" ] || [[ "$SEED_ARG" =~ ^[0-9]+$ ]] || die "--seed must be an integer, got '$SEED_ARG'"

if [ "$RESOURCE" = "cic2018" ]; then RES_DIR="CIC-2018"; else RES_DIR="NSLKDD"; fi
ROBUST_DIR="resources/${RES_DIR}/adv_samples/robust"
REPORT_DIR="reports/${RESOURCE}"      # matches res.REPORT_FOLDER
mkdir -p "$LOG_DIR"

echo "[retrain] repo=$REPO_ROOT resource=$RESOURCE models=${MODELS[*]} device=$DEVICE"
[ -n "$SEED_ARG" ] && echo "[retrain] global seed: $SEED_ARG (per-model SEED_<model> and ADV_IN_<model> still win)"
echo "[retrain] logs -> $LOG_DIR"

declare -a FAILED=()
for m in "${MODELS[@]}"; do
    ext=".pkl"; case "$m" in dnn|tabnet) ext=".pth";; esac

    # --- resolve seed for this model --------------------------------------
    seed="$SEED_ARG"
    ov_seed="SEED_${m}"
    [ -n "${!ov_seed:-}" ] && seed="${!ov_seed}"
    [ -z "$seed" ] || [[ "$seed" =~ ^[0-9]+$ ]] || die "SEED_${m} must be an integer, got '$seed'"

    # --- resolve adversarial CSV ------------------------------------------
    ov_adv="ADV_IN_${m}"
    if [ -n "${!ov_adv:-}" ]; then
        adv="${!ov_adv}"
        src="explicit ADV_IN_${m}"
    elif [ -n "$seed" ]; then
        adv="${ROBUST_DIR}/seed_${seed}/${RESOURCE}_${m}_robust_train_post.csv"
        src="seed ${seed} pool"
    else
        adv="${ROBUST_DIR}/${RESOURCE}_${m}_robust_train_post.csv"
        src="default pool (seed_1337 file)"
    fi
    [ -f "$adv" ] || { echo "[retrain] $m: FAILED - adv set not found: $adv"; FAILED+=("$m:no-adv"); continue; }

    # --- output names (seed suffix keeps runs separate) --------------------
    stem="${RESOURCE}_${m}_robust"
    [ -n "$seed" ] && stem="${stem}_seed${seed}"
    report_stem="${m}_robust"
    [ -n "$seed" ] && report_stem="${report_stem}_seed${seed}"
    model_out="resources/${RES_DIR}/models/${stem}${ext}"
    report_out="${REPORT_DIR}/${m}/${report_stem}_metrics.json"
    cm_out="${REPORT_DIR}/${m}/${report_stem}_cm.png"

    log="$LOG_DIR/retrain_${RESOURCE}_${m}${seed:+_seed$seed}.log"
    echo "[retrain] $m: adv=$src ($adv)"
    echo "[retrain] $m: out=$model_out"
    if "$PYTHON" cli/apelid/5_training/model_training.py \
            -r "$RESOURCE" -m "$m" \
            --params-dir "$PARAMS_DIR" --device "$DEVICE" --adv-train \
            --adv-in "$adv" --model-out "$model_out" --report-out "$report_out" --cm-out "$cm_out" \
            > "$log" 2>&1; then
        acc="$(grep -oE 'accuracy=[0-9.]+' "$log" | tail -1)"
        echo "[retrain] $m: OK ${acc:+($acc)}"
    else
        echo "[retrain] $m: FAILED (see $log)"
        FAILED+=("$m${seed:+:seed$seed}")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[retrain] ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
echo "[retrain] all ${#MODELS[@]} model(s) retrained OK"
