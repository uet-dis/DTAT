#!/usr/bin/env bash
# Sample robust pools with multiple postprocess seeds, one subfolder per seed.
#
# For each requested model it runs postprocess_merge.py with --seed <s> and writes
#   resources/<RES>/adv_samples/robust/seed_<s>/<res>_<model>_robust_train_post.csv
# The merged RAW file (input of postprocess) stays at
#   resources/<RES>/adv_samples/robust/<res>_<model>_robust_train.csv
# and is created automatically for dnn/tabnet (re-merge from adv_samples/train pools)
# if missing. Tree models must already have a raw file (merge_training_samples.py).
#
# Usage:
#   ./cli/apelid/7_generating_samples/sample_robust_seeds.sh <resource> <model> [model ...]
#
# Examples:
#   ./cli/apelid/7_generating_samples/sample_robust_seeds.sh cic2018 xgb catb bagging
#   ./cli/apelid/7_generating_samples/sample_robust_seeds.sh nslkdd xgb catb bagging dnn tabnet gbm
#
# Env overrides:
#   SEEDS="1338 1339 1340"   sampling seeds (default)
#   PYTHON=...               interpreter (default: /home/soc/minhtq/workspace/.conda/bin/python)
#   SKIP_MERGE=1             never auto-merge a missing raw (fail instead)
#   LOG_DIR=...              log directory (default: /tmp/robust_seed_logs)
set -uo pipefail

PYTHON="${PYTHON:-/home/soc/minhtq/workspace/.conda/bin/python}"
SEEDS="${SEEDS:-1338 1339 1340}"
LOG_DIR="${LOG_DIR:-/tmp/robust_seed_logs}"
SKIP_MERGE="${SKIP_MERGE:-0}"

KNOWN_RESOURCES=(cic2018 nslkdd)
KNOWN_MODELS=(xgb dnn tabnet catb bagging gbm)
TREE_MODELS=(xgb catb bagging gbm)

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[ $# -ge 2 ] || usage
RESOURCE="$1"; shift
MODELS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || die "cannot cd to repo root"

case " ${KNOWN_RESOURCES[*]} " in *" $RESOURCE "*) ;; *) die "unknown resource '$RESOURCE' (expected: ${KNOWN_RESOURCES[*]})";; esac
for m in "${MODELS[@]}"; do
    case " ${KNOWN_MODELS[*]} " in *" $m "*) ;; *) die "unknown model '$m' (expected: ${KNOWN_MODELS[*]})";; esac
done

# Dataset-specific sampling targets
if [ "$RESOURCE" = "cic2018" ]; then
    OWN=12000; HALF=6000; FULL=12000
else
    OWN=5000;  HALF=2500; FULL=5000
fi

is_tree() {
    case " ${TREE_MODELS[*]} " in *" $1 "*) return 0;; *) return 1;; esac
}

ROBUST_DIR="resources/$( [ "$RESOURCE" = cic2018 ] && echo CIC-2018 || echo NSLKDD )/adv_samples/robust"
TRAIN_DIR="resources/$( [ "$RESOURCE" = cic2018 ] && echo CIC-2018 || echo NSLKDD )/adv_samples/train"
mkdir -p "$LOG_DIR"
echo "[seeds] repo=$REPO_ROOT resource=$RESOURCE models=${MODELS[*]} seeds=$SEEDS"
echo "[seeds] robust dir=$ROBUST_DIR | logs -> $LOG_DIR"

declare -a FAILED=()

for m in "${MODELS[@]}"; do
    raw="$ROBUST_DIR/${RESOURCE}_${m}_robust_train.csv"

    # --- ensure raw exists -------------------------------------------------
    if [ ! -f "$raw" ]; then
        if [ "$SKIP_MERGE" = "1" ]; then
            echo "[seeds] $m: FAILED - missing raw $raw (SKIP_MERGE=1)"
            FAILED+=("$m:no-raw"); continue
        fi
        if is_tree "$m"; then
            echo "[seeds] $m: FAILED - missing raw $raw; trees are not auto-merged"
            echo "        -> run: python cli/apelid/7_generating_samples/merge_training_samples.py -r $RESOURCE -m $m --adv-in <pools...>"
            FAILED+=("$m:no-raw"); continue
        fi
        log="$LOG_DIR/merge_${RESOURCE}_${m}.log"
        echo "[seeds] $m: raw missing -> re-merge DL pool (log: $log)"
        pools=()
        for atk in fgsm pgd jsma cw deepfool zoo hsja; do
            f="$TRAIN_DIR/${RESOURCE}_${m}_${atk}_adv.csv"
            [ -f "$f" ] && pools+=("$f") || echo "[seeds]   WARN: pool not found: $f"
        done
        if ! "$PYTHON" cli/apelid/7_generating_samples/merge_training_samples.py \
                -r "$RESOURCE" -m "$m" --adv-in "${pools[@]}" > "$log" 2>&1; then
            echo "[seeds] $m: FAILED re-merge (see $log)"
            FAILED+=("$m:merge"); continue
        fi
        [ -f "$raw" ] || { echo "[seeds] $m: FAILED - raw still missing after merge"; FAILED+=("$m:merge"); continue; }
    fi

    # --- postprocess one file per seed -------------------------------------
    for s in $SEEDS; do
        outdir="$ROBUST_DIR/seed_$s"
        mkdir -p "$outdir"
        log="$LOG_DIR/post_${RESOURCE}_${m}_seed${s}.log"
        if is_tree "$m"; then
            tgt=(--sample-threshold "$HALF" --attack-targets "zoo=$OWN" "hsja=$OWN" "default=$HALF")
        else
            tgt=(--sample-threshold "$FULL")
        fi
        echo "[seeds] $m seed=$s: postprocess -> $outdir/${RESOURCE}_${m}_robust_train_post.csv"
        if ! "$PYTHON" cli/apelid/7_generating_samples/postprocess_merge.py \
                -r "$RESOURCE" -m "$m" --seed "$s" \
                --input "$raw" \
                --output "$outdir/${RESOURCE}_${m}_robust_train_post.csv" \
                "${tgt[@]}" --assert-exact > "$log" 2>&1; then
            echo "[seeds] $m seed=$s: FAILED (see $log)"
            FAILED+=("$m:seed$s"); continue
        fi
        rows="$(wc -l < "$outdir/${RESOURCE}_${m}_robust_train_post.csv")"
        echo "[seeds] $m seed=$s: OK rows=$((rows-1))"
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[seeds] ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
echo "[seeds] done: ${#MODELS[@]} model(s) x seeds ${SEEDS}"
