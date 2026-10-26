#!/usr/bin/env bash
# One-command deploy + submit for the CSE Slurm cluster.
# Run this from your Mac with the CSE VPN connected.
#
#   ./submit.sh                                   # config.py defaults
#   ./submit.sh --epochs 10 --batch-size 64       # flags go to train.py
#   ./submit.sh --follow --epochs 2 --limit 128   # submit, wait, fetch history
#   ./submit.sh --predict --epochs 10             # also write submission.csv
#
# Consumed here (everything else is passed to train.py):
#   -f, --follow   wait for the job, print its output, then fetch logs/
#   --fetch        copy remote logs/ and submission.csv back to this machine
#   --predict      run predict.py after training -> submission.csv
#   --no-sync      skip the rsync, submit what is already on the cluster
#   -h, --help     this text
#
# Env overrides: REMOTE (default "linux1"), REMOTE_DIR, SSH_KEY.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE:-linux1}"
REMOTE_DIR="${REMOTE_DIR:-aist4010_hw2_kaggle-1}"
FOLLOW=0
FETCH=0
SYNC=1
PREDICT=0
TRAIN_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -f|--follow) FOLLOW=1; FETCH=1; shift ;;
        --fetch)     FETCH=1; shift ;;
        --predict)   PREDICT=1; shift ;;
        --no-sync)   SYNC=0; shift ;;
        -h|--help)   sed -n '2,17p' "$0"; exit 0 ;;
        *)           TRAIN_ARGS+=("$1"); shift ;;
    esac
done

# --- 0. find a key ---------------------------------------------------------
KEY="${SSH_KEY:-}"
if [ -z "$KEY" ]; then
    for candidate in "$HOME/.ssh/id_ed25519" "$HERE/../aist4010-hw2-kaggle/.ssh/id_ed25519"; do
        if [ -f "$candidate" ]; then KEY="$candidate"; break; fi
    done
fi
if [ -z "$KEY" ] || [ ! -f "$KEY" ]; then
    echo "no ssh key: set SSH_KEY=/path/to/key, or copy it to ~/.ssh/id_ed25519" >&2
    exit 1
fi

SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=10 -o BatchMode=yes)

# --- 1. is the cluster reachable? -----------------------------------------
if ! ssh "${SSH_OPTS[@]}" "$REMOTE" 'true' 2>/dev/null; then
    echo "cannot ssh to $REMOTE - connect to the CSE VPN first (or set REMOTE=...)" >&2
    exit 1
fi

# --- 2. sync the code ------------------------------------------------------
if [ "$SYNC" -eq 1 ]; then
    echo "syncing $HERE/ -> $REMOTE:$REMOTE_DIR/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
        --exclude .git --exclude .ssh --exclude __pycache__ --exclude '*.pyc' \
        --exclude '.DS_Store' --exclude checkpoints --exclude logs \
        --exclude 'slurm-*.out' \
        "$HERE"/ "$REMOTE:$REMOTE_DIR/"
else
    echo "skipping rsync (--no-sync)"
fi

# --- 3. submit, passing every other flag to train.py ----------------------
ARGSTR=""
for arg in ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}; do
    ARGSTR="$ARGSTR$(printf ' %q' "$arg")"
done

JID=$(ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "cd '$REMOTE_DIR' && PREDICT=$PREDICT sbatch --parsable gpu.slurm$ARGSTR")

case "$JID" in
    ''|*[!0-9]*)
        echo "sbatch failed: $JID" >&2
        exit 1 ;;
esac

OUT="slurm-a2-$JID.out"
if [ "$PREDICT" -eq 1 ]; then
    echo "submitted job $JID (training, then submission.csv)"
else
    echo "submitted job $JID"
fi
echo "  queue : ssh $REMOTE 'squeue -j $JID'"
echo "  output: ssh $REMOTE 'cd $REMOTE_DIR && tail -f $OUT'"
echo "  logs  : ssh $REMOTE 'ls $REMOTE_DIR/logs'"

# --- 4. optionally wait and show the result -------------------------------
if [ "$FOLLOW" -eq 1 ]; then
    echo "waiting for job $JID ..."
    while ssh "${SSH_OPTS[@]}" "$REMOTE" "squeue -h -j $JID 2>/dev/null | grep -q ."; do
        sleep 5
    done
    echo "--- $OUT ---"
    ssh "${SSH_OPTS[@]}" "$REMOTE" "cd '$REMOTE_DIR' && tail -n 40 '$OUT'"
fi

# --- 5. pull the training history back to this machine --------------------
if [ "$FETCH" -eq 1 ]; then
    mkdir -p "$HERE/logs"
    echo "fetching logs/ -> $HERE/logs/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
        "$REMOTE:$REMOTE_DIR/logs/" "$HERE/logs/"
    LATEST="$(ls -1dt "$HERE"/logs/*/ 2>/dev/null | head -n 1)"
    if [ -n "$LATEST" ]; then
        echo "latest run: $LATEST"
        for f in history.json history.csv summary.json train.log; do
            [ -f "$LATEST$f" ] && echo "  $LATEST$f"
        done
    fi

    if ssh "${SSH_OPTS[@]}" "$REMOTE" "test -f $REMOTE_DIR/submission.csv"; then
        rsync -az -e "ssh ${SSH_OPTS[*]}" \
            "$REMOTE:$REMOTE_DIR/submission.csv" "$HERE/submission.csv"
        echo "submission.csv -> $HERE/submission.csv ($(($(wc -l < "$HERE/submission.csv") - 1)) rows)"
    fi
fi
