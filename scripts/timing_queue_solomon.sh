#!/bin/bash
# The timing campaign on the reference machine (solomon), sequential, one unit at a time, box otherwise idle
# (owner protocol 2026-09-13, flash-ansr-research/plans/srbf_timing_subset_protocol.md):
#   1. freeze the 262-problem timing subset, nested in the hybrid r-sweep's frozen subset;
#   2. timing ladders (every rung 1..65,536 on the subset) for the base models, and the RL rows when their
#      checkpoints are named;
#   3. hybrid T-curves at r* for T in {10, 100, 1000} s, every size (and the RL rows when named);
#   4. PySR alone (the hybrid at ratio 1: all of the budget to PySR, the same clock) at the three budgets;
#   5. the bootstrap read-out of the ladders.
# Never starts while the r-sweep runs; refuses any host but solomon; needs the GPU free; stops cleanly between
# units when $R/STOP exists. Resumable: every step and unit leaves a marker. Runs detached:
#   RSTAR=0.1 nohup bash scripts/timing_queue_solomon.sh > ~/srbf_clock_kit/timing/logs/queue.out 2>&1 &
set -u
K=${K:-$HOME/srbf_clock_kit}                 # the clock kit: venv, scripts, the r-sweep under $K/hybrid
S=${S:-$K/scripts}
VENV=${VENV:-$K/venv}                        # the venv the campaign runs in (never modified while a run is live)
PYSR=${PYSR:-$HOME/venvs/pysr23/bin/python}
R=${R:-$K/timing}                            # the timing root (FLASH_ANSR_ROOT of every timing run)
RULE=${RULE:-50:40,20:4,10:2}                # 262 problems, nested in the r-sweep's 50:10,10:2
RSTAR=${RSTAR:-}                             # the hybrid ratio the r-sweep picked; required for steps 3-4
BUDGETS=${BUDGETS:-10 100 1000}
K_SEEDS=${K_SEEDS:-100}
REFINER_WORKERS=${REFINER_WORKERS:-16}
SWEEP_DATA=${SWEEP_DATA:-$K/hybrid/root/hybrid_data}
SWEEP_LOG=${SWEEP_LOG:-$K/hybrid/logs/driver.log}
CFG_DIR=${CFG_DIR:-$K/srbf/configs/evaluation/scaling}   # the srbf checkout's scaling configs
# model rows: name=path[=config]; the config defaults to the size's scaling config
M20=${M20:-$HOME/v25_t8_20m_full/runs/v25.0-T8-20M/checkpoint_1500000}
M3=${M3:-$K/models/flash-ansr-v25.0-T8-3M}
M120=${M120:-$K/models/flash-ansr-v25.0-T8-120M}
RL20=${RL20:-}; RL3=${RL3:-}; RL120=${RL120:-}     # RL checkpoint directories (empty: row skipped)
PYSR_REPEATS=${PYSR_REPEATS:-1}

export FLASH_ANSR_ROOT=$R PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PATH=$VENV/bin:$PATH
PY=$VENV/bin/python
mkdir -p $R/logs $R/queue_marks $R/configs
LOG=$R/logs/queue.log
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a $LOG; }
die() { say "FATAL: $*"; exit 1; }
stop_requested() { [ -e $R/STOP ] && { say "STOP file present; leaving the queue between units"; exit 0; }; }
mark() { touch $R/queue_marks/$1.done; }
done_() { [ -e $R/queue_marks/$1.done ]; }

# ---- 0. preflight ------------------------------------------------------------------------------------------------
[ "$(hostname)" = solomon ] || die "this is $(hostname); the reference machine is solomon and solomon only"
pgrep -f 'run_hybrid_swee[p]' > /dev/null && die "the hybrid r-sweep is still running; nothing else runs on solomon meanwhile"
grep -q HYBRID_DONE $SWEEP_LOG 2>/dev/null || grep -q "sweep done" $SWEEP_LOG 2>/dev/null || die "no HYBRID_DONE in $SWEEP_LOG: the r-sweep has not finished"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "${used:-1}" -lt 100 ] && [ "$apps" -eq 0 ] || die "GPU not free (${used} MiB used, ${apps} compute apps): free = 0 MiB, 0 %, no process"
[ -x $PY ] || die "no venv at $VENV"
say "versions: $($PY -c 'import srbf, flash_ansr, simplipy, symbolic_data; print("srbf", srbf.__version__, "flash_ansr", flash_ansr.__version__, "simplipy", simplipy.__version__, "symbolic_data", symbolic_data.__version__)' 2>&1 | tail -1)"
$PY -c 'import flash_ansr_hybrid' 2>/dev/null || say "WARNING: flash_ansr_hybrid not importable in $VENV; the hybrid steps will fail"
for m in M20:$M20 M3:$M3 M120:$M120; do
    [ -d "${m#*:}" ] && say "model ${m%%:*} = ${m#*:}" || say "WARNING: model ${m%%:*} missing at ${m#*:} (its rows are skipped)"
done
[ -d $SWEEP_DATA ] || die "no r-sweep frozen data at $SWEEP_DATA"
say "root $R rule $RULE rstar '${RSTAR}' budgets '$BUDGETS' k_seeds $K_SEEDS refiner_workers $REFINER_WORKERS pysr_repeats $PYSR_REPEATS"

# ---- 1. the frozen timing subset ----------------------------------------------------------------------------------
if ! done_ freeze; then
    $PY $S/freeze_timing_subset.py -c $CFG_DIR/flash-ansr-v25.0-T8-20M_srbf.yaml --from-frozen $SWEEP_DATA \
        --out-dir $R/hybrid_data --subset-rule $RULE 2>&1 | grep -v Warning | tee -a $LOG || die "freeze failed"
    $PY $S/freeze_timing_subset.py --check --out-dir $R/hybrid_data | tee -a $LOG | grep -q "^OK" || die "subset check failed"
    mark freeze
fi

# ---- 2. timing ladders ---------------------------------------------------------------------------------------------
ladder() {   # name path config
    local name=$1 path=$2 cfg=$3
    [ -d "$path" ] || { say "skip ladder $name: no model at $path"; return 0; }
    done_ ladder_$name && return 0
    stop_requested
    say "ladder $name ($path) on $cfg"
    $PY $S/run_timing_ladder.py -c $cfg --data-dir $R/hybrid_data --model-name $name --model-path $path \
        --refiner-workers $REFINER_WORKERS --root $R 2>&1 | grep -v Warning | tee -a $LOG
    ls $R/timing/$name/marks/*.failed > /dev/null 2>&1 && { say "ladder $name has failed units; not marked done"; return 1; }
    mark ladder_$name
}
ladder t8-20m  "$M20"  $CFG_DIR/flash-ansr-v25.0-T8-20M_srbf.yaml
ladder t8-3m   "$M3"   $CFG_DIR/flash-ansr-v25.0-T8-3M_srbf.yaml
ladder t8-120m "$M120" $CFG_DIR/flash-ansr-v25.0-T8-120M_srbf.yaml
[ -n "$RL20" ]  && ladder rl-20m  "$RL20"  $CFG_DIR/flash-ansr-v25.0-T8-20M_srbf.yaml
[ -n "$RL3" ]   && ladder rl-3m   "$RL3"   $CFG_DIR/flash-ansr-v25.0-T8-3M_srbf.yaml
[ -n "$RL120" ] && ladder rl-120m "$RL120" $CFG_DIR/flash-ansr-v25.0-T8-120M_srbf.yaml

# ---- 3./4. hybrid T-curves at r* and PySR alone ---------------------------------------------------------------------
hybrid_cell() {   # name model_path budget ratios
    local name=$1 path=$2 budget=$3 ratios=$4 HR=$R/hybrid/$1
    [ -d "$path" ] || { say "skip hybrid $name: no model at $path"; return 0; }
    done_ hybrid_$name && return 0
    stop_requested
    mkdir -p $HR $R/snapshots/$name $R/logs/pysr_worker/$name
    ln -sfn $R/hybrid_data $HR/hybrid_data          # the frozen subset; the driver reuses existing files
    $PY $S/make_hybrid_config.py --model-path $path --model-name $name --pysr-python $PYSR --budget $budget \
        --ratios $ratios --k-seeds $K_SEEDS --snapshot-dir $R/snapshots/$name --out $R/configs/$name.yaml \
        --device cuda --refiner-workers $REFINER_WORKERS --landing-tolerance 0.01 --pysr-overhead 4.0 \
        --pricing-reserve 0.2 --worker-log $R/logs/pysr_worker/$name | tee -a $LOG
    say "hybrid $name: budget $budget s, ratios $ratios, model $path"
    $PY $S/run_hybrid_sweep.py --config $R/configs/$name.yaml --root $HR --subset-rule $RULE 2>&1 | grep -v "Warning\|warn" | tee -a $LOG
    local n_done=$(ls $HR/hybrid_marks/*.done 2>/dev/null | wc -l)
    [ "$n_done" -ge 29 ] || { say "hybrid $name: only $n_done/29 cells done; not marked"; return 1; }
    mark hybrid_$name
}
if [ -n "$RSTAR" ]; then
    for T in $BUDGETS; do
        hybrid_cell hyb-20m-T$T  "$M20"  $T $RSTAR
        hybrid_cell hyb-3m-T$T   "$M3"   $T $RSTAR
        hybrid_cell hyb-120m-T$T "$M120" $T $RSTAR
        [ -n "$RL20" ]  && hybrid_cell hyb-rl-20m-T$T  "$RL20"  $T $RSTAR
        [ -n "$RL3" ]   && hybrid_cell hyb-rl-3m-T$T   "$RL3"   $T $RSTAR
        [ -n "$RL120" ] && hybrid_cell hyb-rl-120m-T$T "$RL120" $T $RSTAR
    done
    for T in $BUDGETS; do
        for rep in $(seq 1 $PYSR_REPEATS); do
            hybrid_cell pysr-T$T-r$rep "$M20" $T 1        # ratio 1: PySR gets the whole budget, the model is idle
        done
    done
else
    say "RSTAR unset: hybrid T-curves and PySR-alone skipped (rerun with RSTAR=<r*> once the r-sweep is read)"
fi

# ---- 5. read-out --------------------------------------------------------------------------------------------------
models=$(ls -d $R/results/evaluation/timing/*/ 2>/dev/null | xargs -n1 basename | grep -v '^smoke' | paste -sd,)
if [ -n "$models" ]; then
    $PY $S/timing_readout.py --root $R --manifest $R/hybrid_data/timing_subset.json --models $models --reference t8-20m \
        --out $R/REPORT_ladders.md --json $R/REPORT_ladders.json 2>&1 | tail -2 | tee -a $LOG
fi
say "queue finished"
echo TIMING_QUEUE_DONE | tee -a $LOG
