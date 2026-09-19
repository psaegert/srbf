#!/bin/bash
# The timing campaign on the reference machine (solomon), sequential, one unit at a time, box otherwise idle
# (owner protocol 2026-09-13, flash-ansr-research/plans/srbf_timing_subset_protocol.md):
#   1. freeze the 262-problem timing subset, nested in the hybrid r-sweep's frozen subset;
#   2. the baselines' timing ladders on the subset, in the owner's order of 2026-09-16: E2E (candidates_per_bag
#      1..2048), NeSymReS (beam_width 1..512), diffsym (n_samples 1..128) -- E2E and NeSymReS in the legacy venv;
#      ORDER (owner 2026-09-19, superseding 2026-09-18): one PySR run on the whole suite first (step 3b), then the
#      T8 rows with the 120M ahead of the 20M and the 3M (step 4), then NeSymReS's remaining rungs (64, 256, 512;
#      step 4c -- rung 512 alone costs more than every ladder before it; the marks make its finished rungs
#      no-ops), and PySR's second run on the suite after every other row (step 6).
#   3. PySR alone (the hybrid at ratio 1: all of the budget to PySR, the same clock) at the three budgets;
#   4. ONLY WITH SCORING SET (the score study's ruling, owner 2026-09-16: "for our T8 series models we need to decide
#      which scoring we use before we can evaluate them"): the T8 timing ladders (every rung 1..65,536) for the base
#      models and the RL rows when named, then the hybrid T-curves at r* for T in {10, 100, 1000} s, every size;
#   5. the bootstrap read-out of whatever ladders exist.
# Budget (owner 2026-09-16): 100 h of wall time per model row (BUDGET_HOURS); a ladder skips every rung whose projected
# cost would exceed it, so a slow method measures fewer rungs, never more hours. The hybrid cells are not gated
# (PySR alone at the three budgets ~82 h, a model's three T-curves ~81 h, both under the cap by construction).
# Never starts while the r-sweep runs; refuses any host but solomon; needs the GPU free; stops cleanly between
# units when $R/STOP exists. Resumable: every step and unit leaves a marker. Runs detached:
#   VENV=~/srbf_clock_kit/venv_timing nohup bash scripts/timing_queue_solomon.sh > ~/srbf_clock_kit/timing/logs/queue.out 2>&1 &
#   (later, for the T8 rows: SCORING=<ruling> RSTAR=<r*> ... the same command; every finished unit is skipped)
set -u
K=${K:-$HOME/srbf_clock_kit}                 # the clock kit: venv, scripts, the r-sweep under $K/hybrid
S=${S:-$K/scripts}
VENV=${VENV:-$K/venv}                        # the venv the campaign runs in (never modified while a run is live)
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
PYSR=${PYSR:-$HOME/venvs/pysr23/bin/python}          # PySR's own environment: the full-suite runs' worker
PYSR_SUITE_LADDER=${PYSR_SUITE_LADDER:-1,2,4,8,16,32,64,128,256,512,1024}   # iterations, bottom up
PYSR_SUITE_RUNS=${PYSR_SUITE_RUNS:-2}                # owner 2026-09-19: two PySR runs, both on solomon; run 1 ahead of the T8 rows, run 2 last
# the baselines (owner's order 2026-09-16: e2e, nesymres, diffsym, pysr -- before any T8 row)
VENV_LEGACY=${VENV_LEGACY:-$K/venv_legacy}              # srbf[baselines] + patched NeSymReS/E2E clones (build_solomon_baselines.sh)
E2E_MODEL=${E2E_MODEL:-$HOME/Projects/flash-ansr/models/e2e/model1.pt}
NESYMRES_DIR=${NESYMRES_DIR:-$HOME/Projects/flash-ansr/models/nesymres}   # eq_setting.json, config.yaml, 100M.ckpt
DIFFSYM_PY=${DIFFSYM_PY:-$K/diffsym/.venv/bin/python}
DIFFSYM_MODEL=${DIFFSYM_MODEL:-$K/models/diffsym-v4.0/best.pt}
DIFFSYM_CFG=${DIFFSYM_CFG:-$K/diffsym/configs/v4.0}
E2E_LADDER=${E2E_LADDER:-1,2,4,8,16,32,64,128,256}   # default settings only (owner 2026-09-16); extend after the 4090 memory probe
NESYMRES_LADDER=${NESYMRES_LADDER:-1,2,4,8,16,32,64,128,256,512}
DIFFSYM_LADDER=${DIFFSYM_LADDER:-1,2,4,8,16,32,64,128}
SCORING=${SCORING:-}                         # the score study's ruling (e.g. S0 or S1); the T8 rows refuse to run without it
BUDGET_HOURS=${BUDGET_HOURS:-100}            # owner 2026-09-16: 100 h per model row; rungs that do not fit are skipped (run_timing_ladder.py)

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
if [ -x $VENV_LEGACY/bin/python ] && $VENV_LEGACY/bin/python -c 'import srbf, symbolicregression, nesymres' 2>/dev/null; then
    say "legacy venv $VENV_LEGACY: $($VENV_LEGACY/bin/python -c 'import srbf, torch; print("srbf", srbf.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())')"
else say "WARNING: no usable legacy venv at $VENV_LEGACY (srbf + symbolicregression + nesymres); the E2E and NeSymReS rows are skipped"; fi
[ -f "$E2E_MODEL" ] || say "WARNING: no E2E checkpoint at $E2E_MODEL (row skipped)"
[ -f "$NESYMRES_DIR/100M.ckpt" ] || say "WARNING: no NeSymReS checkpoint under $NESYMRES_DIR (row skipped)"
[ -x "$DIFFSYM_PY" ] && [ -f "$DIFFSYM_MODEL" ] || say "WARNING: diffsym env/model missing ($DIFFSYM_PY, $DIFFSYM_MODEL): row skipped"
[ -n "$SCORING" ] && say "SCORING=$SCORING: the T8 rows run after the baselines" || say "SCORING unset: the T8 ladders and the hybrid T-curves wait for the score study's ruling"
say "root $R rule $RULE rstar '${RSTAR}' budgets '$BUDGETS' k_seeds $K_SEEDS refiner_workers $REFINER_WORKERS pysr_repeats $PYSR_REPEATS"

# ---- 1. the frozen timing subset ----------------------------------------------------------------------------------
if ! done_ freeze; then
    $PY $S/freeze_timing_subset.py -c $CFG_DIR/flash-ansr-v25.0-T8-20M_srbf.yaml --from-frozen $SWEEP_DATA \
        --out-dir $R/hybrid_data --subset-rule $RULE 2>&1 | grep -v Warning | tee -a $LOG || die "freeze failed"
    $PY $S/freeze_timing_subset.py --check --out-dir $R/hybrid_data | tee -a $LOG | grep -q "^OK" || die "subset check failed"
    mark freeze
fi

# ---- 2. the baselines' timing ladders (owner's order 2026-09-16) ----------------------------------------------------
baseline_ladder() {   # name config-generator-args... ; runs in $BL_PY (its venv on PATH), config under $R/configs
    local name=$1; shift
    done_ ladder_$name && return 0
    stop_requested
    local cfg=$R/configs/${name}_srbf.yaml
    "$BL_PY" $S/make_baseline_config.py "$@" $cfg $BL_LADDER > /dev/null || { say "config generation for $name failed"; return 1; }
    say "baseline ladder $name (ladder $BL_LADDER) on $cfg, python $BL_PY"
    PATH=$(dirname $BL_PY):$PATH "$BL_PY" $S/run_timing_ladder.py -c $cfg --data-dir $R/hybrid_data --model-name $name \
        --root $R --budget-hours $BUDGET_HOURS 2>&1 | grep -v Warning | tee -a $LOG
    ls $R/timing/$name/marks/*.failed > /dev/null 2>&1 && { say "ladder $name has failed units; not marked done"; return 1; }
    mark ladder_$name
}
if [ -x $VENV_LEGACY/bin/python ] && $VENV_LEGACY/bin/python -c 'import symbolicregression, nesymres' 2>/dev/null; then
    [ -f "$E2E_MODEL" ] && BL_PY=$VENV_LEGACY/bin/python BL_LADDER=$E2E_LADDER baseline_ladder e2e e2e "$E2E_MODEL"
fi
nesymres_row() {   # step 4c calls it: after the T8 rows (owner 2026-09-18/19)
    [ -x $VENV_LEGACY/bin/python ] && $VENV_LEGACY/bin/python -c 'import nesymres' 2>/dev/null || { say "no NeSymReS in $VENV_LEGACY: row skipped"; return 0; }
    [ -f "$NESYMRES_DIR/100M.ckpt" ] || { say "no NeSymReS checkpoint in $NESYMRES_DIR: row skipped"; return 0; }
    BL_PY=$VENV_LEGACY/bin/python BL_LADDER=$NESYMRES_LADDER baseline_ladder nesymres-100M nesymres "$NESYMRES_DIR"
}
if [ -x "$DIFFSYM_PY" ] && [ -f "$DIFFSYM_MODEL" ]; then
    BL_PY=$PY BL_LADDER=$DIFFSYM_LADDER baseline_ladder diffsym-v4.0 diffsym "$DIFFSYM_PY" "$DIFFSYM_MODEL" "$DIFFSYM_CFG"
fi

# ---- 3. PySR alone (the hybrid at ratio 1; needs no r*) -----------------------------------------------------------
hybrid_cell() {   # name model_path budget ratios
    local name=$1 path=$2 budget=$3 ratios=$4 HR=$R/hybrid/$1
    [ -d "$path" ] || { say "skip hybrid $name: no model at $path"; return 0; }
    done_ hybrid_$name && return 0
    stop_requested
    mkdir -p $HR $R/snapshots/$name
    ln -sfn $R/hybrid_data $HR/hybrid_data          # the frozen subset; the driver reuses existing files
    $PY $S/make_hybrid_config.py --model-path $path --model-name $name --budget $budget \
        --ratios $ratios --k-seeds $K_SEEDS --snapshot-dir $R/snapshots/$name --out $R/configs/$name.yaml \
        --device cuda --refiner-workers $REFINER_WORKERS --landing-tolerance 0.01 --pysr-overhead 4.0 \
        --pricing-reserve 0.2 | tee -a $LOG
    say "hybrid $name: budget $budget s, ratios $ratios, model $path"
    $PY $S/run_hybrid_sweep.py --config $R/configs/$name.yaml --root $HR --subset-rule $RULE 2>&1 | grep -v "Warning\|warn" | tee -a $LOG
    local n_done=$(ls $HR/hybrid_marks/*.done 2>/dev/null | wc -l)
    [ "$n_done" -ge 29 ] || { say "hybrid $name: only $n_done/29 cells done; not marked"; return 1; }
    mark hybrid_$name
}
if [ "${RUN_PYSR:-1}" != 1 ]; then
    say "RUN_PYSR=0: PySR alone is deferred (2026-09-19: srbf 0.20.0 cannot build the hybrid adapter; fixed on main 7f30172)"
elif $PY -c 'import flash_ansr_hybrid' 2>/dev/null; then
    for T in $BUDGETS; do
        for rep in $(seq 1 $PYSR_REPEATS); do
            hybrid_cell pysr-T$T-r$rep "$M20" $T 1        # ratio 1: PySR gets the whole budget, the model is idle
        done
    done
else say "flash_ansr_hybrid not importable: PySR-alone skipped"; fi

# ---- 3b. PySR on the whole suite (owner 2026-09-19) ------------------------------------------------------------
# PySR is evaluated on the full srbf suite here, on the reference machine, so its main evaluation IS its measured
# time: no separate timing run. One problem at a time with the whole machine, iterations bottom up; each run is
# its own root (a fresh draw of every catalog, like another FLASH_ANSR_ROOT for a Flash-ANSR draw).
pysr_suite_run() {   # run index
    local i=$1 RR=$K/pysr_suite/run$1
    done_ pysr_suite_run$i && return 0
    stop_requested
    mkdir -p $RR/configs
    $PY $S/make_pysr_suite_config.py --from $CFG_DIR/flash-ansr-v25.0-T8-20M_srbf.yaml --python $PYSR \
        --out $RR/configs/pysr_suite.yaml --ladder $PYSR_SUITE_LADDER | tee -a $LOG
    say "pysr suite run $i: iterations $PYSR_SUITE_LADDER, root $RR, worker $PYSR"
    $PY $S/run_timing_ladder.py -c $RR/configs/pysr_suite.yaml --full-suite --model-name pysr --root $RR 2>&1 | grep -v Warning | tee -a $LOG
    ls $RR/suite/pysr/marks/*.failed > /dev/null 2>&1 && { say "pysr suite run $i has failed units; not marked done"; return 1; }
    mark pysr_suite_run$i
}
if [ "${RUN_PYSR_SUITE:-1}" = 1 ] && [ -x "$PYSR" ]; then
    pysr_suite_run 1                                     # the remaining runs: step 6, after every other row
else say "RUN_PYSR_SUITE=0 or no PySR environment at $PYSR: PySR on the suite skipped"; fi

# ---- 4. the T8 rows: only with the scoring ruling ------------------------------------------------------------------
ladder() {   # name path config
    local name=$1 path=$2 cfg=$3
    [ -d "$path" ] || { say "skip ladder $name: no model at $path"; return 0; }
    done_ ladder_$name && return 0
    stop_requested
    say "ladder $name ($path) on $cfg"
    $PY $S/run_timing_ladder.py -c $cfg --data-dir $R/hybrid_data --model-name $name --model-path $path \
        --refiner-workers $REFINER_WORKERS --root $R --budget-hours $BUDGET_HOURS 2>&1 | grep -v Warning | tee -a $LOG
    ls $R/timing/$name/marks/*.failed > /dev/null 2>&1 && { say "ladder $name has failed units; not marked done"; return 1; }
    mark ladder_$name
}
if [ -z "$SCORING" ]; then
    say "SCORING unset: T8 ladders and hybrid T-curves not run (owner 2026-09-16: decide the scoring first)"
else
say "T8 rows under scoring ruling '$SCORING' (the scaling configs in $CFG_DIR must carry that ranking)"
ladder t8-120m "$M120" $CFG_DIR/flash-ansr-v25.0-T8-120M_srbf.yaml   # owner 2026-09-19: the 120M ahead of the other sizes
ladder t8-20m  "$M20"  $CFG_DIR/flash-ansr-v25.0-T8-20M_srbf.yaml
ladder t8-3m   "$M3"   $CFG_DIR/flash-ansr-v25.0-T8-3M_srbf.yaml
[ -n "$RL20" ]  && ladder rl-20m  "$RL20"  $CFG_DIR/flash-ansr-v25.0-T8-20M_srbf.yaml
[ -n "$RL3" ]   && ladder rl-3m   "$RL3"   $CFG_DIR/flash-ansr-v25.0-T8-3M_srbf.yaml
[ -n "$RL120" ] && ladder rl-120m "$RL120" $CFG_DIR/flash-ansr-v25.0-T8-120M_srbf.yaml

# ---- 4b. hybrid T-curves at r* --------------------------------------------------------------------------------------
if [ -z "$RSTAR" ]; then
    say "RSTAR unset: hybrid T-curves skipped (rerun with RSTAR=<r*> once the r-sweep is read)"
else
    for T in $BUDGETS; do
        hybrid_cell hyb-20m-T$T  "$M20"  $T $RSTAR
        hybrid_cell hyb-3m-T$T   "$M3"   $T $RSTAR
        hybrid_cell hyb-120m-T$T "$M120" $T $RSTAR
        [ -n "$RL20" ]  && hybrid_cell hyb-rl-20m-T$T  "$RL20"  $T $RSTAR
        [ -n "$RL3" ]   && hybrid_cell hyb-rl-3m-T$T   "$RL3"   $T $RSTAR
        [ -n "$RL120" ] && hybrid_cell hyb-rl-120m-T$T "$RL120" $T $RSTAR
    done
fi

# ---- 4c. NeSymReS's remaining rungs, after the T8 rows ------------------------------------------------------------
nesymres_row
fi   # SCORING

# ---- 5. read-out --------------------------------------------------------------------------------------------------
models=$(ls -d $R/results/evaluation/timing/*/ 2>/dev/null | xargs -n1 basename | grep -v '^smoke' | paste -sd,)
if [ -n "$models" ]; then
    ref=t8-20m; echo ",$models," | grep -q ",t8-20m," || ref=${models%%,*}
    $PY $S/timing_readout.py --root $R --manifest $R/hybrid_data/timing_subset.json --models $models --reference $ref \
        --out $R/REPORT_ladders.md --json $R/REPORT_ladders.json 2>&1 | tail -2 | tee -a $LOG
fi

# ---- 6. PySR's remaining runs on the suite: after every other row (owner 2026-09-19) ---------------------------------
if [ "${RUN_PYSR_SUITE:-1}" = 1 ] && [ -x "$PYSR" ]; then
    for i in $(seq 2 $PYSR_SUITE_RUNS); do pysr_suite_run $i || break; done
fi
say "queue finished"
echo TIMING_QUEUE_DONE | tee -a $LOG
