#!/usr/bin/env bash
# Stop the solomon timing queue without corrupting its bookkeeping.
#
# Order matters: ladder() and baseline_ladder() in timing_queue_solomon.sh mark a ladder DONE whenever their
# runner exits without .failed marks, so the queue script dies FIRST, then the runner and every descendant
# (its refinement workers). Processes are matched by their exact argv, never by a pattern that could also match
# this script's own command line (2026-09-19: a pattern hit the launch wrapper instead of the queue, the queue
# survived, and it marked a half-run ladder done). Any queue mark written while stopping is rolled back.
set -u
K=${K:-$HOME/srbf_clock_kit}; QM=$K/timing/queue_marks
before=$(ls "$QM" 2>/dev/null | sort)
desc() { local c; for c in $(pgrep -P "$1"); do echo "$c"; desc "$c"; done; }
argv_pids() {   # pids whose argv (from the program name on) starts with $1
    ps -eo pid=,args= | awk -v s="$1" '{ pid = $1; $1 = ""; sub(/^ +/, ""); if (index($0, s) == 1) print pid }'
}
queue=$(argv_pids "bash $K/scripts/timing_queue_solomon.sh")
runners=$( { argv_pids "$K/venv_timing/bin/python $K/scripts/run_timing_ladder.py"
             argv_pids "$K/venv_legacy/bin/python $K/scripts/run_timing_ladder.py"
             argv_pids "$K/venv_timing/bin/python $K/scripts/run_hybrid_sweep.py"; } | sort -u)
echo "queue: ${queue:-none}   runners: ${runners:-none}"
[ -n "$queue" ] && kill $queue
sleep 2
tree=$(for r in $runners; do echo "$r"; desc "$r"; done | sort -u | tr '\n' ' ')
[ -n "$tree" ] && kill $tree 2>/dev/null
sleep 6
alive=$(for p in $queue $tree; do kill -0 "$p" 2>/dev/null && echo "$p"; done | tr '\n' ' ')
[ -n "$alive" ] && { echo "still alive after TERM, killing: $alive"; kill -9 $alive; sleep 1; }
after=$(ls "$QM" 2>/dev/null | sort)
new=$(comm -13 <(echo "$before") <(echo "$after"))
for m in $new; do rm -f "$QM/$m"; echo "rolled back a mark written while stopping: $m"; done
left=$( { argv_pids "bash $K/scripts/timing_queue_solomon.sh"; for r in $runners; do kill -0 "$r" 2>/dev/null && echo "$r"; done; } | tr '\n' ' ')
[ -n "${left// /}" ] && { echo "NOT STOPPED: $left"; exit 1; }
echo "stopped; queue marks: $(echo $after | tr '\n' ' ')"
