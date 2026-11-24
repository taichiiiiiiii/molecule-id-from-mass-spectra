#!/bin/bash
# Run a heavy local job on the shared Mac and kill it if memory_pressure free% drops below MIN_FREE_PCT.
# Usage: MIN_FREE_PCT=15 memguard_run.sh <log> <cmd...>
# Exit 99 = killed by the guard. The log gets /usr/bin/time -l output (peak RSS) and an exit line.
LOG="$1"; shift
MIN=${MIN_FREE_PCT:-15}
/usr/bin/time -l "$@" > "$LOG" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  if [ -n "$f" ] && [ "$f" -lt "$MIN" ]; then
    echo "$(date '+%F %T') memguard: free ${f}% < ${MIN}% -> kill" >> "$LOG"
    pkill -TERM -P $PID; kill -TERM $PID; sleep 5; pkill -KILL -P $PID 2>/dev/null; kill -KILL $PID 2>/dev/null
    exit 99
  fi
  sleep 5
done
wait $PID; rc=$?
echo "$(date '+%F %T') exit=$rc" >> "$LOG"
exit $rc
