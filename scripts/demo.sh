#!/usr/bin/env bash
# One-command demo: MQTT broker + alert engine + dashboard + PipeGen-generated
# DeepStream app (pipegen_export/run.sh). Ctrl+C stops everything.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { python3 -m venv .venv && .venv/bin/pip -q install -r requirements.txt; }
DASH_PORT="${DASH_PORT:-8080}"
PIDS=()

cleanup() {
  echo; echo "[demo] stopping..."
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
  echo "[demo] done. Alerts: $ROOT/alerts/incidents.jsonl"
}
trap cleanup EXIT INT TERM

# 1. MQTT broker (1883 + websockets 9001), systemd service from setup
if ! systemctl is-active --quiet mosquitto; then sudo systemctl start mosquitto; fi
echo "[demo] mosquitto: $(systemctl is-active mosquitto)"

# 2. raw evidence taps
mosquitto_sub -h localhost -t junction/events -v > logs/mqtt_events.log & PIDS+=($!)
mosquitto_sub -h localhost -t junction/alerts -v > logs/mqtt_alerts.log & PIDS+=($!)

# 3. alert engine (consumes PipeGen output only)
"$PY" alerts/subscriber.py > logs/subscriber.log 2>&1 & PIDS+=($!)

# 4. dashboard
( cd dashboard && exec "$PY" -m http.server "$DASH_PORT" --bind 0.0.0.0 ) > logs/dashboard.log 2>&1 & PIDS+=($!)
echo "[demo] dashboard: http://localhost:$DASH_PORT/"
[ -z "${NO_BROWSER:-}" ] && command -v explorer.exe >/dev/null && explorer.exe "http://localhost:$DASH_PORT/" 2>/dev/null || true

# 5. PipeGen-generated DeepStream app (TensorRT engine, JSON + MQTT out).
# run.sh executes inside PipeGen's own DeepStream container, from PipeGen's
# project directory (where the engine and compiled parser libs live).
CONTAINER="${PIPEGEN_CONTAINER:-craftifai-amd64-v3}"
PG_DIR="${PIPEGEN_DIR:-$(ls -d "$HOME"/.pipegen/*/*-*-*-*-*/pipeline/deepstream 2>/dev/null | head -1)}"
docker start "$CONTAINER" >/dev/null 2>&1 || true
echo "[demo] starting PipeGen pipeline: $PG_DIR/run.sh (container $CONTAINER)"
echo "[demo] note: on RTX 50-series the first CUDA kernels JIT-compile for ~60-75 s before frames flow"
docker exec -e CUDA_CACHE_MAXSIZE=4294967296 "$CONTAINER" bash -c "cd '$PG_DIR' && bash ./run.sh" 2>&1 \
  | grep --line-buffered -vE "GStreamer-WARNING|cannot open shared object|Failed to load plugin" | tee logs/pipegen_run.log
echo "[demo] pipeline finished; leaving alert engine 5s to flush"
sleep 5
