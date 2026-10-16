#!/bin/bash
# Capture 10 real MQTT messages and 10 real JSON metadata lines from a live run.
# Usage (while run.sh is running):
#   bash evidence/capture.sh <json-file-or-dir> [topic]
set -e
cd "$(dirname "$0")"
SRC="${1:?give the JSON output file or folder written by the PipeGen app}"
TOPIC="${2:-junction/counts}"

echo "Capturing 10 MQTT messages on $TOPIC ..."
timeout 120 mosquitto_sub -h localhost -p 1883 -t "$TOPIC" -v -C 10 > sample_mqtt.txt
echo "  -> $(wc -l < sample_mqtt.txt) lines in evidence/sample_mqtt.txt"

if [ -d "$SRC" ]; then
  SRC=$(ls -t "$SRC"/*.json* 2>/dev/null | head -1)
fi
echo "Capturing 10 JSON lines from $SRC ..."
head -n 10 "$SRC" > sample_json.txt
echo "  -> $(wc -l < sample_json.txt) lines in evidence/sample_json.txt"
