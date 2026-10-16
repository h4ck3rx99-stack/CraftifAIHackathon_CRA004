#!/usr/bin/env python3
"""Junction Sentinel alert subscriber.

Subscribes to the MQTT feed published by the PipeGen/DeepStream pipeline,
prints live per-class counts, and raises a CONGESTION alert when the number
of vehicles seen in the last 60 seconds exceeds a threshold. Alerts are
appended to alerts/incidents.jsonl.

Usage:
    python3 alerts/subscriber.py --host localhost --port 1883 \
        --topic junction/counts --threshold 30
"""
import argparse
import json
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

VEHICLE_CLASSES = {"car", "vehicle", "bus", "truck", "motorbike", "motorcycle",
                   "two-wheeler", "two_wheeler", "twowheeler", "bicycle", "scooter"}
INCIDENTS = Path(__file__).resolve().parent / "incidents.jsonl"


def extract_objects(payload):
    """Return a list of (class_name, track_id) from a DeepStream-style payload.

    Handles the DeepStream minimal schema ("objects": ["id|l|t|r|b|class"]),
    the full schema ("object": {"id":..., "vehicle"/"person": {...}}), and
    plain {"counts": {"car": 3, ...}} style messages.
    """
    found = []
    objs = payload.get("objects")
    if isinstance(objs, list):
        for o in objs:
            if isinstance(o, str):
                parts = o.split("|")
                found.append((parts[-1].strip().lower(), parts[0]))
            elif isinstance(o, dict):
                cls = o.get("label") or o.get("class") or o.get("type") or "unknown"
                found.append((str(cls).lower(), str(o.get("id", ""))))
    obj = payload.get("object")
    if isinstance(obj, dict):
        cls = obj.get("label") or obj.get("type")
        if not cls:
            cls = next((k for k in ("vehicle", "person", "bicycle", "face")
                        if k in obj), "unknown")
            if cls == "vehicle" and isinstance(obj.get("vehicle"), dict):
                cls = obj["vehicle"].get("type") or "vehicle"
        found.append((str(cls).lower(), str(obj.get("id", ""))))
    counts = payload.get("counts")
    if isinstance(counts, dict):
        for cls, n in counts.items():
            found.extend((str(cls).lower(), None) for _ in range(int(n)))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="junction/counts")
    ap.add_argument("--threshold", type=int, default=30,
                    help="vehicles per minute that triggers a CONGESTION alert")
    ap.add_argument("--cooldown", type=int, default=30,
                    help="seconds between repeated alerts")
    ap.add_argument("--incidents", default=str(INCIDENTS),
                    help="JSONL file that alerts are appended to")
    args = ap.parse_args()
    incidents = Path(args.incidents)

    totals = Counter()            # unique tracked objects per class
    seen = set()                  # (class, track_id) already counted
    window = deque()              # timestamps of new vehicles, last 60 s
    last_alert = [0.0]

    def on_connect(client, userdata, flags, rc, *extra):
        print(f"[subscriber] connected to {args.host}:{args.port} rc={rc}, "
              f"topic={args.topic}, threshold={args.threshold} veh/min")
        client.subscribe(args.topic)

    def on_message(client, userdata, msg):
        now = time.time()
        try:
            payload = json.loads(msg.payload.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            print(f"[raw] {msg.payload[:200]!r}")
            return
        for cls, tid in extract_objects(payload):
            key = (cls, tid)
            if tid and key in seen:
                continue
            if tid:
                seen.add(key)
            totals[cls] += 1
            if cls in VEHICLE_CLASSES:
                window.append(now)
        while window and now - window[0] > 60:
            window.popleft()
        vpm = len(window)
        summary = ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
        print(f"[{datetime.now():%H:%M:%S}] counts: {summary or '-'} | "
              f"vehicles/min: {vpm}")
        if vpm > args.threshold and now - last_alert[0] > args.cooldown:
            last_alert[0] = now
            alert = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "CONGESTION",
                "vehicles_per_min": vpm,
                "threshold": args.threshold,
                "totals": dict(totals),
            }
            print(f"!!! CONGESTION ALERT: {vpm} vehicles/min > {args.threshold}")
            with incidents.open("a") as f:
                f.write(json.dumps(alert) + "\n")

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:          # paho-mqtt < 2.0
        client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
