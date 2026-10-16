# Bengaluru Junction Sentinel

Live traffic-junction analytics for smart signal timing, built with **CraftifAI PipeGen** (Track A).

## Problem

Bengaluru junctions run on fixed signal timings that ignore what is actually waiting at the light. Cars, two-wheelers, buses and pedestrians pile up unevenly, and nobody sees the congestion until it has already spread. Signal controllers need a live, per-class count of what is crossing the junction, and an alert when it is getting congested.

## Track

Track A: PipeGen. The whole vision pipeline was generated, compiled and deployed through PipeGen's 5 stages: Requirement → Pipeline IR → Model → Compiler → Run.

PipeGen requirement text, used verbatim:

> Detect cars, two-wheelers, buses and pedestrians in a traffic video, count each class crossing a virtual line, and publish live per-class counts and congestion alerts as JSON metadata and MQTT.

## Pipeline

| | |
|---|---|
| Input | Traffic video file (`~/videos/…mp4`) |
| Tasks | _filled in from PipeGen's Pipeline IR_ |
| Model | _filled in from PipeGen's model stage_ |
| Precision | TensorRT FP16 (_confirmed from PipeGen's compiler report_) |
| Outputs | JSON metadata + MQTT (`localhost:1883`, topic `junction/counts`) |

## Architecture

```
traffic video ──► PipeGen-generated DeepStream app (run.sh, TensorRT engine on RTX 5050)
                     │
                     ├──► JSON metadata files
                     └──► MQTT  junction/counts ──► alerts/subscriber.py
                                                     ├─ live per-class counts
                                                     └─ CONGESTION alert → alerts/incidents.jsonl
```

## Repo layout

| Path | What |
|---|---|
| `pipegen_export/` | PipeGen's generated project, copied as-is (includes `run.sh`) |
| `alerts/subscriber.py` | MQTT subscriber: live counts + congestion alerts |
| `evidence/` | Pipeline IR screenshot, TensorRT speed report, live-run screenshot, sample JSON/MQTT |

## How to run

Tested on Windows 11 + WSL2 Ubuntu 22.04, NVIDIA RTX 5050 Laptop GPU (driver 616.92), PipeGen 0.5.1.

```bash
sudo service mosquitto start
pip3 install --user paho-mqtt
python3 alerts/subscriber.py --topic junction/counts --threshold 30   # terminal 1
cd pipegen_export && bash run.sh                                      # terminal 2
```

## Sample output

_Real captured lines go in `evidence/sample_json.txt` and `evidence/sample_mqtt.txt`._

## Status

Work in progress. This section is updated with exactly what ran and what didn't.
