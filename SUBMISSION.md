# Submission text (DRAFT — fields marked [TBD] are filled only from real PipeGen output)

**Bengaluru Junction Sentinel** — Track A (PipeGen)

**Problem:** Bengaluru junctions run fixed signal timings that ignore real demand, so cars, two-wheelers, buses and pedestrians queue unevenly and congestion spreads before anyone notices.

**Pipeline:** Built end to end in CraftifAI PipeGen 0.5.1 from one plain-English requirement: "Detect cars, two-wheelers, buses and pedestrians in a traffic video, count each class crossing a virtual line, and publish live per-class counts and congestion alerts as JSON metadata and MQTT." Input: traffic video file. Tasks: [TBD from Pipeline IR]. Model: [TBD]. Compiled by PipeGen to a TensorRT [TBD: FP16] engine on an RTX 5050 Laptop GPU (WSL2), [TBD fps] per PipeGen's speed report. Outputs: JSON metadata plus MQTT on `junction/counts`; a Python subscriber turns the feed into live per-class counts and CONGESTION alerts (vehicles/min over a threshold), logged to `incidents.jsonl`.

**Bonus:** [TBD: INT8 attempted / not attempted]

Repo: https://github.com/h4ck3rx99-stack/bengaluru-junction-sentinel
