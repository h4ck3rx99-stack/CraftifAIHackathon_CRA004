#!/usr/bin/env python3
"""Bengaluru Junction Sentinel - alert engine.

Consumes the per-frame detection/tracking metadata that the PipeGen-generated
DeepStream app publishes on MQTT (junction/events), and turns it into
actionable output:

  * rolling per-class counts of objects crossing a virtual stop line
  * per-minute flow and PCU demand, live occupancy (queue proxy)
  * congestion level LOW / MEDIUM / HIGH with a suggested signal action
  * publishes junction/stats (1 Hz) and junction/alerts (on change / repeat)
  * appends every alert to alerts/incidents.jsonl

It never runs a model: every detection comes from PipeGen's TensorRT engine.
If PipeGen already emits line-crossing counts (nvdsanalytics), those are used
directly; otherwise crossings are derived from PipeGen's tracker IDs + boxes.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque

import yaml
import paho.mqtt.client as mqtt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLASSES = ["car", "two_wheeler", "bus", "pedestrian"]


# --------------------------------------------------------------------------
# Normalising PipeGen / DeepStream message formats
# --------------------------------------------------------------------------
def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bbox(o):
    """Return (left, top, right, bottom) from the box formats DeepStream /
    PipeGen use, or None."""
    b = o.get("bbox") or o.get("box") or o.get("rect") or o.get("bounding_box")
    if isinstance(b, dict):
        if "topleftx" in b:
            return (_num(b["topleftx"]), _num(b["toplefty"]),
                    _num(b["bottomrightx"]), _num(b["bottomrighty"]))
        l = _num(b.get("left", b.get("x", b.get("x1"))))
        t = _num(b.get("top", b.get("y", b.get("y1"))))
        if "right" in b or "x2" in b:
            return (l, t, _num(b.get("right", b.get("x2"))), _num(b.get("bottom", b.get("y2"))))
        w = _num(b.get("width", b.get("w")))
        h = _num(b.get("height", b.get("h")))
        if None not in (l, t, w, h):
            return (l, t, l + w, t + h)
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        l, t, a, c = (_num(x) for x in b[:4])
        fmt = (o.get("bbox_format") or "").lower()
        if fmt in ("xyxy", "ltrb"):
            return (l, t, a, c)
        return (l, t, l + a, t + c)  # DeepStream rect_params style: left, top, width, height
    if all(k in o for k in ("left", "top", "width", "height")):
        l, t = _num(o["left"]), _num(o["top"])
        return (l, t, l + _num(o["width"]), t + _num(o["height"]))
    return None


def _label(o):
    for k in ("label", "class", "class_name", "obj_label", "type", "objectType"):
        v = o.get(k)
        if isinstance(v, str) and v:
            return v
    # DeepStream full schema: {"vehicle": {...}} or {"person": {...}}
    if "person" in o:
        return "person"
    if isinstance(o.get("vehicle"), dict):
        return o["vehicle"].get("type") or "car"
    return None


def normalise(msg):
    """Return dict: ts, frame, width, height, objects[{label, tid, box}],
    lc (line-crossing counts dict or None)."""
    out = {"ts": time.time(), "frame": None, "width": None, "height": None,
           "objects": [], "lc": None}
    out["frame"] = msg.get("frame_num", msg.get("frameId", msg.get("frame", msg.get("id"))))
    for wk, hk in (("width", "height"), ("frame_width", "frame_height")):
        if wk in msg:
            out["width"], out["height"] = _num(msg.get(wk)), _num(msg.get(hk))

    objs = None
    for k in ("objects", "detections", "objs", "object_list"):
        if isinstance(msg.get(k), list):
            objs = msg[k]
            break
    if objs is None and isinstance(msg.get("object"), dict):  # DS full schema, 1 obj/msg
        objs = [msg["object"]]
    for o in objs or []:
        if isinstance(o, str):  # DS minimal schema "tid|l|t|r|b|label[|...]"
            p = o.split("|")
            if len(p) >= 6:
                out["objects"].append({"label": p[5], "tid": p[0],
                                       "box": tuple(_num(x) for x in p[1:5])})
            continue
        if not isinstance(o, dict):
            continue
        tid = o.get("track_id", o.get("object_id", o.get("tracking_id", o.get("id"))))
        out["objects"].append({"label": _label(o), "tid": tid, "box": _bbox(o)})

    # PipeGen's own per-class analytics (PIR nodes 6-7): {cls}_crossed,
    # {cls}_rolling_60s, {cls}_current, {cls}_congestion, crossing_event_count.
    f = msg.get("fields")
    if isinstance(f, dict) and any(k.endswith("_crossed") for k in f):
        out["pg"] = f
    # Native line-crossing analytics (nvdsanalytics) if PipeGen enabled them.
    for k in ("lc_count", "lineCrossing", "line_crossing", "lc_cum_cnt", "crossings"):
        v = msg.get(k) or (msg.get("analyticsModule") or {}).get(k)
        if isinstance(v, dict) and v:
            out["lc"] = v
            break
    return out


# --------------------------------------------------------------------------
# Alert engine
# --------------------------------------------------------------------------
class Sentinel:
    def __init__(self, cfg, client):
        self.cfg = cfg
        self.client = client
        self.cmap = {k.lower(): v for k, v in cfg["class_map"].items()}
        self.line_y = float(cfg["junction"]["stop_line_y"])
        self.fw = float(cfg["junction"]["frame_width"])
        self.fh = float(cfg["junction"]["frame_height"])
        self.window = float(cfg["window_seconds"])
        self.lock = threading.Lock()
        self.last_y = {}                          # tid -> normalised centroid y
        self.counted = set()                      # tids already counted
        self.crossings = deque()                  # (ts, cls)
        self.totals = defaultdict(int)
        self.lc_prev = None
        self.visible = defaultdict(int)
        self.msgs = 0
        self.frames = 0
        self.pg = None          # latest PipeGen per-class analytics fields
        self.pg_frame = None
        self.level = None
        self.last_alert_ts = 0.0
        self.started = time.time()
        log = cfg["incident_log"]
        self.log_path = log if os.path.isabs(log) else os.path.join(ROOT, log)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    # -- ingest ---------------------------------------------------------
    def on_event(self, raw):
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        msgs = msg if isinstance(msg, list) else [msg]
        with self.lock:
            for m in msgs:
                if isinstance(m, dict):
                    self._ingest(normalise(m))

    def _ingest(self, ev):
        self.msgs += 1
        now = time.time()
        if ev.get("pg") is not None:
            self.pg = ev["pg"]
            self.pg_frame = ev["frame"]
        fw = ev["width"] or self.fw
        fh = ev["height"] or self.fh
        visible = defaultdict(int)
        for o in ev["objects"]:
            cls = self.cmap.get(str(o["label"]).lower()) if o["label"] else None
            if not cls:
                continue
            visible[cls] += 1
            if ev["lc"] is not None or o["tid"] in (None, "", -1) or not o["box"]:
                continue
            l, t, r, b = o["box"]
            scale = fh if max(t, b) > 1.5 else 1.0
            # bottom-centre of the box = where the vehicle touches the road
            y = b / scale
            key = (cls, str(o["tid"]))
            prev = self.last_y.get(key)
            self.last_y[key] = y
            if prev is not None and key not in self.counted:
                if (prev < self.line_y <= y) or (prev > self.line_y >= y):
                    self.counted.add(key)
                    self.crossings.append((now, cls))
                    self.totals[cls] += 1
        if ev["lc"] is not None:
            self._ingest_lc(ev["lc"], now)
        if ev["objects"] or ev["frame"] is not None:
            self.visible = visible
            self.frames += 1

    def _ingest_lc(self, lc, now):
        """nvdsanalytics gives cumulative counts per line(/class)."""
        cur = defaultdict(int)
        for name, n in lc.items():
            cls = None
            for lab, c in self.cmap.items():
                if lab in str(name).lower():
                    cls = c
            cur[cls or "car"] += int(_num(n, 0))
        if self.lc_prev is not None:
            for cls, n in cur.items():
                d = n - self.lc_prev.get(cls, 0)
                for _ in range(max(0, d)):
                    self.crossings.append((now, cls))
                self.totals[cls] += max(0, d)
        self.lc_prev = cur

    # -- analytics ------------------------------------------------------
    def snapshot(self):
        now = time.time()
        with self.lock:
            while self.crossings and now - self.crossings[0][0] > self.window:
                self.crossings.popleft()
            flow = defaultdict(int)
            for _, c in self.crossings:
                flow[c] += 1
            visible = dict(self.visible)
            totals = dict(self.totals)
            msgs, frames = self.msgs, self.frames
        elapsed = min(self.window, max(1.0, now - self.started))
        per_min = {c: round(flow.get(c, 0) * 60.0 / elapsed, 1) for c in CLASSES}
        pg_flags = []
        with self.lock:
            pg, pg_frame = self.pg, self.pg_frame
        if pg:
            # Prefer PipeGen's own line-crossing analytics (PIR nodes 6-7).
            PGN = {"car": "cars", "two_wheeler": "two_wheelers", "bus": "buses", "pedestrian": "pedestrians"}
            totals = {c: int(pg.get(PGN[c] + "_crossed", 0)) for c in CLASSES}
            per_min = {c: float(pg.get(PGN[c] + "_rolling_60s", 0)) for c in CLASSES}
            visible = {c: int(pg.get(PGN[c] + "_current", visible.get(c, 0))) for c in CLASSES}
            pg_flags = [c for c in CLASSES if pg.get(PGN[c] + "_congestion")]
        pcu = round(sum(per_min[c] * self.cfg["pcu"][c] for c in CLASSES), 1)
        occ = sum(v for c, v in visible.items() if c != "pedestrian")
        th = self.cfg["thresholds"]
        lvl_f = "HIGH" if pcu >= th["flow_pcu_per_min"]["high"] else \
            "MEDIUM" if pcu >= th["flow_pcu_per_min"]["medium"] else "LOW"
        lvl_o = "HIGH" if occ >= th["occupancy_vehicles"]["high"] else \
            "MEDIUM" if occ >= th["occupancy_vehicles"]["medium"] else "LOW"
        order = ["LOW", "MEDIUM", "HIGH"]
        level = max(lvl_f, lvl_o, key=order.index)
        if pg_flags:  # PipeGen's per-class threshold alert fired -> at least MEDIUM
            level = max(level, "MEDIUM", key=order.index)
        return {
            "ts": round(now, 3),
            "junction": self.cfg["junction"]["name"],
            "approach": self.cfg["junction"]["approach"],
            "totals_crossed": {c: totals.get(c, 0) for c in CLASSES},
            "flow_per_min": per_min,
            "pcu_per_min": pcu,
            "visible_now": {c: visible.get(c, 0) for c in CLASSES},
            "occupancy_vehicles": occ,
            "congestion_level": level,
            "level_basis": {"flow": lvl_f, "occupancy": lvl_o},
            "advice": self.cfg["actions"][level]["advice"],
            "green_delta_s": self.cfg["actions"][level]["green_delta_s"],
            "pedestrian_phase_request": visible.get("pedestrian", 0) >= self.cfg["pedestrian_alert_threshold"],
            "pipegen_congestion_classes": pg_flags,
            "source": "pipegen_fields" if pg else "derived_from_tracks",
            "pipegen_frame": pg_frame,
            "events_received": msgs,
            "frames_received": frames,
        }

    def tick(self):
        s = self.snapshot()
        self.client.publish(self.cfg["mqtt"]["stats_topic"], json.dumps(s))
        now = time.time()
        changed = s["congestion_level"] != self.level
        repeat = s["congestion_level"] != "LOW" and now - self.last_alert_ts >= 15
        if s["frames_received"] and (changed or repeat):
            alert = {
                "ts": s["ts"],
                "type": "congestion_level_change" if changed else "congestion_persisting",
                "from": self.level, "level": s["congestion_level"],
                "pcu_per_min": s["pcu_per_min"],
                "occupancy_vehicles": s["occupancy_vehicles"],
                "flow_per_min": s["flow_per_min"],
                "totals_crossed": s["totals_crossed"],
                "suggested_signal_action": s["advice"],
                "green_delta_s": s["green_delta_s"],
                "pedestrian_phase_request": s["pedestrian_phase_request"],
                "pipegen_congestion_classes": s["pipegen_congestion_classes"],
                "pipegen_frame": s["pipegen_frame"],
            }
            self.client.publish(self.cfg["mqtt"]["alerts_topic"], json.dumps(alert), qos=1)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(alert) + "\n")
            print(f"[ALERT] {alert['level']:6s} pcu/min={alert['pcu_per_min']:5} "
                  f"occ={alert['occupancy_vehicles']:3} -> {alert['suggested_signal_action']}", flush=True)
            self.level = s["congestion_level"]
            self.last_alert_ts = now


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    host = args.host or cfg["mqtt"]["host"]
    port = args.port or cfg["mqtt"]["port"]

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="junction-sentinel")
    except AttributeError:  # paho-mqtt < 2
        client = mqtt.Client(client_id="junction-sentinel")
    sentinel = Sentinel(cfg, client)

    def on_connect(c, userdata, flags, rc, *a):
        c.subscribe(cfg["mqtt"]["events_topic"])
        print(f"[sentinel] connected to {host}:{port}, listening on {cfg['mqtt']['events_topic']}", flush=True)

    client.on_connect = on_connect
    client.on_message = lambda c, u, m: sentinel.on_event(m.payload.decode("utf-8", "replace"))
    client.connect(host, port, keepalive=30)
    client.loop_start()
    try:
        while True:
            time.sleep(1.0)
            sentinel.tick()
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    sys.exit(main())
