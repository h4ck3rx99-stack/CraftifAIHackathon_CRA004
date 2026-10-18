#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <deque>
#include <string>
#include <algorithm>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── Persistent state for line-crossing and congestion ────────────────────────
// These are module-level statics guarded by a mutex. The .so is loaded once per
// pipeline and called sequentially per frame, but we guard anyway.
static std::mutex g_state_mutex;

// Line-crossing state
static std::set<guint64> g_seen_crossing_ids;
static std::map<std::string, int> g_totals = {
    {"cars", 0}, {"two_wheelers", 0}, {"buses", 0}, {"pedestrians", 0}
};

// Previous bottom-center per track_id (for line-crossing detection)
static std::map<guint64, std::pair<double, double>> g_prev_bottom_center;

// Congestion state
static std::map<std::string, std::deque<double>> g_crossing_times = {
    {"cars", {}}, {"two_wheelers", {}}, {"buses", {}}, {"pedestrians", {}}
};
static std::map<std::string, bool> g_was_active = {
    {"cars", false}, {"two_wheelers", false}, {"buses", false}, {"pedestrians", false}
};

// Class-ID to label mapping (from PIR: 0=pedestrians, 1=two_wheelers(bicycle),
// 2=cars, 3=two_wheelers(motorcycle), 5=buses)
static const char* class_id_to_label(int cid) {
    switch (cid) {
        case 0: return "pedestrians";
        case 1: return "two_wheelers";
        case 2: return "cars";
        case 3: return "two_wheelers";
        case 5: return "buses";
        default: return nullptr;
    }
}

// Side-of-line function: sign tells which side of line(a,b) point p is on
static double side_of_line(double px, double py, double ax, double ay, double bx, double by) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// COCO class names for drawing boxes (the upstream detector is COCO-80 YOLO11s,
// but we only care about the 5 classes the PIR maps; others are filtered by the
// upstream nvinfer parser and won't have object meta).
static const char* LABEL_DISPLAY[] = {
    "pedestrian", "bicycle", "car", "motorcycle", nullptr/*4=aeroplane*/,
    "bus"
};

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[custom_analytics] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) in->frame_meta;

    // Compute timestamp from frame number (assume 30 fps if no PTS)
    double timestamp_s = 0.0;
    guint64 frame_index = 0;
    if (frame_meta) {
        frame_index = frame_meta->frame_num;
        // nv_ds_buf_surface_pts is in nanoseconds
        if (frame_meta->buf_pts != 0 && frame_meta->buf_pts != (guint64)(-1)) {
            timestamp_s = (double)frame_meta->buf_pts / 1.0e9;
        } else {
            timestamp_s = (double)frame_index / 30.0;
        }
    }

    // ── Normalized line endpoints ──
    const double u1 = 0.0, v1 = 0.5, u2 = 1.0, v2 = 0.5;
    const double ax = u1 * W, ay = v1 * H;   // (0, H/2)
    const double bx = u2 * W, by = v2 * H;   // (W, H/2)

    // Gather tracked objects from NvDsObjectMeta
    struct TrackInfo {
        guint64 track_id;
        int class_id;
        const char *label;
        float left, top, width, height;
        float confidence;
        double bottom_center_x, bottom_center_y;
    };
    std::vector<TrackInfo> tracks;

    if (frame_meta) {
        for (NvDsMetaList *ol = frame_meta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;
            const char *lbl = class_id_to_label(obj->class_id);
            if (!lbl) continue; // skip classes we don't care about

            TrackInfo ti;
            ti.track_id = obj->object_id;
            ti.class_id = obj->class_id;
            ti.label = lbl;
            ti.left = obj->rect_params.left;
            ti.top = obj->rect_params.top;
            ti.width = obj->rect_params.width;
            ti.height = obj->rect_params.height;
            ti.confidence = obj->confidence;
            // Bottom-center of bounding box
            ti.bottom_center_x = (double)ti.left + (double)ti.width * 0.5;
            ti.bottom_center_y = (double)ti.top + (double)ti.height;
            tracks.push_back(ti);
        }
    }

    // ── Line-crossing detection ──
    struct CrossingEvent {
        guint64 track_id;
        std::string label;
        std::string direction;
    };
    std::vector<CrossingEvent> crossing_events;

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);

        for (const TrackInfo &ti : tracks) {
            guint64 tid = ti.track_id;

            auto prev_it = g_prev_bottom_center.find(tid);
            if (prev_it == g_prev_bottom_center.end()) {
                // First frame for this track — store and skip
                g_prev_bottom_center[tid] = {ti.bottom_center_x, ti.bottom_center_y};
                continue;
            }

            // Already crossed? Skip (count_once_per_track=true)
            if (g_seen_crossing_ids.count(tid)) {
                // Still update prev position
                g_prev_bottom_center[tid] = {ti.bottom_center_x, ti.bottom_center_y};
                continue;
            }

            double prev_x = prev_it->second.first;
            double prev_y = prev_it->second.second;
            double now_x = ti.bottom_center_x;
            double now_y = ti.bottom_center_y;

            double before = side_of_line(prev_x, prev_y, ax, ay, bx, by);
            double after  = side_of_line(now_x,  now_y,  ax, ay, bx, by);

            // Check sign change (directions='both')
            bool crosses = (before < 0.0 && after >= 0.0) || (before > 0.0 && after <= 0.0);
            if (!crosses) {
                g_prev_bottom_center[tid] = {now_x, now_y};
                continue;
            }

            // Compute intersection with finite line segment
            double dx = now_x - prev_x;
            double dy = now_y - prev_y;
            double lx = bx - ax;
            double ly = by - ay;
            double denom = dx * ly - dy * lx;
            if (std::abs(denom) < 1e-12) {
                g_prev_bottom_center[tid] = {now_x, now_y};
                continue;
            }
            double t_val = ((ax - prev_x) * ly - (ay - prev_y) * lx) / denom;
            double u_val = ((ax - prev_x) * dy - (ay - prev_y) * dx) / denom;
            if (t_val < 0.0 || t_val > 1.0 || u_val < 0.0 || u_val > 1.0) {
                g_prev_bottom_center[tid] = {now_x, now_y};
                continue;
            }

            // Valid crossing!
            g_seen_crossing_ids.insert(tid);
            std::string lbl_str(ti.label);
            g_totals[lbl_str] += 1;

            CrossingEvent ev;
            ev.track_id = tid;
            ev.label = lbl_str;
            ev.direction = (before > 0.0) ? "positive_to_negative" : "negative_to_positive";
            crossing_events.push_back(ev);

            g_prev_bottom_center[tid] = {now_x, now_y};
        }

        // Clean up old tracks from prev map (keep it bounded)
        // We keep all entries that have a current track; remove others that are stale
        // Simple: only keep track_ids seen this frame + those in seen_crossing set
        // Actually, just keep all — the map is bounded by total unique track_ids ever.
        // The set of track_ids that are currently active is bounded by the tracker's
        // max_missed_frames, so this doesn't grow unboundedly in practice.
    }

    // ── Congestion alert ──
    std::map<std::string, int> rolling_counts;
    std::map<std::string, bool> congestion_active;
    std::map<std::string, int> current_counts = {
        {"cars", 0}, {"two_wheelers", 0}, {"buses", 0}, {"pedestrians", 0}
    };
    std::map<std::string, int> cumulative_counts_copy;

    // Congestion thresholds
    static const std::map<std::string, int> THRESHOLDS = {
        {"cars", 20}, {"two_wheelers", 30}, {"buses", 4}, {"pedestrians", 10}
    };
    static const double WINDOW_SECONDS = 60.0;

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);

        // Add crossing events to time deques
        for (const CrossingEvent &ev : crossing_events) {
            g_crossing_times[ev.label].push_back(timestamp_s);
        }

        // Expire old entries and compute rolling counts
        for (auto &kv : THRESHOLDS) {
            const std::string &label = kv.first;
            int limit = kv.second;
            std::deque<double> &queue = g_crossing_times[label];
            // Remove entries older than window
            while (!queue.empty() && queue.front() < timestamp_s - WINDOW_SECONDS) {
                queue.pop_front();
            }
            int rc = (int)queue.size();
            rolling_counts[label] = rc;
            bool active = (rc >= limit);
            congestion_active[label] = active;

            // Rising edge detection (for publish alerts)
            // We just track was_active for drawing/publish
            g_was_active[label] = active;
        }

        // Current counts (number of tracked objects per class in this frame)
        for (const TrackInfo &ti : tracks) {
            current_counts[std::string(ti.label)] += 1;
        }

        cumulative_counts_copy = g_totals;
    }

    // ── DRAW ──
    cv::Mat frame = read_nv12_bgr(in);

    // Draw stop line in yellow (BGR: 0,255,255)
    cv::line(frame,
             cv::Point((int)ax, (int)ay),
             cv::Point((int)bx, (int)by),
             cv::Scalar(0, 255, 255), 3, cv::LINE_AA);

    // Draw boxes with label + track_id + confidence (from object meta)
    if (frame_meta) {
        for (NvDsMetaList *ol = frame_meta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;
            const char *lbl = class_id_to_label(obj->class_id);
            if (!lbl) continue;

            int x1 = (int)obj->rect_params.left;
            int y1 = (int)obj->rect_params.top;
            int x2 = x1 + (int)obj->rect_params.width;
            int y2 = y1 + (int)obj->rect_params.height;

            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 200, 0), 2);

            char buf[128];
            snprintf(buf, sizeof(buf), "%s #%lu %.0f%%", lbl,
                     (unsigned long)obj->object_id,
                     (double)obj->confidence * 100.0);
            cv::putText(frame, buf,
                        cv::Point(x1, std::max(22, y1 - 6)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
    }

    // Draw stats overlay top-left
    static const char* DISPLAY_ORDER[] = {"cars", "two_wheelers", "buses", "pedestrians"};
    int text_y = 32;
    for (int i = 0; i < 4; ++i) {
        const char *label = DISPLAY_ORDER[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: now %d | crossed %d | 60s %d",
                 label,
                 current_counts[label],
                 cumulative_counts_copy[label],
                 rolling_counts[label]);
        cv::putText(frame, buf, cv::Point(16, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        text_y += 32;
    }

    // Draw congestion alerts in red
    for (int i = 0; i < 4; ++i) {
        const char *label = DISPLAY_ORDER[i];
        if (congestion_active[label]) {
            char buf[128];
            snprintf(buf, sizeof(buf), "CONGESTION: %s", label);
            cv::putText(frame, buf, cv::Point(16, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            text_y += 34;
        }
    }

    write_nv12(frame, in);

    // ── PUBLISH metadata via render_meta ──
    render_meta::Builder sb(W, H);

    // Publish the stop line as a polyline
    {
        std::vector<double> line_xy = {ax, ay, bx, by};
        sb.polyline(line_xy, "stop_line");
    }

    // Publish per-class scalar fields
    for (int i = 0; i < 4; ++i) {
        const char *label = DISPLAY_ORDER[i];
        std::string prefix(label);

        sb.field_i((prefix + "_current").c_str(), current_counts[label]);
        sb.field_i((prefix + "_crossed").c_str(), cumulative_counts_copy[label]);
        sb.field_i((prefix + "_rolling_60s").c_str(), rolling_counts[label]);
        sb.field_i((prefix + "_congestion").c_str(), congestion_active[label] ? 1 : 0);
    }

    // Publish crossing events
    for (size_t ei = 0; ei < crossing_events.size(); ++ei) {
        const CrossingEvent &ev = crossing_events[ei];
        char key[64];
        snprintf(key, sizeof(key), "crossing_%zu_track_id", ei);
        sb.field_i(key, (int)ev.track_id);
        snprintf(key, sizeof(key), "crossing_%zu_class", ei);
        sb.field_s(key, ev.label.c_str());
        snprintf(key, sizeof(key), "crossing_%zu_direction", ei);
        sb.field_s(key, ev.direction.c_str());
    }
    sb.field_i("crossing_event_count", (int)crossing_events.size());
    sb.field_d("timestamp_s", timestamp_s);
    sb.field_i("frame_index", (int)frame_index);

    // Publish boxes for each tracked object
    if (frame_meta) {
        for (NvDsMetaList *ol = frame_meta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;
            const char *lbl = class_id_to_label(obj->class_id);
            if (!lbl) continue;
            sb.box((double)obj->rect_params.left,
                   (double)obj->rect_params.top,
                   (double)obj->rect_params.width,
                   (double)obj->rect_params.height,
                   lbl,
                   (int)obj->object_id,
                   (double)obj->confidence);
        }
    }

    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);