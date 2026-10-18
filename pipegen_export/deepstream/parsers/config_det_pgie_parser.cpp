// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[YOLOv11s] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
        for (int dd = 0; dd < layer0.inferDims.numDims; ++dd)
            fprintf(stderr, "[YOLOv11s]   d[%d] = %d\n", dd,
                    (int)layer0.inferDims.d[dd]);
    }

    // ==== BEGIN GENERATED DECODE ====
    // Engine output shape (batch stripped): [84, 8400]
    // layer0.inferDims: numDims=2, d[0]=84, d[1]=8400
    // Layout is [numFeats, numBoxes] where numFeats = 4 + numClasses.
    // We need to "transpose" to iterate per-box.

    const unsigned dim0 = layer0.inferDims.d[0]; // 84  (4 + 80)
    const unsigned dim1 = layer0.inferDims.d[1]; // 8400

    // Identify which dim is the feature dim (== 4 + numClasses) and which is count.
    unsigned numFeats, numBoxes;
    bool transposed; // true if layout is [feats, boxes], false if [boxes, feats]
    if (dim0 == 4u + numClasses) {
        numFeats = dim0;
        numBoxes = dim1;
        transposed = true;
    } else if (dim1 == 4u + numClasses) {
        numFeats = dim1;
        numBoxes = dim0;
        transposed = false;
    } else {
        // Fallback: assume [feats, boxes] (YOLOv8/11 default)
        numFeats = dim0;
        numBoxes = dim1;
        transposed = true;
    }

    for (unsigned i = 0; i < numBoxes; ++i) {
        // Gather cx, cy, w, h and class scores for box i.
        // If transposed (feats-major): data[feat * numBoxes + i]
        // If not transposed (box-major): data[i * numFeats + feat]

        float cx, cy, bw, bh;
        if (transposed) {
            cx = data[0u * numBoxes + i];
            cy = data[1u * numBoxes + i];
            bw = data[2u * numBoxes + i];
            bh = data[3u * numBoxes + i];
        } else {
            cx = data[i * numFeats + 0u];
            cy = data[i * numFeats + 1u];
            bw = data[i * numFeats + 2u];
            bh = data[i * numFeats + 3u];
        }

        // Find best class.
        float maxScore = -1.0f;
        unsigned bestClass = 0;
        unsigned classCount = (numFeats > 4u) ? (numFeats - 4u) : 0u;
        if (classCount > numClasses) classCount = numClasses;

        for (unsigned c = 0; c < classCount; ++c) {
            float s;
            if (transposed) {
                s = data[(4u + c) * numBoxes + i];
            } else {
                s = data[i * numFeats + 4u + c];
            }
            if (s > maxScore) {
                maxScore = s;
                bestClass = c;
            }
        }

        // Threshold — scores are already post-sigmoid probabilities in [0,1].
        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.25f;
        if (maxScore < thr) continue;

        // Convert cx,cy,w,h (in network-input pixel coords) to left,top,width,height.
        float left   = cx - bw * 0.5f;
        float top    = cy - bh * 0.5f;
        float right  = cx + bw * 0.5f;
        float bottom = cy + bh * 0.5f;

        // Clamp to network input.
        left   = std::max<float>(0.0f, std::min<float>(left,   netW));
        top    = std::max<float>(0.0f, std::min<float>(top,    netH));
        right  = std::max<float>(0.0f, std::min<float>(right,  netW));
        bottom = std::max<float>(0.0f, std::min<float>(bottom, netH));

        float w = right - left;
        float h = bottom - top;
        if (w <= 1.0f || h <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = w;
        obj.height = h;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}