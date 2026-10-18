**PipeGen v0.5.1: TensorRT build fails on RTX 50-series (Blackwell), "Unsupported SM: 0xc00"**

Setup: Windows 11 + WSL2 Ubuntu 22.04, NVIDIA driver 616.92, RTX 5050 Laptop GPU (compute capability 12.0). Image `craftifaidocker/deepstream:7.1-amd64-v3`. All startup checks pass, and stages 1 and 2 (Requirement, PIR) work.

Stage 3 (Model) fails on "Compile the model" (YOLO11s ONNX from the catalog, FP16). ONNX inspection reports 321/323 layers supported and 0 unresolved, but ModelStage.log says `builder returned no engine`.

The same build with trtexec inside the container shows why:
```
TensorRT version: 10.3.0   Compute Capability: 12.0
Error[1]: IBuilder::buildSerializedNetwork: Error Code 1: Internal Error (Unsupported SM: 0xc00)
```
TensorRT 10.3 / CUDA 12.6 (DeepStream 7.1) can't target sm_120. Blackwell needs TensorRT 10.8+ and CUDA 12.8+.

Is there a Blackwell-compatible image (DeepStream 8.x / TRT 10.8+) or a supported workaround? Deadline is today. Thanks!
