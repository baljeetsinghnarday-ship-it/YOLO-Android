# UrbanOmniDetect Android

Kotlin Android image inference using the ONNX export of UrbanOmniDetect-2. The default model is
the largest checkpoint, **v2-X at 640 px**. The app expects one model forward pass to produce COCO
boxes/classes and 8 projected cuboid corners for the six road-user classes.

## Export the large model

From the root of the cloned `urbanomnidetect` Python repository, install its requirements, download
the `urbanomnidetect_v2_x_640.pt` checkpoint as described in that repository's README, then export
it as a pose model:

```powershell
python -c "import v2.hybrid_v2; from ultralytics import YOLO; model = YOLO('checkpoints/urbanomnidetect_v2_x_640.pt', task='pose'); print(model.export(format='onnx', imgsz=640, opset=12, dynamic=False, simplify=False, nms=False))"
```

Copy the printed ONNX output to:

```text
app/src/main/assets/urbanomnidetect_v2_x_640.onnx
```

The export must be made from v2-X, not the older pose-only v1 checkpoint or a detection-only
checkpoint. ONNX exports are not checked into this project.

## Inference behavior

The runner implements the image path used by the Python model:

- RGB NCHW float input, normalized to `[0, 1]`, letterboxed to 640 x 640 with value 114
- confidence threshold 0.1 and class-aware NMS at IoU 0.7 for raw YOLO pose outputs
- COCO class ids/names for all 80 detection classes
- 8 keypoints decoded as `(x, y, visibility)` and mapped from letterbox coordinates to original
  image pixels
- cuboids drawn only for COCO ids `0, 1, 2, 3, 5, 7` when all eight points pass the same
  visibility and expanded-box sanity checks used in the Python v2 pipeline
- end-to-end output rows are accepted without a second NMS pass

It accepts the two common Ultralytics pose output layouts:

- raw `[1, 108, candidates]` or `[1, candidates, 108]` (4 box + 80 class + 24 keypoint channels)
- end-to-end `[1, detections, 30]` or `[1, 30, detections]`
  (xyxy + confidence + class + 24 keypoint values)

Unsupported tensor layouts fail with an explicit error instead of returning placeholder detections.
The overlay accounts for the `ImageView` fit-center letterboxing so boxes and cuboids stay aligned
with the displayed source image.

## Build and run

Open this directory in Android Studio, let Gradle sync, then run the `app` configuration on an
Android device or emulator (API 26+). Select an image and tap **Run inference**. The X model is
large and may be slow on CPU; inference is kept off the UI thread.

The Android SDK/Gradle toolchain and the ONNX model are not present in the current environment, so
an APK build and numerical comparison against Python have not yet been run. Parity should be
measured on the same image, 640 input size, confidence threshold, and ONNX export before tuning
preprocessing or output decoding.
