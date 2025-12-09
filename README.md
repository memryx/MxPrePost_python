# PrePost Processing API

The Prepost class provides a unified interface for model pre-processing and post-processing, designed for seamless integration within YOLOv8 - YOLOv11 applications such as  detection, segmentation, and pose estimation.

It handles tasks like:
- Pre-processing: Frame preparation (e.g., resizing, normalization).
- Post-processing: Model output decoding, Non-Maximum Suppression (NMS), class filtering, and result annotation.

This interface is typically used inside the `input and output callbacks` of an application. Under the hood, C++ bindings are used to speed up compute-intensive operations such as NMS.

## General Usage

```python
from memryx import Prepost # import from memryx runtime package

class App:
    def __init__(self):
        # Initialize for a specific task (e.g., detection)
        self.prepost = Prepost(task="detection")


    def in_callback(self):
        frame = cv2.VideoCapture.read()
        frame = self.prepost.preprocess(frame)  # preprocess
        return frame
    
	def out_callback(self, fmaps: list[FeatureMap]):
        results = self.prepost.postprocess(fmaps) # postprocess

        for r in results:
            boxes = r.boxes          # Boxes object for bounding box outputs
            masks = r.masks          # Masks object for segmentation masks outputs
            keypoints = r.keypoints  # Keypoints object for pose outputs
        
        # Do something with results (e.g., display)
```

## Configuration Scenarios

### 1. Default Configuration (Constructor)
You can configure the behavior during initialization. This sets the base configuration for all subsequent calls.

Example:
```python
prepost = Prepost(
    conf=0.5,                   # confidence threshold
    iou=0.5,                    # IoU threshold for NMS
    imgsz=(640, 640),           # model input size (width, height)
    valid_classes=["person", "ball"],
    nms=True,                   # enable/disable NMS
    task="detection",           # detection | segmentation | pose
)
```

### 2. Per frame override (One-shot configuration)

`preprocess()` and `postprocess()` accepts multiple arguments that can be passed at processing time to override the defaults.

For example: Temporarily lower the confidence threshold and disable NMS for frame ID 10.

```python
frame_id += 1
if frame_id == 10:
    # Overrides defaults only for this specific call
    result = prepost.postprocess(fmaps, conf=0.3, nms=False) 
else:
    # Uses the base configuration
    result = prepost.postprocess(fmaps)
```

## Result Visualization

We provide supplementary `draw()` function call for convenient annotation.

```python
results = prepost.postprocess(fmaps)
annotated_image = prepost.draw(results, raw_image)

# User handles display, e.g. cv2.imshow(...)
```