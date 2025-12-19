# Mx Pipeline Package

This package provides a unified interface for model pre-processing and post-processing, designed for seamless integration within YOLOv8 - YOLOv11 applications such as  detection, segmentation, and pose estimation.

It handles tasks like:
- Pre-processing: Frame preparation (e.g., resizing, normalization).
- Post-processing: Model output decoding, Non-Maximum Suppression (NMS), class filtering, and result annotation.

This interface is typically used inside the `input and output callbacks` of an application. Under the hood, C++ bindings are used to speed up compute-intensive operations such as NMS.

## General Usage

```python
from memryx import mxpipe # eventually this package will be placed under runtime

class App:
    def __init__(self):
        # Initialize for a specific task
        self.pipe = mxpipe.Pipeline(task="yolov8_detect")


    def in_callback(self):
        frame = cv2.VideoCapture.read()
        frame = self.pipe.preprocess(frame)  # preprocess
        return frame
    
	def out_callback(self, fmaps: list[FeatureMap]):
        results = self.pipe.postprocess(fmaps) # postprocess

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
pipe = Pipeline(
    conf=0.5,                   # confidence threshold
    iou=0.5,                    # IoU threshold for NMS
    imgsz=(640, 640),           # model input size (width, height)
    valid_classes=["person", "ball"],
    nms=True,                   # enable/disable NMS
    task="yolov8_detect",       # yolov8_detect | yolov8_seg | yolov8_pose ...
)
```

### 2. Per frame override (One-shot configuration)

`preprocess()` and `postprocess()` accepts multiple arguments that can be passed at processing time to override the defaults.

For example: Temporarily lower the confidence threshold and disable NMS for frame ID 10.

```python
frame_id += 1
if frame_id == 10:
    # Overrides defaults only for this specific call
    result = pipe.postprocess(fmaps, conf=0.3, nms=False) 
else:
    # Uses the base configuration
    result = pipe.postprocess(fmaps)
```

## Result Visualization

We provide supplementary `draw()` function call for convenient annotation.

```python
results = pipe.postprocess(fmaps)
annotated_image = pipe.draw(results, raw_image)

# User handles display, e.g. cv2.imshow(...)
```

## Installtion
```bash
# clone
git clone https://github.com/memryx/POST_API.git
cd POST_API
git submodule update --init

# build cpp shared library: libmxpipe.so
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)

# build pymodule: mxpipe.cpython-<python_version>-x86_64-linux-gnu.so
source ~/.mx/bin/activate # or whatever your virtualenv
cd POST_API/pymodule
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug  && make -j$(nproc)
```

## Quick Start

### Python example:
```bash
cd POST_API/samples/yolo_det/python

# Create a symbolic link to the built module, note that python version here is based on your virtualenv
# ex: ln -sv ../../pymodule/build/mxpipe.cpython-310-x86_64-linux-gnu.so
ln -sv ../../../pymodule/build/mxpipe.cpython-<python_version>-x86_64-linux-gnu.so

# ex:
# python run.py -d models/onnx/YOLO_v8_small_640_640_3_onnx.dfp \
#               --video_paths videos/sample.mp4 \
#               --show
#               --old_bind
python run.py [-d <onnx_model>] [--video_paths <video>] [--show] [--old_bind]
```
Notes
- `--old_bind`: Use legacy accl binding (MultiStreamAsyncAccl)
- `--show`: Display results

### C++ example:
```bash
cd POST_API/samples/yolo_det/cpp

mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)

./yolo_detect [-d <dfp_path>] [--video_paths "cam:0,vid:video_path"] [--show]
```
