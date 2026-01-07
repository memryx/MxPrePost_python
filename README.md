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
        self.pipe = mxpipe.Pipeline(task="yolov8_det")


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
    task="yolov8_det",          # [required] yolovX_Y (e.g. yolov8_det, yolov8_seg, yolov11_pose, etc.)
    ori_width=1280,             # [required] original image width
    ori_height=640,             # [required] original image height
    conf=0.5,                   # [optional] confidence threshold
    iou=0.5,                    # [optional] IoU threshold for NMS
    classmap_path="/path/to/classmap.txt",  # [optional] Path to a .txt file containing custom class names (one per line). Defaults to COCO dataset.
    valid_classes=[0],       # [optional] List of class IDs to return. All other detections will be ignored (e.g., [0] for person only in COCO dataset).
)
```

### 2. Per frame override (One-shot configuration)

`preprocess()` and `postprocess()` accepts multiple arguments that can be passed at processing time to override the defaults.

For example: Temporarily lower the confidence threshold and return person class only for frame ID 10.

```python
frame_id += 1
if frame_id == 10:
    # Overrides defaults only for this specific call
    result = pipe.postprocess(fmaps, conf=0.3, valid_classes=[0]) 
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
cd POST_API/samples/python

# Create a symbolic link to the built module, note that python version here is based on your virtualenv
# ex: ln -sv ../../pymodule/build/mxpipe.cpython-310-x86_64-linux-gnu.so
ln -sv ../../pymodule/build/mxpipe.cpython-<python_version>-x86_64-linux-gnu.so

# ex:
# python run.py -d models/onnx/YOLO_v8_small_640_640_3_onnx.dfp \
#               --task yolov8_det \
#               --video_paths videos/sample.mp4 \
#               --show
#               --old_bind
python run.py [--task <task>] [-d <onnx_model>] [--video_paths <video>] [--show] [--old_bind]
```
Notes:
- `--task`: yolovX_Y (e.g. yolov8_det, yolov8_seg, yolov11_pose, etc.)
- `--old_bind`: Use legacy accl binding (MultiStreamAsyncAccl)
- `--show`: Display results

### C++ example:
```bash
cd POST_API/samples/cpp

mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)

./yolo [--task <task>] [-d <dfp_path>] [--video_paths "cam:0,vid:video_path"] [--show]
```

## TODO
- [x] YOLOv8 detection / segmentation / pose estimation
- [x] YOLOv9 detection / ~~segmentation~~ / ~~pose estimation~~ (no model)
- [x] YOLOv10 detection / ~~segmentation / pose estimation~~ (no model)
- [x] YOLOv11 detection / segmentation / pose estimation
- [x] Make detect tasks work for custom dataset
- [ ] Make seg tasks work for custom dataset
- [ ] Make pose tasks work for custom dataset
- [ ] Implement One-shot configuration for specific frame
- [ ] Testing!
- [ ] Maybe align with ultralytics Result format and naming, should we?
- [ ] Complete memryx tutorial

(yolov9c/yolov9e segmentation do not map to mx3)
