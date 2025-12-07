# Post Processing API

The Post class performs model output decoding, NMS, class filtering, and annotations.
It is designed to be used inside the output callback of applications.

## General Usage

```python
class App:
	def __init__(self):
		post = Post()
		
	def out_callback(self, fmaps: list[FeatureMap]):
	    results = post(fmaps)

        # do something with results
```

## Scenarios

### 1) Basic
```python
class App:
	def __init__(self):
		post = Post()
		
	def out_callback(self, fmaps: list[FeatureMap]):
        results = post(fmaps) # return a list of Results objects

        for r in results:
            boxes = r.boxes  # Boxes object for bounding box outputs
            masks = r.masks  # Masks object for segmentation masks outputs
            keypoints = r.keypoints  # Keypoints object for pose outputs
            probs = r.probs  # Probs object for classification outputs
```

### 2) General custom config

Example:
```python
post = Post(
    conf=0.5,             # confidence threshold
    iou=0.5,              # IoU threshold for NMS
    imgsz=(640, 640),           # model input size (width, height)
    valid_classes=["person", "ball"],
    nms=True,                   # enable/disable NMS
    task="detection",           # detection | segmentation | pose
)

results = post(fmaps)
for r in results:
    print(r.boxes) # print the Boxes object containing the detection bounding boxes
```

### 3) Per frame override (One-shot configuration)

`post.apply()` accepts multiple arguments that can be passed at processing time to override defaults.

For example: save certain frame without changing base config:

```python
frame_id += 1
post = Post()

results = post.apply(fmaps, imgsz=320, conf=0.5)
```

### 4) Display annotated image

We provide supplementary `draw` function call.
```python
post = Post()
results = post(fmaps)
annotated_image = post.draw(results, raw_image)

# User handles display, e.g. cv2.imshow(...)
```