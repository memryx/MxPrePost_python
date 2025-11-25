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
```

## Scenarios

### 1) Basic
```python
post = Post()
results = post(fmaps)
```

### 2) General custom config

Result format:
```
[
    {
        "bbox": [x, y, w, h],
        "cls_id": int,
        "cls_name": str,
        "score": float,
        ...  # Additional fields depending on the task (e.g., mask, keypoints)
    },
    ...
]
```

Example:
```python
post = Post(
    conf_thres=0.5,             # confidence threshold
    iou_thres=0.5,              # IoU threshold for NMS
    shape=(640, 640),           # model input size (width, height)
    valid_classes=["person", "ball"],
    nms=True,                   # enable/disable NMS
    task="detection",           # detection | segmentation | pose
    bbox_format="xywh",         # xywh | xyxy
    save=None                   # optional save path
)

results = post(fmaps)
```

### 3) Per frame override (One-shot configuration)
For example: save certain frame without changing base config.

```python
frame_id += 1
post = Post()

results = post(
    fmaps,
    save=f"output_{frame_id}.png"
)
```

### 4) Display annotated image

We provide supplementary `draw` function call.
```python
post = Post()
results = post(fmaps)
annotated_image = post.draw(results, raw_image)

# User handles display, e.g. cv2.imshow(...)
```