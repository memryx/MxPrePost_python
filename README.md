# MxPrepost Package

`mxprepost` provides a unified, high-performance interface for YOLO **pre-processing** and **post-processing**, designed for seamless integration with **MemryX MXA** applications across YOLOv8–YOLOv11 tasks such as **detection**, **segmentation**, and **pose estimation**.  

It handles tasks like:
- Pre-processing: Frame preparation (e.g., resizing, normalization).
- Post-processing: Model output decoding, Non-Maximum Suppression (NMS), class filtering, and result annotation.

It is intended to be used inside the **MXA input/output callbacks** (via `mxapi.MxAccl`). Under the hood, optimized C++ bindings accelerate compute-intensive operations such as **NMS** and output decoding.

---

# General Usage Pattern (MXA Integration)

1. Create `mxapi.MxAccl`
2. Connect input/output callbacks
3. Initialize `mxprepost.MxPrepost` with the accelerator
4. Call:

   * `preprocess(frame)` in input callback
   * `postprocess(mxa_output, ori_width, ori_height)` or `postprocess(mxa_output, ori_frame)` in output callback
   * `draw(frame, result)` for visualization
  
```python
import cv2
import mxprepost
from memryx import mxapi

class App:
    def __init__(self, dfp_path, video_path):
        # Open video source
        self.cap = cv2.VideoCapture(video_path)

        # Store original frame dimensions
        self.ori_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ori_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create MXA accelerator
        self.accl = mxapi.MxAccl(dfp_path, [0], [False, False], False)

		#---------------------------------------------------------------
		#-------------------------- MxPrepost --------------------------
		#---------------------------------------------------------------
        # Initialize MxPrepost
        self.pp = mxprepost.MxPrepost(
            accl=self.accl,
            task="yolov8-det"
        )

        # Connect callbacks
        self.accl.connect_stream(self.in_callback, self.out_callback, stream_id=0)

    def in_callback(self, stream_id):
        ret, frame = self.cap.read()
        if not ret:
            return None
		#---------------------------------------------------------------
		#-------------------------- MxPrepost --------------------------
		#---------------------------------------------------------------
        return self.pp.preprocess(frame)

    def out_callback(self, mxa_output, stream_id):

		#---------------------------------------------------------------
		#-------------------------- MxPrepost --------------------------
		#---------------------------------------------------------------
        # Postprocess using original dimensions
        result = self.pp.postprocess(
            mxa_output,
            self.ori_width,
            self.ori_height
        )

        # Access outputs (depending on task)
		# ---------------------------
        # Detection (Bounding Boxes)
        # ---------------------------
        if result.boxes is not None:
            for box in result.boxes:
                print("xyxy:", box.xyxy)        # [x1, y1, x2, y2]
                print("xywh:", box.xywh)        # [x_center, y_center, w, h]
                print("conf:", box.conf)        # confidence score
                print("cls_id:", box.cls_id)    # class index
                print("cls_name:", box.cls_name)# class label string
                print("----")

        # ---------------------------
        # Segmentation (Masks)
        # ---------------------------
        if result.masks is not None:
            for mask in result.masks:
                # mask is a polygon (list of Point2f), not a binary HxW array
                pts = [(p.x, p.y) for p in mask.xys]
                print("mask cls_id:", int(mask.cls_id))
                print("num polygon points:", len(pts))
                if pts:
                    print("first 5 points:", pts[:5])
                print("----")

        # ---------------------------
        # Pose Estimation (Keypoints)
        # ---------------------------
        if result.keypoints is not None:
            for det_id, kps in enumerate(result.keypoints):
                print(f"detection {det_id} keypoints:", len(kps))
                # each kp has kp.xy (Point2f) and kp.conf
                for kp_id, kp in enumerate(kps):
                    print(f"  kp[{kp_id}] = (x={kp.xy.x:.2f}, y={kp.xy.y:.2f}, conf={kp.conf:.3f})")
                print("----")

        # Optional: draw detections
        # annotated = self.pp.draw(frame, result)
```

---

# Configuration

```python
prepost = mxprepost.MxPrepost(
    accl=accl,                    # [required] mxapi.MxAccl instance
    task="yolov8-det",            # [required] Expected format: yolov<n>-[det|seg|pose]
    conf=0.3,                     # [optional] Default 0.3
    iou=0.4,                      # [optional] Default 0.4
    classmap_path="/path/to/classmap.txt"  # [optional] Path to a .txt file containing custom class names (one per line). Defaults to COCO dataset if not given.
    custom_class_labels=List[str], # [optional] List of custom class labels, instead of using a text file input.
    valid_classes=[0],          # [optional] List of class IDs to return. All other detections will be ignored (e.g., [0] for person only in COCO dataset).
    class_agnostic=True,        # [optional] If False, use class-specific NMS. If True, use class-agnostic NMS. Default True.
    fast_sigmoid=False,         # [optional] If True, use an approximated sigmoid function for slightly faster processing. Default False.
    model_id=0,                 # [optional] model_id in the .dfp file. Default 0
    override_layer_mapping={}   # [optional] Advanced feature to override the per-stride layer name->dfp port mapping.
)
```

### Notes

* `task` examples:

  * `yolov8-det`
  * `yolov8-seg`
  * `yolov11-pose`
* `model_id` is required only if multiple models are compiled into the same DFP.

---

# Python Quick Start

## 1️⃣ Build and Install

```bash
git clone --recurse-submodules git@github.com:memryx/MxPrepost.git
cd MxPrepost
```
```bash
# activate your virtualenv with MemryX SDK 2.2 
source ~/.mx/bin/activate
sh build.sh
```

---

## 2️⃣ Prepare Python Sample

```bash
cd MxPrepost/samples/python

# link mxprepost module in same directory as you targeted python code.
ln -sfv ../../pymodule/build/mxprepost.cpython-*.so .
```
>Copy the .so file to the same path as your python code if working in another directory. 
---

## 3️⃣ Run YOLO Demo

### Webcam Example

```bash
python run.py \
  -d models/yolov8.dfp \
  -t yolov8-det \
  --video_paths /dev/video0
```

### Video File Example

```bash
python run.py \
  -d models/yolov8.dfp \
  -t yolov8-det \
  --video_paths videos/sample.mp4
```

### Multiple Streams Example

```bash
python run.py \
  -d models/yolov8.dfp \
  -t yolov8-det \
  --video_paths /dev/video0 videos/sample.mp4
```

### Disable Display (Benchmark Mode)

```bash
python run.py \
  -d models/yolov8.dfp \
  -t yolov8-det \
  --video_paths /dev/video0 \
  --no-show
```

---

## CLI Arguments

| Argument        | Description                                                  | Default                        |
| --------------- | ------------------------------------------------------------ | ------------------------------ |
| `-d`, `--dfp`   | Path to compiled `.dfp` file                                 | **Required**                   |
| `-t`, `--task`  | YOLO task (`yolov8-det`, `yolov8-seg`, `yolov11-pose`, etc.) | **Required**                   |
| `--video_paths` | One or more input sources (camera or video files)            | `/dev/video0`                  |
| `--no-show`     | Disable display window                                       | Display **enabled** by default |

---

## C++ Usage Pattern (MXA Integration)

The C++ API follows the same pattern as Python:

1. Create `MX::Runtime::MxAccl`
2. Connect input/output callbacks
3. Create `MxPrepost` with the accelerator + task
4. Call:

   * `preprocess(frame)` in the **input callback**
   * `postprocess(outputs, ori_w, ori_h)` (or `postprocess(outputs, result, original_image)`) in the **output callback**
   * `draw(frame, result)` for visualization

### Minimal C++ snippet (mxprepost inside callbacks)

```cpp
#include <memx/accl/MxAccl.h>
#include <memx/prepost/MxPrepost.h>
#include <opencv2/opencv.hpp>

using namespace MX::Runtime;

// Create MxAccl (dfp path, device ids, use_model_shape flags, local)
MxAccl accl{dfp_path, {0}, {false, false}, /*local=*/false};

//---------------------------------------------------------------
//-------------------------- MxPrepost --------------------------
//---------------------------------------------------------------
// Create MxPrepost (task examples: "yolov8-det", "yolov8-seg", "yolov11-pose")
YoloUserConfig cfg; // inits with the defaults for COCO
cfg.conf = 0.3f;
cfg.iou  = 0.4f;
std::unique_ptr<MxPrepost> pp{MxPrepost::create(&accl, task, cfg)};

// Get original dimensions once (from your cv::VideoCapture)
int ori_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
int ori_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

// --- Input callback: capture + preprocess ---
auto in_cb = [&](std::vector<const MX::Types::FeatureMap*> dst, int stream_id) -> bool {
    cv::Mat frame;
    if (!cap.read(frame)) return false;

    //---------------------------------------------------------------
	//-------------------------- MxPrepost --------------------------
	//---------------------------------------------------------------
    std::unique_ptr<MxPrepost> pp;

    // ===========================
    // Method 1: Throwing create()
    // ===========================
    try {
        pp.reset(MxPrepost::create(&accl, task, cfg));   // may throw (e.g., unsupported task)
    } catch (const std::exception& e) {
        std::cerr << "Failed to create MxPrepost: " << e.what() << "\n";
        return 1; // or handle error appropriately
    }

    // ==================================
    // Method 2: No-throw create_safe()
    // ==================================
    // std::string err;
    // if (!MxPrepost::create_safe(&accl, task, cfg, pp, err)) {
    //     std::cerr << "Failed to create MxPrepost: " << err << "\n";
    //     return 1; // or handle error appropriately
    // }

    dst[0]->set_data(reinterpret_cast<float*>(input.data));
    return true;
};

// --- Output callback: postprocess
auto out_cb = [&](std::vector<const MX::Types::FeatureMap*> outs, int stream_id) -> bool {
    // Copy accelerator outputs into your own float buffers, then build float* vector:
    
    //---------------------------------------------------------------
	//-------------------------- MxPrepost --------------------------
	//---------------------------------------------------------------
    Result result;
    pp->postprocess(ofmap_ptrs, result, ori_w, ori_h);   // or: pp->postprocess(ofmap_ptrs, result, original_frame)

    // Access outputs (depending on task)
    // ---------------------------
    // Detection (Bounding Boxes)
    // ---------------------------
    for (const auto& box : result.boxes) {
        // box.xyxy = [x1,y1,x2,y2], box.xywh = [xc,yc,w,h]
        // box.conf, box.cls_id, box.cls_name
    }

    // ---------------------------
    // Segmentation (polygons)
    // ---------------------------
    for (const auto& mask : result.masks) {
        // mask.xys is a polygon: vector<Point2f>
        // mask.cls_id
    }

    // ---------------------------
    // Pose (keypoints)
    // ---------------------------
    for (size_t det_id = 0; det_id < result.keypoints.size(); ++det_id) {
        const auto& kps = result.keypoints[det_id]; // vector<Keypoint>
        // each kp has kp.xy (Point2f) and kp.conf
    }

    // Optional visualization
    // pp->draw(frame, result);

    return true;
};

accl.connect_stream(in_cb, out_cb, /*stream_id=*/0);
accl.start();
accl.wait();
```

---

### Notes

* Prefer `postprocess(ofmaps, result, ori_w, ori_h)` when you already know original frame size.
* Use `postprocess(ofmaps, result, original_frame)` if you have the original `cv::Mat` available in the output callback.


# Compatibility

⚠️ This library does **not** support legacy Accl bindings:

* `SyncAccl`
* `AsyncAccl`
* `MultistreamAsyncAccl`

Only `mxapi.MxAccl` is supported.

---

# Supported Models

* ✅ YOLOv7 (det only)
* ✅ YOLOv8 (det / seg / pose)
* ✅ YOLOv9 (det only)
* ✅ YOLOv10 (det only)
* ✅ YOLOv11 (det / seg / pose)
* ✅ Custom detection datasets for all of the above
* ✅ Custom model resolutions (not just 640x640)


