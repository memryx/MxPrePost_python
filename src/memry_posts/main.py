from __future__ import annotations
import numpy as np
import cv2
from typing import Iterable
from dataclasses import dataclass, field
from typing import List, Optional

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


@dataclass
class Box:
    """Bounding box container."""

    # xyxy: np.ndarray  # shape: (N, 4)
    score: float
    class_id: int


@dataclass
class Results:
    """Unified output container for each frame."""

    boxes: Optional[Boxes] = None
    # masks: Optional[np.ndarray] = None
    # keypoints: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None


# ---------------------------------------------------------------------
#   Utility Functions
# ---------------------------------------------------------------------


def nms(boxes, scores, iou_thr):
    """Simple NMS implementation returning retained indices."""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[1:][iou < iou_thr]

    return keep


# ---------------------------------------------------------------------
#   Main Post Processor
# ---------------------------------------------------------------------


class Post:
    class NumpyPostProcess:
        """
        Super fast numpy implementation of YOLOv8 post-processing.
        """

        def __init__(self, skip_sigmoid: bool = False):
            self.skip_sigmoid = skip_sigmoid

            self.anchors = self._generate_anchors()
            self.scales = self._generate_scales()
            self._weights = np.arange(16, dtype=np.float32)

        def _generate_anchors(self, sizes=[80, 40, 20]):
            yscales = []
            xscales = []
            for s in sizes:
                r = np.arange(s) + 0.5
                yscales.append(np.repeat(r, s))
                xscales.append(np.repeat(r[None, ...], s, axis=0).flatten())

            yscales = np.concatenate(yscales)
            xscales = np.concatenate(xscales)
            anchors = np.stack([xscales, yscales], axis=1)

            return anchors

        def _generate_scales(self, sizes=[80, 40, 20]):
            factors = [8, 16, 32]
            s = np.concatenate(
                [np.ones([int(s * s)]) * f for s, f in zip(sizes, factors)]
            )
            return s[:, None]

        def convert_to_xywh(self, boxes, valid_indices):
            # Distribution Focal Loss decoding
            boxes = self.dfl(boxes)

            # converts distances to actual [x_center, y_center, width, height]
            boxes = self.dist2bbox(
                boxes, self.anchors[valid_indices], self.scales[valid_indices]
            )

            return boxes

        @staticmethod
        def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
            x = x - np.max(x, axis=axis, keepdims=True)
            np.exp(x, out=x)
            x /= np.sum(x, axis=axis, keepdims=True)
            return x

        @staticmethod
        def _sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))

        def dfl(self, x: np.ndarray) -> np.ndarray:
            x = x.reshape(-1, 4, 16)
            p = self._softmax(x, axis=2)
            p = p * self._weights[None, None, :]
            out = np.sum(p, axis=2, keepdims=False)
            return out

        def dist2bbox(
            self, x: np.ndarray, anchors: np.ndarray, scales: np.ndarray
        ) -> np.ndarray:
            lt = x[:, :2]
            rb = x[:, 2:]

            x1y1 = anchors - lt
            x2y2 = anchors + rb

            wh = x2y2 - x1y1
            c_xy = (x1y1 + x2y2) / 2

            out = np.concatenate([c_xy, wh], axis=1)
            out = out * scales

            return out

        def __call__(
            self, lbox, lcls, mbox, mcls, sbox, scls, onnx_format=False
        ) -> np.ndarray:
            if onnx_format:
                lbox = np.moveaxis(lbox, 1, -1)
                lcls = np.moveaxis(lcls, 1, -1)
                mbox = np.moveaxis(mbox, 1, -1)
                mcls = np.moveaxis(mcls, 1, -1)
                sbox = np.moveaxis(sbox, 1, -1)
                scls = np.moveaxis(scls, 1, -1)

            boxes = np.concatenate(
                [lbox.reshape(-1, 64), mbox.reshape(-1, 64), sbox.reshape(-1, 64)],
                axis=0,
            )
            classes = np.concatenate(
                [lcls.reshape(-1, 80), mcls.reshape(-1, 80), scls.reshape(-1, 80)],
                axis=0,
            )

            if not self.skip_sigmoid:
                classes = self._sigmoid(classes)

            return boxes, classes

    def __init__(
        self,
        ori_shape,
        conf=0.25,
        iou=0.5,
        valid_classes=None,
        nms=True,
        task="detection",  # detection | segmentation | pose
    ):
        self.ori_shape = ori_shape  # (height, width)
        self.ori_height, self.ori_width = ori_shape
        self.conf = conf
        self.iou = iou
        self.valid_classes = valid_classes
        self.enable_nms = nms
        self.task = task
        self._post = self.NumpyPostProcess()

    def __call__(self, fmaps: Iterable[np.ndarray]):
        return self.apply(fmaps)

    def apply(self, fmaps, **kwargs):
        # override config
        conf = kwargs.get("conf", self.conf)
        iou = kwargs.get("iou", self.iou)
        valid_classes = kwargs.get("valid_classes", self.valid_classes)
        enable_nms = kwargs.get("nms", self.enable_nms)
        task = kwargs.get("task", self.task)

        boxes, class_scores = self._post(*fmaps)

        # Calculate the scaling factors for the bounding box coordinates
        self.length = max((self.img_height, self.img_width))
        x_factor = self.length / self.ori_width
        y_factor = self.length / self.ori_height

        # Find the class with the highest score for each detection
        max_scores = np.max(
            class_scores, axis=1
        )  # (8400,) - maximum class score for each detection
        class_ids = np.argmax(class_scores, axis=1)  # (8400,) - index of the best class

        # Filter out detections with scores below the confidence threshold
        valid_indices = np.where(max_scores >= self.confidence_thres)[0]
        if len(valid_indices) == 0:
            return []  # Return an empty list if no valid detections

        # Select only valid detections
        valid_boxes = boxes[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_scores = max_scores[valid_indices]

        # NOTE: In order to speed up, do processing for valid detections only
        valid_boxes = self._post.convert_to_xywh(valid_boxes, valid_indices)

        # Convert bounding box coordinates from (x_center, y_center, w, h) to (left, top, width, height)
        valid_boxes[:, 0] = (
            valid_boxes[:, 0] - valid_boxes[:, 2] / 2
        ) * x_factor  # left
        valid_boxes[:, 1] = (
            valid_boxes[:, 1] - valid_boxes[:, 3] / 2
        ) * y_factor  # top
        valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor  # width
        valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor  # height

        # Create detection dictionaries
        detections = [
            {
                "bbox": valid_boxes[i].astype(int).tolist(),
                "class_id": int(valid_class_ids[i]),
                "class": COCO_CLASSES[int(valid_class_ids[i])],
                "score": valid_scores[i],
            }
            for i in range(len(valid_indices))
        ]

        results = []
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        if len(detections) > 0:
            # NMS requires two lists: bounding boxes and confidence scores
            boxes_for_nms = [d["bbox"] for d in detections]
            scores_for_nms = [d["score"] for d in detections]

            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres
            )

            # Check if indices is not empty
            if len(indices) > 0:
                # Flatten indices if they are returned as a list of arrays
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]

                # Filter detections based on NMS
                for i in indices:
                    box = Box()
                    box.score = detections[i]["score"]
                    box.class_id = detections[i]["class_id"]
                    results.append(box)
            else:
                results = []
        else:
            results = []

        # Return the list of final detections
        return results
