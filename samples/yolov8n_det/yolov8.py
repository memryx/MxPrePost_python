"""
============
Information:
============
Project: YOLOv8 example code on MXA
File Name: yolov8.py

============
Description:
============
A script to show how to use the Acclerator API to perform a real-time inference
on MX3 using YOLOv8 model.
"""

###################################################################################################

# Imports

import numpy as np
import cv2

###################################################################################################

COCO_CLASSES = ( "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",)

###################################################################################################
###################################################################################################
###################################################################################################


class YoloV8:
    """
    A helper class to run YOLOv8 pre- and post-proccessing.
    """

    class NumpyPostProcess:
        """
        Super fast numpy implementation of YOLOv8 post-processing.
        """


        def __init__(self, skip_sigmoid: bool = False):
            self.skip_sigmoid = skip_sigmoid

            self.anchors = self._generate_anchors()
            self.scales = self._generate_scales()
            self._weights = np.arange(16, dtype=np.float32)

        def _generate_anchors(self, sizes=[80,40,20]):
            yscales = []
            xscales = []
            for s in sizes:
                r = np.arange(s)+0.5
                yscales.append(np.repeat(r, s))
                xscales.append(np.repeat(r[None,...], s, axis=0).flatten())

            yscales = np.concatenate(yscales)
            xscales = np.concatenate(xscales)
            anchors = np.stack([xscales, yscales], axis=1)

            return anchors

        def _generate_scales(self, sizes=[80,40,20]):
            factors = [8,16,32]
            s = np.concatenate(
                [np.ones([int(s*s)])*f for s,f in zip(sizes, factors)]
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
            lt = x[:,:2]
            rb = x[:,2:]

            x1y1 = anchors - lt
            x2y2 = anchors + rb

            wh = x2y2 - x1y1
            c_xy = (x1y1 + x2y2) / 2

            out = np.concatenate([c_xy, wh], axis=1)
            out = out * scales

            return out
        
        def __call__(self, lbox, lcls, mbox, mcls, sbox, scls, onnx_format=False) -> np.ndarray:

            if onnx_format:
                lbox = np.moveaxis(lbox, 1, -1)
                lcls = np.moveaxis(lcls, 1, -1)
                mbox = np.moveaxis(mbox, 1, -1)
                mcls = np.moveaxis(mcls, 1, -1)
                sbox = np.moveaxis(sbox, 1, -1)
                scls = np.moveaxis(scls, 1, -1) 

            boxes = np.concatenate([
                lbox.reshape(-1, 64), 
                mbox.reshape(-1, 64), 
                sbox.reshape(-1, 64)
            ], axis=0)
            classes = np.concatenate([
                lcls.reshape(-1, 80), 
                mcls.reshape(-1, 80), 
                scls.reshape(-1, 80)
            ], axis=0)

            if not self.skip_sigmoid:
                 classes = self._sigmoid(classes)

            return boxes, classes

###################################################################################################
    def __init__(self, stream_img_size=None, model_type='tflite'):
        """
        The initialization function.
        """

        self.name = 'YoloV8'
        self.post = self.NumpyPostProcess()
        self.input_size = (640,640,3) 
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.4
        self.iou_thres = 0.6
        self.model_type = model_type

        self.stream_mode = False
        if stream_img_size:
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
            self.stream_mode = True

###################################################################################################
    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        self.original_imgage = img

        # Get the height and width of the input image
        [self.img_height, self.img_width, _] = self.original_imgage.shape
        
        # Prepare a square image for inference
        self.length = max((self.img_height, self.img_width))
        self.image = np.zeros((self.length, self.length, 3), np.uint8)
        self.image[0:self.img_height, 0:self.img_width] = self.original_imgage

        # Calculate scale factor
        scale = self.length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(self.image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

        if self.model_type == 'tflite':
            # Assume 'blob' is currently (1, 3, 640, 640)
            blob = blob.transpose(0, 2, 3, 1)  # Change to (1, 640, 640, 3)
        elif self.model_type == 'numpy':
            blob = blob.transpose(2, 3, 0, 1)  # Change to (640, 640, 1, 3)

        # Return the preprocessed image data
        return blob

###################################################################################################
    def postprocess(self, output):
        """
        Performs post-processing on the YOLOv8 model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            list: A list of detections where each detection is a dictionary containing 
                    'bbox', 'class_id', 'class', and 'score'.
        """


        # Transpose the output to shape (8400, 84)
        if self.model_type == 'numpy':
            boxes, class_scores = self.post(*output)
        else:
            outputs = np.transpose(np.squeeze(output[0]))
            
            # Extract the bounding box information and class scores in a vectorized manner
            boxes = outputs[:, :4]  # (8400, 4) - x_center, y_center, width, height
            class_scores = outputs[:, 4:]  # (8400, 80) - class scores for 80 classes

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.length / self.input_width
        y_factor = self.length / self.input_height

        # Find the class with the highest score for each detection
        max_scores = np.max(class_scores, axis=1)  # (8400,) - maximum class score for each detection
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
        if self.model_type == "numpy":
            valid_boxes = self.post.convert_to_xywh(valid_boxes, valid_indices)
        
        # Convert bounding box coordinates from (x_center, y_center, w, h) to (left, top, width, height)
        valid_boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * x_factor  # left
        valid_boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * y_factor  # top
        valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor  # width
        valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor  # height

        # Create detection dictionaries
        detections = [{
            'bbox': valid_boxes[i].astype(int).tolist(),
            'class_id': int(valid_class_ids[i]),
            'class': COCO_CLASSES[int(valid_class_ids[i])],
            'score': valid_scores[i]
        } for i in range(len(valid_indices))]

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        if len(detections) > 0:
            # NMS requires two lists: bounding boxes and confidence scores
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]

            indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres)

            # Check if indices is not empty
            if len(indices) > 0:
                # Flatten indices if they are returned as a list of arrays
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]

                # Filter detections based on NMS
                final_detections = [detections[i] for i in indices]
            else:
                final_detections = []
        else:
            final_detections = []

        # Return the list of final detections
        return final_detections


###################################################################################################
if __name__=="__main__":
    pass

# eof
