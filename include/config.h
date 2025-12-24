#pragma once
#include <array>
#include <iostream>
#include <opencv2/opencv.hpp> /* imshow */
#include <string>
#include <unordered_set>
#include <vector>

namespace MX {
    namespace Pipe {

        constexpr int MODEL_W = 640;         // Model input width to accelerator
        constexpr int MODEL_H = 640;         // Model input height to accelerator
        constexpr int MODEL_CH = 3;          // Model input channel to accelerator
        constexpr int COORD_FMAP_SIZE = 64;  // Number of feature map channels for bbox coordinates
        constexpr int MASK_FMAP_SIZE = 32;  // Number of feature map channels for mask coefficients
        constexpr int MASK_PROTO_H = 160;   // Mask proto height
        constexpr int MASK_PROTO_W = 160;   // Mask proto width

        /**
         * @brief Labels of COCO dataset, COCO 2014 and 2017 uses the same images but
         * different train/val/test splits. Also, COCO defines 91 classes but the data
         * only uses 80 classes.
         */
        constexpr int COCO_CLASS_NUMBER = 80;
        inline const char* COCO_NAMES[COCO_CLASS_NUMBER] = {
                "person",        "bicycle",       "car",           "motorbike",
                "aeroplane",     "bus",           "train",         "truck",
                "boat",          "traffic light", "fire hydrant",  "stop sign",
                "parking meter", "bench",         "bird",          "cat",
                "dog",           "horse",         "sheep",         "cow",
                "elephant",      "bear",          "zebra",         "giraffe",
                "backpack",      "umbrella",      "handbag",       "tie",
                "suitcase",      "frisbee",       "skis",          "snowboard",
                "sports ball",   "kite",          "baseball bat",  "baseball glove",
                "skateboard",    "surfboard",     "tennis racket", "bottle",
                "wine glass",    "cup",           "fork",          "knife",
                "spoon",         "bowl",          "banana",        "apple",
                "sandwich",      "orange",        "broccoli",      "carrot",
                "hot dog",       "pizza",         "donut",         "cake",
                "chair",         "sofa",          "pottedplant",   "bed",
                "diningtable",   "toilet",        "tvmonitor",     "laptop",
                "mouse",         "remote",        "keyboard",      "cell phone",
                "microwave",     "oven",          "toaster",       "sink",
                "refrigerator",  "book",          "clock",         "vase",
                "scissors",      "teddy bear",    "hair drier",    "toothbrush",
        };

        const std::vector<cv::Scalar> COCO_TEXT_COLORS = {
                {0, 0, 0},
                {255, 255, 255},
                {255, 255, 255},
                {255, 255, 255},
                {255, 215, 0},
        };

        const std::vector<cv::Scalar> COCO_BOX_COLORS = {
                {255, 255, 0, 0.6},
                {26, 35, 126, 0.6},
                {255, 50, 50, 0.6},
                {0, 0, 0, 0.6},
                {51, 51, 51, 0.6},
        };

        struct BBox {
            float x_min;
            float y_min;
            float x_max;
            float y_max;

            // Bounding box representations
            std::array<float, 4> xyxy;  // (left, top, right, bottom)
            std::array<float, 4> xywh;  // (x_center, y_center, width, height)

            float conf = 0.0f;     // confidence score
            int cls_id = -1;       // class index
            std::string cls_name;  // class name (optional)

            // Default constructor
            BBox() = default;

            // Constructor with xyxy
            BBox(float x_min_,
                 float y_min_,
                 float x_max_,
                 float y_max_,
                 float conf_,
                 int cls_id_,
                 const std::string& cls_name_ = "") :
                x_min(x_min_), y_min(y_min_), x_max(x_max_), y_max(y_max_), conf(conf_),
                cls_id(cls_id_), cls_name(cls_name_) {
                update_xyxy();
                update_xywh();
            }

          private:
            void update_xyxy() {
                xyxy[0] = x_min;
                xyxy[1] = y_min;
                xyxy[2] = x_max;
                xyxy[3] = y_max;
            }

            void update_xywh() {
                xywh[0] = (x_min + x_max) * 0.5f;  // x_center
                xywh[1] = (y_min + y_max) * 0.5f;  // y_center
                xywh[2] = x_max - x_min;           // width
                xywh[3] = y_max - y_min;           // height
            }
        };

        struct Point {
            int x;
            int y;
        };

        struct Mask {
            std::vector<Point> xy;
            int cls_id;
        };

        struct Result {
            std::vector<BBox> boxes;
            std::vector<Mask> masks;
            std::vector<std::vector<std::pair<float, float>>> keypoints;
        };

        struct YoloConfig {
            int ori_width = -1;                     // [Required] Original image width
            int ori_height = -1;                    // [Required] Original image height
            float conf = 0.3f;                      // [Optional] Confidence threshold
            float iou = 0.4f;                       // [Optional] IOU threshold for NMS
            std::unordered_set<int> valid_classes;  // [Optional] List of valid class names
            bool fast_sigmoid = false;              // [Optional] Use fast sigmoid or not
        };
    }
}