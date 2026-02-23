#pragma once
#include <array>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_set>
#include <vector>

namespace MX {
    namespace Prepost {

        constexpr int NUM_KEYPOINTS = 17;   // Number of keypoints for pose estimation

        // Pairs of keypoints for drawing skeleton
        constexpr std::array<std::pair<int, int>, 18> KEYPOINT_PAIRS = {{
                {0, 1},
                {0, 2},
                {1, 3},
                {2, 4},
                {0, 5},
                {0, 6},
                {5, 7},
                {7, 9},
                {6, 8},
                {8, 10},
                {5, 6},
                {5, 11},
                {6, 12},
                {11, 12},
                {11, 13},
                {13, 15},
                {12, 14},
                {14, 16},
        }};

        // Color list for drawing keypoints
        const std::vector<cv::Scalar> KEYPOINT_COLORS = {
                cv::Scalar(128, 255, 0),   cv::Scalar(255, 128, 50),  cv::Scalar(128, 0, 255),
                cv::Scalar(255, 255, 0),   cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255),
                cv::Scalar(51, 153, 255),  cv::Scalar(255, 153, 153), cv::Scalar(255, 51, 51),
                cv::Scalar(153, 255, 153), cv::Scalar(51, 255, 51),   cv::Scalar(0, 255, 0),
                cv::Scalar(255, 0, 51),    cv::Scalar(153, 0, 153),   cv::Scalar(51, 0, 51),
                cv::Scalar(0, 0, 0),       cv::Scalar(0, 102, 255),   cv::Scalar(0, 51, 255),
                cv::Scalar(0, 153, 255),   cv::Scalar(0, 153, 153)};

        /**
         * @brief Labels of COCO dataset, COCO 2014 and 2017 uses the same images but
         * different train/val/test splits. Also, COCO defines 91 classes but the data
         * only uses 80 classes.
         */
        constexpr int COCO_CLASS_NUMBER = 80;
        inline const char* COCO_NAMES[COCO_CLASS_NUMBER]{
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

        struct Point2f {
            float x;
            float y;
        };

        struct Mask {
            std::vector<Point2f> xys;
            int cls_id;
        };

        struct Keypoint {
            Point2f xy;
            float conf;

            Keypoint(Point2f xy, float conf) : xy{xy}, conf(conf) {
            }
            Keypoint(float x, float y, float conf) : xy{Point2f{x, y}}, conf(conf) {
            }
        };

        struct Result {
            std::vector<BBox> boxes;
            std::vector<Mask> masks;
            std::vector<std::vector<Keypoint>> keypoints;
        };

        struct YoloUserConfig {
            float conf = 0.3f;  // [Optional] Confidence threshold
            float iou = 0.4f;   // [Optional] IOU threshold for NMS

            // [Optional] Path to a .txt file containing custom class names (one per line).
            // Defaults to COCO dataset.
            std::string classmap_path;

            //  [optional] List of class IDs to return. All other detections will be ignored (e.g.,
            //  [0] for person only in COCO dataset).
            std::unordered_set<int> valid_classes;
            bool class_agnostic = false;  // [Optional] Use class agnostic or not
            bool fast_sigmoid = false;    // [Optional] Use fast sigmoid or not
            int model_id = 0;
        };

        struct YoloFinalConfig {
            float conf;
            float iou;
            std::vector<int> valid_classes;
            bool class_agnostic;
            bool fast_sigmoid;

            // extra params for compared to YoloUserConfig
            std::vector<std::string> class_labels;

            int model_w;
            int model_h;

            int model_id;
        };
    }
}
