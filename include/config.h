#pragma once
#include <array>
#include <iostream>
#include <list>
#include <queue>
#include <vector>

namespace MX {
    namespace Proc {
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
                x_min(x_min_),
                y_min(y_min_), x_max(x_max_), y_max(y_max_), conf(conf_), cls_id(cls_id_),
                cls_name(cls_name_) {
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

        struct Result {
            std::list<BBox> bboxes;
            std::list<std::vector<float>> masks;
            std::vector<std::vector<std::pair<float, float>>> keypoints;
        };

        struct YoloDetectConfig {
            int ori_width;
            int ori_height;
            float conf_thres;
            float iou_thres;
        };
    }
}