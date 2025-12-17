#pragma once
#include <iostream>
#include <queue>
#include <vector>

namespace MX {
    namespace Proc {
        struct BBox {
            int class_index;    // class index with maximum confident
            float class_score;  // class confident(score)
            float x_min;        // global top-left x relates to model's input feature map size width
            float y_min;        // global top-left y relates to model's input feature map size height
            float x_max;        // global bottom-right x relates to model's input feature map size width
            float y_max;        // global bottom-right y relates to model's input feature map size height

            // Default constructor
            BBox() : class_index(-1), class_score(-1), x_min(-1), y_min(-1), x_max(-1), y_max(-1) {
            }
            // Parameterized constructor
            BBox(int _class_index, float _class_socre, float _x_min, float _y_min, float _x_max, float _y_max) :
                class_index(_class_index), class_score(_class_socre), x_min(_x_min), y_min(_y_min), x_max(_x_max), y_max(_y_max) {
            }
        };

        struct YOLOv8Result {
            std::queue<BBox> bboxes;
            std::queue<std::vector<std::pair<float, float>>> keypoints;
            std::queue<std::vector<float>> mask_features;
        };

        struct YoloDetectConfig {
            int ori_width;
            int ori_height;
            float conf_thres;
            float iou_thres;
        };
    }
}