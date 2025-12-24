#include "pipeline.h"

#include "yoloultralytics_detect.h"
#include "yoloultralytics_pose.h"
#include "yoloultralytics_segment.h"

using namespace MX::Pipe;

Pipeline* Pipeline::create(const std::string& task, const YoloConfig& config) {
    if (task == "yolov8_det") {
        return new YoloUltralyticsDetect(config);
    } else if (task == "yolov8_seg") {
        return new YoloUltralyticsSegment(config);
    } else if (task == "yolov8_pose") {
        return new YoloUltralyticsPose(config);
    } else if (task == "yolov11_det") {
        return new YoloUltralyticsDetect(config);
    } else if (task == "yolov11_seg") {
        return new YoloUltralyticsSegment(config);
    } else if (task == "yolov11_pose") {
        return new YoloUltralyticsPose(config);
    }

    throw std::runtime_error("Unsupported task: " + task);
}
