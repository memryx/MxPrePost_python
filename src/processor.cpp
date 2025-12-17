#include "processor.h"

#include "yoloultralytics_detect.h"

using namespace MX::Proc;

Processor* Processor::create(const std::string& task, const YoloDetectConfig& config) {
    if (task == "yolov8_detect") {
        return new YoloUltralyticsDetect(config);
    } else if (task == "yolov8_seg") {
        // TODO:
        return nullptr;
    } else if (task == "yolov8_pose") {
        // TODO:
        return nullptr;
    } else if (task == "yolov11_detect") {
        return new YoloUltralyticsDetect(config);
    }

    throw std::runtime_error("Unsupported task: " + task);
}
