#include "MxPrepost.h"

#include "yolo10_detect.h"
#include "yolo7_detect.h"
#include "yoloultralytics_detect.h"
#include "yoloultralytics_pose.h"
#include "yoloultralytics_segment.h"

using namespace MX::Runtime;

MxPrepost* MxPrepost::create(MX::Runtime::MxAccl* accl,
                             const std::string& task,
                             const YoloUserConfig& config) {
    if (task == "yolov7-det") {
        return new Yolo7Detect(accl, config, task);
    } else if (task == "yolov8-det") {
        return new YoloUltralyticsDetect(accl, config, task);
    } else if (task == "yolov8-seg") {
        return new YoloUltralyticsSegment(accl, config, task);
    } else if (task == "yolov8-pose") {
        return new YoloUltralyticsPose(accl, config, task);
    } else if (task == "yolov9-det") {
        return new YoloUltralyticsDetect(accl, config, task);
    } else if (task == "yolov10-det") {
        return new Yolo10Detect(accl, config, task);
    } else if (task == "yolov11-det") {
        return new YoloUltralyticsDetect(accl, config, task);
    } else if (task == "yolov11-seg") {
        return new YoloUltralyticsSegment(accl, config, task);
    } else if (task == "yolov11-pose") {
        return new YoloUltralyticsPose(accl, config, task);
    }

    throw std::runtime_error("Unsupported task: " + task);
}
