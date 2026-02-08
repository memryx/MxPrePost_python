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
    if (task == "yolov7_det") {
        return new Yolo7Detect(accl, config);
    } else if (task == "yolov8_det") {
        return new YoloUltralyticsDetect(accl, config);
    } else if (task == "yolov8_seg") {
        return new YoloUltralyticsSegment(accl, config);
    } else if (task == "yolov8_pose") {
        return new YoloUltralyticsPose(accl, config);
    } else if (task == "yolov9_det") {
        return new YoloUltralyticsDetect(accl, config);
    } else if (task == "yolov10_det") {
        return new Yolo10Detect(accl, config);
    } else if (task == "yolov11_det") {
        return new YoloUltralyticsDetect(accl, config);
    } else if (task == "yolov11_seg") {
        return new YoloUltralyticsSegment(accl, config);
    } else if (task == "yolov11_pose") {
        return new YoloUltralyticsPose(accl, config);
    }

    throw std::runtime_error("Unsupported task: " + task);
}
