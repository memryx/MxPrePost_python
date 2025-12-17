#include "processor.h"
#include "yolov8.h"

using namespace MX::Proc;

Processor *Processor::create(const std::string &task,
                             const YoloDetectConfig &config)
{
    if (task == "yolov8_detect")
    {
        return new YOLOv8(config);
    }
    if (task == "yolov8_seg")
    {
        // TODO:
        return nullptr;
    }

    throw std::runtime_error("Unsupported task: " + task);
}
