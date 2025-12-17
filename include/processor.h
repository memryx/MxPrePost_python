#pragma once
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "config.h"

namespace MX
{
    namespace Proc
    {
        class Processor
        {
        public:
            virtual ~Processor() = default;

            // Pure virtual methods to be implemented by derived classes
            virtual cv::Mat preprocess(const cv::Mat &input) = 0;
            virtual void postprocess(const std::vector<float *> &outputs,
                                     YOLOv8Result &result) = 0;
            // virtual void draw(cv::Mat& image) = 0;

            // Factory method
            static Processor *create(const std::string &task,
                                     const YoloDetectConfig &config);
        };

    }
}