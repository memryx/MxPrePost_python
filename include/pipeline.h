#pragma once
#include "config.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

namespace MX {
    namespace Pipe {
        class Pipeline {
          public:
            virtual ~Pipeline() = default;

            // Pure virtual methods to be implemented by derived classes
            virtual cv::Mat preprocess(const cv::Mat& input) = 0;
            virtual void postprocess(const std::vector<float*>& outputs, Result& result) = 0;
            virtual void draw(cv::Mat& image, const Result& result) = 0;

            // Factory method
            static Pipeline* create(const std::string& task, const YoloConfig& config);
        };

    }
}