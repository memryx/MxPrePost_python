#pragma once
#include "config.h"

// Forward declaration
namespace MX::Runtime {
    class MxAccl;
}

namespace MX {
    namespace Runtime {
        class MxPrepost {
          public:
            virtual ~MxPrepost() = default;

            // Pure virtual methods to be implemented by derived classes
            virtual cv::Mat preprocess(const cv::Mat& input) = 0;
            virtual void postprocess(const std::vector<float*>& outputs, Result& result) = 0;
            virtual void draw(cv::Mat& image, const Result& result) = 0;

            // Factory method
            static MxPrepost* create(MX::Runtime::MxAccl* accl,
                                     const std::string& task,
                                     const YoloUserConfig& config);
        };

    }
}