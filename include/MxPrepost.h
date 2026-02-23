#pragma once
#include "config.h"

namespace MX {
    namespace Prepost {
        class MxError : public std::runtime_error {
          public:
            using std::runtime_error::runtime_error;
        };

        class UnsupportedTaskError : public MxError {
          public:
            using MxError::MxError;
        };

        class MxPrepost {
          public:
            /**
             * Create a pre/post-processing object for the given task.
             *
             * @param accl   Accelerator runtime object (must be non-null).
             * @param task   Task string like "yolov8-det".
             * @param config User config.
             *
             * @return Raw pointer owned by caller (or wrap into std::unique_ptr).
             *
             * @throws UnsupportedTaskError if task is not recognized.
             */
            virtual ~MxPrepost() = default;

            // Pure virtual methods to be implemented by derived classes
            virtual cv::Mat preprocess(const cv::Mat& input) = 0;

            // Legacy API (keep for compatibility; derived YOLO classes can throw here)
            virtual void postprocess(const std::vector<float*>& outputs, Result& result) = 0;

            // NEW API: postprocess requires original shape context
            virtual void postprocess(const std::vector<float*>& outputs,
                                     Result& result,
                                     int ori_w,
                                     int ori_h) = 0;

            // NEW API: convenience overload using original image
            virtual void postprocess(const std::vector<float*>& outputs,
                                     Result& result,
                                     const cv::Mat& original_image) = 0;

            virtual void draw(cv::Mat& image, const Result& result) = 0;

            // Factory method (may throw std::runtime_error on invalid task)
            static MxPrepost* create(MX::Runtime::MxAccl* accl,
                                     const std::string& task,
                                     const YoloUserConfig& config);

            // No-throw factory (for library users who don't want exceptions)
            static bool create_safe(MX::Runtime::MxAccl* accl,
                                    const std::string& task,
                                    const YoloUserConfig& config,
                                    std::unique_ptr<MxPrepost>& out,
                                    std::string& err) noexcept;
        };

    }  // namespace Prepost
}  // namespace MX
