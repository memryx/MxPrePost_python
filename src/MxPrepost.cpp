#include "MxPrepost.h"

//#include "yolo10_detect.h"
#include "yolo7_detect.h"
#include "yoloultralytics_detect.h"
//#include "yoloultralytics_pose.h"
//#include "yoloultralytics_segment.h"

#include <limits>
#include <string>
#include <unordered_map>

using namespace MX::Prepost;
using namespace MX::Runtime;

namespace {

    using CreatorFn =
            std::function<MxPrepost*(MxAccl*, const YoloUserConfig&, const std::string&)>;

    const std::unordered_map<std::string, CreatorFn> kRegistry = {
            {"yolov7-det",
             [](auto* a, const auto& c, const auto& t) { return new Yolo7Detect(a, c, t); }},
            {"yolov8-det",
             [](auto* a, const auto& c, const auto& t) {
                 return new YoloUltralyticsDetect(a, c, t);
             }},
            //{"yolov8-seg",
            // [](auto* a, const auto& c, const auto& t) {
            //     return new YoloUltralyticsSegment(a, c, t);
            // }},
            //{"yolov8-pose",
            // [](auto* a, const auto& c, const auto& t) {
            //     return new YoloUltralyticsPose(a, c, t);
            // }},
            {"yolov9-det",
             [](auto* a, const auto& c, const auto& t) {
                 return new YoloUltralyticsDetect(a, c, t);
             }},
            //{"yolov10-det",
            // [](auto* a, const auto& c, const auto& t) { return new Yolo10Detect(a, c, t); }},
            {"yolov11-det",
             [](auto* a, const auto& c, const auto& t) {
                 return new YoloUltralyticsDetect(a, c, t);
             }},
            //{"yolov11-seg",
            // [](auto* a, const auto& c, const auto& t) {
            //     return new YoloUltralyticsSegment(a, c, t);
            // }},
            //{"yolov11-pose",
            // [](auto* a, const auto& c, const auto& t) {
            //     return new YoloUltralyticsPose(a, c, t);
            // }},
    };

    static std::string suggest_task(const std::string& input) {
        const std::string in = MX::Prepost::Util::normalize(input);

        std::size_t best_dist = std::numeric_limits<std::size_t>::max();
        std::string best;

        for (const auto& kv : kRegistry) {
            const std::string candidate = MX::Prepost::Util::normalize(kv.first);
            const std::size_t d = MX::Prepost::Util::levenshtein(in, candidate);
            if (d < best_dist) {
                best_dist = d;
                best = kv.first;
            }
        }

        return (best_dist <= 2) ? best : std::string{};
    }

    static std::string valid_task_list() {
        std::ostringstream oss;
        std::size_t i = 0;
        for (const auto& kv : kRegistry) {
            oss << kv.first;
            if (++i < kRegistry.size())
                oss << ", ";
        }
        return oss.str();
    }

}  // anonymous namespace

// ------------------------
// Throwing factory
// ------------------------
MxPrepost* MxPrepost::create(MxAccl* accl, const std::string& task, const YoloUserConfig& config) {
    auto it = kRegistry.find(task);
    if (it != kRegistry.end()) {
        return it->second(accl, config, task);
    }

    std::ostringstream msg;
    msg << "MxPrepost - Unsupported Task Error: '" << termcolor::red << task << termcolor::reset
        << "'.\n";

    if (auto suggestion = suggest_task(task); !suggestion.empty()) {
        // bold+green suggestion, then reset
        msg << "Did you mean: '"
            << "\033[1m" << suggestion << termcolor::reset
            << "'? Diff: " << MX::Prepost::Util::colored_diff(task, suggestion) << "\n";
    }

    msg << "Expected format: yolov<n>-[det|seg|pose].\n";
    msg << "Available tasks: " << valid_task_list() << ".\n";

    throw UnsupportedTaskError(msg.str());
}
