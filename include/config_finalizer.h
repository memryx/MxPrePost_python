#pragma once

#include "config.h"

namespace MX {
    namespace Pipe {
        class ConfigFinalizer {
          public:
            static YoloFinalConfig finalize(const YoloUserConfig& config);
        };
    }
}
