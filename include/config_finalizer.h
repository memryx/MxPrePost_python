#pragma once

#include "config.h"

// forward declaration
namespace MX::Runtime {
    class MxAcclBase;
}

namespace MX {
    namespace Runtime {
        class ConfigFinalizer {
          public:
            static YoloFinalConfig finalize(MX::Runtime::MxAcclBase* accl,
                                            const YoloUserConfig& config);
        };
    }
}
