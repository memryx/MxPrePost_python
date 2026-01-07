#pragma once

#include "config.h"

// forward declaration
namespace MX::Runtime {
    class MxAccl;
}

namespace MX {
    namespace Pipe {
        class ConfigFinalizer {
          public:
            static YoloFinalConfig finalize(MX::Runtime::MxAccl* accl,
                                            const YoloUserConfig& config);
        };
    }
}
