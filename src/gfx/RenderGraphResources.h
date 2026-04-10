#pragma once

#include "core/Types.h"

#include <limits>
#include <string>

namespace enigma::gfx {

// Opaque handle to a registered image resource in RenderGraph.
// Index == kInvalid means "no resource".
struct RGImageHandle {
    static constexpr u32 kInvalid = std::numeric_limits<u32>::max();
    u32 index = kInvalid;
    bool valid() const { return index != kInvalid; }
};

// Opaque handle to a registered buffer resource in RenderGraph.
// (Not yet used in Phase 0B; reserved for Phase 1 acceleration structures.)
struct RGBufferHandle {
    static constexpr u32 kInvalid = std::numeric_limits<u32>::max();
    u32 index = kInvalid;
    bool valid() const { return index != kInvalid; }
};

} // namespace enigma::gfx
