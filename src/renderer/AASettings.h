#pragma once

#include "core/Types.h"

namespace enigma {

// Anti-aliasing settings exposed to the UI and owned by Renderer.
// All toggles take effect on the next drawFrame() call.
struct AASettings {
    // SMAA 1x — luma-based morphological AA.
    // Always on by default; three raster passes after post-process tonemap.
    bool smaaEnabled = true;

    // Optional MSAA on the ClusteredForwardPass (transparent geometry only).
    // Opaque geometry goes through the visibility buffer and is unaffected.
    bool msaaEnabled  = false;
    u32  msaaSamples  = 4;   // 2 or 4; 8 requires separate allocation

    // DLAA (future) — ML-based AA; preferred over SMAA for complex content.
    // Not yet implemented; reserved for a future milestone.
    bool dlaaEnabled = false;
};

} // namespace enigma
