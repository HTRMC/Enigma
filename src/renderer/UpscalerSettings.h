#pragma once

#include "core/Types.h"
#include "renderer/Upscaler.h"
#include "gfx/Device.h"

namespace enigma {

struct UpscalerSettings {
    UpscalerBackend  backend         = UpscalerBackend::None; // None = use auto-select
    UpscalerQuality  quality         = UpscalerQuality::Quality;
    f32              sharpness       = 0.0f;   // 0 = off, 1 = max
    bool             frameGenEnabled = false;   // DLSS 3.x MFG (RTX 40+ only)
    bool             autoSelect      = true;    // override backend when false

    // Returns the effective backend (auto-selects if autoSelect == true).
    UpscalerBackend effectiveBackend(const VkPhysicalDeviceProperties& props, gfx::GpuTier tier) const;
};

} // namespace enigma
