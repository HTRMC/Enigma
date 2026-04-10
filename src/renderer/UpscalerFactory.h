#pragma once

#include "renderer/Upscaler.h"
#include "gfx/Device.h"

#include <memory>

namespace enigma {

// Factory: instantiates the correct IUpscaler backend based on vendor + GPU tier.
// Auto-selection: DLSS on NVIDIA RTX, XeSS on Intel Arc, FSR on all others.
// Override: pass explicit UpscalerBackend::* to force a specific backend.
class UpscalerFactory {
public:
    static std::unique_ptr<IUpscaler> create(
        const VkPhysicalDeviceProperties& props,
        gfx::GpuTier tier,
        UpscalerBackend override = UpscalerBackend::None);

    // Returns the auto-selected backend for a given device.
    static UpscalerBackend autoSelect(const VkPhysicalDeviceProperties& props, gfx::GpuTier tier);
};

} // namespace enigma
