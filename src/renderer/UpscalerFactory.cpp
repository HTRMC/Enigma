#include "renderer/UpscalerFactory.h"

#include "core/Log.h"
#include "renderer/DLSSUpscaler.h"
#include "renderer/XeSSUpscaler.h"
#include "renderer/FSRUpscaler.h"

#include <algorithm>
#include <cstring>

#define ENIGMA_HAS_DLSS 0
#define ENIGMA_HAS_XESS 0
#define ENIGMA_HAS_FSR  0

namespace enigma {

VkExtent2D upscalerRenderResolution(VkExtent2D displayRes, UpscalerQuality quality) {
    f32 scale = 1.0f;
    switch (quality) {
        case UpscalerQuality::UltraPerformance: scale = 0.33f; break;
        case UpscalerQuality::Performance:      scale = 0.50f; break;
        case UpscalerQuality::Balanced:          scale = 0.59f; break;
        case UpscalerQuality::Quality:           scale = 0.67f; break;
        case UpscalerQuality::UltraQuality:      scale = 0.77f; break;
        case UpscalerQuality::NativeAA:          scale = 1.00f; break;
    }
    return {
        std::max(1u, static_cast<u32>(static_cast<f32>(displayRes.width)  * scale)),
        std::max(1u, static_cast<u32>(static_cast<f32>(displayRes.height) * scale)),
    };
}

namespace {

constexpr u32 kVendorNvidia = 0x10DE;
constexpr u32 kVendorIntel  = 0x8086;
// constexpr u32 kVendorAMD = 0x1002; // Falls through to FSR

bool deviceNameContains(const char* deviceName, const char* substr) {
    // Simple case-sensitive substring search.
    return std::strstr(deviceName, substr) != nullptr;
}

} // namespace

UpscalerBackend UpscalerFactory::autoSelect(const VkPhysicalDeviceProperties& props,
                                            gfx::GpuTier /*tier*/) {
    // NVIDIA RTX -> DLSS (if SDK available)
    if (props.vendorID == kVendorNvidia &&
        deviceNameContains(props.deviceName, "RTX")) {
#if ENIGMA_HAS_DLSS
        return UpscalerBackend::DLSS;
#else
        ENIGMA_LOG_INFO("[upscaler] NVIDIA RTX detected but DLSS SDK not linked, falling back to FSR");
#endif
    }

    // Intel Arc -> XeSS (if SDK available)
    if (props.vendorID == kVendorIntel &&
        deviceNameContains(props.deviceName, "Arc")) {
#if ENIGMA_HAS_XESS
        return UpscalerBackend::XeSS;
#else
        ENIGMA_LOG_INFO("[upscaler] Intel Arc detected but XeSS SDK not linked, falling back to FSR");
#endif
    }

    // All other cases (AMD, non-RTX NVIDIA, etc.) -> FSR
    return UpscalerBackend::FSR;
}

std::unique_ptr<IUpscaler> UpscalerFactory::create(
    const VkPhysicalDeviceProperties& props,
    gfx::GpuTier tier,
    UpscalerBackend override) {

    const UpscalerBackend selected = (override != UpscalerBackend::None)
        ? override
        : autoSelect(props, tier);

    switch (selected) {
        case UpscalerBackend::DLSS:
            ENIGMA_LOG_INFO("[upscaler] creating DLSS backend");
            return std::make_unique<DLSSUpscaler>();
        case UpscalerBackend::XeSS:
            ENIGMA_LOG_INFO("[upscaler] creating XeSS backend");
            return std::make_unique<XeSSUpscaler>();
        case UpscalerBackend::FSR:
        case UpscalerBackend::None:
        default:
            ENIGMA_LOG_INFO("[upscaler] creating FSR backend");
            return std::make_unique<FSRUpscaler>();
    }
}

} // namespace enigma
