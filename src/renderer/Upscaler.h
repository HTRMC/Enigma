#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma {

enum class UpscalerBackend {
    None,    // pass-through / bilinear blit
    DLSS,    // NVIDIA DLSS 3.7 via Streamline SDK
    XeSS,    // Intel XeSS 1.3
    FSR,     // AMD FSR 3.1 via FidelityFX SDK
};

enum class UpscalerQuality {
    UltraPerformance, // ~33% render res
    Performance,      // ~50% render res
    Balanced,         // ~59% render res
    Quality,          // ~67% render res
    UltraQuality,     // ~77% render res
    NativeAA,         // 100% render res (TAA only)
};

// Returns the recommended render resolution for a given display resolution and quality mode.
VkExtent2D upscalerRenderResolution(VkExtent2D displayRes, UpscalerQuality quality);

class IUpscaler {
public:
    virtual ~IUpscaler() = default;

    virtual UpscalerBackend backend() const = 0;

    // Initialize for a given display resolution and quality mode.
    virtual void init(VkDevice device, VkPhysicalDevice physical,
                      VkExtent2D displayRes, UpscalerQuality quality) = 0;

    // Re-initialize after quality mode change (may be called mid-frame after vkQueueWaitIdle).
    virtual void reinit(VkExtent2D displayRes, UpscalerQuality quality) = 0;

    // Evaluate upscaling.
    //   inputColor:    VkImageView of the rendered scene (render resolution, RGBA16F)
    //   depth:         VkImageView of depth (render resolution, D32F)
    //   motionVectors: VkImageView of motion vectors (render resolution, RG16F)
    //   outputColor:   VkImageView of the upscaled output (display resolution, RGBA16F)
    //   cmd:           command buffer to record into
    //   extent:        render resolution (input extent)
    //   displayExtent: display resolution (output extent)
    //   jitterX/Y:     Halton sub-pixel jitter offsets for current frame
    //   sharpness:     0.0-1.0 (backend-specific sharpening, 0 = off)
    //   reset:         true on camera cut / teleport (discard temporal history)
    virtual void evaluate(VkCommandBuffer cmd,
                          VkImageView inputColor, VkImageLayout inputLayout,
                          VkImageView depth,      VkImageLayout depthLayout,
                          VkImageView motionVectors, VkImageLayout mvLayout,
                          VkImageView outputColor, VkImageLayout outputLayout,
                          VkExtent2D extent, VkExtent2D displayExtent,
                          f32 jitterX, f32 jitterY,
                          f32 sharpness = 0.0f,
                          bool reset = false) = 0;

    virtual void shutdown() = 0;

    // Returns the render resolution this upscaler is currently configured for.
    virtual VkExtent2D renderResolution() const = 0;
};

} // namespace enigma
