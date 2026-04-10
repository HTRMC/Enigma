#pragma once

#include "renderer/Upscaler.h"

namespace enigma {

// FSR 3.1 upscaler backend (AMD FidelityFX SDK).
// SDK path: vendor/fidelityfx/ (not yet present -- stub implementation).
// Define ENIGMA_HAS_FSR=1 and provide the FidelityFX SDK headers to activate.
// Guaranteed fallback: implements a simple TAA stub (temporal accumulation)
// so Min-tier devices get temporal smoothing even without the full SDK.
class FSRUpscaler : public IUpscaler {
public:
    UpscalerBackend backend() const override { return UpscalerBackend::FSR; }
    void init(VkDevice device, VkPhysicalDevice physical,
              VkExtent2D displayRes, UpscalerQuality quality) override;
    void reinit(VkExtent2D displayRes, UpscalerQuality quality) override;
    void evaluate(VkCommandBuffer cmd,
                  VkImageView inputColor, VkImageLayout inputLayout,
                  VkImageView depth,      VkImageLayout depthLayout,
                  VkImageView motionVectors, VkImageLayout mvLayout,
                  VkImageView outputColor, VkImageLayout outputLayout,
                  VkExtent2D extent, VkExtent2D displayExtent,
                  f32 jitterX, f32 jitterY,
                  f32 sharpness, bool reset) override;
    void shutdown() override;
    VkExtent2D renderResolution() const override;

private:
    VkDevice        m_device     = VK_NULL_HANDLE;
    VkExtent2D      m_renderRes  = {};
    VkExtent2D      m_displayRes = {};
    UpscalerQuality m_quality    = UpscalerQuality::Quality;
    // TODO: FfxFsr3UpscalerContext m_context; -- FidelityFX SDK context
};

} // namespace enigma
