#pragma once

#include "renderer/Upscaler.h"

namespace enigma {

// XeSS 1.3 upscaler backend (Intel XeSS SDK).
// SDK path: vendor/xess/ (not yet present -- stub implementation).
// Define ENIGMA_HAS_XESS=1 and provide the XeSS SDK headers to activate.
class XeSSUpscaler : public IUpscaler {
public:
    UpscalerBackend backend() const override { return UpscalerBackend::XeSS; }
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
    // TODO: xess_context_handle_t m_context; -- XeSS SDK handle
};

} // namespace enigma
