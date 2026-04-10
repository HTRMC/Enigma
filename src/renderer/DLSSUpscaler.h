#pragma once

#include "renderer/Upscaler.h"

namespace enigma {

// DLSS 3.7 upscaler backend (NVIDIA Streamline SDK).
// SDK path: vendor/streamline/ (not yet present -- stub implementation).
// Define ENIGMA_HAS_DLSS=1 and provide the Streamline SDK headers to activate.
class DLSSUpscaler : public IUpscaler {
public:
    UpscalerBackend backend() const override { return UpscalerBackend::DLSS; }
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
    // TODO: sl::Feature m_dlssFeature; -- Streamline SDK context
};

} // namespace enigma
