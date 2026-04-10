#include "renderer/DLSSUpscaler.h"

#include "core/Log.h"

#define ENIGMA_HAS_DLSS 0

namespace enigma {

void DLSSUpscaler::init(VkDevice device, VkPhysicalDevice /*physical*/,
                        VkExtent2D displayRes, UpscalerQuality quality) {
    m_device     = device;
    m_displayRes = displayRes;
    m_quality    = quality;
    m_renderRes  = upscalerRenderResolution(displayRes, quality);

    // TODO: replace with real SDK call — sl::dlssInit(m_dlssFeature, ...)
    ENIGMA_LOG_INFO("[upscaler] DLSS stub initialized (render {}x{} -> display {}x{})",
                    m_renderRes.width, m_renderRes.height,
                    m_displayRes.width, m_displayRes.height);
}

void DLSSUpscaler::reinit(VkExtent2D displayRes, UpscalerQuality quality) {
    m_displayRes = displayRes;
    m_quality    = quality;
    m_renderRes  = upscalerRenderResolution(displayRes, quality);

    // TODO: replace with real SDK call — sl::dlssSetQuality(m_dlssFeature, ...)
    ENIGMA_LOG_INFO("[upscaler] DLSS stub reinit (quality changed, render {}x{})",
                    m_renderRes.width, m_renderRes.height);
}

void DLSSUpscaler::evaluate(VkCommandBuffer /*cmd*/,
                            VkImageView /*inputColor*/, VkImageLayout /*inputLayout*/,
                            VkImageView /*depth*/,      VkImageLayout /*depthLayout*/,
                            VkImageView /*motionVectors*/, VkImageLayout /*mvLayout*/,
                            VkImageView /*outputColor*/, VkImageLayout /*outputLayout*/,
                            VkExtent2D /*extent*/, VkExtent2D /*displayExtent*/,
                            f32 /*jitterX*/, f32 /*jitterY*/,
                            f32 /*sharpness*/, bool /*reset*/) {
    // TODO: replace with real SDK call — sl::dlssEvaluate(cmd, inputColor, depth,
    //       motionVectors, outputColor, extent, displayExtent, jitterX, jitterY,
    //       sharpness, reset)
    //
    // Stub: bilinear blit inputColor -> outputColor would go here.
    // Without real VkImage handles (only VkImageViews are passed), we cannot
    // record a vkCmdBlitImage. The real SDK will use its own dispatch.
    ENIGMA_LOG_INFO("[upscaler] DLSS stub: bilinear blit (SDK not linked)");
}

void DLSSUpscaler::shutdown() {
    // TODO: replace with real SDK call — sl::dlssShutdown(m_dlssFeature)
    m_device = VK_NULL_HANDLE;
}

VkExtent2D DLSSUpscaler::renderResolution() const {
    return m_renderRes;
}

} // namespace enigma
