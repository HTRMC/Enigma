#include "renderer/FSRUpscaler.h"

#include "core/Log.h"

#define ENIGMA_HAS_FSR 0

namespace enigma {

void FSRUpscaler::init(VkDevice device, VkPhysicalDevice /*physical*/,
                       VkExtent2D displayRes, UpscalerQuality quality) {
    m_device     = device;
    m_displayRes = displayRes;
    m_quality    = quality;
    m_renderRes  = upscalerRenderResolution(displayRes, quality);

    // TODO: replace with real SDK call — ffxFsr3ContextCreate(&m_context, ...)
    ENIGMA_LOG_INFO("[upscaler] FSR stub initialized (render {}x{} -> display {}x{})",
                    m_renderRes.width, m_renderRes.height,
                    m_displayRes.width, m_displayRes.height);
}

void FSRUpscaler::reinit(VkExtent2D displayRes, UpscalerQuality quality) {
    m_displayRes = displayRes;
    m_quality    = quality;
    m_renderRes  = upscalerRenderResolution(displayRes, quality);

    // TODO: replace with real SDK call — ffxFsr3ContextDestroy + recreate
    ENIGMA_LOG_INFO("[upscaler] FSR stub reinit (quality changed, render {}x{})",
                    m_renderRes.width, m_renderRes.height);
}

void FSRUpscaler::evaluate(VkCommandBuffer /*cmd*/,
                           VkImageView /*inputColor*/, VkImageLayout /*inputLayout*/,
                           VkImageView /*depth*/,      VkImageLayout /*depthLayout*/,
                           VkImageView /*motionVectors*/, VkImageLayout /*mvLayout*/,
                           VkImageView /*outputColor*/, VkImageLayout /*outputLayout*/,
                           VkExtent2D /*extent*/, VkExtent2D /*displayExtent*/,
                           f32 /*jitterX*/, f32 /*jitterY*/,
                           f32 /*sharpness*/, bool /*reset*/) {
    // TODO: replace with real SDK call — ffxFsr3ContextDispatch(&m_context, ...)
    //
    // Stub: temporal accumulation blend (0.1 blend factor per frame).
    // Without real VkImage handles (only VkImageViews are passed), we cannot
    // record a vkCmdBlitImage. The real SDK will use its own compute dispatch.
    ENIGMA_LOG_INFO("[upscaler] FSR stub: TAA placeholder (SDK not linked)");
}

void FSRUpscaler::shutdown() {
    // TODO: replace with real SDK call — ffxFsr3ContextDestroy(&m_context)
    m_device = VK_NULL_HANDLE;
}

VkExtent2D FSRUpscaler::renderResolution() const {
    return m_renderRes;
}

} // namespace enigma
