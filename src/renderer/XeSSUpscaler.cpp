#include "renderer/XeSSUpscaler.h"

#include "core/Log.h"

#define ENIGMA_HAS_XESS 0

namespace enigma {

void XeSSUpscaler::init(VkDevice device, VkPhysicalDevice /*physical*/,
                        VkExtent2D displayRes, UpscalerQuality quality) {
    m_device     = device;
    m_displayRes = displayRes;
    m_quality    = quality;
    m_renderRes  = upscalerRenderResolution(displayRes, quality);

    // TODO: replace with real SDK call — xessInit(m_context, ...)
    ENIGMA_LOG_INFO("[upscaler] XeSS stub initialized (render {}x{} -> display {}x{})",
                    m_renderRes.width, m_renderRes.height,
                    m_displayRes.width, m_displayRes.height);
}

void XeSSUpscaler::reinit(VkExtent2D displayRes, UpscalerQuality quality) {
    m_displayRes = displayRes;
    m_quality    = quality;
    m_renderRes  = upscalerRenderResolution(displayRes, quality);

    // TODO: replace with real SDK call — xessSetQuality(m_context, ...)
    ENIGMA_LOG_INFO("[upscaler] XeSS stub reinit (quality changed, render {}x{})",
                    m_renderRes.width, m_renderRes.height);
}

void XeSSUpscaler::evaluate(VkCommandBuffer /*cmd*/,
                            VkImageView /*inputColor*/, VkImageLayout /*inputLayout*/,
                            VkImageView /*depth*/,      VkImageLayout /*depthLayout*/,
                            VkImageView /*motionVectors*/, VkImageLayout /*mvLayout*/,
                            VkImageView /*outputColor*/, VkImageLayout /*outputLayout*/,
                            VkExtent2D /*extent*/, VkExtent2D /*displayExtent*/,
                            f32 /*jitterX*/, f32 /*jitterY*/,
                            f32 /*sharpness*/, bool /*reset*/) {
    // TODO: replace with real SDK call — xessExecute(m_context, cmd, inputColor,
    //       depth, motionVectors, outputColor, ...)
    //
    // Stub: bilinear blit inputColor -> outputColor would go here.
    ENIGMA_LOG_INFO("[upscaler] XeSS stub: bilinear blit (SDK not linked)");
}

void XeSSUpscaler::shutdown() {
    // TODO: replace with real SDK call — xessDestroyContext(m_context)
    m_device = VK_NULL_HANDLE;
}

VkExtent2D XeSSUpscaler::renderResolution() const {
    return m_renderRes;
}

} // namespace enigma
