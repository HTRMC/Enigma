#pragma once

#include "core/Math.h"
#include "core/Types.h"
#include "renderer/AtmosphereSettings.h"

#include <volk.h>

#include <filesystem>

namespace enigma::gfx {
class Device;
class Pipeline;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// PostProcessPass
// ===============
// Final screen-space pass producing the display-ready image from the
// HDR linear intermediate. Runs as the last render graph pass, writing
// to the swapchain colour attachment.
//
// Pipeline (in order):
//   1. Aerial Perspective apply  — blends AP volume into geometry pixels
//   2. Exposure                  — pow(2, exposureEV) scale
//   3. Bloom                     — threshold + 13-tap star-pattern single-pass
//   4. Tone mapping              — AgX (default) or ACES
//
// Uses the AP read descriptor set (set=1) provided by AtmospherePass.
// The pipeline layout is built with globalSetLayout (set=0) +
// apReadSetLayout (set=1) via Pipeline::CreateInfo::additionalSetLayout.
class PostProcessPass {
public:
    explicit PostProcessPass(gfx::Device& device);
    ~PostProcessPass();

    PostProcessPass(const PostProcessPass&)            = delete;
    PostProcessPass& operator=(const PostProcessPass&) = delete;
    PostProcessPass(PostProcessPass&&)                 = delete;
    PostProcessPass& operator=(PostProcessPass&&)      = delete;

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkDescriptorSetLayout apReadSetLayout,
                       VkFormat colorAttachmentFormat);

    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkDescriptorSet apReadSet,
                VkExtent2D extent,
                u32 hdrColorSlot,
                u32 depthSlot,
                u32 cameraSlot,
                u32 samplerSlot,
                const AtmosphereSettings& settings,
                vec4 cameraWorldPosKm);  // xyz = km from planet centre

    void registerHotReload(gfx::ShaderHotReload& reloader);

private:
    void rebuildPipeline();

    gfx::Device*          m_device          = nullptr;
    gfx::Pipeline*        m_pipeline        = nullptr;

    gfx::ShaderManager*   m_shaderManager   = nullptr;
    VkDescriptorSetLayout m_globalSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_apReadSetLayout  = VK_NULL_HANDLE;
    VkFormat              m_colorFormat      = VK_FORMAT_UNDEFINED;
    std::filesystem::path m_shaderPath;
};

} // namespace enigma
