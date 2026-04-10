#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>

#include <filesystem>

namespace enigma::gfx {
class Device;
class Pipeline;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// LightingPass
// ============
// Fullscreen deferred lighting pass. Reads the G-buffer (albedo, normal,
// metalRough, depth) via bindless sampled images and evaluates the same
// Cook-Torrance BRDF as the forward mesh.hlsl pass. Writes to the swapchain
// colour attachment.
//
// The pass draws a single fullscreen triangle (3 vertices, no vertex buffer)
// with depth testing disabled.
class LightingPass {
public:
    explicit LightingPass(gfx::Device& device);
    ~LightingPass();

    LightingPass(const LightingPass&)            = delete;
    LightingPass& operator=(const LightingPass&) = delete;
    LightingPass(LightingPass&&)                 = delete;
    LightingPass& operator=(LightingPass&&)      = delete;

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkFormat colorAttachmentFormat);

    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent,
                u32 albedoSlot,
                u32 normalSlot,
                u32 metalRoughSlot,
                u32 depthSlot,
                u32 cameraSlot,
                u32 samplerSlot,
                vec4 lightDirIntensity,
                vec4 lightColor);

    void registerHotReload(gfx::ShaderHotReload& reloader);

private:
    void rebuildPipeline();

    gfx::Device*          m_device          = nullptr;
    gfx::Pipeline*        m_pipeline        = nullptr;

    gfx::ShaderManager*   m_shaderManager   = nullptr;
    VkDescriptorSetLayout m_globalSetLayout  = VK_NULL_HANDLE;
    VkFormat              m_colorFormat      = VK_FORMAT_UNDEFINED;
    std::filesystem::path m_shaderPath;
};

} // namespace enigma
