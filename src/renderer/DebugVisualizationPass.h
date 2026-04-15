#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>
#include <filesystem>
#include <memory>

namespace enigma::gfx {
class Device;
class Pipeline;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// DebugVisualizationPass
// ======================
// Fullscreen triangle debug visualization modes: Unlit, Detail Lighting,
// Clusters (meshlet colors), and HDR blit (for LitWireframe base layer).
// All modes write directly to the swapchain, bypassing post-processing.
class DebugVisualizationPass {
public:
    explicit DebugVisualizationPass(gfx::Device& device);
    ~DebugVisualizationPass();

    DebugVisualizationPass(const DebugVisualizationPass&)            = delete;
    DebugVisualizationPass& operator=(const DebugVisualizationPass&) = delete;

    void buildPipelines(gfx::ShaderManager& shaderManager,
                        VkDescriptorSetLayout globalSetLayout,
                        VkFormat swapchainFormat);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Unlit: outputs raw G-buffer albedo.
    void recordUnlit(VkCommandBuffer cmd,
                     VkDescriptorSet globalSet,
                     VkExtent2D extent,
                     u32 albedoSlot,
                     u32 depthSlot,
                     u32 samplerSlot);

    // Detail Lighting: full Cook-Torrance on white material.
    void recordDetailLighting(VkCommandBuffer cmd,
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

    // Clusters: colors each meshlet by deterministic hash of its ID.
    void recordClusters(VkCommandBuffer cmd,
                        VkDescriptorSet globalSet,
                        VkExtent2D extent,
                        u32 visBufferSlot,
                        u32 samplerSlot);

    // Blit: copies HDR intermediate to swapchain with simple Reinhard tonemap.
    // Used as the lit base layer for Lit Wireframe mode.
    void recordBlit(VkCommandBuffer cmd,
                    VkDescriptorSet globalSet,
                    VkExtent2D extent,
                    u32 hdrSampledSlot,
                    u32 samplerSlot);

private:
    void rebuildPipelines();

    gfx::Device*          m_device        = nullptr;
    gfx::ShaderManager*   m_shaderManager = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    VkFormat              m_swapchainFormat = VK_FORMAT_UNDEFINED;

    std::unique_ptr<gfx::Pipeline> m_unlitPipeline;
    std::unique_ptr<gfx::Pipeline> m_detailLightingPipeline;
    std::unique_ptr<gfx::Pipeline> m_clustersPipeline;
    std::unique_ptr<gfx::Pipeline> m_blitPipeline;

    std::filesystem::path m_unlitShaderPath;
    std::filesystem::path m_detailLightingShaderPath;
    std::filesystem::path m_clustersShaderPath;
    std::filesystem::path m_blitShaderPath;
};

} // namespace enigma
