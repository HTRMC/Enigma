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

    // MicropolyRasterClass (M4.6): per-pixel R/G decoded from the 2-bit
    // rasterClassBits field of the 64-bit Micropoly vis image. Red = HW,
    // green = SW, blue = reserved class, black = empty pixel. Renderer
    // only dispatches this when Device::supportsShaderImageInt64() is true.
    void recordMicropolyRasterClass(VkCommandBuffer cmd,
                                    VkDescriptorSet globalSet,
                                    VkExtent2D extent,
                                    u32 visImage64Bindless);

    // MicropolyLodHeatmap (M6.1): decodes clusterIdx from the vis-pack,
    // loads the DAG node's lodLevel, and writes a blue→red gradient.
    // Falls back to magenta when the DAG SSBO isn't wired (defensive).
    void recordMicropolyLodHeatmap(VkCommandBuffer cmd,
                                   VkDescriptorSet globalSet,
                                   VkExtent2D extent,
                                   u32 visImage64Bindless,
                                   u32 dagBufferBindless,
                                   u32 dagNodeCount);

    // MicropolyResidencyHeatmap (M6.1): decodes cluster → pageId → slot
    // index; green = resident, magenta = non-resident, yellow = missing
    // DAG/pageToSlot bindless slot, black = empty vis pixel.
    void recordMicropolyResidencyHeatmap(VkCommandBuffer cmd,
                                         VkDescriptorSet globalSet,
                                         VkExtent2D extent,
                                         u32 visImage64Bindless,
                                         u32 dagBufferBindless,
                                         u32 pageToSlotBindless,
                                         u32 dagNodeCount,
                                         u32 pageCount);

    // MicropolyBounds (M6.2b): per-cluster bounding-sphere wireframe
    // overlay. Iterates the DAG and draws each sphere's projected outline
    // with a per-cluster hue. Does NOT read the 64-bit vis image — the
    // availability gate on the Renderer side is DAG-wiring only
    // (shaderImageInt64 not required).
    void recordMicropolyBounds(VkCommandBuffer cmd,
                               VkDescriptorSet globalSet,
                               VkExtent2D extent,
                               u32 dagBufferBindless,
                               u32 dagNodeCount,
                               u32 cameraSlot,
                               u32 screenWidth,
                               u32 screenHeight);

    // MicropolyBinOverflowHeat (M6 plan §3.M6): per-pixel SW-raster tile
    // bin fill-level heat. Reads the tileBinCount + spillBuffer SSBOs
    // written by MicropolySwRasterPass. Does NOT read the vis image.
    void recordMicropolyBinOverflow(VkCommandBuffer cmd,
                                    VkDescriptorSet globalSet,
                                    VkExtent2D extent,
                                    u32 tileBinCountBindless,
                                    u32 spillBufferBindless,
                                    u32 tilesX,
                                    u32 tilesY,
                                    u32 screenWidth,
                                    u32 screenHeight);

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
    std::unique_ptr<gfx::Pipeline> m_micropolyRasterClassPipeline;
    std::unique_ptr<gfx::Pipeline> m_micropolyLodHeatmapPipeline;
    std::unique_ptr<gfx::Pipeline> m_micropolyResidencyHeatmapPipeline;
    std::unique_ptr<gfx::Pipeline> m_micropolyBoundsPipeline;
    std::unique_ptr<gfx::Pipeline> m_micropolyBinOverflowPipeline;

    std::filesystem::path m_unlitShaderPath;
    std::filesystem::path m_detailLightingShaderPath;
    std::filesystem::path m_clustersShaderPath;
    std::filesystem::path m_blitShaderPath;
    std::filesystem::path m_micropolyRasterClassShaderPath;
    std::filesystem::path m_micropolyLodHeatmapShaderPath;
    std::filesystem::path m_micropolyResidencyHeatmapShaderPath;
    std::filesystem::path m_micropolyBoundsShaderPath;
    std::filesystem::path m_micropolyBinOverflowShaderPath;
};

} // namespace enigma
