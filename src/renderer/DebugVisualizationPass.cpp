#include "renderer/DebugVisualizationPass.h"

#include "core/Math.h"
#include "core/Paths.h"
#include "core/Types.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

namespace enigma {

// 64-byte push block shared across all debug fullscreen passes.
// Fields used vary per mode — unused fields are ignored by each shader.
struct DebugVisPushBlock {
    u32    albedoSlot;
    u32    normalSlot;
    u32    metalRoughSlot;
    u32    depthSlot;
    u32    cameraSlot;
    u32    samplerSlot;
    u32    visOrHdrSlot;  // visBufferSlot for Clusters, hdrSlot for Blit
    u32    _pad;
    vec4   lightDirIntensity;
    vec4   lightColor;
};
static_assert(sizeof(DebugVisPushBlock) == 64);

// M6.1: dedicated push structs for the two heatmap overlays. Kept small
// so the shader surface is explicit — these shaders read nothing outside
// the vis image + one or two SSBOs and don't pretend to share the 64-byte
// DebugVisPushBlock shape with the other modes.
struct DebugMpLodHeatmapPushBlock {
    u32 visImage64Bindless;
    u32 dagBufferBindless;
    u32 dagNodeCount;
    u32 _pad;
};
static_assert(sizeof(DebugMpLodHeatmapPushBlock) == 16);

struct DebugMpResidencyHeatmapPushBlock {
    u32 visImage64Bindless;
    u32 dagBufferBindless;
    u32 pageToSlotBindless;
    u32 dagNodeCount;
    u32 pageCount;
    u32 _pad0;
    u32 _pad1;
    u32 _pad2;
};
static_assert(sizeof(DebugMpResidencyHeatmapPushBlock) == 32);

// M6.2b: bounding-sphere wireframe overlay push block. Pure DAG-driven —
// no vis image, no pageToSlot. Needs the camera matrices (for projection)
// and the viewport dimensions (for NDC -> pixel scale).
struct DebugMpBoundsPushBlock {
    u32 dagBufferBindless;
    u32 dagNodeCount;
    u32 cameraSlot;
    u32 screenWidth;
    u32 screenHeight;
    u32 _pad0;
    u32 _pad1;
    u32 _pad2;
};
static_assert(sizeof(DebugMpBoundsPushBlock) == 32);

// M6 plan §3.M6: SW-raster tile bin fill-level heat. The shader reads the
// tileBinCount + spillBuffer SSBOs written by MicropolySwRasterPass, so
// the push surface is small and disjoint from the other debug overlays.
struct DebugMpBinOverflowPushBlock {
    u32 tileBinCountBindless;
    u32 spillBufferBindless;
    u32 tilesX;
    u32 tilesY;
    u32 screenWidth;
    u32 screenHeight;
    u32 _pad0;
    u32 _pad1;
};
static_assert(sizeof(DebugMpBinOverflowPushBlock) == 32);

DebugVisualizationPass::DebugVisualizationPass(gfx::Device& device)
    : m_device(&device)
{}

DebugVisualizationPass::~DebugVisualizationPass() = default;

void DebugVisualizationPass::buildPipelines(gfx::ShaderManager& shaderManager,
                                             VkDescriptorSetLayout globalSetLayout,
                                             VkFormat swapchainFormat) {
    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_swapchainFormat = swapchainFormat;

    m_unlitShaderPath                       = Paths::shaderSourceDir() / "debug_unlit.hlsl";
    m_detailLightingShaderPath              = Paths::shaderSourceDir() / "debug_detail_lighting.hlsl";
    m_clustersShaderPath                    = Paths::shaderSourceDir() / "debug_clusters.hlsl";
    m_blitShaderPath                        = Paths::shaderSourceDir() / "debug_blit.hlsl";
    m_micropolyRasterClassShaderPath        = Paths::shaderSourceDir() / "debug_micropoly_raster_class.hlsl";
    m_micropolyLodHeatmapShaderPath         = Paths::shaderSourceDir() / "debug_micropoly_lod_heatmap.hlsl";
    m_micropolyResidencyHeatmapShaderPath   = Paths::shaderSourceDir() / "debug_micropoly_residency_heatmap.hlsl";
    m_micropolyBoundsShaderPath             = Paths::shaderSourceDir() / "debug_micropoly_bounds.hlsl";
    m_micropolyBinOverflowShaderPath        = Paths::shaderSourceDir() / "debug_micropoly_bin_overflow.hlsl";

    rebuildPipelines();
}

void DebugVisualizationPass::rebuildPipelines() {
    using Stage = gfx::ShaderManager::Stage;

    const auto buildFullscreen = [&](const std::filesystem::path& path,
                                     u32 pushSize) {
        VkShaderModule vs = m_shaderManager->compile(path, Stage::Vertex,   "VSMain");
        VkShaderModule fs = m_shaderManager->compile(path, Stage::Fragment, "PSMain");

        gfx::Pipeline::CreateInfo ci{};
        ci.vertShader            = vs;
        ci.fragShader            = fs;
        ci.globalSetLayout       = m_globalSetLayout;
        ci.colorAttachmentFormat = m_swapchainFormat;
        ci.pushConstantSize      = pushSize;

        auto pipeline = std::make_unique<gfx::Pipeline>(*m_device, ci);

        vkDestroyShaderModule(m_device->logical(), vs, nullptr);
        vkDestroyShaderModule(m_device->logical(), fs, nullptr);

        return pipeline;
    };

    const u32 sharedPush      = static_cast<u32>(sizeof(DebugVisPushBlock));
    const u32 lodPush         = static_cast<u32>(sizeof(DebugMpLodHeatmapPushBlock));
    const u32 residencyPush   = static_cast<u32>(sizeof(DebugMpResidencyHeatmapPushBlock));
    const u32 boundsPush      = static_cast<u32>(sizeof(DebugMpBoundsPushBlock));
    const u32 binOverflowPush = static_cast<u32>(sizeof(DebugMpBinOverflowPushBlock));

    m_unlitPipeline                      = buildFullscreen(m_unlitShaderPath,                     sharedPush);
    m_detailLightingPipeline             = buildFullscreen(m_detailLightingShaderPath,            sharedPush);
    m_clustersPipeline                   = buildFullscreen(m_clustersShaderPath,                  sharedPush);
    m_blitPipeline                       = buildFullscreen(m_blitShaderPath,                      sharedPush);
    m_micropolyRasterClassPipeline       = buildFullscreen(m_micropolyRasterClassShaderPath,      sharedPush);
    m_micropolyLodHeatmapPipeline        = buildFullscreen(m_micropolyLodHeatmapShaderPath,       lodPush);
    m_micropolyResidencyHeatmapPipeline  = buildFullscreen(m_micropolyResidencyHeatmapShaderPath, residencyPush);
    m_micropolyBoundsPipeline            = buildFullscreen(m_micropolyBoundsShaderPath,           boundsPush);
    m_micropolyBinOverflowPipeline       = buildFullscreen(m_micropolyBinOverflowShaderPath,      binOverflowPush);
}

void DebugVisualizationPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    // Hot-reload any debug shader -> rebuild all pipelines.
    reloader.watchGroup({m_unlitShaderPath, m_detailLightingShaderPath,
                         m_clustersShaderPath, m_blitShaderPath,
                         m_micropolyRasterClassShaderPath,
                         m_micropolyLodHeatmapShaderPath,
                         m_micropolyResidencyHeatmapShaderPath,
                         m_micropolyBoundsShaderPath,
                         m_micropolyBinOverflowShaderPath},
                        [this]() { rebuildPipelines(); });
}

static void recordFullscreenTriangle(VkCommandBuffer cmd, VkExtent2D extent,
                                     VkPipeline pipeline, VkPipelineLayout layout,
                                     VkDescriptorSet globalSet,
                                     const DebugVisPushBlock& pc) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            layout, 0, 1, &globalSet, 0, nullptr);
    vkCmdPushConstants(cmd, layout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle
}

void DebugVisualizationPass::recordUnlit(VkCommandBuffer cmd,
                                          VkDescriptorSet globalSet,
                                          VkExtent2D extent,
                                          u32 albedoSlot,
                                          u32 depthSlot,
                                          u32 samplerSlot) {
    DebugVisPushBlock pc{};
    pc.albedoSlot  = albedoSlot;
    pc.depthSlot   = depthSlot;
    pc.samplerSlot = samplerSlot;
    recordFullscreenTriangle(cmd, extent,
                              m_unlitPipeline->handle(), m_unlitPipeline->layout(),
                              globalSet, pc);
}

void DebugVisualizationPass::recordDetailLighting(VkCommandBuffer cmd,
                                                   VkDescriptorSet globalSet,
                                                   VkExtent2D extent,
                                                   u32 albedoSlot,
                                                   u32 normalSlot,
                                                   u32 metalRoughSlot,
                                                   u32 depthSlot,
                                                   u32 cameraSlot,
                                                   u32 samplerSlot,
                                                   vec4 lightDirIntensity,
                                                   vec4 lightColor) {
    DebugVisPushBlock pc{};
    pc.albedoSlot       = albedoSlot;
    pc.normalSlot       = normalSlot;
    pc.metalRoughSlot   = metalRoughSlot;
    pc.depthSlot        = depthSlot;
    pc.cameraSlot       = cameraSlot;
    pc.samplerSlot      = samplerSlot;
    pc.lightDirIntensity = lightDirIntensity;
    pc.lightColor        = lightColor;
    recordFullscreenTriangle(cmd, extent,
                              m_detailLightingPipeline->handle(), m_detailLightingPipeline->layout(),
                              globalSet, pc);
}

void DebugVisualizationPass::recordClusters(VkCommandBuffer cmd,
                                             VkDescriptorSet globalSet,
                                             VkExtent2D extent,
                                             u32 visBufferSlot,
                                             u32 samplerSlot) {
    DebugVisPushBlock pc{};
    pc.samplerSlot  = samplerSlot;
    pc.visOrHdrSlot = visBufferSlot;
    recordFullscreenTriangle(cmd, extent,
                              m_clustersPipeline->handle(), m_clustersPipeline->layout(),
                              globalSet, pc);
}

void DebugVisualizationPass::recordBlit(VkCommandBuffer cmd,
                                         VkDescriptorSet globalSet,
                                         VkExtent2D extent,
                                         u32 hdrSampledSlot,
                                         u32 samplerSlot) {
    DebugVisPushBlock pc{};
    pc.samplerSlot  = samplerSlot;
    pc.visOrHdrSlot = hdrSampledSlot;
    recordFullscreenTriangle(cmd, extent,
                              m_blitPipeline->handle(), m_blitPipeline->layout(),
                              globalSet, pc);
}

void DebugVisualizationPass::recordMicropolyRasterClass(VkCommandBuffer cmd,
                                                         VkDescriptorSet globalSet,
                                                         VkExtent2D extent,
                                                         u32 visImage64Bindless) {
    // The fragment shader reads the R64_UINT Micropoly vis image through the
    // g_storageImages64 bindless alias (same binding slot as g_storageImages
    // for R32 float4). Passing the slot via DebugVisPushBlock::visOrHdrSlot
    // keeps the 64-byte push shape identical to all other debug modes.
    DebugVisPushBlock pc{};
    pc.visOrHdrSlot = visImage64Bindless;
    recordFullscreenTriangle(cmd, extent,
                              m_micropolyRasterClassPipeline->handle(),
                              m_micropolyRasterClassPipeline->layout(),
                              globalSet, pc);
}

// M6.1 heatmaps use dedicated (non-DebugVisPushBlock) push structs, so
// they bind the pipeline + descriptor set + push constants inline rather
// than via the shared recordFullscreenTriangle helper. The remaining
// fullscreen-triangle setup (viewport / scissor / vkCmdDraw) is
// intentionally duplicated — pulling it into a template would cost
// clarity without any size benefit (two sites, 6 lines each).

void DebugVisualizationPass::recordMicropolyLodHeatmap(VkCommandBuffer cmd,
                                                        VkDescriptorSet globalSet,
                                                        VkExtent2D extent,
                                                        u32 visImage64Bindless,
                                                        u32 dagBufferBindless,
                                                        u32 dagNodeCount) {
    DebugMpLodHeatmapPushBlock pc{};
    pc.visImage64Bindless = visImage64Bindless;
    pc.dagBufferBindless  = dagBufferBindless;
    pc.dagNodeCount       = dagNodeCount;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_micropolyLodHeatmapPipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_micropolyLodHeatmapPipeline->layout(),
                            0, 1, &globalSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_micropolyLodHeatmapPipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void DebugVisualizationPass::recordMicropolyResidencyHeatmap(VkCommandBuffer cmd,
                                                              VkDescriptorSet globalSet,
                                                              VkExtent2D extent,
                                                              u32 visImage64Bindless,
                                                              u32 dagBufferBindless,
                                                              u32 pageToSlotBindless,
                                                              u32 dagNodeCount,
                                                              u32 pageCount) {
    DebugMpResidencyHeatmapPushBlock pc{};
    pc.visImage64Bindless = visImage64Bindless;
    pc.dagBufferBindless  = dagBufferBindless;
    pc.pageToSlotBindless = pageToSlotBindless;
    pc.dagNodeCount       = dagNodeCount;
    pc.pageCount          = pageCount;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_micropolyResidencyHeatmapPipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_micropolyResidencyHeatmapPipeline->layout(),
                            0, 1, &globalSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_micropolyResidencyHeatmapPipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void DebugVisualizationPass::recordMicropolyBinOverflow(VkCommandBuffer cmd,
                                                         VkDescriptorSet globalSet,
                                                         VkExtent2D extent,
                                                         u32 tileBinCountBindless,
                                                         u32 spillBufferBindless,
                                                         u32 tilesX,
                                                         u32 tilesY,
                                                         u32 screenWidth,
                                                         u32 screenHeight) {
    DebugMpBinOverflowPushBlock pc{};
    pc.tileBinCountBindless = tileBinCountBindless;
    pc.spillBufferBindless  = spillBufferBindless;
    pc.tilesX               = tilesX;
    pc.tilesY               = tilesY;
    pc.screenWidth          = screenWidth;
    pc.screenHeight         = screenHeight;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_micropolyBinOverflowPipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_micropolyBinOverflowPipeline->layout(),
                            0, 1, &globalSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_micropolyBinOverflowPipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void DebugVisualizationPass::recordMicropolyBounds(VkCommandBuffer cmd,
                                                    VkDescriptorSet globalSet,
                                                    VkExtent2D extent,
                                                    u32 dagBufferBindless,
                                                    u32 dagNodeCount,
                                                    u32 cameraSlot,
                                                    u32 screenWidth,
                                                    u32 screenHeight) {
    DebugMpBoundsPushBlock pc{};
    pc.dagBufferBindless = dagBufferBindless;
    pc.dagNodeCount      = dagNodeCount;
    pc.cameraSlot        = cameraSlot;
    pc.screenWidth       = screenWidth;
    pc.screenHeight      = screenHeight;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_micropolyBoundsPipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_micropolyBoundsPipeline->layout(),
                            0, 1, &globalSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_micropolyBoundsPipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace enigma
