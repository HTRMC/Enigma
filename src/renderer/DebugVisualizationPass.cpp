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

    m_unlitShaderPath          = Paths::shaderSourceDir() / "debug_unlit.hlsl";
    m_detailLightingShaderPath = Paths::shaderSourceDir() / "debug_detail_lighting.hlsl";
    m_clustersShaderPath       = Paths::shaderSourceDir() / "debug_clusters.hlsl";
    m_blitShaderPath           = Paths::shaderSourceDir() / "debug_blit.hlsl";

    rebuildPipelines();
}

void DebugVisualizationPass::rebuildPipelines() {
    using Stage = gfx::ShaderManager::Stage;

    const auto buildFullscreen = [&](const std::filesystem::path& path) {
        VkShaderModule vs = m_shaderManager->compile(path, Stage::Vertex,   "VSMain");
        VkShaderModule fs = m_shaderManager->compile(path, Stage::Fragment, "PSMain");

        gfx::Pipeline::CreateInfo ci{};
        ci.vertShader            = vs;
        ci.fragShader            = fs;
        ci.globalSetLayout       = m_globalSetLayout;
        ci.colorAttachmentFormat = m_swapchainFormat;
        ci.pushConstantSize      = static_cast<u32>(sizeof(DebugVisPushBlock));

        auto pipeline = std::make_unique<gfx::Pipeline>(*m_device, ci);

        vkDestroyShaderModule(m_device->logical(), vs, nullptr);
        vkDestroyShaderModule(m_device->logical(), fs, nullptr);

        return pipeline;
    };

    m_unlitPipeline          = buildFullscreen(m_unlitShaderPath);
    m_detailLightingPipeline = buildFullscreen(m_detailLightingShaderPath);
    m_clustersPipeline       = buildFullscreen(m_clustersShaderPath);
    m_blitPipeline           = buildFullscreen(m_blitShaderPath);
}

void DebugVisualizationPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    // Hot-reload any debug shader -> rebuild all pipelines.
    reloader.watchGroup({m_unlitShaderPath, m_detailLightingShaderPath,
                         m_clustersShaderPath, m_blitShaderPath},
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

} // namespace enigma
