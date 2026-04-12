#include "renderer/PostProcessPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

namespace enigma {

// Push constant layout — must match post_process.hlsl PushBlock exactly.
struct PostProcessPushBlock {
    u32  hdrColorSlot;    //  4
    u32  depthSlot;       //  4
    u32  cameraSlot;      //  4
    u32  samplerSlot;     //  4
    vec4 cameraWorldPosKm; // 16
    f32  exposureEV;      //  4
    f32  bloomThreshold;  //  4
    f32  bloomIntensity;  //  4
    u32  tonemapMode;     //  4  (0=AgX, 1=ACES)
    u32  bloomEnabled;    //  4
    u32  apEnabled;       //  4
    u32  _pad0;           //  4
    u32  _pad1;           //  4
};                        // Total: 64 bytes

static_assert(sizeof(PostProcessPushBlock) == 64);

PostProcessPass::PostProcessPass(gfx::Device& device)
    : m_device(&device) {}

PostProcessPass::~PostProcessPass() {
    delete m_pipeline;
}

void PostProcessPass::buildPipeline(gfx::ShaderManager& shaderManager,
                                     VkDescriptorSetLayout globalSetLayout,
                                     VkDescriptorSetLayout apReadSetLayout,
                                     VkFormat colorAttachmentFormat) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "PostProcessPass::buildPipeline called twice");

    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_apReadSetLayout = apReadSetLayout;
    m_colorFormat     = colorAttachmentFormat;
    m_shaderPath      = Paths::shaderSourceDir() / "post_process.hlsl";

    VkShaderModule vert = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    VkShaderModule frag = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader            = vert;
    ci.vertEntryPoint        = "VSMain";
    ci.fragShader            = frag;
    ci.fragEntryPoint        = "PSMain";
    ci.globalSetLayout       = globalSetLayout;
    ci.additionalSetLayout   = apReadSetLayout; // AP volume at set=1
    ci.colorAttachmentFormat = colorAttachmentFormat;
    ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    ci.pushConstantSize      = sizeof(PostProcessPushBlock);
    ci.depthCompareOp        = VK_COMPARE_OP_ALWAYS;
    ci.cullMode              = VK_CULL_MODE_NONE;

    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[post-process] pipeline built");
}

void PostProcessPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    ENIGMA_ASSERT(m_shaderManager != nullptr);

    VkShaderModule vert = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    if (vert == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[post-process] hot-reload: VS compile failed"); return; }
    VkShaderModule frag = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");
    if (frag == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[post-process] hot-reload: PS compile failed");
        vkDestroyShaderModule(m_device->logical(), vert, nullptr);
        return;
    }

    vkDeviceWaitIdle(m_device->logical());
    delete m_pipeline;

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader            = vert;
    ci.vertEntryPoint        = "VSMain";
    ci.fragShader            = frag;
    ci.fragEntryPoint        = "PSMain";
    ci.globalSetLayout       = m_globalSetLayout;
    ci.additionalSetLayout   = m_apReadSetLayout;
    ci.colorAttachmentFormat = m_colorFormat;
    ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    ci.pushConstantSize      = sizeof(PostProcessPushBlock);
    ci.depthCompareOp        = VK_COMPARE_OP_ALWAYS;
    ci.cullMode              = VK_CULL_MODE_NONE;
    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[post-process] hot-reload: pipeline rebuilt");
}

void PostProcessPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

void PostProcessPass::record(VkCommandBuffer cmd,
                              VkDescriptorSet globalSet,
                              VkDescriptorSet apReadSet,
                              VkExtent2D extent,
                              u32 hdrColorSlot,
                              u32 depthSlot,
                              u32 cameraSlot,
                              u32 samplerSlot,
                              const AtmosphereSettings& settings,
                              vec4 cameraWorldPosKm) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "PostProcessPass::record before buildPipeline");

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->handle());

    // Bind set 0 (global bindless) and set 1 (AP volume read).
    const VkDescriptorSet sets[2] = { globalSet, apReadSet };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->layout(), 0, 2, sets, 0, nullptr);

    PostProcessPushBlock pc{};
    pc.hdrColorSlot   = hdrColorSlot;
    pc.depthSlot      = depthSlot;
    pc.cameraSlot     = cameraSlot;
    pc.samplerSlot    = samplerSlot;
    pc.cameraWorldPosKm = cameraWorldPosKm;
    pc.exposureEV     = settings.exposureEV;
    pc.bloomThreshold = settings.bloomThreshold;
    pc.bloomIntensity = settings.bloomIntensity;
    pc.tonemapMode    = static_cast<u32>(settings.tonemapMode);
    pc.bloomEnabled   = settings.bloomEnabled  ? 1u : 0u;
    pc.apEnabled      = settings.aerialPerspectiveEnabled ? 1u : 0u;

    vkCmdPushConstants(cmd, m_pipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    // Fullscreen triangle: 3 vertices, no vertex buffer.
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace enigma
