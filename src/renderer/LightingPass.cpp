#include "renderer/LightingPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

namespace enigma {

// Push constant layout — must match lighting.hlsl PushBlock exactly.
struct LightingPushBlock {
    u32  albedoSlot;        //  4
    u32  normalSlot;        //  4
    u32  metalRoughSlot;    //  4
    u32  depthSlot;         //  4
    u32  cameraSlot;        //  4
    u32  samplerSlot;       //  4
    u32  _pad0;             //  4
    u32  _pad1;             //  4  (32 bytes total)
    vec4 lightDirIntensity; // 16
    vec4 lightColor;        // 16
};                          // Total: 64 bytes

static_assert(sizeof(LightingPushBlock) == 64);

LightingPass::LightingPass(gfx::Device& device)
    : m_device(&device) {}

LightingPass::~LightingPass() {
    delete m_pipeline;
}

void LightingPass::buildPipeline(gfx::ShaderManager& shaderManager,
                                  VkDescriptorSetLayout globalSetLayout,
                                  VkFormat colorAttachmentFormat) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "LightingPass::buildPipeline called twice");

    m_shaderManager  = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_colorFormat    = colorAttachmentFormat;
    m_shaderPath     = Paths::shaderSourceDir() / "lighting.hlsl";

    VkShaderModule vert = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    VkShaderModule frag = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader            = vert;
    ci.vertEntryPoint        = "VSMain";
    ci.fragShader            = frag;
    ci.fragEntryPoint        = "PSMain";
    ci.globalSetLayout       = globalSetLayout;
    ci.colorAttachmentFormat = colorAttachmentFormat;
    ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED; // no depth write in lighting pass
    ci.pushConstantSize      = sizeof(LightingPushBlock);
    ci.depthCompareOp        = VK_COMPARE_OP_ALWAYS;
    ci.cullMode              = VK_CULL_MODE_NONE;   // fullscreen triangle

    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[lighting] pipeline built");
}

void LightingPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    ENIGMA_ASSERT(m_shaderManager != nullptr);

    VkShaderModule vert = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    if (vert == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[lighting] hot-reload: VS compile failed"); return; }
    VkShaderModule frag = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");
    if (frag == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[lighting] hot-reload: PS compile failed");
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
    ci.colorAttachmentFormat = m_colorFormat;
    ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    ci.pushConstantSize      = sizeof(LightingPushBlock);
    ci.depthCompareOp        = VK_COMPARE_OP_ALWAYS;
    ci.cullMode              = VK_CULL_MODE_NONE;
    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[lighting] hot-reload: pipeline rebuilt");
}

void LightingPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

void LightingPass::record(VkCommandBuffer cmd,
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
    ENIGMA_ASSERT(m_pipeline != nullptr && "LightingPass::record before buildPipeline");

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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    LightingPushBlock pc{};
    pc.albedoSlot       = albedoSlot;
    pc.normalSlot       = normalSlot;
    pc.metalRoughSlot   = metalRoughSlot;
    pc.depthSlot        = depthSlot;
    pc.cameraSlot       = cameraSlot;
    pc.samplerSlot      = samplerSlot;
    pc.lightDirIntensity = lightDirIntensity;
    pc.lightColor        = lightColor;

    vkCmdPushConstants(cmd, m_pipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    // Fullscreen triangle: 3 vertices, no vertex buffer.
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace enigma
