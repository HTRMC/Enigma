#include "renderer/MeshPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Math.h"
#include "core/Paths.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "scene/Scene.h"

namespace enigma {

// Push constant layout — must match mesh.hlsl PushBlock exactly.
struct MeshPushBlock {
    mat4 model;               // 64 bytes
    u32  vertexSlot;          //  4
    u32  cameraSlot;          //  4
    u32  materialBufferSlot;  //  4
    u32  materialIndex;       //  4
    vec4 lightDirIntensity;   // 16  (xyz=direction, w=intensity)
    vec4 lightColor;          // 16  (xyz=color, w=unused)
};                            // Total: 112 bytes (≤ 128 Vulkan minimum guarantee)

static_assert(sizeof(MeshPushBlock) == 112);

MeshPass::MeshPass(gfx::Device& device)
    : m_device(&device) {}

MeshPass::~MeshPass() {
    delete m_pipeline;
}

void MeshPass::buildPipeline(gfx::ShaderManager& shaderManager,
                             VkDescriptorSetLayout globalSetLayout,
                             VkFormat colorAttachmentFormat,
                             VkFormat depthAttachmentFormat) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "MeshPass::buildPipeline called twice");

    m_shaderManager  = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_colorFormat    = colorAttachmentFormat;
    m_depthFormat    = depthAttachmentFormat;
    m_shaderPath     = Paths::shaderSourceDir() / "mesh.hlsl";

    VkShaderModule vert = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Vertex, "VSMain");
    VkShaderModule frag = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader            = vert;
    ci.vertEntryPoint        = "VSMain";
    ci.fragShader            = frag;
    ci.fragEntryPoint        = "PSMain";
    ci.globalSetLayout       = globalSetLayout;
    ci.colorAttachmentFormat = colorAttachmentFormat;
    ci.depthAttachmentFormat = depthAttachmentFormat;
    ci.pushConstantSize      = sizeof(MeshPushBlock);
    ci.depthCompareOp        = VK_COMPARE_OP_GREATER_OR_EQUAL; // reverse-Z
    ci.cullMode              = VK_CULL_MODE_BACK_BIT;
    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[mesh] pipeline built");
}

void MeshPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr && "rebuildPipeline before initial build");
    ENIGMA_ASSERT(m_shaderManager != nullptr);

    VkShaderModule vert =
        m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Vertex, "VSMain");
    if (vert == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[mesh] hot-reload: VSMain compile failed, keeping previous pipeline");
        return;
    }
    VkShaderModule frag =
        m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");
    if (frag == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[mesh] hot-reload: PSMain compile failed, keeping previous pipeline");
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
    ci.depthAttachmentFormat = m_depthFormat;
    ci.pushConstantSize      = sizeof(MeshPushBlock);
    ci.depthCompareOp        = VK_COMPARE_OP_GREATER_OR_EQUAL;
    ci.cullMode              = VK_CULL_MODE_BACK_BIT;
    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[mesh] hot-reload: pipeline rebuilt successfully");
}

void MeshPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "registerHotReload called before buildPipeline");
    reloader.watchGroup({m_shaderPath},
                        [this]() { rebuildPipeline(); });
}

void MeshPass::record(VkCommandBuffer cmd,
                      VkDescriptorSet globalSet,
                      VkExtent2D extent,
                      const Scene& scene,
                      u32 cameraSlot,
                      vec4 lightDirIntensity,
                      vec4 lightColor) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "MeshPass::record before buildPipeline");

    // Viewport + scissor.
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind pipeline + global descriptor set.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    // Draw each node's primitives.
    for (const auto& node : scene.nodes) {
        for (u32 primIdx : node.primitiveIndices) {
            const auto& prim = scene.primitives[primIdx];

            MeshPushBlock pc{};
            pc.model               = node.worldTransform;
            pc.vertexSlot          = prim.vertexBufferSlot;
            pc.cameraSlot          = cameraSlot;
            pc.materialBufferSlot  = scene.materialBufferSlot;
            pc.materialIndex       = prim.materialIndex >= 0
                                         ? static_cast<u32>(prim.materialIndex)
                                         : 0u;
            pc.lightDirIntensity   = lightDirIntensity;
            pc.lightColor          = lightColor;

            vkCmdPushConstants(cmd, m_pipeline->layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            vkCmdBindIndexBuffer(cmd, prim.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, prim.indexCount, 1, 0, 0, 0);
        }
    }
}

} // namespace enigma
