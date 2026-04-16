#include "renderer/ClusteredForwardPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "scene/Scene.h"

namespace enigma {

// Push block — must match PushBlock in clustered_forward.hlsl exactly.
// 112 bytes total (fits within the 128-byte minimum guarantee).
struct ClusteredForwardPushBlock {
    mat4  model;               // 64 bytes — world transform
    vec4  sunDirIntensity;     // 16 bytes — xyz=world-space sun dir, w=intensity
    vec4  sunColor;            // 16 bytes — xyz=linear RGB, w=unused
    u32   vertexSlot;          //  4 bytes
    u32   cameraSlot;          //  4 bytes
    u32   materialBufferSlot;  //  4 bytes
    u32   materialIndex;       //  4 bytes
};

static_assert(sizeof(ClusteredForwardPushBlock) == 112);

// Material flags — mirrors Scene.h Material::flags bit layout.
static constexpr u32 kFlagBlend = Material::kFlagBlend;
static_assert(kFlagBlend == 0x1u, "kFlagBlend mismatch with Scene.h");

ClusteredForwardPass::ClusteredForwardPass(gfx::Device& device)
    : m_device(&device) {}

ClusteredForwardPass::~ClusteredForwardPass() {
    delete m_pipeline;
}

void ClusteredForwardPass::buildPipeline(gfx::ShaderManager& shaderManager,
                                          VkDescriptorSetLayout globalSetLayout,
                                          VkFormat hdrColorFormat) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "ClusteredForwardPass::buildPipeline called twice");

    m_shaderManager  = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_hdrColorFormat  = hdrColorFormat;
    m_shaderPath      = Paths::shaderSourceDir() / "clustered_forward.hlsl";

    VkShaderModule vs = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    VkShaderModule fs = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader             = vs;
    ci.vertEntryPoint         = "VSMain";
    ci.fragShader             = fs;
    ci.fragEntryPoint         = "PSMain";
    ci.globalSetLayout        = globalSetLayout;
    ci.colorAttachmentFormat  = hdrColorFormat;
    ci.depthAttachmentFormat  = VK_FORMAT_D32_SFLOAT;
    ci.depthCompareOp         = VK_COMPARE_OP_GREATER_OR_EQUAL; // reverse-Z
    ci.depthWriteEnable       = false;  // read-only depth test
    ci.blendEnable            = true;   // src-alpha blending
    ci.cullMode               = VK_CULL_MODE_BACK_BIT;
    ci.pushConstantSize       = sizeof(ClusteredForwardPushBlock);

    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), vs, nullptr);
    vkDestroyShaderModule(m_device->logical(), fs, nullptr);

    ENIGMA_LOG_INFO("[clustered_forward] pipeline built");
}

void ClusteredForwardPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr);

    VkShaderModule vs = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    if (vs == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[clustered_forward] hot-reload: VS compile failed"); return; }
    VkShaderModule fs = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");
    if (fs == VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device->logical(), vs, nullptr);
        ENIGMA_LOG_ERROR("[clustered_forward] hot-reload: PS compile failed"); return;
    }

    vkDeviceWaitIdle(m_device->logical());
    delete m_pipeline;

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader             = vs;
    ci.vertEntryPoint         = "VSMain";
    ci.fragShader             = fs;
    ci.fragEntryPoint         = "PSMain";
    ci.globalSetLayout        = m_globalSetLayout;
    ci.colorAttachmentFormat  = m_hdrColorFormat;
    ci.depthAttachmentFormat  = VK_FORMAT_D32_SFLOAT;
    ci.depthCompareOp         = VK_COMPARE_OP_GREATER_OR_EQUAL;
    ci.depthWriteEnable       = false;
    ci.blendEnable            = true;
    ci.cullMode               = VK_CULL_MODE_BACK_BIT;
    ci.pushConstantSize       = sizeof(ClusteredForwardPushBlock);

    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), vs, nullptr);
    vkDestroyShaderModule(m_device->logical(), fs, nullptr);

    ENIGMA_LOG_INFO("[clustered_forward] hot-reload: pipeline rebuilt");
}

void ClusteredForwardPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

void ClusteredForwardPass::record(VkCommandBuffer cmd,
                                   VkDescriptorSet globalSet,
                                   VkExtent2D      extent,
                                   const Scene&    scene,
                                   u32             cameraSlot,
                                   u32             materialBufferSlot,
                                   vec3            sunDir,
                                   vec3            sunColor,
                                   float           sunIntensity) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "ClusteredForwardPass::record before buildPipeline");

    // Check if any transparent primitives exist. Skip the pass if not.
    bool hasTransparents = false;
    for (const auto& node : scene.nodes) {
        for (u32 primIdx : node.primitiveIndices) {
            const auto& prim = scene.primitives[primIdx];
            if (prim.materialIndex >= 0 &&
                (scene.materials[static_cast<u32>(prim.materialIndex)].flags & kFlagBlend) != 0) {
                hasTransparents = true;
                break;
            }
        }
        if (hasTransparents) break;
    }
    if (!hasTransparents) return;

    // The RenderGraph calls vkCmdBeginRendering before invoking this execute lambda,
    // using the color/depth attachment info from the RasterPassDesc. No explicit
    // barriers needed here — the graph handles layout transitions between passes.

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    // -------------------------------------------------------------------
    // Draw transparent primitives.
    // -------------------------------------------------------------------
    ClusteredForwardPushBlock pc{};
    pc.sunDirIntensity = vec4(sunDir.x, sunDir.y, sunDir.z, sunIntensity);
    pc.sunColor        = vec4(sunColor.x, sunColor.y, sunColor.z, 0.0f);
    pc.cameraSlot           = cameraSlot;
    pc.materialBufferSlot   = materialBufferSlot;

    for (const auto& node : scene.nodes) {
        for (u32 primIdx : node.primitiveIndices) {
            const auto& prim = scene.primitives[primIdx];
            if (prim.materialIndex < 0) continue;
            const auto& mat = scene.materials[static_cast<u32>(prim.materialIndex)];
            if ((mat.flags & kFlagBlend) == 0) continue;

            pc.model         = node.worldTransform;
            pc.vertexSlot    = prim.vertexBufferSlot;
            pc.materialIndex = static_cast<u32>(prim.materialIndex);

            vkCmdPushConstants(cmd, m_pipeline->layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            vkCmdBindIndexBuffer(cmd, prim.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, prim.indexCount, 1, 0, 0, 0);
        }
    }

}

} // namespace enigma
