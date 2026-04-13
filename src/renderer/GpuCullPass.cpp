#include "renderer/GpuCullPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "renderer/GpuMeshletBuffer.h"
#include "renderer/GpuSceneBuffer.h"
#include "renderer/IndirectDrawBuffer.h"

namespace enigma {

// Push block matching gpu_cull.comp.hlsl PushBlock exactly.
struct GpuCullPushBlock {
    u32 instanceBufferSlot;  // GpuInstance[] at binding 2
    u32 meshletBufferSlot;   // Meshlet[] at binding 2
    u32 commandsBufferSlot;  // RWByteAddressBuffer at binding 5 — draw commands
    u32 countBufferSlot;     // RWByteAddressBuffer at binding 5 — atomic counter
    u32 survivingIdsSlot;    // RWByteAddressBuffer at binding 5 — surviving meshlet IDs
    u32 cameraSlot;          // CameraData at binding 2
    u32 totalMeshlets;       // total meshlet count across all instances
    u32 instanceCount;       // number of GpuInstance entries
};

static_assert(sizeof(GpuCullPushBlock) == 32);

GpuCullPass::GpuCullPass(gfx::Device& device)
    : m_device(&device) {}

GpuCullPass::~GpuCullPass() {
    delete m_pipeline;
}

void GpuCullPass::buildPipeline(gfx::ShaderManager& shaderManager,
                                 VkDescriptorSetLayout globalSetLayout) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "GpuCullPass::buildPipeline called twice");

    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_shaderPath      = Paths::shaderSourceDir() / "gpu_cull.comp.hlsl";

    VkShaderModule cs = shaderManager.compile(m_shaderPath,
                                              gfx::ShaderManager::Stage::Compute, "CSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.computeShader     = cs;
    ci.computeEntryPoint = "CSMain";
    ci.globalSetLayout   = globalSetLayout;
    ci.pushConstantSize  = sizeof(GpuCullPushBlock);

    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    ENIGMA_LOG_INFO("[gpu_cull] pipeline built");
}

void GpuCullPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr);

    VkShaderModule cs = m_shaderManager->tryCompile(m_shaderPath,
                                                    gfx::ShaderManager::Stage::Compute, "CSMain");
    if (cs == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[gpu_cull] hot-reload: CS compile failed"); return; }

    vkDeviceWaitIdle(m_device->logical());
    delete m_pipeline;

    gfx::Pipeline::CreateInfo ci{};
    ci.computeShader     = cs;
    ci.computeEntryPoint = "CSMain";
    ci.globalSetLayout   = m_globalSetLayout;
    ci.pushConstantSize  = sizeof(GpuCullPushBlock);
    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    ENIGMA_LOG_INFO("[gpu_cull] hot-reload: pipeline rebuilt");
}

void GpuCullPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

void GpuCullPass::record(VkCommandBuffer           cmd,
                          VkDescriptorSet           globalSet,
                          const GpuSceneBuffer&     scene,
                          const GpuMeshletBuffer&   meshlets,
                          const IndirectDrawBuffer& indirect,
                          u32                       cameraSlot) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "GpuCullPass::record before buildPipeline");

    const u32 totalMeshlets = meshlets.total_meshlet_count();
    if (totalMeshlets == 0) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    GpuCullPushBlock pc{};
    pc.instanceBufferSlot = scene.slot();
    pc.meshletBufferSlot  = meshlets.meshlets_slot();
    pc.commandsBufferSlot = indirect.commands_slot();
    pc.countBufferSlot    = indirect.count_slot();
    pc.survivingIdsSlot   = indirect.surviving_slot();
    pc.cameraSlot         = cameraSlot;
    pc.totalMeshlets      = totalMeshlets;
    pc.instanceCount      = static_cast<u32>(scene.instance_count());

    vkCmdPushConstants(cmd, m_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    // One thread per meshlet, 64 threads per workgroup.
    const u32 groups = (totalMeshlets + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);
}

} // namespace enigma
