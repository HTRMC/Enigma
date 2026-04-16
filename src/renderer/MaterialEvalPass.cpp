#include "renderer/MaterialEvalPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "renderer/GpuMeshletBuffer.h"
#include "renderer/GpuSceneBuffer.h"

namespace enigma {

// Push block matching material_eval.comp.hlsl PushBlock exactly.
struct MaterialEvalPushBlock {
    u32 visBufferSlot;        // Texture2D<uint> — vis buffer
    u32 depthBufferSlot;      // Texture2D<float> — depth
    u32 instanceBufferSlot;   // StructuredBuffer<GpuInstance>
    u32 meshletBufferSlot;    // StructuredBuffer<Meshlet>
    u32 meshletVerticesSlot;  // StructuredBuffer<uint>   — vertex index remapping
    u32 meshletTrianglesSlot; // RWByteAddressBuffer — packed u8 triangle indices
    u32 materialBufferSlot;   // StructuredBuffer<float4> — GpuMaterial[]
    u32 cameraSlot;
    u32 albedoStorageSlot;    // RWTexture2D — G-buffer write targets
    u32 normalStorageSlot;
    u32 metalRoughStorageSlot;
    u32 motionVecStorageSlot;
    u32 screenWidth;
    u32 screenHeight;
    u32 instanceCount;  // number of GpuInstance entries — for meshlet→instance walk
};

static_assert(sizeof(MaterialEvalPushBlock) == 60);

MaterialEvalPass::MaterialEvalPass(gfx::Device& device)
    : m_device(&device)
{}

MaterialEvalPass::~MaterialEvalPass() {
    delete m_pipeline;
}

void MaterialEvalPass::prepare(VkExtent2D extent,
                               VkImage albedoImage,     VkImage normalImage,
                               VkImage metalRoughImage, VkImage motionVecImage,
                               VkImage depthImage,
                               u32 albedoStorageSlot,     u32 normalStorageSlot,
                               u32 metalRoughStorageSlot, u32 motionVecStorageSlot,
                               u32 depthSampledSlot) {
    m_extent = extent;

    m_albedo_image     = albedoImage;
    m_normal_image     = normalImage;
    m_metalRough_image = metalRoughImage;
    m_motionVec_image  = motionVecImage;
    m_depth_image      = depthImage;

    m_albedo_slot     = albedoStorageSlot;
    m_normal_slot     = normalStorageSlot;
    m_metalRough_slot = metalRoughStorageSlot;
    m_motionVec_slot  = motionVecStorageSlot;
    m_depth_slot      = depthSampledSlot;
}

void MaterialEvalPass::buildPipeline(gfx::ShaderManager& shaderManager,
                                     VkDescriptorSetLayout globalSetLayout) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "MaterialEvalPass::buildPipeline called twice");

    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_shaderPath      = Paths::shaderSourceDir() / "material_eval.comp.hlsl";

    VkShaderModule cs = shaderManager.compile(m_shaderPath,
                                              gfx::ShaderManager::Stage::Compute, "CSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.computeShader     = cs;
    ci.computeEntryPoint = "CSMain";
    ci.globalSetLayout   = globalSetLayout;
    ci.pushConstantSize  = sizeof(MaterialEvalPushBlock);

    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    ENIGMA_LOG_INFO("[material_eval] pipeline built");
}

void MaterialEvalPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr);

    VkShaderModule cs = m_shaderManager->tryCompile(m_shaderPath,
                                                    gfx::ShaderManager::Stage::Compute, "CSMain");
    if (cs == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[material_eval] hot-reload: CS compile failed");
        return;
    }

    vkDeviceWaitIdle(m_device->logical());
    delete m_pipeline;

    gfx::Pipeline::CreateInfo ci{};
    ci.computeShader     = cs;
    ci.computeEntryPoint = "CSMain";
    ci.globalSetLayout   = m_globalSetLayout;
    ci.pushConstantSize  = sizeof(MaterialEvalPushBlock);
    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    ENIGMA_LOG_INFO("[material_eval] hot-reload: pipeline rebuilt");
}

void MaterialEvalPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

void MaterialEvalPass::record(VkCommandBuffer         cmd,
                              VkDescriptorSet         globalSet,
                              VkExtent2D              extent,
                              u32                     visBufferSlot,
                              const GpuSceneBuffer&   scene,
                              const GpuMeshletBuffer& meshlets,
                              u32                     materialBufferSlot,
                              u32                     cameraSlot) {
    ENIGMA_ASSERT(m_pipeline    != nullptr       && "MaterialEvalPass::record before buildPipeline");
    ENIGMA_ASSERT(m_depth_image != VK_NULL_HANDLE && "MaterialEvalPass::record before prepare");

    // Pre-dispatch barriers:
    //   G-buffer color images:  UNDEFINED → GENERAL          (storage write)
    //   Depth image:            DEPTH_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    VkImageMemoryBarrier2 preBarriers[5]{};

    for (int i = 0; i < 4; ++i) {
        preBarriers[i].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        preBarriers[i].srcStageMask     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        preBarriers[i].srcAccessMask    = VK_ACCESS_2_NONE;
        preBarriers[i].dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        preBarriers[i].dstAccessMask    = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        preBarriers[i].oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        preBarriers[i].newLayout        = VK_IMAGE_LAYOUT_GENERAL;
        preBarriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    }
    preBarriers[0].image = m_albedo_image;
    preBarriers[1].image = m_normal_image;
    preBarriers[2].image = m_metalRough_image;
    preBarriers[3].image = m_motionVec_image;

    // Depth: already in SHADER_READ_ONLY_OPTIMAL — VisibilityBufferPass::record()
    // and recordTerrain() both emit a post-pass barrier that transitions depth
    // from DEPTH_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL before returning.
    // This barrier is a layout-identical no-op that satisfies validation; the
    // real memory visibility was established by the VB post-pass barrier.
    preBarriers[4].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    preBarriers[4].srcStageMask     = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    preBarriers[4].srcAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    preBarriers[4].dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    preBarriers[4].dstAccessMask    = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    preBarriers[4].oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    preBarriers[4].newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    preBarriers[4].image            = m_depth_image;
    preBarriers[4].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

    VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    preDep.imageMemoryBarrierCount = 5;
    preDep.pImageMemoryBarriers    = preBarriers;
    vkCmdPipelineBarrier2(cmd, &preDep);

    // Dispatch.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    MaterialEvalPushBlock pc{};
    pc.visBufferSlot         = visBufferSlot;
    pc.depthBufferSlot       = m_depth_slot;
    pc.instanceBufferSlot    = scene.slot();
    pc.meshletBufferSlot     = meshlets.meshlets_slot();
    pc.meshletVerticesSlot   = meshlets.vertices_slot();
    pc.meshletTrianglesSlot  = meshlets.triangles_slot();
    pc.materialBufferSlot    = materialBufferSlot;
    pc.cameraSlot            = cameraSlot;
    pc.albedoStorageSlot     = m_albedo_slot;
    pc.normalStorageSlot     = m_normal_slot;
    pc.metalRoughStorageSlot = m_metalRough_slot;
    pc.motionVecStorageSlot  = m_motionVec_slot;
    pc.screenWidth           = extent.width;
    pc.screenHeight          = extent.height;
    pc.instanceCount         = static_cast<u32>(scene.instance_count());

    vkCmdPushConstants(cmd, m_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    const u32 groupsX = (extent.width  + 7) / 8;
    const u32 groupsY = (extent.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    // Post-dispatch: G-buffer images GENERAL → SHADER_READ_ONLY_OPTIMAL
    // so LightingPass, RT passes, and post-process can sample them.
    VkImageMemoryBarrier2 postBarriers[4]{};
    for (int i = 0; i < 4; ++i) {
        postBarriers[i].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        postBarriers[i].srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postBarriers[i].srcAccessMask    = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        postBarriers[i].dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                         | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        postBarriers[i].dstAccessMask    = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        postBarriers[i].oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
        postBarriers[i].newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        postBarriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    }
    postBarriers[0].image = m_albedo_image;
    postBarriers[1].image = m_normal_image;
    postBarriers[2].image = m_metalRough_image;
    postBarriers[3].image = m_motionVec_image;

    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.imageMemoryBarrierCount = 4;
    postDep.pImageMemoryBarriers    = postBarriers;
    vkCmdPipelineBarrier2(cmd, &postDep);
}

} // namespace enigma
