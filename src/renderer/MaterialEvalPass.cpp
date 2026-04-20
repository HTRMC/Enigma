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
    u32 meshletToInstanceSlot; // StructuredBuffer<float4> — u32 per globalMeshletId
    // M3.4 — appended at END of the block so the prefix layout is
    // byte-identical to the pre-M3.4 push block. The MP_ENABLE=false
    // pipeline variant dead-code-eliminates every read of this field at
    // the SPIR-V level; we still push a value so the Vulkan-side layout
    // stays stable between the two spec-constant variants.
    u32 vis64BufferSlot;       // RWTexture2D<uint64_t> — Micropoly vis image
    // M5 — fields required to walk mp geometry in the mpWins branch
    // (DAG node -> page -> on-disk cluster -> vertices). The MP_ENABLE=false
    // shader never declares these — DXC's preprocessor strips them — but
    // the C++ side always pushes them so the VkPipelineLayout push range
    // matches both pipeline variants.
    u32 mpDagBufferSlot;
    u32 mpPageToSlotSlot;
    u32 mpPageCacheSlot;
    u32 mpPageFirstDagNodeSlot;
    u32 mpPageSlotBytes;
    u32 mpPageCount;
    u32 mpDagNodeCount;
};

static_assert(sizeof(MaterialEvalPushBlock) == 92);

MaterialEvalPass::MaterialEvalPass(gfx::Device& device)
    : m_device(&device)
{}

MaterialEvalPass::~MaterialEvalPass() {
    delete m_pipelineMpDisabled;
    delete m_pipelineMpEnabled;
}

// Build one pipeline variant. The two variants are compiled from the same
// HLSL source with different -D defines so DXC produces distinct SPIR-V
// blobs: the =false compile sees no MP_ENABLE define (preprocessor strips
// all Int64 code) giving SPIR-V byte-identical to the pre-M3.4 golden
// (Principle 1). The =true compile passes -D MP_ENABLE=1.
gfx::Pipeline* MaterialEvalPass::buildPipeline_(VkShaderModule module,
                                                bool /*enableMp*/) const {
    gfx::Pipeline::CreateInfo ci{};
    ci.computeShader     = module;
    ci.computeEntryPoint = "CSMain";
    ci.globalSetLayout   = m_globalSetLayout;
    ci.pushConstantSize  = sizeof(MaterialEvalPushBlock);
    return new gfx::Pipeline(*m_device, ci);
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
    ENIGMA_ASSERT(m_pipelineMpDisabled == nullptr
        && "MaterialEvalPass::buildPipeline called twice");

    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_shaderPath      = Paths::shaderSourceDir() / "material_eval.comp.hlsl";

    // MP_ENABLE=false: compile with no extra defines — preprocessor strips all
    // Int64/micropoly code, producing SPIR-V byte-identical to the pre-M3.4
    // golden (Principle 1). MP_ENABLE=true is built lazily below.
    VkShaderModule cs = shaderManager.compile(m_shaderPath,
                                              gfx::ShaderManager::Stage::Compute, "CSMain");

    // Build the MP_ENABLE=false variant eagerly — this is the hot path for
    // devices where micropoly is disabled (capability row Disabled or
    // MicropolyConfig::enabled=false). MP_ENABLE=true is built lazily the
    // first time record(enableMp=true) fires, so disabled devices never
    // pay the second pipeline compile cost.
    m_pipelineMpDisabled = buildPipeline_(cs, /*enableMp=*/false);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    ENIGMA_LOG_INFO("[material_eval] pipeline built (MP_ENABLE=false)");
}

void MaterialEvalPass::ensureMpEnabledPipeline_() {
    if (m_pipelineMpEnabled != nullptr) return;

    // Compile the MP_ENABLE=true variant with -D MP_ENABLE=1 so DXC includes
    // the 64-bit vis image read and micropoly merge path. The preprocessor
    // approach (rather than spec constants) guarantees the =false blob stays
    // byte-identical to the pre-M3.4 golden by construction.
    //
    // Use tryCompile (non-fatal) rather than compile: a broken MP_ENABLE=1
    // source should not crash the engine on the first frame a micropoly
    // device ever renders. If compilation fails we leave m_pipelineMpEnabled
    // null, log once, and record() falls back to the baseline pipeline with
    // a throttled warning.
    ENIGMA_ASSERT(m_shaderManager != nullptr);
    VkShaderModule cs = m_shaderManager->tryCompile(
        m_shaderPath, gfx::ShaderManager::Stage::Compute, "CSMain",
        {"MP_ENABLE=1"});
    if (cs == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[material_eval] MP_ENABLE=1 compile failed; falling back to disabled pipeline");
        return;  // leave m_pipelineMpEnabled null
    }

    m_pipelineMpEnabled = buildPipeline_(cs, /*enableMp=*/true);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    ENIGMA_LOG_INFO("[material_eval] pipeline built (MP_ENABLE=true, lazy)");
}

void MaterialEvalPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipelineMpDisabled != nullptr);

    // Baseline: no defines.
    VkShaderModule cs = m_shaderManager->tryCompile(m_shaderPath,
                                                    gfx::ShaderManager::Stage::Compute, "CSMain");
    if (cs == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[material_eval] hot-reload: CS compile failed");
        return;
    }

    vkDeviceWaitIdle(m_device->logical());

    // Rebuild the baseline pipeline unconditionally.
    delete m_pipelineMpDisabled;
    m_pipelineMpDisabled = buildPipeline_(cs, /*enableMp=*/false);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    // Only rebuild the MP-enabled variant if it had already been built —
    // otherwise stay lazy. A hot-reload shouldn't eagerly create a pipeline
    // the app isn't using.
    if (m_pipelineMpEnabled != nullptr) {
        VkShaderModule csMp = m_shaderManager->tryCompile(
            m_shaderPath, gfx::ShaderManager::Stage::Compute, "CSMain",
            {"MP_ENABLE=1"});
        if (csMp != VK_NULL_HANDLE) {
            delete m_pipelineMpEnabled;
            m_pipelineMpEnabled = buildPipeline_(csMp, /*enableMp=*/true);
            vkDestroyShaderModule(m_device->logical(), csMp, nullptr);
        } else {
            ENIGMA_LOG_ERROR("[material_eval] hot-reload: MP_ENABLE=1 compile failed, keeping old pipeline");
        }
    }

    ENIGMA_LOG_INFO("[material_eval] hot-reload: pipeline(s) rebuilt");
}

void MaterialEvalPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipelineMpDisabled != nullptr);
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

void MaterialEvalPass::record(VkCommandBuffer         cmd,
                              VkDescriptorSet         globalSet,
                              VkExtent2D              extent,
                              u32                     visBufferSlot,
                              const GpuSceneBuffer&   scene,
                              const GpuMeshletBuffer& meshlets,
                              u32                     materialBufferSlot,
                              u32                     cameraSlot,
                              u32                     vis64BufferSlot,
                              bool                    enableMp,
                              const MpResolveInputs&  mp) {
    ENIGMA_ASSERT(m_pipelineMpDisabled != nullptr
        && "MaterialEvalPass::record before buildPipeline");
    ENIGMA_ASSERT(m_depth_image != VK_NULL_HANDLE && "MaterialEvalPass::record before prepare");

    // Lazy-build the MP-enabled variant the first time the caller asks for
    // it. Disabled devices never take this branch and never pay the cost.
    if (enableMp) {
        ensureMpEnabledPipeline_();
    }
    // The baseline (=false) pipeline handle is the single source of truth
    // for the Principle 1 SPIR-V byte-identity claim. Selecting between the
    // two variants here is a single pointer assignment — dispatch semantics
    // (barriers, descriptor sets, push constants) stay identical between
    // the two paths.
    //
    // Graceful fallback: if enableMp=true but the lazy MP_ENABLE=1 compile
    // failed (e.g. shader hot-reload dropped a syntax error into the source
    // the first time we touched it), fall back to the baseline pipeline.
    // The rendered frame loses the micropoly merge but every other G-buffer
    // channel renders correctly and the engine stays up. Log once per pass
    // lifetime to avoid spamming the console at 60 Hz.
    const gfx::Pipeline* activePipeline = m_pipelineMpDisabled;
    if (enableMp) {
        if (m_pipelineMpEnabled != nullptr) {
            activePipeline = m_pipelineMpEnabled;
        } else if (!m_mpFallbackLogged) {
            ENIGMA_LOG_ERROR("[material_eval] enableMp=true but MP pipeline unavailable; falling back to baseline");
            m_mpFallbackLogged = true;
        }
    }
    ENIGMA_ASSERT(activePipeline != nullptr);

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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, activePipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            activePipeline->layout(), 0, 1, &globalSet, 0, nullptr);

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
    pc.meshletToInstanceSlot = scene.meshlet_to_instance_slot();
    // When enableMp=false the shader never reads vis64BufferSlot (the DXC
    // dead-code pass removes the load entirely), so UINT32_MAX is a safe
    // sentinel that surfaces misuse via the bindless validator if a future
    // change ever actually reads it.
    pc.vis64BufferSlot       = enableMp ? vis64BufferSlot : UINT32_MAX;
    // M5 mp geometry walk inputs — only meaningful for enableMp=true.
    pc.mpDagBufferSlot        = enableMp ? mp.dagBufferSlot        : UINT32_MAX;
    pc.mpPageToSlotSlot       = enableMp ? mp.pageToSlotSlot       : UINT32_MAX;
    pc.mpPageCacheSlot        = enableMp ? mp.pageCacheSlot        : UINT32_MAX;
    pc.mpPageFirstDagNodeSlot = enableMp ? mp.pageFirstDagNodeSlot : UINT32_MAX;
    pc.mpPageSlotBytes        = enableMp ? mp.pageSlotBytes        : 0u;
    pc.mpPageCount            = enableMp ? mp.pageCount            : 0u;
    pc.mpDagNodeCount         = enableMp ? mp.dagNodeCount         : 0u;

    vkCmdPushConstants(cmd, activePipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
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
