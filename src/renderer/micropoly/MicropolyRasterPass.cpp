// MicropolyRasterPass.cpp
// ========================
// See MicropolyRasterPass.h for the contract. This TU owns the Task+Mesh+
// Fragment pipeline construction and the per-frame draw recording. Pattern
// mirrors VisibilityBufferPass (task/mesh pipeline assembly) + MicropolyCullPass
// (bindless + hot-reload wiring).

#include "renderer/micropoly/MicropolyRasterPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

#include <array>
#include <cstring>
#include <utility>

namespace enigma::renderer::micropoly {

// Push block layout — must match PushBlock in both mp_raster.task.hlsl and
// mp_raster.mesh.hlsl. Task + Mesh + Fragment all see the identical block
// via a single push-constant range covering all three stages.
struct MicropolyRasterPushBlock {
    u32 indirectBufferBindlessIndex;
    u32 dagBufferBindlessIndex;
    u32 pageToSlotBufferBindlessIndex;
    u32 pageCacheBufferBindlessIndex;
    u32 cameraSlot;
    u32 visImageBindlessIndex;
    u32 pageSlotBytes;
    u32 pageCount;
    u32 dagNodeCount;
    u32 rasterClassBufferBindlessIndex; // M4.4: u32 rasterClass per drawSlot
    u32 pageFirstDagNodeBufferBindlessIndex; // M4.5: pageId -> firstDagNodeIdx
};

static_assert(sizeof(MicropolyRasterPushBlock) == 44,
    "MicropolyRasterPushBlock must stay 44 bytes — mirror in shaders/micropoly/mp_raster.task.hlsl (mesh shader sees push via pipeline layout but reads from task payload)");

// micropolyRasterErrorKindString is defined in MicropolyRasterPassError.cpp —
// split into its own TU so headless smoke tests can link the stringifier
// without pulling ShaderManager + Paths + Log + gfx::Allocator into their
// link graph. Matches peer pattern (out-of-line definition) while keeping
// test link graphs small.

// --- ctor / dtor ------------------------------------------------------------

MicropolyRasterPass::MicropolyRasterPass(gfx::Device& device,
                                         gfx::DescriptorAllocator& descriptors)
    : m_device(&device), m_descriptors(&descriptors) {}

MicropolyRasterPass::~MicropolyRasterPass() {
    destroy_();
}

MicropolyRasterPass::MicropolyRasterPass(MicropolyRasterPass&& other) noexcept
    : m_device(other.m_device),
      m_descriptors(other.m_descriptors),
      m_shaderManager(other.m_shaderManager),
      m_globalSetLayout(other.m_globalSetLayout),
      m_taskShaderPath(std::move(other.m_taskShaderPath)),
      m_meshShaderPath(std::move(other.m_meshShaderPath)),
      m_pipelineLayout(other.m_pipelineLayout),
      m_pipeline(other.m_pipeline) {
    other.m_device          = nullptr;
    other.m_descriptors     = nullptr;
    other.m_shaderManager   = nullptr;
    other.m_globalSetLayout = VK_NULL_HANDLE;
    other.m_pipelineLayout  = VK_NULL_HANDLE;
    other.m_pipeline        = VK_NULL_HANDLE;
}

MicropolyRasterPass& MicropolyRasterPass::operator=(MicropolyRasterPass&& other) noexcept {
    if (this == &other) return *this;
    destroy_();

    m_device          = other.m_device;
    m_descriptors     = other.m_descriptors;
    m_shaderManager   = other.m_shaderManager;
    m_globalSetLayout = other.m_globalSetLayout;
    m_taskShaderPath  = std::move(other.m_taskShaderPath);
    m_meshShaderPath  = std::move(other.m_meshShaderPath);
    m_pipelineLayout  = other.m_pipelineLayout;
    m_pipeline        = other.m_pipeline;

    other.m_device          = nullptr;
    other.m_descriptors     = nullptr;
    other.m_shaderManager   = nullptr;
    other.m_globalSetLayout = VK_NULL_HANDLE;
    other.m_pipelineLayout  = VK_NULL_HANDLE;
    other.m_pipeline        = VK_NULL_HANDLE;
    return *this;
}

void MicropolyRasterPass::destroy_() {
    if (m_device == nullptr) return;
    VkDevice dev = m_device->logical();
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
}

// --- pipeline rebuild -------------------------------------------------------

bool MicropolyRasterPass::rebuildPipeline_() {
    ENIGMA_ASSERT(m_shaderManager != nullptr);
    ENIGMA_ASSERT(m_device != nullptr);

    // Try-compile each stage before touching the live pipeline so a typo in
    // one shader doesn't leave the engine with a half-destroyed pipeline.
    VkShaderModule taskMod = m_shaderManager->tryCompile(
        m_taskShaderPath, gfx::ShaderManager::Stage::Task, "ASMain");
    if (taskMod == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[micropoly_raster] task shader compile failed: {}",
                         m_taskShaderPath.string());
        return false;
    }
    VkShaderModule meshMod = m_shaderManager->tryCompile(
        m_meshShaderPath, gfx::ShaderManager::Stage::Mesh, "MSMain");
    if (meshMod == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[micropoly_raster] mesh shader compile failed: {}",
                         m_meshShaderPath.string());
        vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
        return false;
    }
    VkShaderModule fragMod = m_shaderManager->tryCompile(
        m_meshShaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");
    if (fragMod == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[micropoly_raster] fragment shader compile failed: {}",
                         m_meshShaderPath.string());
        vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
        vkDestroyShaderModule(m_device->logical(), meshMod, nullptr);
        return false;
    }

    // Wait-idle before tearing the live pipeline down. Mirrors
    // MicropolyCullPass::rebuildPipeline_ — hot reload may fire while a
    // previous frame's pipeline is still in flight.
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        vkDestroyPipeline(m_device->logical(), m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }

    // Build pipeline layout on first call; on hot-reload we reuse the
    // layout (push-constant shape is stable).
    if (m_pipelineLayout == VK_NULL_HANDLE) {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT
                             | VK_SHADER_STAGE_MESH_BIT_EXT
                             | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0u;
        pushRange.size   = sizeof(MicropolyRasterPushBlock);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount         = 1u;
        layoutInfo.pSetLayouts            = &m_globalSetLayout;
        layoutInfo.pushConstantRangeCount = 1u;
        layoutInfo.pPushConstantRanges    = &pushRange;

        if (vkCreatePipelineLayout(m_device->logical(), &layoutInfo,
                                   nullptr, &m_pipelineLayout) != VK_SUCCESS) {
            ENIGMA_LOG_ERROR("[micropoly_raster] vkCreatePipelineLayout failed");
            vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
            vkDestroyShaderModule(m_device->logical(), meshMod, nullptr);
            vkDestroyShaderModule(m_device->logical(), fragMod, nullptr);
            return false;
        }
    }

    const std::array<VkPipelineShaderStageCreateInfo, 3> stages = {{
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u,
          VK_SHADER_STAGE_TASK_BIT_EXT, taskMod, "ASMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u,
          VK_SHADER_STAGE_MESH_BIT_EXT, meshMod, "MSMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u,
          VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "PSMain", nullptr },
    }};

    // Dynamic state: viewport + scissor; set per-frame in record().
    const std::array<VkDynamicState, 2> dynamicStates = {{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    }};
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = static_cast<u32>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1u;
    viewportState.scissorCount  = 1u;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    // No back-face culling: the cull shader already applies normal-cone
    // backface rejection at cluster granularity, and the asset bake may
    // produce triangles with mixed winding. Relying on the fragment
    // atomic-min to pick the nearest sample keeps correctness without
    // needing a bake-time winding normalisation step.
    rasterizer.cullMode  = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // No color attachments: the fragment shader writes via storage image
    // atomic-min. colorBlend.attachmentCount=0 is the "dummy" state Vulkan
    // expects for a no-color-target pipeline.
    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 0u;
    colorBlend.pAttachments    = nullptr;

    // No depth either — standard HW depth test would interfere with the
    // atomic-min-based occlusion used by the vis image. M4 may revisit
    // this for a discard-based fast path.
    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable       = VK_FALSE;
    depthStencil.depthWriteEnable      = VK_FALSE;
    depthStencil.depthCompareOp        = VK_COMPARE_OP_ALWAYS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable     = VK_FALSE;

    VkPipelineRenderingCreateInfo renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount    = 0u;
    renderingInfo.pColorAttachmentFormats = nullptr;
    renderingInfo.depthAttachmentFormat   = VK_FORMAT_UNDEFINED;

    // Mesh pipeline: pVertexInputState + pInputAssemblyState MUST be null.
    VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = nullptr;
    pipelineInfo.pInputAssemblyState = nullptr;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = m_pipelineLayout;

    const VkResult r = vkCreateGraphicsPipelines(m_device->logical(),
                                                 VK_NULL_HANDLE,
                                                 1u, &pipelineInfo,
                                                 nullptr, &m_pipeline);

    vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), meshMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), fragMod, nullptr);

    if (r != VK_SUCCESS) {
        ENIGMA_LOG_ERROR("[micropoly_raster] vkCreateGraphicsPipelines failed: {}",
                         static_cast<int>(r));
        return false;
    }
    return true;
}

// --- create -----------------------------------------------------------------

std::expected<MicropolyRasterPass, MicropolyRasterError>
MicropolyRasterPass::create(gfx::Device& device,
                            gfx::DescriptorAllocator& descriptors,
                            gfx::ShaderManager& shaderManager) {
    if (!device.supportsMeshShaders()) {
        return std::unexpected(MicropolyRasterError{
            MicropolyRasterErrorKind::MeshShadersUnsupported,
            "VK_EXT_mesh_shader not available"});
    }
    if (!device.supportsShaderImageInt64()) {
        // R64_UINT atomic-min requires SPV_EXT_shader_image_int64. The
        // fallback R32G32_UINT + CAS loop is not implemented in M3.3 —
        // plan defers to M4 alongside SW raster (same constraint).
        return std::unexpected(MicropolyRasterError{
            MicropolyRasterErrorKind::Int64ImageUnsupported,
            "VK_EXT_shader_image_atomic_int64 not available — M3.3 native path requires R64_UINT atomic-min"});
    }

    MicropolyRasterPass pass(device, descriptors);
    pass.m_shaderManager   = &shaderManager;
    pass.m_globalSetLayout = descriptors.layout();
    pass.m_taskShaderPath  = Paths::shaderSourceDir()
                           / "micropoly" / "mp_raster.task.hlsl";
    pass.m_meshShaderPath  = Paths::shaderSourceDir()
                           / "micropoly" / "mp_raster.mesh.hlsl";

    if (!pass.rebuildPipeline_()) {
        return std::unexpected(MicropolyRasterError{
            MicropolyRasterErrorKind::PipelineBuildFailed,
            "rebuildPipeline_ failed"});
    }

    ENIGMA_LOG_INFO("[micropoly_raster] pipeline built (task+mesh+frag, R64_UINT vis)");
    return pass;
}

// --- record ---------------------------------------------------------------

void MicropolyRasterPass::record(const DispatchInputs& inputs) {
    ENIGMA_ASSERT(m_pipeline != VK_NULL_HANDLE
        && "MicropolyRasterPass::record before create()");
    ENIGMA_ASSERT(inputs.cmd != VK_NULL_HANDLE);
    ENIGMA_ASSERT(inputs.globalSet != VK_NULL_HANDLE);

    // No-op guards.
    if (inputs.extent.width == 0u || inputs.extent.height == 0u) return;
    if (inputs.indirectBuffer == VK_NULL_HANDLE) return;
    if (inputs.maxClusters == 0u) return;
    if (inputs.dagBufferBindlessIndex == UINT32_MAX) return;

    VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    renderingInfo.renderArea           = { {0, 0}, inputs.extent };
    renderingInfo.layerCount           = 1u;
    renderingInfo.colorAttachmentCount = 0u;
    renderingInfo.pColorAttachments    = nullptr;
    renderingInfo.pDepthAttachment     = nullptr;
    renderingInfo.pStencilAttachment   = nullptr;

    vkCmdBeginRendering(inputs.cmd, &renderingInfo);

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(inputs.extent.width);
    viewport.height   = static_cast<float>(inputs.extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(inputs.cmd, 0u, 1u, &viewport);

    VkRect2D scissor{ {0, 0}, inputs.extent };
    vkCmdSetScissor(inputs.cmd, 0u, 1u, &scissor);

    vkCmdBindPipeline(inputs.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(inputs.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipelineLayout, 0u, 1u, &inputs.globalSet,
                            0u, nullptr);

    MicropolyRasterPushBlock pc{};
    pc.indirectBufferBindlessIndex    = inputs.indirectBufferBindlessIndex;
    pc.dagBufferBindlessIndex         = inputs.dagBufferBindlessIndex;
    pc.pageToSlotBufferBindlessIndex  = inputs.pageToSlotBufferBindlessIndex;
    pc.pageCacheBufferBindlessIndex   = inputs.pageCacheBufferBindlessIndex;
    pc.cameraSlot                     = inputs.cameraSlot;
    pc.visImageBindlessIndex          = inputs.visImageBindlessIndex;
    pc.pageSlotBytes                  = inputs.pageSlotBytes;
    pc.pageCount                      = inputs.pageCount;
    pc.dagNodeCount                   = inputs.dagNodeCount;
    pc.rasterClassBufferBindlessIndex = inputs.rasterClassBufferBindlessIndex;
    pc.pageFirstDagNodeBufferBindlessIndex = inputs.pageFirstDagNodeBufferBindlessIndex;

    vkCmdPushConstants(inputs.cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_TASK_BIT_EXT
                       | VK_SHADER_STAGE_MESH_BIT_EXT
                       | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0u, sizeof(pc), &pc);

    // vkCmdDrawMeshTasksIndirectCountEXT: reads the draw-count from
    // `indirectBuffer` at offset 0 (the header u32 the cull shader
    // InterlockedAdd'd onto), then reads up to maxClusters commands
    // starting at offset 16 with stride 16 — matching the layout emitted
    // by mp_cluster_cull.comp.hlsl::emitDrawCmd.
    vkCmdDrawMeshTasksIndirectCountEXT(inputs.cmd,
                                       inputs.indirectBuffer, 16u,  // draws at offset 16
                                       inputs.indirectBuffer, 0u,   // count at offset 0
                                       inputs.maxClusters,
                                       16u /* stride */);

    vkCmdEndRendering(inputs.cmd);

    // Post-draw barrier: fragment-shader SHADER_WRITE -> compute-shader
    // SHADER_READ on the vis image, so the downstream MaterialEvalPass
    // merge (M3.4) sees the completed atomic-min writes. The image stays
    // in GENERAL layout across this barrier.
    //
    // We don't name the specific VkImage here — MicropolyPass owns the
    // image handle and the caller supplies the bindless index. A full-
    // memory barrier is sufficient because the downstream reader is a
    // compute shader that binds the same storage-image slot.
    VkMemoryBarrier2 imgBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    imgBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    imgBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    imgBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1u;
    dep.pMemoryBarriers    = &imgBarrier;
    vkCmdPipelineBarrier2(inputs.cmd, &dep);
}

// --- hot reload -------------------------------------------------------------

void MicropolyRasterPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != VK_NULL_HANDLE
        && "registerHotReload before create()");
    reloader.watchGroup({m_taskShaderPath, m_meshShaderPath}, [this]() {
        if (!rebuildPipeline_()) {
            ENIGMA_LOG_ERROR("[micropoly_raster] hot-reload rebuild failed");
        } else {
            ENIGMA_LOG_INFO("[micropoly_raster] hot-reload: pipeline rebuilt");
        }
    });
}

} // namespace enigma::renderer::micropoly
