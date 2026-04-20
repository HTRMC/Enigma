// MicropolyCullPass.cpp
// ======================
// See MicropolyCullPass.h for the contract. This TU owns:
//   - Compute pipeline construction (mp_cluster_cull.comp.hlsl).
//   - Cull-stats + indirect-draw VkBuffer allocation.
//   - The dispatch() record — push-constant packing + barrier emit.
//
// Pattern mirror: GpuCullPass.cpp + PageCache.cpp for the Vulkan/VMA shape.

#include "renderer/micropoly/MicropolyCullPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

// VMA types only — Allocator.cpp stamps VMA_IMPLEMENTATION exactly once.
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4100 4127 4189 4324 4505)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include <cstring>
#include <utility>

namespace enigma::renderer::micropoly {

// Push block layout — must match PushBlock in mp_cluster_cull.comp.hlsl
// field-by-field.
struct MicropolyCullPushBlock {
    u32 totalClusterCount;
    u32 dagBufferBindlessIndex;
    // (residencyBitmapBindlessIndex removed — residency now routes through
    // pageToSlotBuffer, see MicropolyCullPass.h::DispatchInputs.)
    u32 requestQueueBindlessIndex;
    u32 indirectBufferBindlessIndex;
    u32 cullStatsBindlessIndex;
    u32 hiZBindlessIndex;
    u32 cameraSlot;
    f32 hiZMipCount;
    f32 screenSpaceErrorThreshold;
    u32 maxIndirectClusters;         // Security HIGH-1: indirect-draw slot cap.
    u32 pageCount;                   // Security HIGH-2: residency bitmap cap.
    u32 rasterClassBufferBindlessIndex;  // M4.4: u32 rasterClass per drawSlot.
    u32 screenHeight;                // M4.4: viewport height (px) for classifier.
    u32 pageToSlotBufferBindlessIndex;   // M4.4: pageId -> slotIndex for triCount.
    u32 pageCacheBufferBindlessIndex;    // M4.4: page pool for triCount fetch.
    u32 pageSlotBytes;               // M4.4: PageCache slot stride for triCount.
    u32 pageFirstDagNodeBufferBindlessIndex; // M4.5: pageId -> firstDagNodeIdx.
};

static_assert(sizeof(MicropolyCullPushBlock) == 68,
    "MicropolyCullPushBlock size must match PushBlock in mp_cluster_cull.comp.hlsl");

const char* micropolyCullErrorKindString(MicropolyCullErrorKind kind) {
    switch (kind) {
        case MicropolyCullErrorKind::PipelineBuildFailed:        return "PipelineBuildFailed";
        case MicropolyCullErrorKind::BufferCreationFailed:       return "BufferCreationFailed";
        case MicropolyCullErrorKind::BindlessRegistrationFailed: return "BindlessRegistrationFailed";
    }
    return "?";
}

// --- ctor / dtor ------------------------------------------------------------

MicropolyCullPass::MicropolyCullPass(gfx::Device& device,
                                     gfx::Allocator& allocator,
                                     gfx::DescriptorAllocator& descriptors)
    : m_device(&device), m_allocator(&allocator), m_descriptors(&descriptors) {}

MicropolyCullPass::~MicropolyCullPass() {
    destroy_();
}

MicropolyCullPass::MicropolyCullPass(MicropolyCullPass&& other) noexcept
    : m_device(other.m_device),
      m_allocator(other.m_allocator),
      m_descriptors(other.m_descriptors),
      m_shaderManager(other.m_shaderManager),
      m_pipeline(other.m_pipeline),
      m_globalSetLayout(other.m_globalSetLayout),
      m_shaderPath(std::move(other.m_shaderPath)),
      m_cullStatsBuffer(other.m_cullStatsBuffer),
      m_cullStatsAllocation(other.m_cullStatsAllocation),
      m_cullStatsBufferBytes(other.m_cullStatsBufferBytes),
      m_cullStatsBindlessSlot(other.m_cullStatsBindlessSlot),
      m_cullStatsMapped(other.m_cullStatsMapped),
      m_indirectDrawBuffer(other.m_indirectDrawBuffer),
      m_indirectDrawAllocation(other.m_indirectDrawAllocation),
      m_indirectDrawBufferBytes(other.m_indirectDrawBufferBytes),
      m_indirectDrawBindlessSlot(other.m_indirectDrawBindlessSlot),
      m_rasterClassBuffer(other.m_rasterClassBuffer),
      m_rasterClassAllocation(other.m_rasterClassAllocation),
      m_rasterClassBufferBytes(other.m_rasterClassBufferBytes),
      m_rasterClassBufferBindlessSlot(other.m_rasterClassBufferBindlessSlot) {
    other.m_device = nullptr;
    other.m_allocator = nullptr;
    other.m_descriptors = nullptr;
    other.m_shaderManager = nullptr;
    other.m_pipeline = nullptr;
    other.m_globalSetLayout = VK_NULL_HANDLE;
    other.m_cullStatsBuffer = VK_NULL_HANDLE;
    other.m_cullStatsAllocation = VK_NULL_HANDLE;
    other.m_cullStatsBufferBytes = 0ull;
    other.m_cullStatsBindlessSlot = UINT32_MAX;
    other.m_cullStatsMapped = nullptr;
    other.m_indirectDrawBuffer = VK_NULL_HANDLE;
    other.m_indirectDrawAllocation = VK_NULL_HANDLE;
    other.m_indirectDrawBufferBytes = 0ull;
    other.m_indirectDrawBindlessSlot = UINT32_MAX;
    other.m_rasterClassBuffer = VK_NULL_HANDLE;
    other.m_rasterClassAllocation = VK_NULL_HANDLE;
    other.m_rasterClassBufferBytes = 0ull;
    other.m_rasterClassBufferBindlessSlot = UINT32_MAX;
}

MicropolyCullPass& MicropolyCullPass::operator=(MicropolyCullPass&& other) noexcept {
    if (this == &other) return *this;
    destroy_();

    m_device                   = other.m_device;
    m_allocator                = other.m_allocator;
    m_descriptors              = other.m_descriptors;
    m_shaderManager            = other.m_shaderManager;
    m_pipeline                 = other.m_pipeline;
    m_globalSetLayout          = other.m_globalSetLayout;
    m_shaderPath               = std::move(other.m_shaderPath);
    m_cullStatsBuffer          = other.m_cullStatsBuffer;
    m_cullStatsAllocation      = other.m_cullStatsAllocation;
    m_cullStatsBufferBytes     = other.m_cullStatsBufferBytes;
    m_cullStatsBindlessSlot    = other.m_cullStatsBindlessSlot;
    m_cullStatsMapped          = other.m_cullStatsMapped;
    m_indirectDrawBuffer       = other.m_indirectDrawBuffer;
    m_indirectDrawAllocation   = other.m_indirectDrawAllocation;
    m_indirectDrawBufferBytes  = other.m_indirectDrawBufferBytes;
    m_indirectDrawBindlessSlot = other.m_indirectDrawBindlessSlot;
    m_rasterClassBuffer             = other.m_rasterClassBuffer;
    m_rasterClassAllocation         = other.m_rasterClassAllocation;
    m_rasterClassBufferBytes        = other.m_rasterClassBufferBytes;
    m_rasterClassBufferBindlessSlot = other.m_rasterClassBufferBindlessSlot;

    other.m_device = nullptr;
    other.m_allocator = nullptr;
    other.m_descriptors = nullptr;
    other.m_shaderManager = nullptr;
    other.m_pipeline = nullptr;
    other.m_globalSetLayout = VK_NULL_HANDLE;
    other.m_cullStatsBuffer = VK_NULL_HANDLE;
    other.m_cullStatsAllocation = VK_NULL_HANDLE;
    other.m_cullStatsBufferBytes = 0ull;
    other.m_cullStatsBindlessSlot = UINT32_MAX;
    other.m_cullStatsMapped = nullptr;
    other.m_indirectDrawBuffer = VK_NULL_HANDLE;
    other.m_indirectDrawAllocation = VK_NULL_HANDLE;
    other.m_indirectDrawBufferBytes = 0ull;
    other.m_indirectDrawBindlessSlot = UINT32_MAX;
    other.m_rasterClassBuffer = VK_NULL_HANDLE;
    other.m_rasterClassAllocation = VK_NULL_HANDLE;
    other.m_rasterClassBufferBytes = 0ull;
    other.m_rasterClassBufferBindlessSlot = UINT32_MAX;
    return *this;
}

void MicropolyCullPass::destroy_() {
    // Pipeline first — it transitively holds references to the layout.
    if (m_pipeline != nullptr) {
        delete m_pipeline;
        m_pipeline = nullptr;
    }

    // Release bindless slots.
    if (m_descriptors != nullptr) {
        if (m_cullStatsBindlessSlot != UINT32_MAX) {
            m_descriptors->releaseUavBuffer(m_cullStatsBindlessSlot);
            m_cullStatsBindlessSlot = UINT32_MAX;
        }
        if (m_indirectDrawBindlessSlot != UINT32_MAX) {
            m_descriptors->releaseUavBuffer(m_indirectDrawBindlessSlot);
            m_indirectDrawBindlessSlot = UINT32_MAX;
        }
        if (m_rasterClassBufferBindlessSlot != UINT32_MAX) {
            m_descriptors->releaseUavBuffer(m_rasterClassBufferBindlessSlot);
            m_rasterClassBufferBindlessSlot = UINT32_MAX;
        }
    }

    // Destroy VkBuffers.
    if (m_allocator != nullptr) {
        if (m_cullStatsBuffer != VK_NULL_HANDLE) {
            // vmaDestroyBuffer unmaps persistently-mapped allocations
            // automatically, so no explicit vmaUnmapMemory is needed.
            vmaDestroyBuffer(m_allocator->handle(), m_cullStatsBuffer,
                             m_cullStatsAllocation);
            m_cullStatsBuffer     = VK_NULL_HANDLE;
            m_cullStatsAllocation = VK_NULL_HANDLE;
            m_cullStatsMapped     = nullptr;
        }
        if (m_indirectDrawBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), m_indirectDrawBuffer,
                             m_indirectDrawAllocation);
            m_indirectDrawBuffer     = VK_NULL_HANDLE;
            m_indirectDrawAllocation = VK_NULL_HANDLE;
        }
        if (m_rasterClassBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), m_rasterClassBuffer,
                             m_rasterClassAllocation);
            m_rasterClassBuffer     = VK_NULL_HANDLE;
            m_rasterClassAllocation = VK_NULL_HANDLE;
        }
    }
    m_cullStatsBufferBytes    = 0ull;
    m_indirectDrawBufferBytes = 0ull;
    m_rasterClassBufferBytes  = 0ull;
}

// --- pipeline rebuild -------------------------------------------------------

bool MicropolyCullPass::rebuildPipeline_() {
    ENIGMA_ASSERT(m_shaderManager != nullptr);
    ENIGMA_ASSERT(m_device != nullptr);

    // Try-compile so hot-reload doesn't blow up the engine on a typo.
    VkShaderModule cs = m_shaderManager->tryCompile(
        m_shaderPath, gfx::ShaderManager::Stage::Compute, "CSMain");
    if (cs == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[micropoly_cull] shader compile failed: {}",
                         m_shaderPath.string());
        return false;
    }

    // Wait-idle before rebinding the pipeline. This mirrors HiZPass /
    // GpuCullPass — the compute pipeline can be in flight when hot-reload
    // fires, and swapping it without an idle is UB.
    if (m_pipeline != nullptr) {
        vkDeviceWaitIdle(m_device->logical());
        delete m_pipeline;
        m_pipeline = nullptr;
    }

    gfx::Pipeline::CreateInfo ci{};
    ci.computeShader     = cs;
    ci.computeEntryPoint = "CSMain";
    ci.globalSetLayout   = m_globalSetLayout;
    ci.pushConstantSize  = sizeof(MicropolyCullPushBlock);

    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);
    return true;
}

// --- create -----------------------------------------------------------------

std::expected<MicropolyCullPass, MicropolyCullError>
MicropolyCullPass::create(gfx::Device& device,
                          gfx::Allocator& allocator,
                          gfx::DescriptorAllocator& descriptors,
                          gfx::ShaderManager& shaderManager) {
    MicropolyCullPass pass(device, allocator, descriptors);
    pass.m_shaderManager   = &shaderManager;
    pass.m_globalSetLayout = descriptors.layout();
    pass.m_shaderPath      = Paths::shaderSourceDir()
                           / "micropoly" / "mp_cluster_cull.comp.hlsl";

    // --- 1. Cull-stats buffer -----------------------------------------------
    // 7 u32 counters, round up to 32 bytes for 16-byte alignment cleanliness.
    // Allocated HOST_VISIBLE + HOST_COHERENT + persistently-mapped so the
    // debug HUD (MicropolySettingsPanel) can read the counters each frame
    // without an explicit readback copy + fence. Atomic ops on host-visible
    // memory are Vulkan-spec-legal; the perf cost (~thousands of atomics
    // per frame through the PCIe bar on discrete GPUs) is sub-microsecond
    // and only paid on debug/settings builds.
    pass.m_cullStatsBufferBytes = 32ull;

    VkBufferCreateInfo statsBI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    statsBI.size  = pass.m_cullStatsBufferBytes;
    statsBI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    statsBI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo statsAI{};
    statsAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    statsAI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo statsAllocInfo{};
    if (vmaCreateBuffer(allocator.handle(), &statsBI, &statsAI,
                        &pass.m_cullStatsBuffer, &pass.m_cullStatsAllocation,
                        &statsAllocInfo) != VK_SUCCESS) {
        return std::unexpected(MicropolyCullError{
            MicropolyCullErrorKind::BufferCreationFailed,
            "vmaCreateBuffer for cullStatsBuffer failed"});
    }
    // VMA guarantees pMappedData is non-null when VMA_ALLOCATION_CREATE_MAPPED_BIT
    // is set on a host-visible allocation (the only memory type AUTO_PREFER_HOST
    // picks), so this pointer is valid for the lifetime of the allocation.
    pass.m_cullStatsMapped = statsAllocInfo.pMappedData;

    // --- 2. Indirect-draw buffer --------------------------------------------
    // Layout: 16-byte header + kMpMaxIndirectDrawClusters * 16-byte commands.
    // Command shape is {groupCountX, groupCountY, groupCountZ, clusterIdx}
    // matching VkDrawMeshTasksIndirectCommandEXT (12 bytes) + a per-draw
    // cluster index (4 bytes). M3.3 consumers read stride=16, offset=16.
    pass.m_indirectDrawBufferBytes =
        16ull + static_cast<VkDeviceSize>(kMpMaxIndirectDrawClusters) * 16ull;

    VkBufferCreateInfo drawBI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    drawBI.size = pass.m_indirectDrawBufferBytes;
    drawBI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    drawBI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo drawAI{};
    drawAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator.handle(), &drawBI, &drawAI,
                        &pass.m_indirectDrawBuffer,
                        &pass.m_indirectDrawAllocation, nullptr) != VK_SUCCESS) {
        // Destroy the already-allocated stats buffer so the error path is
        // leak-free. The expected<> return will trigger destroy_() via the
        // ~MicropolyCullPass() running on the return-value's moved-out
        // state anyway, but being explicit here is cheap insurance.
        vmaDestroyBuffer(allocator.handle(), pass.m_cullStatsBuffer,
                         pass.m_cullStatsAllocation);
        pass.m_cullStatsBuffer     = VK_NULL_HANDLE;
        pass.m_cullStatsAllocation = VK_NULL_HANDLE;
        return std::unexpected(MicropolyCullError{
            MicropolyCullErrorKind::BufferCreationFailed,
            "vmaCreateBuffer for indirectDrawBuffer failed"});
    }

    // --- 2b. M4.4 rasterClassBuffer ----------------------------------------
    // One u32 per survivor drawSlot — written by the cull shader next to the
    // indirect-draw emit. kMpMaxIndirectDrawClusters * 4 = 256 KiB at cap.
    pass.m_rasterClassBufferBytes =
        static_cast<VkDeviceSize>(kMpMaxIndirectDrawClusters) * 4ull;

    VkBufferCreateInfo classBI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    classBI.size  = pass.m_rasterClassBufferBytes;
    classBI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    classBI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo classAI{};
    classAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator.handle(), &classBI, &classAI,
                        &pass.m_rasterClassBuffer,
                        &pass.m_rasterClassAllocation, nullptr) != VK_SUCCESS) {
        vmaDestroyBuffer(allocator.handle(), pass.m_cullStatsBuffer,
                         pass.m_cullStatsAllocation);
        pass.m_cullStatsBuffer     = VK_NULL_HANDLE;
        pass.m_cullStatsAllocation = VK_NULL_HANDLE;
        vmaDestroyBuffer(allocator.handle(), pass.m_indirectDrawBuffer,
                         pass.m_indirectDrawAllocation);
        pass.m_indirectDrawBuffer     = VK_NULL_HANDLE;
        pass.m_indirectDrawAllocation = VK_NULL_HANDLE;
        return std::unexpected(MicropolyCullError{
            MicropolyCullErrorKind::BufferCreationFailed,
            "vmaCreateBuffer for rasterClassBuffer failed"});
    }

    // --- 3. Bindless registration -------------------------------------------
    pass.m_cullStatsBindlessSlot = descriptors.registerUavBuffer(
        pass.m_cullStatsBuffer, pass.m_cullStatsBufferBytes);
    pass.m_indirectDrawBindlessSlot = descriptors.registerUavBuffer(
        pass.m_indirectDrawBuffer, pass.m_indirectDrawBufferBytes);
    pass.m_rasterClassBufferBindlessSlot = descriptors.registerUavBuffer(
        pass.m_rasterClassBuffer, pass.m_rasterClassBufferBytes);
    if (pass.m_cullStatsBindlessSlot == UINT32_MAX
        || pass.m_indirectDrawBindlessSlot == UINT32_MAX
        || pass.m_rasterClassBufferBindlessSlot == UINT32_MAX) {
        return std::unexpected(MicropolyCullError{
            MicropolyCullErrorKind::BindlessRegistrationFailed,
            "registerUavBuffer returned UINT32_MAX"});
    }

    // --- 4. Pipeline --------------------------------------------------------
    if (!pass.rebuildPipeline_()) {
        return std::unexpected(MicropolyCullError{
            MicropolyCullErrorKind::PipelineBuildFailed,
            "rebuildPipeline_ failed"});
    }

    ENIGMA_LOG_INFO(
        "[micropoly_cull] pipeline built; indirectBuffer={} bytes, cullStats={} bytes, "
        "rasterClassBuffer={} bytes, bindless(indirect={}, stats={}, class={})",
        pass.m_indirectDrawBufferBytes, pass.m_cullStatsBufferBytes,
        pass.m_rasterClassBufferBytes,
        pass.m_indirectDrawBindlessSlot, pass.m_cullStatsBindlessSlot,
        pass.m_rasterClassBufferBindlessSlot);
    return pass;
}

// --- per-frame reset --------------------------------------------------------

void MicropolyCullPass::resetCounters(VkCommandBuffer cmd) const {
    ENIGMA_ASSERT(cmd != VK_NULL_HANDLE);
    if (m_cullStatsBuffer == VK_NULL_HANDLE) return;

    // Zero the whole cull-stats buffer + the first 16 bytes (header) of
    // the indirect-draw buffer. The draw-command region beyond the header
    // is left as-is; the shader overwrites whatever stale bytes survive.
    vkCmdFillBuffer(cmd, m_cullStatsBuffer, 0u,
                    m_cullStatsBufferBytes, 0u);
    vkCmdFillBuffer(cmd, m_indirectDrawBuffer, 0u, 16u, 0u);
    // M4.4: zero the per-drawSlot rasterClass tags. A stale tag would route
    // a now-HW cluster through the SW path (or vice versa) after a frame
    // where the survivor set shrinks. Full-buffer fill is cheap (256 KiB).
    if (m_rasterClassBuffer != VK_NULL_HANDLE) {
        vkCmdFillBuffer(cmd, m_rasterClassBuffer, 0u,
                        m_rasterClassBufferBytes, 0u);
    }
}

// --- dispatch ---------------------------------------------------------------

void MicropolyCullPass::dispatch(const DispatchInputs& inputs) {
    ENIGMA_ASSERT(m_pipeline != nullptr
        && "MicropolyCullPass::dispatch before create()");
    ENIGMA_ASSERT(inputs.cmd != VK_NULL_HANDLE);
    ENIGMA_ASSERT(inputs.globalSet != VK_NULL_HANDLE);

    // Zero-cluster dispatch is a legal no-op; still emit the reset so the
    // buffers start from a known state.
    if (inputs.totalClusterCount == 0u) {
        return;
    }

    vkCmdBindPipeline(inputs.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_pipeline->handle());
    vkCmdBindDescriptorSets(inputs.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipeline->layout(), 0u, 1u, &inputs.globalSet,
                            0u, nullptr);

    MicropolyCullPushBlock pc{};
    pc.totalClusterCount           = inputs.totalClusterCount;
    pc.dagBufferBindlessIndex      = inputs.dagBufferBindlessIndex;
    pc.requestQueueBindlessIndex   = inputs.requestQueueBindlessIndex;
    pc.indirectBufferBindlessIndex =
        (inputs.indirectBufferBindlessIndexOverride != UINT32_MAX)
            ? inputs.indirectBufferBindlessIndexOverride
            : m_indirectDrawBindlessSlot;
    pc.cullStatsBindlessIndex =
        (inputs.cullStatsBindlessIndexOverride != UINT32_MAX)
            ? inputs.cullStatsBindlessIndexOverride
            : m_cullStatsBindlessSlot;
    pc.hiZBindlessIndex            = inputs.hiZBindlessIndex;
    pc.cameraSlot                  = inputs.cameraSlot;
    pc.hiZMipCount                 = inputs.hiZMipCount;
    pc.screenSpaceErrorThreshold   = inputs.screenSpaceErrorThreshold;
    pc.maxIndirectClusters         = kMpMaxIndirectDrawClusters;
    pc.pageCount                   = inputs.pageCount;
    pc.rasterClassBufferBindlessIndex = m_rasterClassBufferBindlessSlot;
    pc.screenHeight                = inputs.screenHeight;
    pc.pageToSlotBufferBindlessIndex = inputs.pageToSlotBufferBindlessIndex;
    pc.pageCacheBufferBindlessIndex  = inputs.pageCacheBufferBindlessIndex;
    pc.pageSlotBytes               = inputs.pageSlotBytes;
    pc.pageFirstDagNodeBufferBindlessIndex = inputs.pageFirstDagNodeBufferBindlessIndex;

    vkCmdPushConstants(inputs.cmd, m_pipeline->layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc), &pc);

    const u32 groups = (inputs.totalClusterCount + 63u) / 64u;
    vkCmdDispatch(inputs.cmd, groups, 1u, 1u);

    // Barrier: compute SSBO writes → downstream readers. Indirect-draw
    // buffer feeds vkCmdDrawMeshTasksIndirectEXT (DRAW_INDIRECT stage) and
    // is also consumed by the task/mesh shaders that read the clusterIdx
    // payload (TASK_SHADER / MESH_SHADER stages). Cull-stats is HOST-read
    // at frame end for debug UI (HOST stage).
    // M4.4: rasterClassBuffer is read by the HW raster task shader
    // (TASK_SHADER) and the SW raster bin compute (COMPUTE_SHADER); emit a
    // parallel barrier so both consumers see coherent class tags.
    const VkBufferMemoryBarrier2 barriers[3] = {
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_indirectDrawBuffer, 0u, VK_WHOLE_SIZE
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_cullStatsBuffer, 0u, VK_WHOLE_SIZE
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_rasterClassBuffer, 0u, VK_WHOLE_SIZE
        },
    };

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = 3u;
    dep.pBufferMemoryBarriers    = barriers;
    vkCmdPipelineBarrier2(inputs.cmd, &dep);
}

// --- readback ---------------------------------------------------------------

CullStats MicropolyCullPass::readbackStats() const {
    // Null when micropoly is disabled (pass never constructed) or between
    // destroy_() and a subsequent create(). Callers use this as a "stats
    // unavailable" signal via the zero-initialised struct.
    if (m_cullStatsBuffer == VK_NULL_HANDLE || m_cullStatsMapped == nullptr) {
        return CullStats{};
    }

    // HOST_COHERENT host-visible memory: no explicit invalidate needed.
    // A torn read (one counter from frame N, another from frame N+1) is
    // possible on paper but harmless for the HUD — values only ever
    // advance monotonically within a frame and the shader resets them
    // to zero at frame start via vkCmdFillBuffer in resetCounters().
    const u32* src = static_cast<const u32*>(m_cullStatsMapped);
    CullStats s{};
    s.totalDispatched = src[0];
    s.culledLOD       = src[1];
    s.culledResidency = src[2];
    s.culledFrustum   = src[3];
    s.culledBackface  = src[4];
    s.culledHiZ       = src[5];
    s.visible         = src[6];
    return s;
}

// --- hot reload -------------------------------------------------------------

void MicropolyCullPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr
        && "registerHotReload before create()");
    reloader.watchGroup({m_shaderPath}, [this]() {
        if (!rebuildPipeline_()) {
            ENIGMA_LOG_ERROR("[micropoly_cull] hot-reload rebuild failed");
        } else {
            ENIGMA_LOG_INFO("[micropoly_cull] hot-reload: pipeline rebuilt");
        }
    });
}

} // namespace enigma::renderer::micropoly
