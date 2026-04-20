// MicropolySwRasterPass.cpp
// ===========================
// See MicropolySwRasterPass.h for the contract. Structural mirror of
// MicropolyCullPass.cpp (compute-pipeline + VMA buffer + bindless lifecycle)
// and MicropolyRasterPass.cpp (push-block + record pattern). Error-kind
// stringifier lives in MicropolySwRasterPassError.cpp to keep the test
// binary's link graph minimal.

#include "renderer/micropoly/MicropolySwRasterPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "renderer/micropoly/MicropolyCullPass.h" // kMpMaxIndirectDrawClusters

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

// --- Push blocks ----------------------------------------------------------

// Must match PushBlock in shaders/micropoly/sw_raster_bin.comp.hlsl
// field-by-field.
struct MicropolySwRasterBinPushBlock {
    u32 indirectBufferBindlessIndex;
    u32 dagBufferBindlessIndex;
    u32 pageToSlotBufferBindlessIndex;
    u32 pageCacheBufferBindlessIndex;
    u32 cameraSlot;

    u32 tileBinCountBindlessIndex;
    u32 tileBinEntriesBindlessIndex;
    u32 spillBufferBindlessIndex;
    u32 spillHeadsBufferBindlessIndex; // M4.6: per-tile linked-list heads

    u32 viewportWidth;
    u32 viewportHeight;
    u32 tilesX;
    u32 tilesY;

    u32 pageSlotBytes;
    u32 pageCount;
    u32 dagNodeCount;
    u32 rasterClassBufferBindlessIndex; // M4.4: rasterClass per drawSlot
    u32 pageFirstDagNodeBufferBindlessIndex; // M4.5: pageId -> firstDagNodeIdx
};

static_assert(sizeof(MicropolySwRasterBinPushBlock) == 72,
    "MicropolySwRasterBinPushBlock size must match PushBlock in "
    "shaders/micropoly/sw_raster_bin.comp.hlsl");

// Must match PushBlock in sw_raster_bin_prep.comp.hlsl field-by-field. 4
// u32 fields = 16 bytes (3 load-bearing + 1 pad to keep the push block
// 16-byte aligned). The maxDispatchGroups field caps the dispatchIndirect
// header at the cull pass's slot cap — see the shader comment for the
// underlying InterlockedAdd race.
struct MicropolySwRasterBinPrepPushBlock {
    u32 indirectBufferBindlessIndex;
    u32 dispatchIndirectBufferBindlessIndex;
    u32 maxDispatchGroups;
    u32 _pad0;
};

static_assert(sizeof(MicropolySwRasterBinPrepPushBlock) == 16,
    "MicropolySwRasterBinPrepPushBlock size must match PushBlock in "
    "shaders/micropoly/sw_raster_bin_prep.comp.hlsl");

// M4.3: SW raster fragment push block. Matches PushBlock in
// shaders/micropoly/sw_raster.comp.hlsl field-by-field. 16 u32 fields = 64 B.
// Distinct from MicropolySwRasterBinPushBlock: this shader READS the bin
// SSBOs and WRITES the vis image; the binning shader is the inverse.
struct MicropolySwRasterPushBlock {
    u32 tileBinCountBindless;
    u32 tileBinEntriesBindless;
    u32 spillBufferBindless;
    u32 dagBufferBindless;
    u32 pageToSlotBindless;
    u32 pageCacheBindless;
    u32 indirectBufferBindless;
    u32 visImage64Bindless;
    u32 cameraSlot;
    u32 screenWidth;
    u32 screenHeight;
    u32 tilesX;
    u32 pageSlotBytes;
    u32 dagNodeCount;
    u32 pageCount;
    u32 pageFirstDagNodeBufferBindlessIndex; // M4.5: pageId -> firstDagNodeIdx
    u32 spillHeadsBufferBindless;            // M4.6: per-tile linked-list heads
};

static_assert(sizeof(MicropolySwRasterPushBlock) == 68,
    "MicropolySwRasterPushBlock size must match PushBlock in "
    "shaders/micropoly/sw_raster.comp.hlsl");

// --- ctor / dtor ----------------------------------------------------------

MicropolySwRasterPass::MicropolySwRasterPass(gfx::Device& device,
                                             gfx::Allocator& allocator,
                                             gfx::DescriptorAllocator& descriptors)
    : m_device(&device), m_allocator(&allocator), m_descriptors(&descriptors) {}

MicropolySwRasterPass::~MicropolySwRasterPass() {
    destroy_();
}

void MicropolySwRasterPass::destroy_() {
    // Pipelines first — they transitively hold the layout.
    if (m_binPipeline != nullptr) {
        delete m_binPipeline;
        m_binPipeline = nullptr;
    }
    if (m_prepPipeline != nullptr) {
        delete m_prepPipeline;
        m_prepPipeline = nullptr;
    }
    if (m_rasterPipeline != nullptr) {
        delete m_rasterPipeline;
        m_rasterPipeline = nullptr;
    }

    // Release bindless slots.
    if (m_descriptors != nullptr) {
        if (m_tileBinCountBindlessSlot != UINT32_MAX) {
            m_descriptors->releaseUavBuffer(m_tileBinCountBindlessSlot);
            m_tileBinCountBindlessSlot = UINT32_MAX;
        }
        if (m_tileBinEntriesBindlessSlot != UINT32_MAX) {
            m_descriptors->releaseUavBuffer(m_tileBinEntriesBindlessSlot);
            m_tileBinEntriesBindlessSlot = UINT32_MAX;
        }
        if (m_spillBufferBindlessSlot != UINT32_MAX) {
            m_descriptors->releaseUavBuffer(m_spillBufferBindlessSlot);
            m_spillBufferBindlessSlot = UINT32_MAX;
        }
        if (m_spillHeadsBufferBindlessSlot != UINT32_MAX) {
            m_descriptors->releaseUavBuffer(m_spillHeadsBufferBindlessSlot);
            m_spillHeadsBufferBindlessSlot = UINT32_MAX;
        }
        if (m_dispatchIndirectBindlessSlot != UINT32_MAX) {
            m_descriptors->releaseUavBuffer(m_dispatchIndirectBindlessSlot);
            m_dispatchIndirectBindlessSlot = UINT32_MAX;
        }
    }

    // Destroy VkBuffers.
    if (m_allocator != nullptr) {
        if (m_tileBinCountBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(),
                             m_tileBinCountBuffer,
                             m_tileBinCountAllocation);
            m_tileBinCountBuffer     = VK_NULL_HANDLE;
            m_tileBinCountAllocation = VK_NULL_HANDLE;
        }
        if (m_tileBinEntriesBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(),
                             m_tileBinEntriesBuffer,
                             m_tileBinEntriesAllocation);
            m_tileBinEntriesBuffer     = VK_NULL_HANDLE;
            m_tileBinEntriesAllocation = VK_NULL_HANDLE;
        }
        if (m_spillBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(),
                             m_spillBuffer,
                             m_spillAllocation);
            m_spillBuffer     = VK_NULL_HANDLE;
            m_spillAllocation = VK_NULL_HANDLE;
        }
        if (m_spillHeadsBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(),
                             m_spillHeadsBuffer,
                             m_spillHeadsAllocation);
            m_spillHeadsBuffer     = VK_NULL_HANDLE;
            m_spillHeadsAllocation = VK_NULL_HANDLE;
        }
        if (m_dispatchIndirectBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(),
                             m_dispatchIndirectBuffer,
                             m_dispatchIndirectAllocation);
            m_dispatchIndirectBuffer     = VK_NULL_HANDLE;
            m_dispatchIndirectAllocation = VK_NULL_HANDLE;
        }
    }
    m_tileBinCountBufferBytes     = 0ull;
    m_tileBinEntriesBufferBytes   = 0ull;
    m_spillBufferBytes            = 0ull;
    m_spillHeadsBufferBytes       = 0ull;
    m_dispatchIndirectBufferBytes = 0ull;
}

// --- pipeline rebuild -----------------------------------------------------

bool MicropolySwRasterPass::rebuildPipelines_() {
    ENIGMA_ASSERT(m_shaderManager != nullptr);
    ENIGMA_ASSERT(m_device != nullptr);

    // Try-compile all three stages first so a typo in one doesn't leave the
    // pass with only a partial pipeline rebuild.
    VkShaderModule prepMod = m_shaderManager->tryCompile(
        m_prepShaderPath, gfx::ShaderManager::Stage::Compute, "CSMain");
    if (prepMod == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[micropoly_sw_raster] prep shader compile failed: {}",
                         m_prepShaderPath.string());
        return false;
    }
    VkShaderModule binMod = m_shaderManager->tryCompile(
        m_binShaderPath, gfx::ShaderManager::Stage::Compute, "CSMain");
    if (binMod == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[micropoly_sw_raster] bin shader compile failed: {}",
                         m_binShaderPath.string());
        vkDestroyShaderModule(m_device->logical(), prepMod, nullptr);
        return false;
    }
    VkShaderModule rasterMod = m_shaderManager->tryCompile(
        m_rasterShaderPath, gfx::ShaderManager::Stage::Compute, "CSMain");
    if (rasterMod == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[micropoly_sw_raster] raster shader compile failed: {}",
                         m_rasterShaderPath.string());
        vkDestroyShaderModule(m_device->logical(), prepMod, nullptr);
        vkDestroyShaderModule(m_device->logical(), binMod,  nullptr);
        return false;
    }

    // Wait-idle before rebinding pipelines (hot-reload may fire mid-flight).
    if (m_binPipeline != nullptr || m_prepPipeline != nullptr || m_rasterPipeline != nullptr) {
        vkDeviceWaitIdle(m_device->logical());
        if (m_binPipeline != nullptr) {
            delete m_binPipeline;
            m_binPipeline = nullptr;
        }
        if (m_prepPipeline != nullptr) {
            delete m_prepPipeline;
            m_prepPipeline = nullptr;
        }
        if (m_rasterPipeline != nullptr) {
            delete m_rasterPipeline;
            m_rasterPipeline = nullptr;
        }
    }

    // Prep pipeline — push block is 16 B.
    {
        gfx::Pipeline::CreateInfo ci{};
        ci.computeShader     = prepMod;
        ci.computeEntryPoint = "CSMain";
        ci.globalSetLayout   = m_globalSetLayout;
        ci.pushConstantSize  = sizeof(MicropolySwRasterBinPrepPushBlock);
        m_prepPipeline = new gfx::Pipeline(*m_device, ci);
    }

    // Bin pipeline — push block is 64 B.
    {
        gfx::Pipeline::CreateInfo ci{};
        ci.computeShader     = binMod;
        ci.computeEntryPoint = "CSMain";
        ci.globalSetLayout   = m_globalSetLayout;
        ci.pushConstantSize  = sizeof(MicropolySwRasterBinPushBlock);
        m_binPipeline = new gfx::Pipeline(*m_device, ci);
    }

    // M4.3: raster fragment pipeline — push block is 64 B (distinct shape
    // from the bin pipeline despite the same size, see MicropolySwRasterPushBlock).
    {
        gfx::Pipeline::CreateInfo ci{};
        ci.computeShader     = rasterMod;
        ci.computeEntryPoint = "CSMain";
        ci.globalSetLayout   = m_globalSetLayout;
        ci.pushConstantSize  = sizeof(MicropolySwRasterPushBlock);
        m_rasterPipeline = new gfx::Pipeline(*m_device, ci);
    }

    vkDestroyShaderModule(m_device->logical(), prepMod,   nullptr);
    vkDestroyShaderModule(m_device->logical(), binMod,    nullptr);
    vkDestroyShaderModule(m_device->logical(), rasterMod, nullptr);
    return true;
}

// --- create ---------------------------------------------------------------

std::expected<std::unique_ptr<MicropolySwRasterPass>, MicropolySwRasterError>
MicropolySwRasterPass::create(gfx::Device& device,
                              gfx::Allocator& allocator,
                              gfx::DescriptorAllocator& descriptors,
                              gfx::ShaderManager& shaderManager,
                              VkExtent2D extent) {
    // Gate: mirror MicropolyRasterPass so SW raster sits under the same
    // capability umbrella. The fragment pipeline writes R64_UINT via
    // InterlockedMax, requiring shaderImageInt64; the bin + prep compute
    // pipelines only need mesh-shader umbrella wiring. Gate on both so a
    // device with mesh shaders but no Int64 image atomics doesn't spin up
    // a broken fragment pipeline.
    if (!device.supportsMeshShaders()) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::MeshShadersUnsupported,
            "VK_EXT_mesh_shader not available — SW raster gates off the same "
            "capability umbrella as the HW raster pass for M4.2"});
    }
    if (!device.supportsShaderImageInt64()) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::Int64ImageUnsupported,
            "VK_EXT_shader_image_atomic_int64 not available — SW raster "
            "fragment pipeline writes R64_UINT via InterlockedMax"});
    }

    auto pass = std::unique_ptr<MicropolySwRasterPass>(
        new MicropolySwRasterPass(device, allocator, descriptors));
    pass->m_shaderManager   = &shaderManager;
    pass->m_globalSetLayout = descriptors.layout();
    pass->m_binShaderPath    = Paths::shaderSourceDir()
                             / "micropoly" / "sw_raster_bin.comp.hlsl";
    pass->m_prepShaderPath   = Paths::shaderSourceDir()
                             / "micropoly" / "sw_raster_bin_prep.comp.hlsl";
    // M4.3: fragment raster shader path. Compiled eagerly alongside the
    // bin + prep pipelines so a missing/mistyped file surfaces at create()
    // rather than at first record().
    pass->m_rasterShaderPath = Paths::shaderSourceDir()
                             / "micropoly" / "sw_raster.comp.hlsl";

    pass->m_extent = extent;
    pass->m_tilesX = (extent.width  + kMpSwTileX - 1u) / kMpSwTileX;
    pass->m_tilesY = (extent.height + kMpSwTileY - 1u) / kMpSwTileY;
    pass->m_numTiles = pass->m_tilesX * pass->m_tilesY;

    // Guard against zero-sized extent at construction. Higher-level gating
    // in Renderer.cpp should prevent this; double-cover defensively so tests
    // that pass a zero extent get a clean error instead of a 0-byte buffer.
    if (pass->m_numTiles == 0u) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BufferAllocFailed,
            "extent yields zero tiles — SW raster refuses to size bin buffers "
            "from a zero viewport"});
    }

    // --- 1. tileBinCountBuffer ---------------------------------------------
    pass->m_tileBinCountBufferBytes =
        static_cast<VkDeviceSize>(pass->m_numTiles) * 4ull;

    VkBufferCreateInfo countBI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    countBI.size  = pass->m_tileBinCountBufferBytes;
    countBI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    countBI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo countAI{};
    countAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator.handle(), &countBI, &countAI,
                        &pass->m_tileBinCountBuffer,
                        &pass->m_tileBinCountAllocation, nullptr) != VK_SUCCESS) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BufferAllocFailed,
            "vmaCreateBuffer for tileBinCountBuffer failed"});
    }

    // --- 2. tileBinEntriesBuffer -------------------------------------------
    pass->m_tileBinEntriesBufferBytes =
        static_cast<VkDeviceSize>(pass->m_numTiles)
        * static_cast<VkDeviceSize>(kMpSwTileBinCap) * 4ull;

    VkBufferCreateInfo entriesBI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    entriesBI.size  = pass->m_tileBinEntriesBufferBytes;
    entriesBI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    entriesBI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo entriesAI{};
    entriesAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator.handle(), &entriesBI, &entriesAI,
                        &pass->m_tileBinEntriesBuffer,
                        &pass->m_tileBinEntriesAllocation, nullptr) != VK_SUCCESS) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BufferAllocFailed,
            "vmaCreateBuffer for tileBinEntriesBuffer failed"});
    }

    // --- 3. spillBuffer ----------------------------------------------------
    // Header: u32 spillCount + u32 spillDroppedCount (8 B). Array: spillCap
    // * {u32 tileIdx, u32 triRef} = spillCap * 8 B.
    pass->m_spillBufferBytes =
        8ull + static_cast<VkDeviceSize>(kMpSwSpillCap) * 8ull;

    VkBufferCreateInfo spillBI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    spillBI.size  = pass->m_spillBufferBytes;
    spillBI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    spillBI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo spillAI{};
    spillAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator.handle(), &spillBI, &spillAI,
                        &pass->m_spillBuffer,
                        &pass->m_spillAllocation, nullptr) != VK_SUCCESS) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BufferAllocFailed,
            "vmaCreateBuffer for spillBuffer failed"});
    }

    // --- 3b. spillHeadsBuffer ---------------------------------------------
    // M4.6: per-tile linked-list head. u32 * numTiles. Reset to
    // UINT32_MAX each frame via vkCmdFillBuffer.
    pass->m_spillHeadsBufferBytes =
        static_cast<VkDeviceSize>(pass->m_numTiles) * 4ull;
    VkBufferCreateInfo headsBI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    headsBI.size  = pass->m_spillHeadsBufferBytes;
    headsBI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    headsBI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo headsAI{};
    headsAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateBuffer(allocator.handle(), &headsBI, &headsAI,
                        &pass->m_spillHeadsBuffer,
                        &pass->m_spillHeadsAllocation, nullptr) != VK_SUCCESS) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BufferAllocFailed,
            "vmaCreateBuffer for spillHeadsBuffer failed"});
    }

    // --- 4. dispatchIndirectBuffer -----------------------------------------
    // {groupCountX, groupCountY, groupCountZ} = 12 B; round to 16 B for
    // alignment friendliness.
    pass->m_dispatchIndirectBufferBytes = 16ull;

    VkBufferCreateInfo diBI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    diBI.size  = pass->m_dispatchIndirectBufferBytes;
    diBI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
               | VK_BUFFER_USAGE_TRANSFER_DST_BIT
               | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    diBI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo diAI{};
    diAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator.handle(), &diBI, &diAI,
                        &pass->m_dispatchIndirectBuffer,
                        &pass->m_dispatchIndirectAllocation, nullptr) != VK_SUCCESS) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BufferAllocFailed,
            "vmaCreateBuffer for dispatchIndirectBuffer failed"});
    }

    // --- 5. Bindless registration ------------------------------------------
    pass->m_tileBinCountBindlessSlot = descriptors.registerUavBuffer(
        pass->m_tileBinCountBuffer, pass->m_tileBinCountBufferBytes);
    pass->m_tileBinEntriesBindlessSlot = descriptors.registerUavBuffer(
        pass->m_tileBinEntriesBuffer, pass->m_tileBinEntriesBufferBytes);
    pass->m_spillBufferBindlessSlot = descriptors.registerUavBuffer(
        pass->m_spillBuffer, pass->m_spillBufferBytes);
    pass->m_spillHeadsBufferBindlessSlot = descriptors.registerUavBuffer(
        pass->m_spillHeadsBuffer, pass->m_spillHeadsBufferBytes);
    pass->m_dispatchIndirectBindlessSlot = descriptors.registerUavBuffer(
        pass->m_dispatchIndirectBuffer, pass->m_dispatchIndirectBufferBytes);
    if (pass->m_tileBinCountBindlessSlot    == UINT32_MAX
        || pass->m_tileBinEntriesBindlessSlot == UINT32_MAX
        || pass->m_spillBufferBindlessSlot    == UINT32_MAX
        || pass->m_spillHeadsBufferBindlessSlot == UINT32_MAX
        || pass->m_dispatchIndirectBindlessSlot == UINT32_MAX) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BindlessRegistrationFailed,
            "registerUavBuffer returned UINT32_MAX"});
    }

    // --- 6. Pipelines ------------------------------------------------------
    if (!pass->rebuildPipelines_()) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::PipelineBuildFailed,
            "rebuildPipelines_ failed"});
    }

    ENIGMA_LOG_INFO(
        "[micropoly_sw_raster] pipelines built (prep+bin+raster); tiles={}x{} ({}), "
        "tileBinCount={} B, tileBinEntries={} B, spill={} B, "
        "bindless(count={}, entries={}, spill={}, dispatch={})",
        pass->m_tilesX, pass->m_tilesY, pass->m_numTiles,
        pass->m_tileBinCountBufferBytes, pass->m_tileBinEntriesBufferBytes,
        pass->m_spillBufferBytes,
        pass->m_tileBinCountBindlessSlot, pass->m_tileBinEntriesBindlessSlot,
        pass->m_spillBufferBindlessSlot, pass->m_dispatchIndirectBindlessSlot);
    return pass;
}

// --- record ---------------------------------------------------------------

void MicropolySwRasterPass::record(const DispatchInputs& inputs) {
    ENIGMA_ASSERT(m_binPipeline    != nullptr
        && "MicropolySwRasterPass::record before create()");
    ENIGMA_ASSERT(m_prepPipeline   != nullptr);
    ENIGMA_ASSERT(m_rasterPipeline != nullptr);
    ENIGMA_ASSERT(inputs.cmd != VK_NULL_HANDLE);
    ENIGMA_ASSERT(inputs.globalSet != VK_NULL_HANDLE);

    // No-op guards (match MicropolyRasterPass).
    if (inputs.extent.width == 0u || inputs.extent.height == 0u) return;
    if (inputs.indirectBuffer == VK_NULL_HANDLE) return;
    if (inputs.dagBufferBindlessIndex == UINT32_MAX) return;
    // M4.3 Phase 4 security: if a prior resize() failed it may have torn
    // down bin SSBOs; skip rather than dereference null handles.
    if (m_tileBinCountBuffer == VK_NULL_HANDLE
        || m_tileBinEntriesBuffer == VK_NULL_HANDLE
        || m_spillBuffer == VK_NULL_HANDLE
        || m_spillHeadsBuffer == VK_NULL_HANDLE) return;
    // M4.3: without a valid vis image the fragment pipeline would write
    // into an invalid bindless slot. Skip entirely (matches the style of
    // the other guards — callers that don't have a vis image get the
    // binning dispatch too, since it has no observable effect on other
    // passes). Extend the gate if M4.4's dispatcher wants a per-cluster
    // skip instead of a pass-level skip.
    const bool fragmentReady = inputs.visImage64Bindless != UINT32_MAX;

    // Step 1 + 2: zero tile-count header + spill header. The tileBinEntries
    // buffer is NOT zeroed because the count is the authoritative valid-
    // range marker; stale entries past the count are ignored by M4.3.
    vkCmdFillBuffer(inputs.cmd, m_tileBinCountBuffer, 0u,
                    m_tileBinCountBufferBytes, 0u);
    vkCmdFillBuffer(inputs.cmd, m_spillBuffer, 0u, 8u, 0u);
    // M4.6: reset per-tile linked-list heads to UINT32_MAX so an empty
    // tile's walker sees "no chain". Must happen before any bin dispatch
    // touches the heads via InterlockedExchange.
    vkCmdFillBuffer(inputs.cmd, m_spillHeadsBuffer, 0u,
                    m_spillHeadsBufferBytes, 0xFFFFFFFFu);

    // Step 3: TRANSFER -> COMPUTE_SHADER read/write barrier.
    VkMemoryBarrier2 clearBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    clearBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT
                               | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    clearBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                               | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo clearDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    clearDep.memoryBarrierCount = 1u;
    clearDep.pMemoryBarriers    = &clearBarrier;
    vkCmdPipelineBarrier2(inputs.cmd, &clearDep);

    // Step 4: prep compute — copy cull count -> dispatchIndirect {count,1,1}.
    vkCmdBindPipeline(inputs.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_prepPipeline->handle());
    vkCmdBindDescriptorSets(inputs.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_prepPipeline->layout(), 0u, 1u,
                            &inputs.globalSet, 0u, nullptr);
    {
        MicropolySwRasterBinPrepPushBlock prepPc{};
        prepPc.indirectBufferBindlessIndex         = inputs.indirectBufferBindlessIndex;
        prepPc.dispatchIndirectBufferBindlessIndex = m_dispatchIndirectBindlessSlot;
        // Security clamp — must match the cull pass's maxIndirectClusters.
        // See MicropolyCullPass.h kMpMaxIndirectDrawClusters (65,536). The
        // cull shader InterlockedAdds the counter UNCONDITIONALLY, so its
        // header may overshoot; without this clamp we'd hand an oversized
        // groupCountX to vkCmdDispatchIndirect.
        prepPc.maxDispatchGroups                   = kMpMaxIndirectDrawClusters;
        prepPc._pad0                               = 0u;
        vkCmdPushConstants(inputs.cmd, m_prepPipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           sizeof(prepPc), &prepPc);
    }
    vkCmdDispatch(inputs.cmd, 1u, 1u, 1u);

    // Step 5: prep write -> DRAW_INDIRECT read on the dispatchIndirect buffer.
    {
        VkBufferMemoryBarrier2 prepBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        prepBarrier.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        prepBarrier.srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        prepBarrier.dstStageMask        = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        prepBarrier.dstAccessMask       = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        prepBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepBarrier.buffer              = m_dispatchIndirectBuffer;
        prepBarrier.offset              = 0u;
        prepBarrier.size                = VK_WHOLE_SIZE;
        VkDependencyInfo prepDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        prepDep.bufferMemoryBarrierCount = 1u;
        prepDep.pBufferMemoryBarriers    = &prepBarrier;
        vkCmdPipelineBarrier2(inputs.cmd, &prepDep);
    }

    // Step 6: bin compute — one workgroup per cluster, indirect dispatch.
    vkCmdBindPipeline(inputs.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_binPipeline->handle());
    vkCmdBindDescriptorSets(inputs.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_binPipeline->layout(), 0u, 1u,
                            &inputs.globalSet, 0u, nullptr);
    {
        MicropolySwRasterBinPushBlock pc{};
        pc.indirectBufferBindlessIndex    = inputs.indirectBufferBindlessIndex;
        pc.dagBufferBindlessIndex         = inputs.dagBufferBindlessIndex;
        pc.pageToSlotBufferBindlessIndex  = inputs.pageToSlotBufferBindlessIndex;
        pc.pageCacheBufferBindlessIndex   = inputs.pageCacheBufferBindlessIndex;
        pc.cameraSlot                     = inputs.cameraSlot;
        pc.tileBinCountBindlessIndex      = m_tileBinCountBindlessSlot;
        pc.tileBinEntriesBindlessIndex    = m_tileBinEntriesBindlessSlot;
        pc.spillBufferBindlessIndex       = m_spillBufferBindlessSlot;
        pc.spillHeadsBufferBindlessIndex  = m_spillHeadsBufferBindlessSlot;
        pc.viewportWidth                  = inputs.extent.width;
        pc.viewportHeight                 = inputs.extent.height;
        pc.tilesX                         = m_tilesX;
        pc.tilesY                         = m_tilesY;
        pc.pageSlotBytes                  = inputs.pageSlotBytes;
        pc.pageCount                      = inputs.pageCount;
        pc.dagNodeCount                   = inputs.dagNodeCount;
        pc.rasterClassBufferBindlessIndex = inputs.rasterClassBufferBindlessIndex;
        pc.pageFirstDagNodeBufferBindlessIndex = inputs.pageFirstDagNodeBufferBindlessIndex;
        vkCmdPushConstants(inputs.cmd, m_binPipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           sizeof(pc), &pc);
    }
    vkCmdDispatchIndirect(inputs.cmd, m_dispatchIndirectBuffer, 0u);

    // Step 7: bin SSBO writes -> downstream compute readers (M4.3).
    const VkBufferMemoryBarrier2 postBarriers[4] = {
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_tileBinCountBuffer, 0u, VK_WHOLE_SIZE
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_tileBinEntriesBuffer, 0u, VK_WHOLE_SIZE
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_spillBuffer, 0u, VK_WHOLE_SIZE
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_spillHeadsBuffer, 0u, VK_WHOLE_SIZE
        },
    };
    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.bufferMemoryBarrierCount = 4u;
    postDep.pBufferMemoryBarriers    = postBarriers;
    vkCmdPipelineBarrier2(inputs.cmd, &postDep);

    // --- M4.3: Step 8 — SW raster fragment dispatch. ----------------------
    // Skip when we don't have a target vis image yet (e.g. early bringup
    // before MicropolyPass has created the image). The binning still ran
    // so downstream consumers (M4.4's dispatcher? M4.6 debug overlay?)
    // can still inspect the bin SSBOs.
    if (!fragmentReady) return;

    // The vis image may have been written by MicropolyRasterPass (HW raster)
    // earlier this frame. That pass emitted a FRAGMENT_SHADER write ->
    // COMPUTE_SHADER read barrier (MaterialEvalPass merge). For SW raster
    // we need COMPUTE_SHADER read + write on the same image — both paths
    // commutatively atomic-max the pixel, but visibility of prior writes
    // still needs a dependency. A memory barrier is sufficient because
    // the image layout stays GENERAL end-to-end (no image-layout
    // transition needed).
    {
        VkMemoryBarrier2 visBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        visBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        visBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        visBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        visBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                 | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo visDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        visDep.memoryBarrierCount = 1u;
        visDep.pMemoryBarriers    = &visBarrier;
        vkCmdPipelineBarrier2(inputs.cmd, &visDep);
    }

    // Bind the raster pipeline + descriptor set.
    vkCmdBindPipeline(inputs.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_rasterPipeline->handle());
    vkCmdBindDescriptorSets(inputs.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_rasterPipeline->layout(), 0u, 1u,
                            &inputs.globalSet, 0u, nullptr);

    {
        MicropolySwRasterPushBlock rpc{};
        rpc.tileBinCountBindless    = m_tileBinCountBindlessSlot;
        rpc.tileBinEntriesBindless  = m_tileBinEntriesBindlessSlot;
        rpc.spillBufferBindless     = m_spillBufferBindlessSlot;
        rpc.dagBufferBindless       = inputs.dagBufferBindlessIndex;
        rpc.pageToSlotBindless      = inputs.pageToSlotBufferBindlessIndex;
        rpc.pageCacheBindless       = inputs.pageCacheBufferBindlessIndex;
        rpc.indirectBufferBindless  = inputs.indirectBufferBindlessIndex;
        rpc.visImage64Bindless      = inputs.visImage64Bindless;
        rpc.cameraSlot              = inputs.cameraSlot;
        rpc.screenWidth             = inputs.extent.width;
        rpc.screenHeight            = inputs.extent.height;
        // M4.3 Phase 4: derive tile grid from per-frame extent, not cached
        // m_tilesX, so push block + dispatch grid agree even if a resize()
        // has not yet caught up to a new swapchain extent.
        rpc.tilesX                  = (inputs.extent.width  + kMpSwTileX - 1u) / kMpSwTileX;
        rpc.pageSlotBytes           = inputs.pageSlotBytes;
        rpc.dagNodeCount            = inputs.dagNodeCount;
        rpc.pageCount               = inputs.pageCount;
        rpc.pageFirstDagNodeBufferBindlessIndex = inputs.pageFirstDagNodeBufferBindlessIndex;
        rpc.spillHeadsBufferBindless = m_spillHeadsBufferBindlessSlot;
        vkCmdPushConstants(inputs.cmd, m_rasterPipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           sizeof(rpc), &rpc);
    }

    // Direct dispatch — one workgroup per 8x8 tile. Derive the grid from
    // inputs.extent so a transient m_tilesX/m_tilesY drift (failed
    // resize() / not-yet-called resize()) cannot mis-size the dispatch
    // relative to the push block. The shader's in-viewport check drops
    // right/bottom-edge threads that would shoot past width/height.
    const u32 dispatchTilesX = (inputs.extent.width  + kMpSwTileX - 1u) / kMpSwTileX;
    const u32 dispatchTilesY = (inputs.extent.height + kMpSwTileY - 1u) / kMpSwTileY;
    vkCmdDispatch(inputs.cmd, dispatchTilesX, dispatchTilesY, 1u);

    // Post-raster barrier: COMPUTE_SHADER write -> COMPUTE_SHADER read for
    // material_eval (M3.4 + follow-ups). MaterialEvalPass reads the vis
    // image via the same bindless slot; a memory barrier covers the
    // cross-stage ordering. MaterialEval's own FRAGMENT->COMPUTE barrier
    // at the start of its record() handles the HW-path ordering, but a
    // COMPUTE->COMPUTE dep isn't covered by that and we emit it here.
    {
        VkMemoryBarrier2 postRasterBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        postRasterBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postRasterBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        postRasterBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postRasterBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo postRasterDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        postRasterDep.memoryBarrierCount = 1u;
        postRasterDep.pMemoryBarriers    = &postRasterBarrier;
        vkCmdPipelineBarrier2(inputs.cmd, &postRasterDep);
    }
}

// --- resize ---------------------------------------------------------------

std::expected<void, MicropolySwRasterError>
MicropolySwRasterPass::resize(VkExtent2D newExtent) {
    ENIGMA_ASSERT(m_device    != nullptr);
    ENIGMA_ASSERT(m_allocator != nullptr);
    ENIGMA_ASSERT(m_descriptors != nullptr);

    // Zero-extent guard — matches create(). Callers shouldn't pass a
    // minimized-window extent; if they do, the bin SSBOs simply keep
    // their previous size (safe: the record() no-op guard catches this).
    const u32 newTilesX = (newExtent.width  + kMpSwTileX - 1u) / kMpSwTileX;
    const u32 newTilesY = (newExtent.height + kMpSwTileY - 1u) / kMpSwTileY;
    const u32 newNumTiles = newTilesX * newTilesY;
    if (newNumTiles == 0u) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BufferAllocFailed,
            "resize: newExtent yields zero tiles"});
    }

    // Early-out if the tile count hasn't changed (common: resize handler
    // fires on every swapchain recreate, which may be a no-op rotation).
    if (newTilesX == m_tilesX && newTilesY == m_tilesY
        && m_tileBinCountBuffer != VK_NULL_HANDLE) {
        m_extent = newExtent;
        return {};
    }

    // Drain any in-flight dispatch that might still be referencing the
    // old bin buffers. resizeGBuffer already idles for the swapchain
    // recreate path, but defence-in-depth — this pass may get called
    // from other paths as the engine evolves.
    vkDeviceWaitIdle(m_device->logical());

    // Release the four extent-dependent bindless slots + destroy the
    // VkBuffers. The dispatchIndirectBuffer is extent-independent (12 B
    // header) so we leave it alone.
    if (m_tileBinCountBindlessSlot != UINT32_MAX) {
        m_descriptors->releaseUavBuffer(m_tileBinCountBindlessSlot);
        m_tileBinCountBindlessSlot = UINT32_MAX;
    }
    if (m_tileBinEntriesBindlessSlot != UINT32_MAX) {
        m_descriptors->releaseUavBuffer(m_tileBinEntriesBindlessSlot);
        m_tileBinEntriesBindlessSlot = UINT32_MAX;
    }
    if (m_spillBufferBindlessSlot != UINT32_MAX) {
        m_descriptors->releaseUavBuffer(m_spillBufferBindlessSlot);
        m_spillBufferBindlessSlot = UINT32_MAX;
    }
    if (m_spillHeadsBufferBindlessSlot != UINT32_MAX) {
        m_descriptors->releaseUavBuffer(m_spillHeadsBufferBindlessSlot);
        m_spillHeadsBufferBindlessSlot = UINT32_MAX;
    }
    if (m_tileBinCountBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(),
                         m_tileBinCountBuffer,
                         m_tileBinCountAllocation);
        m_tileBinCountBuffer     = VK_NULL_HANDLE;
        m_tileBinCountAllocation = VK_NULL_HANDLE;
    }
    if (m_tileBinEntriesBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(),
                         m_tileBinEntriesBuffer,
                         m_tileBinEntriesAllocation);
        m_tileBinEntriesBuffer     = VK_NULL_HANDLE;
        m_tileBinEntriesAllocation = VK_NULL_HANDLE;
    }
    if (m_spillBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(),
                         m_spillBuffer,
                         m_spillAllocation);
        m_spillBuffer     = VK_NULL_HANDLE;
        m_spillAllocation = VK_NULL_HANDLE;
    }
    if (m_spillHeadsBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(),
                         m_spillHeadsBuffer,
                         m_spillHeadsAllocation);
        m_spillHeadsBuffer     = VK_NULL_HANDLE;
        m_spillHeadsAllocation = VK_NULL_HANDLE;
    }
    m_tileBinCountBufferBytes   = 0ull;
    m_tileBinEntriesBufferBytes = 0ull;
    m_spillBufferBytes          = 0ull;
    m_spillHeadsBufferBytes     = 0ull;

    // Commit new sizing.
    m_extent   = newExtent;
    m_tilesX   = newTilesX;
    m_tilesY   = newTilesY;
    m_numTiles = newNumTiles;

    // Re-allocate — same VMA call pattern as create().
    m_tileBinCountBufferBytes =
        static_cast<VkDeviceSize>(m_numTiles) * 4ull;
    {
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size  = m_tileBinCountBufferBytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateBuffer(m_allocator->handle(), &bi, &ai,
                            &m_tileBinCountBuffer,
                            &m_tileBinCountAllocation, nullptr) != VK_SUCCESS) {
            return std::unexpected(MicropolySwRasterError{
                MicropolySwRasterErrorKind::BufferAllocFailed,
                "resize: tileBinCountBuffer alloc failed"});
        }
    }

    m_tileBinEntriesBufferBytes =
        static_cast<VkDeviceSize>(m_numTiles)
        * static_cast<VkDeviceSize>(kMpSwTileBinCap) * 4ull;
    {
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size  = m_tileBinEntriesBufferBytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateBuffer(m_allocator->handle(), &bi, &ai,
                            &m_tileBinEntriesBuffer,
                            &m_tileBinEntriesAllocation, nullptr) != VK_SUCCESS) {
            return std::unexpected(MicropolySwRasterError{
                MicropolySwRasterErrorKind::BufferAllocFailed,
                "resize: tileBinEntriesBuffer alloc failed"});
        }
    }

    // Spill buffer bytes are tile-count-independent (capacity-driven), but
    // destroy_ + re-create symmetry keeps the lifecycle uniform.
    m_spillBufferBytes = 8ull
        + static_cast<VkDeviceSize>(kMpSwSpillCap) * 8ull;
    {
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size  = m_spillBufferBytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateBuffer(m_allocator->handle(), &bi, &ai,
                            &m_spillBuffer,
                            &m_spillAllocation, nullptr) != VK_SUCCESS) {
            return std::unexpected(MicropolySwRasterError{
                MicropolySwRasterErrorKind::BufferAllocFailed,
                "resize: spillBuffer alloc failed"});
        }
    }

    // M4.6: spillHeads is extent-dependent (one u32 per tile).
    m_spillHeadsBufferBytes =
        static_cast<VkDeviceSize>(m_numTiles) * 4ull;
    {
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size  = m_spillHeadsBufferBytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateBuffer(m_allocator->handle(), &bi, &ai,
                            &m_spillHeadsBuffer,
                            &m_spillHeadsAllocation, nullptr) != VK_SUCCESS) {
            return std::unexpected(MicropolySwRasterError{
                MicropolySwRasterErrorKind::BufferAllocFailed,
                "resize: spillHeadsBuffer alloc failed"});
        }
    }

    // Re-register all four extent-dependent SSBOs as bindless UAVs.
    m_tileBinCountBindlessSlot = m_descriptors->registerUavBuffer(
        m_tileBinCountBuffer, m_tileBinCountBufferBytes);
    m_tileBinEntriesBindlessSlot = m_descriptors->registerUavBuffer(
        m_tileBinEntriesBuffer, m_tileBinEntriesBufferBytes);
    m_spillBufferBindlessSlot = m_descriptors->registerUavBuffer(
        m_spillBuffer, m_spillBufferBytes);
    m_spillHeadsBufferBindlessSlot = m_descriptors->registerUavBuffer(
        m_spillHeadsBuffer, m_spillHeadsBufferBytes);
    if (m_tileBinCountBindlessSlot    == UINT32_MAX
        || m_tileBinEntriesBindlessSlot == UINT32_MAX
        || m_spillBufferBindlessSlot    == UINT32_MAX
        || m_spillHeadsBufferBindlessSlot == UINT32_MAX) {
        return std::unexpected(MicropolySwRasterError{
            MicropolySwRasterErrorKind::BindlessRegistrationFailed,
            "resize: registerUavBuffer returned UINT32_MAX"});
    }

    ENIGMA_LOG_INFO(
        "[micropoly_sw_raster] resize: tiles={}x{} ({}), "
        "tileBinCount={} B, tileBinEntries={} B, spill={} B, "
        "bindless(count={}, entries={}, spill={})",
        m_tilesX, m_tilesY, m_numTiles,
        m_tileBinCountBufferBytes, m_tileBinEntriesBufferBytes,
        m_spillBufferBytes,
        m_tileBinCountBindlessSlot, m_tileBinEntriesBindlessSlot,
        m_spillBufferBindlessSlot);
    return {};
}

// --- hot reload -----------------------------------------------------------

void MicropolySwRasterPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_binPipeline    != nullptr
        && "registerHotReload before create()");
    ENIGMA_ASSERT(m_prepPipeline   != nullptr);
    ENIGMA_ASSERT(m_rasterPipeline != nullptr);
    reloader.watchGroup({m_binShaderPath, m_prepShaderPath, m_rasterShaderPath},
        [this]() {
            if (!rebuildPipelines_()) {
                ENIGMA_LOG_ERROR("[micropoly_sw_raster] hot-reload rebuild failed");
            } else {
                ENIGMA_LOG_INFO("[micropoly_sw_raster] hot-reload: pipelines rebuilt");
            }
        });
}

} // namespace enigma::renderer::micropoly
