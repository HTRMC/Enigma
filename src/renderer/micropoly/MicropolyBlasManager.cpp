// MicropolyBlasManager.cpp
// =========================
// See MicropolyBlasManager.h for the contract. This TU owns:
//   - DAG level-3 cluster extraction from an MpAssetReader.
//   - CPU-side zstd decompression of the involved pages (one-shot per
//     asset-load; not per-frame — so we borrow the reader's fetchPage
//     API directly rather than tapping the GPU PageCache).
//   - Triangle gather into DEVICE_LOCAL vertex + u32 index buffers via
//     a staging upload.
//   - vkCmdBuildAccelerationStructuresKHR on the graphics queue with a
//     fence-wait (mirrors gfx::BLAS::build).
//
// Pattern mirror: gfx::BLAS::build for the Vulkan/VMA shape;
// MicropolyCullPass.cpp for the factory + destroy_() idiom.
//
// Capability gate: create() on a device without supportsRayTracing()
// returns an OK stub. The stub's buildForAsset() returns NotSupported
// and the stub never calls any RT-extension entrypoint — Principle 1.

#include "renderer/micropoly/MicropolyBlasManager.h"

#include "asset/MpAssetFormat.h"
#include "asset/MpAssetReader.h"
#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"

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

#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace enigma::renderer::micropoly {

namespace {

// Position-only vertex stride for the shadow BLAS. Three floats; no UVs,
// normals, or material info — RT shadow tracing cares only about
// geometry. Keeps the BLAS memory footprint minimal.
constexpr VkDeviceSize kShadowVertexStride = sizeof(f32) * 3u;

// Hard cap on aggregate triangle count per per-asset-proxy BLAS. Defense
// against a crafted .mpa advertising an unreasonable triangle count — the
// build would otherwise allocate AS storage proportional to the claim.
// Sized for realistic AAA content: real-world BMW GT3 asset has ~2.5M
// triangles total with DAG level-3 pulling ~500K-1M; 5M is still a
// defensive cap (a legitimate asset wouldn't have a billion-tri BLAS)
// while allowing genuine content through. Previously 200K which was too
// low for anything beyond toy scenes.
constexpr u32 kMaxBlasTriangles = 5'000'000u;

// Helper: create a DEVICE_LOCAL buffer with the given usage and fill it
// from a host source via a temporary staging buffer. Returns the
// destination VkBuffer + VmaAllocation; writes nothing on failure.
// Submits on the graphics queue + waits idle (acceptable at asset-load).
bool uploadDeviceLocalBuffer(gfx::Device&        device,
                             gfx::Allocator&     allocator,
                             const void*         src,
                             VkDeviceSize        bytes,
                             VkBufferUsageFlags  usage,
                             VkBuffer&           outBuffer,
                             VmaAllocation&      outAllocation,
                             std::string&        outDetail) {
    outBuffer     = VK_NULL_HANDLE;
    outAllocation = nullptr;

    // Staging — host-visible, sequentially-written, mapped.
    VkBufferCreateInfo stagingCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stagingCI.size  = bytes;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAI{};
    stagingAI.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer          stagingBuf   = VK_NULL_HANDLE;
    VmaAllocation     stagingAlloc = nullptr;
    VmaAllocationInfo stagingInfo{};
    if (vmaCreateBuffer(allocator.handle(), &stagingCI, &stagingAI,
                        &stagingBuf, &stagingAlloc, &stagingInfo) != VK_SUCCESS) {
        outDetail = "vmaCreateBuffer (staging) failed";
        return false;
    }
    std::memcpy(stagingInfo.pMappedData, src, static_cast<std::size_t>(bytes));

    // Destination — DEVICE_LOCAL, transfer dst + caller-requested usage.
    VkBufferCreateInfo dstCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    dstCI.size  = bytes;
    dstCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage;

    VmaAllocationCreateInfo dstAI{};
    dstAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator.handle(), &dstCI, &dstAI,
                        &outBuffer, &outAllocation, nullptr) != VK_SUCCESS) {
        vmaDestroyBuffer(allocator.handle(), stagingBuf, stagingAlloc);
        outDetail = "vmaCreateBuffer (device-local) failed";
        return false;
    }

    // Immediate single-use cmd buffer for the copy.
    VkCommandPoolCreateInfo poolCI{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = device.graphicsQueueFamily();

    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device.logical(), &poolCI, nullptr, &pool) != VK_SUCCESS) {
        vmaDestroyBuffer(allocator.handle(), stagingBuf, stagingAlloc);
        vmaDestroyBuffer(allocator.handle(), outBuffer, outAllocation);
        outBuffer = VK_NULL_HANDLE; outAllocation = nullptr;
        outDetail = "vkCreateCommandPool failed";
        return false;
    }

    VkCommandBufferAllocateInfo cmdAI{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cmdAI.commandPool        = pool;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1u;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device.logical(), &cmdAI, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(device.logical(), pool, nullptr);
        vmaDestroyBuffer(allocator.handle(), stagingBuf, stagingAlloc);
        vmaDestroyBuffer(allocator.handle(), outBuffer, outAllocation);
        outBuffer = VK_NULL_HANDLE; outAllocation = nullptr;
        outDetail = "vkAllocateCommandBuffers failed";
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    (void)vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy region{};
    region.srcOffset = 0u;
    region.dstOffset = 0u;
    region.size      = bytes;
    vkCmdCopyBuffer(cmd, stagingBuf, outBuffer, 1u, &region);

    (void)vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1u;
    submit.pCommandBuffers    = &cmd;
    (void)vkQueueSubmit(device.graphicsQueue(), 1u, &submit, VK_NULL_HANDLE);
    (void)vkQueueWaitIdle(device.graphicsQueue());

    vkDestroyCommandPool(device.logical(), pool, nullptr);
    vmaDestroyBuffer(allocator.handle(), stagingBuf, stagingAlloc);
    return true;
}

// Extract level-`dagLodLevel` triangles from an open reader into the
// caller-provided vertex/index vectors (positions only, u32 indices).
// `outDetail` carries a diagnostic when the return is false.
//
// The baker emits up to 128 vertices per cluster; we concat them into a
// single per-asset vertex buffer and produce u32 indices adjusted by the
// running base-vertex offset.
bool gatherLevelTriangles(asset::MpAssetReader& reader,
                          u32                   dagLodLevel,
                          std::vector<f32>&     outPositions,  // xyz * N
                          std::vector<u32>&     outIndices,
                          std::string&          outDetail) {
    outPositions.clear();
    outIndices.clear();

    if (!reader.isOpen()) {
        outDetail = "reader not open";
        return false;
    }

    // Collect the unique pageIds carrying level-`dagLodLevel` clusters.
    // A page may carry multiple clusters across different LOD levels; we
    // still only need to decompress each page once.
    std::unordered_set<u32> pagesNeeded;
    pagesNeeded.reserve(32u);
    const auto dagNodes = reader.dagNodes();
    // The DAG node itself does not carry lodLevel — that lives on the
    // ClusterOnDisk record per page. So we can't short-circuit on the
    // DAG; we must decompress every page referenced by a DAG node and
    // filter by ClusterOnDisk::dagLodLevel. In practice the baker writes
    // one group per page so the set of touched pages equals the set of
    // groups in the DAG.
    for (const auto& node : dagNodes) {
        pagesNeeded.insert(node.pageId);
    }

    std::vector<u8> decomp;
    u32 clustersAtLevel = 0u;

    for (u32 pageId : pagesNeeded) {
        auto pv = reader.fetchPage(pageId, decomp);
        if (!pv.has_value()) {
            outDetail = std::string{"fetchPage("} + std::to_string(pageId)
                      + ") -> " + asset::mpReadErrorKindString(pv.error().kind)
                      + ": " + pv.error().detail;
            return false;
        }

        // For each cluster at the requested LOD, append its triangles.
        // Per-vertex stride is kMpVertexStride (32 B: pos3 + norm3 + uv2);
        // we only copy the first 12 B (pos3).
        const auto& view = *pv;
        const u8* vertexBase = reinterpret_cast<const u8*>(view.vertexBlob.data());
        const u8* triBase    = reinterpret_cast<const u8*>(view.triangleBlob.data());

        for (u32 c = 0u; c < view.clusterCount; ++c) {
            const auto& cluster = view.clusters[c];
            if (cluster.dagLodLevel != dagLodLevel) continue;

            // MpAssetReader::fetchPage has already validated that the
            // cluster's vertex/triangle slices fit inside the page's
            // blobs, so we can index directly without re-checking.
            const u32 baseVertex = static_cast<u32>(outPositions.size() / 3u);
            outPositions.reserve(outPositions.size()
                                 + static_cast<std::size_t>(cluster.vertexCount) * 3u);

            const u8* clusterVerts = vertexBase + cluster.vertexOffset;
            for (u32 v = 0u; v < cluster.vertexCount; ++v) {
                f32 pos[3];
                std::memcpy(&pos[0], clusterVerts + v * asset::kMpVertexStride,
                            sizeof(pos));
                outPositions.push_back(pos[0]);
                outPositions.push_back(pos[1]);
                outPositions.push_back(pos[2]);
            }

            outIndices.reserve(outIndices.size()
                               + static_cast<std::size_t>(cluster.triangleCount) * 3u);
            const u8* clusterTris = triBase + cluster.triangleOffset;
            for (u32 t = 0u; t < cluster.triangleCount; ++t) {
                const u8 i0 = clusterTris[t * 3u + 0u];
                const u8 i1 = clusterTris[t * 3u + 1u];
                const u8 i2 = clusterTris[t * 3u + 2u];
                // Validate local indices (u8) against per-cluster vertex
                // count. A crafted .mpa could otherwise point the BLAS at
                // arbitrary bytes past the cluster's vertex range.
                if (i0 >= cluster.vertexCount
                 || i1 >= cluster.vertexCount
                 || i2 >= cluster.vertexCount) {
                    outDetail = "triangle local index out of range in cluster";
                    return false;
                }
                outIndices.push_back(baseVertex + static_cast<u32>(i0));
                outIndices.push_back(baseVertex + static_cast<u32>(i1));
                outIndices.push_back(baseVertex + static_cast<u32>(i2));
            }
            ++clustersAtLevel;
        }
    }

    if (clustersAtLevel == 0u) {
        outDetail = std::string{"no clusters found at dagLodLevel "}
                  + std::to_string(dagLodLevel);
        return false;
    }
    return true;
}

} // namespace

// --- ctor / dtor ------------------------------------------------------------

MicropolyBlasManager::MicropolyBlasManager(gfx::Device&    device,
                                           gfx::Allocator& allocator,
                                           bool            rtCapable)
    : m_device(&device), m_allocator(&allocator), m_rtCapable(rtCapable) {}

MicropolyBlasManager::~MicropolyBlasManager() {
    destroy_();
}

void MicropolyBlasManager::destroy_() noexcept {
    if (m_device == nullptr || m_allocator == nullptr) return;

    VkDevice dev = m_device->logical();
    for (auto& built : m_built) {
        if (built.as != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(dev, built.as, nullptr);
            built.as = VK_NULL_HANDLE;
        }
        if (built.asBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), built.asBuffer, built.asAllocation);
            built.asBuffer     = VK_NULL_HANDLE;
            built.asAllocation = nullptr;
        }
        if (built.vertexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), built.vertexBuffer, built.vertexAlloc);
            built.vertexBuffer = VK_NULL_HANDLE;
            built.vertexAlloc  = nullptr;
        }
        if (built.indexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), built.indexBuffer, built.indexAlloc);
            built.indexBuffer = VK_NULL_HANDLE;
            built.indexAlloc  = nullptr;
        }
    }
    m_built.clear();
    m_instances.clear();
    m_alreadyBuilt.clear();
}

// --- factory ----------------------------------------------------------------

std::expected<std::unique_ptr<MicropolyBlasManager>, MicropolyBlasError>
MicropolyBlasManager::create(gfx::Device& device, gfx::Allocator& allocator) {
    const bool rt = device.supportsRayTracing();
    // Always return OK — a non-RT device gets a stub whose instances()
    // is empty. Plan constraint: "non-RT devices see zero BLAS work,
    // zero memory allocation" beyond the stub object itself.
    auto mgr = std::unique_ptr<MicropolyBlasManager>(
        new MicropolyBlasManager(device, allocator, rt));
    if (!rt) {
        ENIGMA_LOG_INFO("[micropoly_blas] stub manager created on non-RT device");
    } else {
        ENIGMA_LOG_INFO("[micropoly_blas] manager created (RT capable)");
    }
    return mgr;
}

// --- build ------------------------------------------------------------------

std::expected<void, MicropolyBlasError>
MicropolyBlasManager::buildForAsset(asset::MpAssetReader& reader,
                                    u32                   dagLodLevel) {
    if (!m_rtCapable) {
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::NotSupported,
            "Device::supportsRayTracing() == false; stub manager"});
    }
    if (!reader.isOpen()) {
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::ReaderNotOpen,
            "MpAssetReader is not open"});
    }
    // Cache — plan: "rerunning for the same reader is a no-op (cached)."
    for (const auto* r : m_alreadyBuilt) {
        if (r == &reader) {
            return {};
        }
    }

    // ---- 1. CPU gather -----------------------------------------------------
    std::vector<f32> positions;  // xyz * N
    std::vector<u32> indices;
    std::string      detail;
    if (!gatherLevelTriangles(reader, dagLodLevel, positions, indices, detail)) {
        // Distinguish "no clusters at this LOD" from true decompress
        // failures — the former is a soft skip, the latter is a hard
        // error the caller should surface.
        const MicropolyBlasErrorKind kind =
            (detail.rfind("no clusters", 0) == 0)
                ? MicropolyBlasErrorKind::NoLevel3Clusters
                : MicropolyBlasErrorKind::PageDecompressFailed;
        return std::unexpected(MicropolyBlasError{kind, std::move(detail)});
    }
    const u32 vertexCount = static_cast<u32>(positions.size() / 3u);
    const u32 indexCount  = static_cast<u32>(indices.size());
    ENIGMA_ASSERT(vertexCount > 0u && indexCount > 0u && indexCount % 3u == 0u);

    // Aggregate triangle cap — defend against a crafted .mpa claiming an
    // unreasonable tri count across all level-N clusters. See
    // kMaxBlasTriangles definition at TU top.
    if ((indexCount / 3u) > kMaxBlasTriangles) {
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "aggregate triangle count exceeds BLAS cap"});
    }

    // ---- 2. Upload vertex + index buffers ---------------------------------
    BuiltBlas built{};
    const VkBufferUsageFlags blasInputUsage =
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
      | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    const VkDeviceSize vBytes = static_cast<VkDeviceSize>(positions.size())
                              * sizeof(f32);
    const VkDeviceSize iBytes = static_cast<VkDeviceSize>(indices.size())
                              * sizeof(u32);

    std::string upDetail;
    if (!uploadDeviceLocalBuffer(*m_device, *m_allocator,
                                 positions.data(), vBytes, blasInputUsage,
                                 built.vertexBuffer, built.vertexAlloc,
                                 upDetail)) {
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            std::string{"vertex upload: "} + upDetail});
    }
    if (!uploadDeviceLocalBuffer(*m_device, *m_allocator,
                                 indices.data(), iBytes, blasInputUsage,
                                 built.indexBuffer, built.indexAlloc,
                                 upDetail)) {
        vmaDestroyBuffer(m_allocator->handle(),
                         built.vertexBuffer, built.vertexAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            std::string{"index upload: "} + upDetail});
    }

    // ---- 3. BLAS build ----------------------------------------------------
    VkDevice dev = m_device->logical();

    VkBufferDeviceAddressInfo vAddrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    vAddrInfo.buffer = built.vertexBuffer;
    const VkDeviceAddress vAddr = vkGetBufferDeviceAddress(dev, &vAddrInfo);

    VkBufferDeviceAddressInfo iAddrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    iAddrInfo.buffer = built.indexBuffer;
    const VkDeviceAddress iAddr = vkGetBufferDeviceAddress(dev, &iAddrInfo);

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vAddr;
    triangles.vertexStride  = kShadowVertexStride;
    triangles.maxVertex     = vertexCount - 1u;
    triangles.indexType     = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = iAddr;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;  // plan v1: opaque only.
    geometry.geometry.triangles = triangles;

    const u32 primitiveCount = indexCount / 3u;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1u;
    buildInfo.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    // AS storage buffer.
    VkBufferCreateInfo asBufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    asBufCI.size  = sizeInfo.accelerationStructureSize;
    asBufCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VmaAllocationCreateInfo asAllocCI{};
    asAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(m_allocator->handle(), &asBufCI, &asAllocCI,
                        &built.asBuffer, &built.asAllocation, nullptr) != VK_SUCCESS) {
        vmaDestroyBuffer(m_allocator->handle(), built.vertexBuffer, built.vertexAlloc);
        vmaDestroyBuffer(m_allocator->handle(), built.indexBuffer,  built.indexAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vmaCreateBuffer (AS storage) failed"});
    }

    VkAccelerationStructureCreateInfoKHR asCreateInfo{};
    asCreateInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = built.asBuffer;
    asCreateInfo.size   = sizeInfo.accelerationStructureSize;
    asCreateInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR(dev, &asCreateInfo, nullptr,
                                         &built.as) != VK_SUCCESS) {
        vmaDestroyBuffer(m_allocator->handle(), built.asBuffer, built.asAllocation);
        vmaDestroyBuffer(m_allocator->handle(), built.vertexBuffer, built.vertexAlloc);
        vmaDestroyBuffer(m_allocator->handle(), built.indexBuffer,  built.indexAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vkCreateAccelerationStructureKHR failed"});
    }

    // Scratch buffer. Mirror gfx::BLAS::build's 256 B alignment cushion.
    constexpr VkDeviceSize kScratchAlign = 256u;
    VkBufferCreateInfo scratchCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    scratchCI.size  = sizeInfo.buildScratchSize + kScratchAlign - 1u;
    scratchCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VmaAllocationCreateInfo scratchAI{};
    scratchAI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkBuffer      scratchBuf   = VK_NULL_HANDLE;
    VmaAllocation scratchAlloc = nullptr;
    if (vmaCreateBuffer(m_allocator->handle(), &scratchCI, &scratchAI,
                        &scratchBuf, &scratchAlloc, nullptr) != VK_SUCCESS) {
        vkDestroyAccelerationStructureKHR(dev, built.as, nullptr);
        vmaDestroyBuffer(m_allocator->handle(), built.asBuffer, built.asAllocation);
        vmaDestroyBuffer(m_allocator->handle(), built.vertexBuffer, built.vertexAlloc);
        vmaDestroyBuffer(m_allocator->handle(), built.indexBuffer,  built.indexAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vmaCreateBuffer (scratch) failed"});
    }

    VkBufferDeviceAddressInfo scratchAddrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    scratchAddrInfo.buffer = scratchBuf;
    const VkDeviceAddress rawScratchAddr =
        vkGetBufferDeviceAddress(dev, &scratchAddrInfo);
    const VkDeviceAddress scratchAddr =
        (rawScratchAddr + (kScratchAlign - 1u)) & ~(VkDeviceAddress)(kScratchAlign - 1u);

    buildInfo.dstAccelerationStructure  = built.as;
    buildInfo.scratchData.deviceAddress = scratchAddr;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    // Immediate single-use command buffer for the build + barrier.
    VkCommandPoolCreateInfo poolCI{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = m_device->graphicsQueueFamily();

    // Cleanup helper — tears down any partial state before returning a
    // BlasBuildFailed error. Ordering note: AS handle MUST be destroyed
    // before its backing buffer; scratch is independent; vertex/index
    // buffers are independent. Built.as may still be VK_NULL_HANDLE-free
    // at this point so the guard is safe.
    auto teardownOnBuildFailure = [&](VkCommandPool p, VkCommandBuffer c,
                                      VkBuffer sb, VmaAllocation sa) {
        if (c != VK_NULL_HANDLE && p != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(dev, p, 1u, &c);
        }
        if (p != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev, p, nullptr);
        }
        if (sb != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), sb, sa);
        }
        // AS before AS-buffer.
        if (built.as != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(dev, built.as, nullptr);
        }
        if (built.asBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), built.asBuffer, built.asAllocation);
        }
        if (built.vertexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), built.vertexBuffer, built.vertexAlloc);
        }
        if (built.indexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), built.indexBuffer, built.indexAlloc);
        }
    };

    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(dev, &poolCI, nullptr, &pool) != VK_SUCCESS) {
        teardownOnBuildFailure(VK_NULL_HANDLE, VK_NULL_HANDLE, scratchBuf, scratchAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vkCreateCommandPool failed"});
    }

    VkCommandBufferAllocateInfo cmdAI{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cmdAI.commandPool        = pool;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1u;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(dev, &cmdAI, &cmd) != VK_SUCCESS) {
        teardownOnBuildFailure(pool, VK_NULL_HANDLE, scratchBuf, scratchAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vkAllocateCommandBuffers failed"});
    }

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        teardownOnBuildFailure(pool, cmd, scratchBuf, scratchAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vkBeginCommandBuffer failed"});
    }

    vkCmdBuildAccelerationStructuresKHR(cmd, 1u, &buildInfo, &pRangeInfo);

    // Barrier: AS build writes → RT shader reads (next TLAS build step).
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                          | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1u;
    dep.pMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        teardownOnBuildFailure(pool, cmd, scratchBuf, scratchAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vkEndCommandBuffer failed"});
    }

    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1u;
    submit.pCommandBuffers    = &cmd;
    if (vkQueueSubmit(m_device->graphicsQueue(), 1u, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
        teardownOnBuildFailure(pool, cmd, scratchBuf, scratchAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vkQueueSubmit failed"});
    }
    if (vkQueueWaitIdle(m_device->graphicsQueue()) != VK_SUCCESS) {
        teardownOnBuildFailure(pool, cmd, scratchBuf, scratchAlloc);
        return std::unexpected(MicropolyBlasError{
            MicropolyBlasErrorKind::BlasBuildFailed,
            "vkQueueWaitIdle failed"});
    }

    vkDestroyCommandPool(dev, pool, nullptr);
    vmaDestroyBuffer(m_allocator->handle(), scratchBuf, scratchAlloc);

    // Query AS device address.
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = built.as;
    const VkDeviceAddress blasAddr = vkGetAccelerationStructureDeviceAddressKHR(dev, &addrInfo);

    // ---- 4. Record instance entry ----------------------------------------
    const u32 customIndex = static_cast<u32>(m_instances.size());
    MicropolyBlasInstanceEntry entry{};
    entry.blas        = built.as;
    entry.blasAddress = blasAddr;
    entry.customIndex = customIndex;
    entry.mask        = 0xFFu;
    entry.transform   = mat4(1.0f);  // M5a: identity; per-instance is M5-later.

    m_built.push_back(built);
    m_instances.push_back(entry);
    m_alreadyBuilt.push_back(&reader);

    ENIGMA_LOG_INFO(
        "[micropoly_blas] built per-asset-proxy BLAS "
        "(lod={}, vertices={}, triangles={}, size={} B)",
        dagLodLevel, vertexCount, primitiveCount,
        sizeInfo.accelerationStructureSize);
    return {};
}

} // namespace enigma::renderer::micropoly
