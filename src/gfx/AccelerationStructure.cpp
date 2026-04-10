#include "gfx/AccelerationStructure.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <cstring>

namespace enigma::gfx {

// ---------------------------------------------------------------------------
// BLAS
// ---------------------------------------------------------------------------

BLAS BLAS::build(Device& device, Allocator& allocator,
                 VkBuffer vertexBuffer, u32 vertexCount, VkDeviceSize vertexStride,
                 VkBuffer indexBuffer, u32 indexCount,
                 bool allowCompaction) {
    VkDevice dev = device.logical();

    // Get buffer device addresses for vertex and index data.
    VkBufferDeviceAddressInfo vertAddrInfo{};
    vertAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    vertAddrInfo.buffer = vertexBuffer;
    const VkDeviceAddress vertAddress = vkGetBufferDeviceAddress(dev, &vertAddrInfo);

    VkBufferDeviceAddressInfo idxAddrInfo{};
    idxAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    idxAddrInfo.buffer = indexBuffer;
    const VkDeviceAddress idxAddress = vkGetBufferDeviceAddress(dev, &idxAddrInfo);

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertAddress;
    triangles.vertexStride  = vertexStride;
    triangles.maxVertex     = vertexCount - 1;
    triangles.indexType     = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = idxAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;

    const u32 primitiveCount = indexCount / 3;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (allowCompaction) {
        buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    // Create AS buffer.
    VkBufferCreateInfo asBufInfo{};
    asBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    asBufInfo.size  = sizeInfo.accelerationStructureSize;
    asBufInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo asAllocInfo{};
    asAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    BLAS result{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &asBufInfo, &asAllocInfo,
                                    &result.m_buffer, &result.m_allocation, nullptr));

    // Create the acceleration structure object.
    VkAccelerationStructureCreateInfoKHR asCreateInfo{};
    asCreateInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = result.m_buffer;
    asCreateInfo.size   = sizeInfo.accelerationStructureSize;
    asCreateInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    ENIGMA_VK_CHECK(vkCreateAccelerationStructureKHR(dev, &asCreateInfo, nullptr, &result.m_as));

    // Create scratch buffer.
    VkBufferCreateInfo scratchBufInfo{};
    scratchBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    scratchBufInfo.size  = sizeInfo.buildScratchSize;
    scratchBufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                         | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo scratchAllocInfo{};
    scratchAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkBuffer scratchBuf = VK_NULL_HANDLE;
    VmaAllocation scratchAlloc = nullptr;
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &scratchBufInfo, &scratchAllocInfo,
                                    &scratchBuf, &scratchAlloc, nullptr));

    VkBufferDeviceAddressInfo scratchAddrInfo{};
    scratchAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    scratchAddrInfo.buffer = scratchBuf;
    const VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(dev, &scratchAddrInfo);

    // Build on the device using immediate submit on the graphics queue.
    buildInfo.dstAccelerationStructure  = result.m_as;
    buildInfo.scratchData.deviceAddress = scratchAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount  = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex     = 0;
    rangeInfo.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    // Immediate single-use command buffer for BLAS build.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = device.graphicsQueueFamily();

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateCommandPool(dev, &poolCI, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool        = cmdPool;
    cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkAllocateCommandBuffers(dev, &cmdAllocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);

    ENIGMA_VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    ENIGMA_VK_CHECK(vkQueueSubmit(device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    ENIGMA_VK_CHECK(vkQueueWaitIdle(device.graphicsQueue()));

    vkDestroyCommandPool(dev, cmdPool, nullptr);
    vmaDestroyBuffer(allocator.handle(), scratchBuf, scratchAlloc);

    // Query the device address of the built AS.
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = result.m_as;
    result.m_address = vkGetAccelerationStructureDeviceAddressKHR(dev, &addrInfo);

    return result;
}

void BLAS::destroy(Device& device, Allocator& allocator) {
    if (m_as != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(device.logical(), m_as, nullptr);
        m_as = VK_NULL_HANDLE;
    }
    if (m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator.handle(), m_buffer, m_allocation);
        m_buffer     = VK_NULL_HANDLE;
        m_allocation = nullptr;
    }
    m_address = 0;
}

// ---------------------------------------------------------------------------
// TLAS
// ---------------------------------------------------------------------------

TLAS::TLAS(Device& device, Allocator& allocator, u32 maxInstances)
    : m_maxInstances(maxInstances) {
    VkDevice dev = device.logical();

    // Instance staging buffer (host-visible, persistently mapped).
    const VkDeviceSize instanceBufSize =
        static_cast<VkDeviceSize>(maxInstances) * sizeof(VkAccelerationStructureInstanceKHR);

    VkBufferCreateInfo instBufCI{};
    instBufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    instBufCI.size  = instanceBufSize;
    instBufCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo instAllocCI{};
    instAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
    instAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo instAllocResult{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &instBufCI, &instAllocCI,
                                    &m_instanceBuf, &m_instanceAlloc, &instAllocResult));
    m_instances = static_cast<VkAccelerationStructureInstanceKHR*>(instAllocResult.pMappedData);
    ENIGMA_ASSERT(m_instances != nullptr);

    // Zero-initialize instances (all inactive by default).
    std::memset(m_instances, 0, instanceBufSize);

    // Pre-query build sizes for the max capacity TLAS.
    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &maxInstances, &sizeInfo);

    // Create AS buffer.
    VkBufferCreateInfo asBufCI{};
    asBufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    asBufCI.size  = sizeInfo.accelerationStructureSize;
    asBufCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo asAllocCI{};
    asAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &asBufCI, &asAllocCI,
                                    &m_buffer, &m_allocation, nullptr));

    VkAccelerationStructureCreateInfoKHR asCI{};
    asCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = m_buffer;
    asCI.size   = sizeInfo.accelerationStructureSize;
    asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    ENIGMA_VK_CHECK(vkCreateAccelerationStructureKHR(dev, &asCI, nullptr, &m_as));

    // Query AS device address.
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = m_as;
    m_address = vkGetAccelerationStructureDeviceAddressKHR(dev, &addrInfo);

    // Create scratch buffer.
    VkBufferCreateInfo scratchCI{};
    scratchCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    scratchCI.size  = sizeInfo.buildScratchSize;
    scratchCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo scratchAllocCI{};
    scratchAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &scratchCI, &scratchAllocCI,
                                    &m_scratchBuf, &m_scratchAlloc, nullptr));

    ENIGMA_LOG_INFO("[accel] TLAS created (max {} instances, AS={} bytes, scratch={} bytes)",
                    maxInstances, sizeInfo.accelerationStructureSize, sizeInfo.buildScratchSize);
}

void TLAS::destroy(Device& device, Allocator& allocator) {
    if (m_as != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(device.logical(), m_as, nullptr);
        m_as = VK_NULL_HANDLE;
    }
    if (m_scratchBuf != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator.handle(), m_scratchBuf, m_scratchAlloc);
        m_scratchBuf   = VK_NULL_HANDLE;
        m_scratchAlloc = nullptr;
    }
    if (m_instanceBuf != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator.handle(), m_instanceBuf, m_instanceAlloc);
        m_instanceBuf   = VK_NULL_HANDLE;
        m_instanceAlloc = nullptr;
        m_instances     = nullptr;
    }
    if (m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator.handle(), m_buffer, m_allocation);
        m_buffer     = VK_NULL_HANDLE;
        m_allocation = nullptr;
    }
    m_address = 0;
}

u32 TLAS::allocateInstanceSlot() {
    if (!m_freeSlots.empty()) {
        const u32 slot = m_freeSlots.back();
        m_freeSlots.pop_back();
        return slot;
    }
    ENIGMA_ASSERT(m_nextSlot < m_maxInstances);
    return m_nextSlot++;
}

void TLAS::releaseInstanceSlot(u32 slot) {
    // Zero out the released instance so it does not contribute to the TLAS.
    std::memset(&m_instances[slot], 0, sizeof(VkAccelerationStructureInstanceKHR));
    m_freeSlots.push_back(slot);
}

void TLAS::setInstance(u32 slot, const VkAccelerationStructureInstanceKHR& inst) {
    ENIGMA_ASSERT(slot < m_nextSlot);
    m_instances[slot] = inst;
}

void TLAS::build(VkCommandBuffer cmd, Device& device, Allocator& allocator) {
    (void)allocator; // scratch is pre-allocated

    VkDevice dev = device.logical();

    VkBufferDeviceAddressInfo instAddrInfo{};
    instAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    instAddrInfo.buffer = m_instanceBuf;
    const VkDeviceAddress instAddress = vkGetBufferDeviceAddress(dev, &instAddrInfo);

    VkBufferDeviceAddressInfo scratchAddrInfo{};
    scratchAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    scratchAddrInfo.buffer = m_scratchBuf;
    const VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(dev, &scratchAddrInfo);

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers    = VK_FALSE;
    instancesData.data.deviceAddress = instAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type                      = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure  = m_as;
    buildInfo.geometryCount             = 1;
    buildInfo.pGeometries               = &geometry;
    buildInfo.scratchData.deviceAddress = scratchAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = m_nextSlot; // all allocated slots (active + zeroed)
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);

    // Barrier: AS build writes must complete before any RT dispatch reads.
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo depInfo{};
    depInfo.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.memoryBarrierCount      = 1;
    depInfo.pMemoryBarriers         = &barrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

} // namespace enigma::gfx
