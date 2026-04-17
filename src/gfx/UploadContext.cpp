#include "gfx/UploadContext.h"

#include "core/Assert.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <cstring>

namespace enigma::gfx {

UploadContext::UploadContext(Device& device, Allocator& allocator)
    : m_device(&device), m_allocator(&allocator) {

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = device.graphicsQueueFamily();
    ENIGMA_VK_CHECK(vkCreateCommandPool(device.logical(), &poolInfo, nullptr, &m_commandPool));

    VkCommandBufferAllocateInfo cbInfo{};
    cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool        = m_commandPool;
    cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    ENIGMA_VK_CHECK(vkAllocateCommandBuffers(device.logical(), &cbInfo, &m_commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(m_commandBuffer, &beginInfo));

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ENIGMA_VK_CHECK(vkCreateFence(device.logical(), &fenceInfo, nullptr, &m_fence));
}

UploadContext::~UploadContext() {
    VkDevice dev = m_device->logical();
    if (!m_submitted) {
        // Discard without submitting — still need to clean up.
        vkEndCommandBuffer(m_commandBuffer);
    }
    for (auto& s : m_stagingBuffers) {
        vmaDestroyBuffer(m_allocator->handle(), s.buffer, s.allocation);
    }
    vkDestroyFence(dev, m_fence, nullptr);
    vkDestroyCommandPool(dev, m_commandPool, nullptr);
}

void UploadContext::uploadBuffer(VkBuffer dst, const void* data, VkDeviceSize size) {
    ENIGMA_ASSERT(!m_submitted);

    // Create staging buffer.
    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size        = size;
    stagingInfo.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    StagingEntry entry{};
    VmaAllocationInfo allocResult{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stagingInfo, &allocInfo,
                                    &entry.buffer, &entry.allocation, &allocResult));
    ENIGMA_ASSERT(allocResult.pMappedData != nullptr);
    std::memcpy(allocResult.pMappedData, data, static_cast<usize>(size));
    m_stagingBuffers.push_back(entry);

    // Record copy.
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(m_commandBuffer, entry.buffer, dst, 1, &region);
}

void UploadContext::uploadImage(VkImage dst, VkExtent3D extent, VkFormat /*format*/,
                                const void* pixels, VkDeviceSize size) {
    ENIGMA_ASSERT(!m_submitted);

    // Create staging buffer.
    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size        = size;
    stagingInfo.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    StagingEntry entry{};
    VmaAllocationInfo allocResult{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stagingInfo, &allocInfo,
                                    &entry.buffer, &entry.allocation, &allocResult));
    ENIGMA_ASSERT(allocResult.pMappedData != nullptr);
    std::memcpy(allocResult.pMappedData, pixels, static_cast<usize>(size));
    m_stagingBuffers.push_back(entry);

    // UNDEFINED → TRANSFER_DST_OPTIMAL
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask       = 0;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = dst;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(m_commandBuffer, &dep);
    }

    // Buffer → Image copy.
    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.imageExtent      = extent;
    vkCmdCopyBufferToImage(m_commandBuffer, entry.buffer, dst,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = dst;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(m_commandBuffer, &dep);
    }
}

void UploadContext::uploadImageWithMipchain(VkImage dst, u32 width, u32 height,
                                             VkFormat /*format*/, u32 mipLevels,
                                             const void* basePixels, VkDeviceSize baseSize)
{
    ENIGMA_ASSERT(!m_submitted);
    ENIGMA_ASSERT(mipLevels >= 1);

    // --- Stage the base mip level ---
    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size        = baseSize;
    stagingInfo.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    StagingEntry entry{};
    VmaAllocationInfo allocResult{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stagingInfo, &allocInfo,
                                    &entry.buffer, &entry.allocation, &allocResult));
    ENIGMA_ASSERT(allocResult.pMappedData != nullptr);
    std::memcpy(allocResult.pMappedData, basePixels, static_cast<usize>(baseSize));
    m_stagingBuffers.push_back(entry);

    auto makeBarrier = [&](u32 baseMip, u32 levelCount,
                            VkImageLayout oldLayout, VkImageLayout newLayout,
                            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = srcStage;
        b.srcAccessMask       = srcAccess;
        b.dstStageMask        = dstStage;
        b.dstAccessMask       = dstAccess;
        b.oldLayout           = oldLayout;
        b.newLayout           = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = dst;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, baseMip, levelCount, 0, 1 };

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(m_commandBuffer, &dep);
    };

    // All mips UNDEFINED → TRANSFER_DST_OPTIMAL (base will be written, others will be blitted into).
    makeBarrier(0, mipLevels,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Copy staging → base mip.
    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.imageExtent      = { width, height, 1 };
    vkCmdCopyBufferToImage(m_commandBuffer, entry.buffer, dst,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // Generate each subsequent mip from the previous one.
    i32 mipW = static_cast<i32>(width);
    i32 mipH = static_cast<i32>(height);
    for (u32 i = 1; i < mipLevels; ++i) {
        // Previous level TRANSFER_DST_OPTIMAL → TRANSFER_SRC_OPTIMAL so we can blit from it.
        makeBarrier(i - 1, 1,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT,  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BLIT_BIT,  VK_ACCESS_2_TRANSFER_READ_BIT);

        const i32 dstW = std::max(1, mipW / 2);
        const i32 dstH = std::max(1, mipH / 2);

        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
        blit.srcOffsets[0]  = { 0, 0, 0 };
        blit.srcOffsets[1]  = { mipW, mipH, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
        blit.dstOffsets[0]  = { 0, 0, 0 };
        blit.dstOffsets[1]  = { dstW, dstH, 1 };

        vkCmdBlitImage(m_commandBuffer,
                       dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);

        mipW = dstW;
        mipH = dstH;
    }

    // Final transitions: levels [0, mipLevels-1) are in TRANSFER_SRC, the last is in TRANSFER_DST.
    if (mipLevels > 1) {
        makeBarrier(0, mipLevels - 1,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    makeBarrier(mipLevels - 1, 1,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void UploadContext::submitAndWait() {
    ENIGMA_ASSERT(!m_submitted);
    m_submitted = true;

    ENIGMA_VK_CHECK(vkEndCommandBuffer(m_commandBuffer));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &m_commandBuffer;
    ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, m_fence));
    ENIGMA_VK_CHECK(vkWaitForFences(m_device->logical(), 1, &m_fence, VK_TRUE, UINT64_MAX));
}

} // namespace enigma::gfx
