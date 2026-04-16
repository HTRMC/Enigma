#include "world/HeightmapLoader.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"

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
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace enigma {

HeightmapLoader::HeightmapLoader(gfx::Device& device,
                                 gfx::Allocator& allocator,
                                 gfx::DescriptorAllocator& descriptors)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptors(&descriptors) {}

HeightmapLoader::~HeightmapLoader() {
    VkDevice dev = m_device->logical();

    // Staging buffer might still be alive if the caller forgot to call
    // releaseStaging — safety-net cleanup.
    if (m_staging != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_staging, m_stagingAlloc);
        m_staging      = VK_NULL_HANDLE;
        m_stagingAlloc = nullptr;
    }

    if (m_samplerSlot != UINT32_MAX) {
        m_descriptors->releaseSampler(m_samplerSlot);
        m_samplerSlot = UINT32_MAX;
    }
    if (m_texSlot != UINT32_MAX) {
        m_descriptors->releaseSampledImage(m_texSlot);
        m_texSlot = UINT32_MAX;
    }
    if (m_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(dev, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
    if (m_imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(dev, m_imageView, nullptr);
        m_imageView = VK_NULL_HANDLE;
    }
    if (m_image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_image, m_imageAlloc);
        m_image      = VK_NULL_HANDLE;
        m_imageAlloc = nullptr;
    }
}

// ---------------------------------------------------------------------------

bool HeightmapLoader::load(const HeightmapDesc& desc, VkCommandBuffer uploadCmd) {
    ENIGMA_ASSERT(m_image == VK_NULL_HANDLE && "HeightmapLoader::load called twice");
    ENIGMA_ASSERT(desc.sampleCount > 1);
    ENIGMA_ASSERT(desc.worldSize > 0.0f);

    m_desc   = desc;
    m_origin = vec3(-desc.worldSize * 0.5f, 0.0f, -desc.worldSize * 0.5f);

    const usize sampleCount = static_cast<usize>(desc.sampleCount) * desc.sampleCount;
    m_heights.assign(sampleCount, 0.0f);

    bool fileOk = false;
    {
        std::ifstream in(desc.path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            ENIGMA_LOG_ERROR("[heightmap] failed to open '{}' — using zeros",
                             desc.path.string());
        } else {
            const std::streamsize fileSize = in.tellg();
            in.seekg(0, std::ios::beg);

            const std::string ext = desc.path.extension().string();
            if (ext == ".r32f") {
                const std::streamsize expected =
                    static_cast<std::streamsize>(sampleCount * sizeof(f32));
                if (fileSize != expected) {
                    ENIGMA_LOG_ERROR(
                        "[heightmap] '{}' size {} != expected {} for {}x{} r32f — using zeros",
                        desc.path.string(),
                        static_cast<i64>(fileSize),
                        static_cast<i64>(expected),
                        desc.sampleCount, desc.sampleCount);
                } else {
                    in.read(reinterpret_cast<char*>(m_heights.data()), expected);
                    fileOk = in.good() || in.eof();
                    if (!fileOk) {
                        ENIGMA_LOG_ERROR("[heightmap] read error on '{}' — using zeros",
                                         desc.path.string());
                        std::fill(m_heights.begin(), m_heights.end(), 0.0f);
                    }
                }
            } else if (ext == ".raw") {
                const std::streamsize expected =
                    static_cast<std::streamsize>(sampleCount * sizeof(u16));
                if (fileSize != expected) {
                    ENIGMA_LOG_ERROR(
                        "[heightmap] '{}' size {} != expected {} for {}x{} r16le — using zeros",
                        desc.path.string(),
                        static_cast<i64>(fileSize),
                        static_cast<i64>(expected),
                        desc.sampleCount, desc.sampleCount);
                } else {
                    std::vector<u16> raw(sampleCount);
                    in.read(reinterpret_cast<char*>(raw.data()), expected);
                    if (!(in.good() || in.eof())) {
                        ENIGMA_LOG_ERROR("[heightmap] read error on '{}' — using zeros",
                                         desc.path.string());
                    } else {
                        const f32 range = desc.maxHeight - desc.minHeight;
                        for (usize i = 0; i < sampleCount; ++i) {
                            const f32 t = static_cast<f32>(raw[i]) / 65535.0f;
                            m_heights[i] = desc.minHeight + t * range;
                        }
                        fileOk = true;
                    }
                }
            } else {
                ENIGMA_LOG_ERROR(
                    "[heightmap] '{}' has unsupported extension '{}' — expected .raw or .r32f, using zeros",
                    desc.path.string(), ext);
            }

            if (fileOk) {
                f32 minH =  std::numeric_limits<f32>::infinity();
                f32 maxH = -std::numeric_limits<f32>::infinity();
                for (f32 h : m_heights) {
                    minH = std::min(minH, h);
                    maxH = std::max(maxH, h);
                }
                ENIGMA_LOG_INFO(
                    "[heightmap] loaded '{}' — {}x{} samples, {} bytes, min/max = {:.2f} / {:.2f} m",
                    desc.path.string(), desc.sampleCount, desc.sampleCount,
                    static_cast<i64>(fileSize), minH, maxH);
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU image (R32_SFLOAT, 1 mip, sampled + transfer dst).
    // -----------------------------------------------------------------------
    VkImageCreateInfo imageCI{};
    imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType     = VK_IMAGE_TYPE_2D;
    imageCI.format        = VK_FORMAT_R32_SFLOAT;
    imageCI.extent        = { desc.sampleCount, desc.sampleCount, 1 };
    imageCI.mipLevels     = 1;
    imageCI.arrayLayers   = 1;
    imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo imgAlloc{};
    imgAlloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    imgAlloc.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imageCI, &imgAlloc,
                                   &m_image, &m_imageAlloc, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image                           = m_image;
    viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format                          = VK_FORMAT_R32_SFLOAT;
    viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = 1;
    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &m_imageView));

    // -----------------------------------------------------------------------
    // Staging buffer — outlives load() (freed via releaseStaging() after fence).
    // -----------------------------------------------------------------------
    const VkDeviceSize stagingSize = sampleCount * sizeof(f32);

    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size        = stagingSize;
    stagingInfo.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stagingAlloc{};
    stagingAlloc.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAlloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                       | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo stagingResult{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stagingInfo, &stagingAlloc,
                                    &m_staging, &m_stagingAlloc, &stagingResult));
    ENIGMA_ASSERT(stagingResult.pMappedData != nullptr);
    std::memcpy(stagingResult.pMappedData, m_heights.data(), stagingSize);

    // -----------------------------------------------------------------------
    // Record layout transition + copy + shader-read transition.
    // -----------------------------------------------------------------------
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
        barrier.image               = m_image;
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(uploadCmd, &dep);
    }

    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.imageExtent      = { desc.sampleCount, desc.sampleCount, 1 };
    vkCmdCopyBufferToImage(uploadCmd, m_staging, m_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT
                                    | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = m_image;
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(uploadCmd, &dep);
    }

    // -----------------------------------------------------------------------
    // Sampler: linear, clamp-to-edge, no mips.
    // -----------------------------------------------------------------------
    VkSamplerCreateInfo samplerCI{};
    samplerCI.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter    = VK_FILTER_LINEAR;
    samplerCI.minFilter    = VK_FILTER_LINEAR;
    samplerCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.minLod       = 0.0f;
    samplerCI.maxLod       = 0.0f;
    samplerCI.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    ENIGMA_VK_CHECK(vkCreateSampler(m_device->logical(), &samplerCI, nullptr, &m_sampler));

    // -----------------------------------------------------------------------
    // Bindless registration.
    // -----------------------------------------------------------------------
    m_texSlot     = m_descriptors->registerSampledImage(m_imageView,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_samplerSlot = m_descriptors->registerSampler(m_sampler);

    ENIGMA_LOG_INFO("[heightmap] GPU texture ready — bindless tex slot {}, sampler slot {}",
                    m_texSlot, m_samplerSlot);

    return fileOk;
}

void HeightmapLoader::releaseStaging() {
    if (m_staging != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_staging, m_stagingAlloc);
        m_staging      = VK_NULL_HANDLE;
        m_stagingAlloc = nullptr;
    }
}

// ---------------------------------------------------------------------------

f32 HeightmapLoader::sampleBilinear(f32 worldX, f32 worldZ) const {
    if (!std::isfinite(worldX) || !std::isfinite(worldZ)) return 0.0f;
    if (m_heights.empty()) return 0.0f;

    f32 u = (worldX - m_origin.x) / m_desc.worldSize;
    f32 v = (worldZ - m_origin.z) / m_desc.worldSize;
    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);

    const f32 fx = u * static_cast<f32>(m_desc.sampleCount - 1);
    const f32 fz = v * static_cast<f32>(m_desc.sampleCount - 1);

    u32 x0 = static_cast<u32>(fx);
    u32 z0 = static_cast<u32>(fz);
    x0 = std::min(x0, m_desc.sampleCount - 2u);
    z0 = std::min(z0, m_desc.sampleCount - 2u);
    const u32 x1 = x0 + 1u;
    const u32 z1 = z0 + 1u;

    const f32 tx = fx - static_cast<f32>(x0);
    const f32 tz = fz - static_cast<f32>(z0);

    const auto idx = [&](u32 x, u32 z) {
        return static_cast<usize>(z) * m_desc.sampleCount + x;
    };
    const f32 h00 = m_heights[idx(x0, z0)];
    const f32 h10 = m_heights[idx(x1, z0)];
    const f32 h01 = m_heights[idx(x0, z1)];
    const f32 h11 = m_heights[idx(x1, z1)];

    return glm::mix(glm::mix(h00, h10, tx), glm::mix(h01, h11, tx), tz);
}

} // namespace enigma
