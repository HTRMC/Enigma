#include "renderer/HiZPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cmath>

namespace enigma {

// Push block matching hiz_build.comp.hlsl PushBlock.
struct HiZPushBlock {
    u32 srcSlot;
    u32 dstSlot;
    u32 srcWidth;
    u32 srcHeight;
};

static_assert(sizeof(HiZPushBlock) == 16);

HiZPass::HiZPass(gfx::Device& device, gfx::Allocator& allocator,
                 gfx::DescriptorAllocator& descriptors)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptors(&descriptors)
{}

HiZPass::~HiZPass() {
    delete m_pipeline;
    destroyImage();
}

void HiZPass::destroyImage() {
    for (u32 slot : m_mip_slots) {
        m_descriptors->releaseStorageImage(slot);
    }
    m_mip_slots.clear();

    for (VkImageView view : m_mip_views) {
        vkDestroyImageView(m_device->logical(), view, nullptr);
    }
    m_mip_views.clear();

    if (m_image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_image, m_alloc);
        m_image = VK_NULL_HANDLE;
        m_alloc = nullptr;
    }
}

void HiZPass::allocate(VkExtent2D extent) {
    if (m_image != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroyImage();
    }
    m_extent = extent;

    // Compute mip count: log2(max(w,h)) + 1.
    const u32 maxDim  = std::max(extent.width, extent.height);
    const u32 mipCount = static_cast<u32>(std::floor(std::log2(static_cast<float>(maxDim)))) + 1u;

    // R32_SFLOAT: one float per pixel, storage + sampled for HiZ reads.
    VkImageCreateInfo imageCI{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageCI.imageType   = VK_IMAGE_TYPE_2D;
    imageCI.format      = VK_FORMAT_R32_SFLOAT;
    imageCI.extent      = { extent.width, extent.height, 1 };
    imageCI.mipLevels   = mipCount;
    imageCI.arrayLayers = 1;
    imageCI.samples     = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                        | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imageCI, &allocCI,
                                   &m_image, &m_alloc, nullptr));

    // Create one VkImageView per mip level; register each as a storage image.
    m_mip_views.resize(mipCount);
    m_mip_slots.resize(mipCount);

    for (u32 mip = 0; mip < mipCount; ++mip) {
        VkImageViewCreateInfo viewCI{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewCI.image                           = m_image;
        viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format                          = VK_FORMAT_R32_SFLOAT;
        viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.baseMipLevel   = mip;
        viewCI.subresourceRange.levelCount     = 1;
        viewCI.subresourceRange.baseArrayLayer = 0;
        viewCI.subresourceRange.layerCount     = 1;

        ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr,
                                          &m_mip_views[mip]));

        m_mip_slots[mip] = m_descriptors->registerStorageImage(m_mip_views[mip]);
    }

    ENIGMA_LOG_INFO("[hiz] allocated {}x{} ({} mips)", extent.width, extent.height, mipCount);
}

void HiZPass::buildPipeline(gfx::ShaderManager& shaderManager,
                             VkDescriptorSetLayout globalSetLayout) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "HiZPass::buildPipeline called twice");

    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_shaderPath      = Paths::shaderSourceDir() / "hiz_build.comp.hlsl";

    VkShaderModule cs = shaderManager.compile(m_shaderPath,
                                              gfx::ShaderManager::Stage::Compute, "CSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.computeShader    = cs;
    ci.computeEntryPoint = "CSMain";
    ci.globalSetLayout  = globalSetLayout;
    ci.pushConstantSize = sizeof(HiZPushBlock);

    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    ENIGMA_LOG_INFO("[hiz] pipeline built");
}

void HiZPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    VkShaderModule cs = m_shaderManager->tryCompile(m_shaderPath,
                                                    gfx::ShaderManager::Stage::Compute, "CSMain");
    if (cs == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[hiz] hot-reload: CS compile failed"); return; }

    vkDeviceWaitIdle(m_device->logical());
    delete m_pipeline;

    gfx::Pipeline::CreateInfo ci{};
    ci.computeShader    = cs;
    ci.computeEntryPoint = "CSMain";
    ci.globalSetLayout  = m_globalSetLayout;
    ci.pushConstantSize = sizeof(HiZPushBlock);
    m_pipeline = new gfx::Pipeline(*m_device, ci);
    vkDestroyShaderModule(m_device->logical(), cs, nullptr);

    ENIGMA_LOG_INFO("[hiz] hot-reload: pipeline rebuilt");
}

void HiZPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

u32 HiZPass::mip_slot(u32 level) const {
    ENIGMA_ASSERT(level < m_mip_slots.size());
    return m_mip_slots[level];
}

void HiZPass::record(VkCommandBuffer cmd, VkDescriptorSet globalSet) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "HiZPass::record before buildPipeline");
    ENIGMA_ASSERT(m_image != VK_NULL_HANDLE && "HiZPass::record before allocate");

    const u32 mipCount = static_cast<u32>(m_mip_slots.size());
    if (mipCount < 2) return; // nothing to downsample

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    // Transition all mips to GENERAL so they can be used as storage images.
    VkImageMemoryBarrier2 initBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    initBarrier.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    initBarrier.srcAccessMask       = VK_ACCESS_2_NONE;
    initBarrier.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    initBarrier.dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                    | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    initBarrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    initBarrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    initBarrier.image               = m_image;
    initBarrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipCount, 0, 1 };

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &initBarrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    // Downsample mip N → mip N+1.
    u32 srcW = m_extent.width;
    u32 srcH = m_extent.height;

    for (u32 mip = 1; mip < mipCount; ++mip) {
        // Barrier: ensure previous write to (mip-1) is visible as read for mip.
        VkImageMemoryBarrier2 mipBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        mipBarrier.srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mipBarrier.srcAccessMask    = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mipBarrier.dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mipBarrier.dstAccessMask    = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        mipBarrier.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
        mipBarrier.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
        mipBarrier.image            = m_image;
        mipBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 1, 0, 1 };

        VkDependencyInfo mipDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        mipDep.imageMemoryBarrierCount = 1;
        mipDep.pImageMemoryBarriers    = &mipBarrier;
        vkCmdPipelineBarrier2(cmd, &mipDep);

        HiZPushBlock pc{};
        pc.srcSlot   = m_mip_slots[mip - 1];
        pc.dstSlot   = m_mip_slots[mip];
        pc.srcWidth  = srcW;
        pc.srcHeight = srcH;

        vkCmdPushConstants(cmd, m_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        const u32 dstW = std::max(1u, srcW / 2);
        const u32 dstH = std::max(1u, srcH / 2);
        vkCmdDispatch(cmd, (dstW + 7) / 8, (dstH + 7) / 8, 1);

        srcW = dstW;
        srcH = dstH;
    }
}

} // namespace enigma
