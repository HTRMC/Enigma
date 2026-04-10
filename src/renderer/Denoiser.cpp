#include "renderer/Denoiser.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

namespace enigma {

// Spatial denoise push constants — must match denoise_spatial.hlsl PushBlock.
struct DenoiseSpatialPushBlock {
    u32 inputSlot;
    u32 outputSlot;
    u32 normalSlot;   // reserved for edge-aware weighting (future)
    u32 depthSlot;    // reserved for edge-aware weighting (future)
    u32 screenWidth;
    u32 screenHeight;
    u32 stepWidth;    // A-trous step width (1, 2, 4, 8, 16)
    u32 _pad0;
};
static_assert(sizeof(DenoiseSpatialPushBlock) == 32);

// Temporal denoise push constants — must match denoise_temporal.hlsl PushBlock.
struct DenoiseTemporalPushBlock {
    u32 inputSlot;
    u32 historySlot;
    u32 motionVecSlot;
    u32 outputSlot;
    u32 screenWidth;
    u32 screenHeight;
    u32 _pad0;
    u32 _pad1;
};
static_assert(sizeof(DenoiseTemporalPushBlock) == 32);

Denoiser::Denoiser(gfx::Device& device, gfx::Allocator& allocator)
    : m_device(&device)
    , m_allocator(&allocator) {}

Denoiser::~Denoiser() {
    destroyOutput();
    delete m_spatialPipeline;
    delete m_temporalPipeline;
}

void Denoiser::buildPipelines(gfx::ShaderManager& shaderManager,
                               VkDescriptorSetLayout globalSetLayout,
                               VkFormat effectFormat) {
    m_shaderManager  = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_format          = effectFormat;

    // Spatial A-trous wavelet filter (compute).
    m_spatialPath = Paths::shaderSourceDir() / "denoise_spatial.hlsl";
    VkShaderModule spatialCS = shaderManager.compile(m_spatialPath, gfx::ShaderManager::Stage::Compute, "CSMain");

    gfx::Pipeline::CreateInfo spatialCI{};
    spatialCI.globalSetLayout  = globalSetLayout;
    spatialCI.pushConstantSize = sizeof(DenoiseSpatialPushBlock);
    spatialCI.computeShader    = spatialCS;
    spatialCI.computeEntryPoint = "CSMain";

    m_spatialPipeline = new gfx::Pipeline(*m_device, spatialCI);
    vkDestroyShaderModule(m_device->logical(), spatialCS, nullptr);

    // Temporal accumulation with neighborhood clamping (compute).
    m_temporalPath = Paths::shaderSourceDir() / "denoise_temporal.hlsl";
    VkShaderModule temporalCS = shaderManager.compile(m_temporalPath, gfx::ShaderManager::Stage::Compute, "CSMain");

    gfx::Pipeline::CreateInfo temporalCI{};
    temporalCI.globalSetLayout  = globalSetLayout;
    temporalCI.pushConstantSize = sizeof(DenoiseTemporalPushBlock);
    temporalCI.computeShader    = temporalCS;
    temporalCI.computeEntryPoint = "CSMain";

    m_temporalPipeline = new gfx::Pipeline(*m_device, temporalCI);
    vkDestroyShaderModule(m_device->logical(), temporalCS, nullptr);

    ENIGMA_LOG_INFO("[denoiser] compute pipelines built (format={})", static_cast<u32>(effectFormat));
}

void Denoiser::allocate(VkExtent2D extent, VkFormat effectFormat) {
    if (m_output.image != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroyOutput();
    }
    m_format = effectFormat;

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = effectFormat;
    imgCI.extent        = {extent.width, extent.height, 1};
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imgCI, &allocCI,
                                   &m_output.image, &m_output.allocation, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = m_output.image;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = effectFormat;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &m_output.view));
}

void Denoiser::record(VkCommandBuffer cmd,
                       VkDescriptorSet globalSet,
                       VkExtent2D extent,
                       u32 inputSlot,
                       u32 motionVecSlot,
                       u32 historySlot,
                       u32 outputSlotVal) {
    // Transition output to GENERAL for compute writes.
    VkImageMemoryBarrier2 toGeneral{};
    toGeneral.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toGeneral.srcStageMask        = VK_PIPELINE_STAGE_2_NONE;
    toGeneral.srcAccessMask       = VK_ACCESS_2_NONE;
    toGeneral.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toGeneral.dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    toGeneral.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.image               = m_output.image;
    toGeneral.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{};
    depInfo.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount  = 1;
    depInfo.pImageMemoryBarriers     = &toGeneral;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    const u32 groupCountX = (extent.width  + 7) / 8;
    const u32 groupCountY = (extent.height + 7) / 8;

    // --- Spatial pass (A-trous wavelet, 5 iterations) ---
    // For simplicity, the spatial pass reads from inputSlot and writes
    // to outputSlot. A production implementation would ping-pong between
    // two intermediate images across 5 iterations. Phase 3 will add the
    // ping-pong buffers; for now a single pass approximation is used.
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_spatialPipeline->handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_spatialPipeline->layout(), 0, 1, &globalSet, 0, nullptr);

        DenoiseSpatialPushBlock pc{};
        pc.inputSlot    = inputSlot;
        pc.outputSlot   = outputSlotVal;
        pc.screenWidth  = extent.width;
        pc.screenHeight = extent.height;
        pc.stepWidth    = 1; // single iteration for now

        vkCmdPushConstants(cmd, m_spatialPipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
    }

    // Barrier between spatial and temporal.
    VkMemoryBarrier2 computeBarrier{};
    computeBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    computeBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo barrierInfo{};
    barrierInfo.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    barrierInfo.memoryBarrierCount      = 1;
    barrierInfo.pMemoryBarriers         = &computeBarrier;
    vkCmdPipelineBarrier2(cmd, &barrierInfo);

    // --- Temporal pass ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline->handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_temporalPipeline->layout(), 0, 1, &globalSet, 0, nullptr);

        DenoiseTemporalPushBlock pc{};
        pc.inputSlot     = outputSlotVal; // spatially-filtered result
        pc.historySlot   = historySlot;
        pc.motionVecSlot = motionVecSlot;
        pc.outputSlot    = outputSlotVal;
        pc.screenWidth   = extent.width;
        pc.screenHeight  = extent.height;

        vkCmdPushConstants(cmd, m_temporalPipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
    }

    // Transition to SHADER_READ_ONLY for downstream passes.
    VkImageMemoryBarrier2 toRead{};
    toRead.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toRead.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toRead.srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    toRead.dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                               | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toRead.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT;
    toRead.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.image               = m_output.image;
    toRead.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo2{};
    depInfo2.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo2.imageMemoryBarrierCount  = 1;
    depInfo2.pImageMemoryBarriers     = &toRead;
    vkCmdPipelineBarrier2(cmd, &depInfo2);
}

void Denoiser::registerHotReload(gfx::ShaderHotReload& reloader) {
    reloader.watchGroup({m_spatialPath, m_temporalPath}, [this]() {
        ENIGMA_LOG_INFO("[denoiser] hot-reload triggered");
    });
}

void Denoiser::destroyOutput() {
    if (m_output.view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), m_output.view, nullptr);
        m_output.view = VK_NULL_HANDLE;
    }
    if (m_output.image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_output.image, m_output.allocation);
        m_output.image      = VK_NULL_HANDLE;
        m_output.allocation = nullptr;
    }
}

} // namespace enigma
