#include "renderer/RTReflectionPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/RTPipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

namespace enigma {

// RT push constants — must match reflection.rgen.hlsl PushBlock exactly.
struct RTReflectionPushBlock {
    u32  normalSlot;            //  4
    u32  depthSlot;             //  4
    u32  cameraSlot;            //  4
    u32  samplerSlot;           //  4
    u32  tlasSlot;              //  4
    u32  outputSlot;            //  4
    u32  skyViewLutSlot;        //  4
    u32  transmittanceLutSlot;  //  4  = 32 bytes
    vec4 sunWorldDirIntensity;  // 16
    vec4 cameraWorldPosKm;      // 16
};                              // Total: 64 bytes
static_assert(sizeof(RTReflectionPushBlock) == 64);

// SSR push constants — must match ssr.hlsl PushBlock.
struct SSRPushBlock {
    u32 normalSlot;
    u32 depthSlot;
    u32 cameraSlot;
    u32 samplerSlot;
    u32 outputSlot;
    u32 screenWidth;
    u32 screenHeight;
    u32 _pad0;
};
static_assert(sizeof(SSRPushBlock) == 32);

RTReflectionPass::RTReflectionPass(gfx::Device& device, gfx::Allocator& allocator)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_useRT(device.gpuTier() >= gfx::GpuTier::Recommended) {}

RTReflectionPass::~RTReflectionPass() {
    destroyOutput();
    delete m_rtPipeline;
    delete m_ssrPipeline;
}

void RTReflectionPass::buildPipeline(gfx::ShaderManager& shaderManager,
                                      VkDescriptorSetLayout globalSetLayout,
                                      VkFormat /*outputFormat*/) {
    m_shaderManager  = &shaderManager;
    m_globalSetLayout = globalSetLayout;

    if (m_useRT) {
        m_rgenPath  = Paths::shaderSourceDir() / "rt" / "reflection.rgen.hlsl";
        m_rchitPath = Paths::shaderSourceDir() / "rt" / "reflection.rchit.hlsl";
        m_rmissPath = Paths::shaderSourceDir() / "rt" / "reflection.rmiss.hlsl";

        VkShaderModule rgen  = shaderManager.compile(m_rgenPath,  gfx::ShaderManager::Stage::RayGeneration, "RayGenMain");
        VkShaderModule rchit = shaderManager.compile(m_rchitPath, gfx::ShaderManager::Stage::ClosestHit,    "ClosestHitMain");
        VkShaderModule rmiss = shaderManager.compile(m_rmissPath, gfx::ShaderManager::Stage::Miss,          "MissMain");

        gfx::RTPipeline::CreateInfo ci{};
        ci.raygenModule      = rgen;
        ci.raygenEntry       = "RayGenMain";
        ci.missModule        = rmiss;
        ci.missEntry         = "MissMain";
        ci.closestHitModule  = rchit;
        ci.closestHitEntry   = "ClosestHitMain";
        ci.globalSetLayout   = globalSetLayout;
        ci.pushConstantSize  = sizeof(RTReflectionPushBlock); // 64 bytes
        ci.maxRecursionDepth = 1;

        m_rtPipeline = new gfx::RTPipeline(*m_device, *m_allocator, ci);

        vkDestroyShaderModule(m_device->logical(), rgen,  nullptr);
        vkDestroyShaderModule(m_device->logical(), rchit, nullptr);
        vkDestroyShaderModule(m_device->logical(), rmiss, nullptr);

        ENIGMA_LOG_INFO("[rt-reflection] RT pipeline built");
    } else {
        m_ssrPath = Paths::shaderSourceDir() / "ssr.hlsl";

        VkShaderModule vert = shaderManager.compile(m_ssrPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
        VkShaderModule frag = shaderManager.compile(m_ssrPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

        gfx::Pipeline::CreateInfo ci{};
        ci.vertShader            = vert;
        ci.vertEntryPoint        = "VSMain";
        ci.fragShader            = frag;
        ci.fragEntryPoint        = "PSMain";
        ci.globalSetLayout       = globalSetLayout;
        ci.colorAttachmentFormat = kOutputFormat;
        ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        ci.pushConstantSize      = sizeof(SSRPushBlock);
        ci.depthCompareOp        = VK_COMPARE_OP_ALWAYS;
        ci.cullMode              = VK_CULL_MODE_NONE;

        m_ssrPipeline = new gfx::Pipeline(*m_device, ci);

        vkDestroyShaderModule(m_device->logical(), vert, nullptr);
        vkDestroyShaderModule(m_device->logical(), frag, nullptr);

        ENIGMA_LOG_INFO("[rt-reflection] SSR fallback pipeline built");
    }
}

void RTReflectionPass::allocate(VkExtent2D extent) {
    if (m_output.image != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroyOutput();
    }
    m_extent = extent;

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = kOutputFormat;
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
    viewCI.format           = kOutputFormat;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &m_output.view));
}

void RTReflectionPass::record(VkCommandBuffer cmd,
                               VkDescriptorSet globalSet,
                               VkExtent2D extent,
                               u32 normalSlot,
                               u32 depthSlot,
                               u32 cameraSlot,
                               u32 samplerSlot,
                               u32 tlasSlot,
                               u32 outputSlotVal,
                               u32 skyViewLutSlot,
                               u32 transmittanceLutSlot,
                               vec4 sunWorldDirIntensity,
                               vec4 cameraWorldPosKm) {
    // Transition reflection image to GENERAL for storage writes.
    VkImageMemoryBarrier2 toGeneral{};
    toGeneral.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toGeneral.srcStageMask        = VK_PIPELINE_STAGE_2_NONE;
    toGeneral.srcAccessMask       = VK_ACCESS_2_NONE;
    toGeneral.dstStageMask        = m_useRT ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
                                            : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
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

    if (m_useRT) {
        ENIGMA_ASSERT(m_rtPipeline != nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline->handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                m_rtPipeline->layout(), 0, 1, &globalSet, 0, nullptr);

        RTReflectionPushBlock pc{};
        pc.normalSlot           = normalSlot;
        pc.depthSlot            = depthSlot;
        pc.cameraSlot           = cameraSlot;
        pc.samplerSlot          = samplerSlot;
        pc.tlasSlot             = tlasSlot;
        pc.outputSlot           = outputSlotVal;
        pc.skyViewLutSlot       = skyViewLutSlot;
        pc.transmittanceLutSlot = transmittanceLutSlot;
        pc.sunWorldDirIntensity = sunWorldDirIntensity;
        pc.cameraWorldPosKm     = cameraWorldPosKm;

        vkCmdPushConstants(cmd, m_rtPipeline->layout(),
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                           | VK_SHADER_STAGE_MISS_BIT_KHR,
                           0, sizeof(pc), &pc);

        vkCmdTraceRaysKHR(cmd,
                          &m_rtPipeline->raygenRegion(),
                          &m_rtPipeline->missRegion(),
                          &m_rtPipeline->hitGroupRegion(),
                          &m_rtPipeline->callableRegion(),
                          extent.width, extent.height, 1);
    } else {
        // SSR fallback: the record is handled outside the render graph
        // as a standalone compute-like fullscreen pass writing to storage image.
        // For now this is a no-op placeholder; the SSR pipeline needs a
        // render pass context which the render graph would provide.
        // Phase 2 will integrate SSR properly into the render graph.
        (void)globalSet;
        (void)normalSlot;
        (void)depthSlot;
        (void)cameraSlot;
        (void)samplerSlot;
        (void)outputSlotVal;
        (void)skyViewLutSlot;
        (void)transmittanceLutSlot;
        (void)sunWorldDirIntensity;
        (void)cameraWorldPosKm;
    }

    // Transition to SHADER_READ_ONLY for the lighting pass to sample.
    VkImageMemoryBarrier2 toRead{};
    toRead.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toRead.srcStageMask        = m_useRT ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
                                         : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    toRead.srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    toRead.dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
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

void RTReflectionPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    if (m_useRT) {
        reloader.watchGroup({m_rgenPath, m_rchitPath, m_rmissPath}, [this]() {
            ENIGMA_LOG_INFO("[rt-reflection] hot-reload triggered (RT shaders)");
            // RT pipeline rebuild requires full recreation. For now, log only.
            // Phase 2 will implement full RT pipeline hot-reload.
        });
    } else {
        reloader.watchGroup({m_ssrPath}, [this]() {
            ENIGMA_LOG_INFO("[rt-reflection] hot-reload triggered (SSR shader)");
        });
    }
}

void RTReflectionPass::destroyOutput() {
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
