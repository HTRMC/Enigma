#include "renderer/GBufferPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Math.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "scene/Scene.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

namespace enigma {

// Push constant layout — must match gbuffer.hlsl PushBlock exactly.
struct GBufferPushBlock {
    mat4 model;              // 64 bytes
    u32  vertexSlot;         //  4
    u32  cameraSlot;         //  4
    u32  materialBufferSlot; //  4
    u32  materialIndex;      //  4
};                           // Total: 80 bytes

static_assert(sizeof(GBufferPushBlock) == 80);

GBufferPass::GBufferPass(gfx::Device& device, gfx::Allocator& allocator)
    : m_device(&device)
    , m_allocator(&allocator) {}

GBufferPass::~GBufferPass() {
    delete m_pipeline;
    destroyImages();
}

// ---------------------------------------------------------------------------

void GBufferPass::createImage(VkFormat format, VkImageUsageFlags usage,
                               VkImageAspectFlags aspect, GBufferImage& out) {
    VkImageCreateInfo imageCI{};
    imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType     = VK_IMAGE_TYPE_2D;
    imageCI.format        = format;
    imageCI.extent        = {m_extent.width, m_extent.height, 1};
    imageCI.mipLevels     = 1;
    imageCI.arrayLayers   = 1;
    imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage         = usage;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocCI.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imageCI, &allocCI,
                                   &out.image, &out.allocation, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image                           = out.image;
    viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format                          = format;
    viewCI.subresourceRange.aspectMask     = aspect;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = 1;

    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &out.view));
}

void GBufferPass::destroyImage(GBufferImage& img) {
    if (img.view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), img.view, nullptr);
        img.view = VK_NULL_HANDLE;
    }
    if (img.image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), img.image, img.allocation);
        img.image      = VK_NULL_HANDLE;
        img.allocation = nullptr;
    }
}

void GBufferPass::destroyImages() {
    destroyImage(m_albedo);
    destroyImage(m_normal);
    destroyImage(m_metalRough);
    destroyImage(m_motionVec);
    destroyImage(m_depth);
}

void GBufferPass::allocate(VkExtent2D extent) {
    if (m_albedo.image != VK_NULL_HANDLE) {
        // Wait for GPU to finish before destroying in-use images.
        vkDeviceWaitIdle(m_device->logical());
        destroyImages();
    }
    m_extent = extent;

    createImage(kAlbedoFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, m_albedo);

    createImage(kNormalFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, m_normal);

    createImage(kMetalRoughFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, m_metalRough);

    createImage(kMotionVecFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, m_motionVec);

    createImage(kDepthFormat,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT, m_depth);

    ENIGMA_LOG_INFO("[gbuffer] allocated {}x{}", extent.width, extent.height);
}

// ---------------------------------------------------------------------------

void GBufferPass::buildPipeline(gfx::ShaderManager& shaderManager,
                                 VkDescriptorSetLayout globalSetLayout) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "GBufferPass::buildPipeline called twice");

    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_shaderPath      = Paths::shaderSourceDir() / "gbuffer.hlsl";

    VkShaderModule vert = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    VkShaderModule frag = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader              = vert;
    ci.vertEntryPoint          = "VSMain";
    ci.fragShader              = frag;
    ci.fragEntryPoint          = "PSMain";
    ci.globalSetLayout         = globalSetLayout;
    ci.colorAttachmentFormats[0] = kAlbedoFormat;
    ci.colorAttachmentFormats[1] = kNormalFormat;
    ci.colorAttachmentFormats[2] = kMetalRoughFormat;
    ci.colorAttachmentFormats[3] = kMotionVecFormat;
    ci.colorAttachmentCount    = 4;
    ci.depthAttachmentFormat   = kDepthFormat;
    ci.pushConstantSize        = sizeof(GBufferPushBlock);
    ci.depthCompareOp          = VK_COMPARE_OP_GREATER_OR_EQUAL; // reverse-Z
    ci.cullMode                = VK_CULL_MODE_BACK_BIT;

    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[gbuffer] pipeline built");
}

void GBufferPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    ENIGMA_ASSERT(m_shaderManager != nullptr);

    VkShaderModule vert = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    if (vert == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[gbuffer] hot-reload: VS compile failed"); return; }
    VkShaderModule frag = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");
    if (frag == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[gbuffer] hot-reload: PS compile failed");
        vkDestroyShaderModule(m_device->logical(), vert, nullptr);
        return;
    }

    vkDeviceWaitIdle(m_device->logical());
    delete m_pipeline;

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader              = vert;
    ci.vertEntryPoint          = "VSMain";
    ci.fragShader              = frag;
    ci.fragEntryPoint          = "PSMain";
    ci.globalSetLayout         = m_globalSetLayout;
    ci.colorAttachmentFormats[0] = kAlbedoFormat;
    ci.colorAttachmentFormats[1] = kNormalFormat;
    ci.colorAttachmentFormats[2] = kMetalRoughFormat;
    ci.colorAttachmentFormats[3] = kMotionVecFormat;
    ci.colorAttachmentCount    = 4;
    ci.depthAttachmentFormat   = kDepthFormat;
    ci.pushConstantSize        = sizeof(GBufferPushBlock);
    ci.depthCompareOp          = VK_COMPARE_OP_GREATER_OR_EQUAL;
    ci.cullMode                = VK_CULL_MODE_BACK_BIT;
    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[gbuffer] hot-reload: pipeline rebuilt");
}

void GBufferPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

// ---------------------------------------------------------------------------

void GBufferPass::record(VkCommandBuffer cmd,
                          VkDescriptorSet globalSet,
                          VkExtent2D extent,
                          const Scene& scene,
                          u32 cameraSlot) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "GBufferPass::record before buildPipeline");

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    for (const auto& node : scene.nodes) {
        for (u32 primIdx : node.primitiveIndices) {
            const auto& prim = scene.primitives[primIdx];

            GBufferPushBlock pc{};
            pc.model              = node.worldTransform;
            pc.vertexSlot         = prim.vertexBufferSlot;
            pc.cameraSlot         = cameraSlot;
            pc.materialBufferSlot = scene.materialBufferSlot;
            pc.materialIndex      = prim.materialIndex >= 0
                                        ? static_cast<u32>(prim.materialIndex)
                                        : 0u;

            vkCmdPushConstants(cmd, m_pipeline->layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            vkCmdBindIndexBuffer(cmd, prim.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, prim.indexCount, 1, 0, 0, 0);
        }
    }
}

} // namespace enigma
