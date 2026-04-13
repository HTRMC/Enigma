#include "renderer/SMAAPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

// VMA is driven via volk; include after volk.h.
#include <volk.h>
#include <vk_mem_alloc.h>

namespace enigma {

// ---------------------------------------------------------------------------
// Push-constant blocks — must match the HLSL PushBlock structs exactly.
// ---------------------------------------------------------------------------

struct SMAAEdgePushBlock {
    u32 colorSlot;
    u32 samplerSlot;
    u32 width;
    u32 height;
}; // 16 bytes

struct SMAABlendPushBlock {
    u32 edgeSlot;
    u32 samplerSlot;
    u32 width;
    u32 height;
}; // 16 bytes

struct SMAANeighborPushBlock {
    u32 colorSlot;
    u32 weightSlot;
    u32 samplerSlot;
    u32 width;
    u32 height;
    u32 _pad;
}; // 24 bytes

static_assert(sizeof(SMAAEdgePushBlock)    == 16);
static_assert(sizeof(SMAABlendPushBlock)   == 16);
static_assert(sizeof(SMAANeighborPushBlock)== 24);

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

SMAAPass::SMAAPass(gfx::Device& device, gfx::Allocator& allocator)
    : m_device(&device)
    , m_allocator(&allocator) {}

SMAAPass::~SMAAPass() {
    delete m_neighborPipeline;
    delete m_blendPipeline;
    delete m_edgePipeline;
    // Note: caller must have already called free() to destroy textures.
    ENIGMA_ASSERT(m_edgeImage   == VK_NULL_HANDLE && "SMAAPass destroyed without free()");
    ENIGMA_ASSERT(m_weightImage == VK_NULL_HANDLE && "SMAAPass destroyed without free()");
}

// ---------------------------------------------------------------------------
// Intermediate texture management
// ---------------------------------------------------------------------------

static void createTexture(VkDevice           dev,
                          VmaAllocator       vma,
                          VkExtent2D         extent,
                          VkFormat           format,
                          VkImage&           outImage,
                          VkImageView&       outView,
                          VmaAllocation&     outAlloc)
{
    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = format;
    imgCI.extent        = {extent.width, extent.height, 1};
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateImage(vma, &imgCI, &allocCI, &outImage, &outAlloc, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = outImage;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = format;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    ENIGMA_VK_CHECK(vkCreateImageView(dev, &viewCI, nullptr, &outView));
}

void SMAAPass::allocate(VkExtent2D extent, gfx::DescriptorAllocator& descriptorAllocator) {
    ENIGMA_ASSERT(m_edgeImage == VK_NULL_HANDLE && "Call free() before reallocating");
    m_extent = extent;

    VkDevice     dev = m_device->logical();
    VmaAllocator vma = m_allocator->handle();

    createTexture(dev, vma, extent, VK_FORMAT_R8G8_UNORM,
                  m_edgeImage,   m_edgeView,   m_edgeAlloc);
    createTexture(dev, vma, extent, VK_FORMAT_R8G8B8A8_UNORM,
                  m_weightImage, m_weightView, m_weightAlloc);

    m_edgeSampledSlot   = descriptorAllocator.registerSampledImage(
        m_edgeView,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_weightSampledSlot = descriptorAllocator.registerSampledImage(
        m_weightView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    ENIGMA_LOG_INFO("[SMAAPass] allocated {}x{} (edge={}, weight={})",
                    extent.width, extent.height,
                    m_edgeSampledSlot, m_weightSampledSlot);
}

void SMAAPass::free(gfx::DescriptorAllocator& descriptorAllocator) {
    VkDevice     dev = m_device->logical();
    VmaAllocator vma = m_allocator->handle();

    // Release bindless slots before destroying the views — slots go back onto
    // the free-list so the next allocate() reuses them, keeping numbering stable.
    if (m_edgeSampledSlot != UINT32_MAX) {
        descriptorAllocator.releaseSampledImage(m_edgeSampledSlot);
        m_edgeSampledSlot = UINT32_MAX;
    }
    if (m_weightSampledSlot != UINT32_MAX) {
        descriptorAllocator.releaseSampledImage(m_weightSampledSlot);
        m_weightSampledSlot = UINT32_MAX;
    }

    if (m_edgeView   != VK_NULL_HANDLE) {
        vkDestroyImageView(dev, m_edgeView, nullptr);
        m_edgeView = VK_NULL_HANDLE;
    }
    if (m_edgeImage  != VK_NULL_HANDLE) {
        vmaDestroyImage(vma, m_edgeImage, m_edgeAlloc);
        m_edgeImage = VK_NULL_HANDLE;
        m_edgeAlloc = nullptr;
    }
    if (m_weightView  != VK_NULL_HANDLE) {
        vkDestroyImageView(dev, m_weightView, nullptr);
        m_weightView = VK_NULL_HANDLE;
    }
    if (m_weightImage != VK_NULL_HANDLE) {
        vmaDestroyImage(vma, m_weightImage, m_weightAlloc);
        m_weightImage = VK_NULL_HANDLE;
        m_weightAlloc = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Pipeline management
// ---------------------------------------------------------------------------

void SMAAPass::buildPipelines(gfx::ShaderManager&   shaderManager,
                               VkDescriptorSetLayout globalSetLayout,
                               VkFormat              ldrFormat) {
    ENIGMA_ASSERT(m_edgePipeline == nullptr && "buildPipelines called twice");

    m_shaderManager  = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_ldrFormat       = ldrFormat;

    m_edgePath     = Paths::shaderSourceDir() / "smaa_edge.hlsl";
    m_blendPath    = Paths::shaderSourceDir() / "smaa_blend.hlsl";
    m_neighborPath = Paths::shaderSourceDir() / "smaa_neighborhood.hlsl";

    rebuildPipelines();
    ENIGMA_LOG_INFO("[SMAAPass] pipelines built");
}

void SMAAPass::rebuildPipelines() {
    auto compile = [&](const auto& path, gfx::ShaderManager::Stage stage, const char* entry) {
        return m_shaderManager->compile(path, stage, entry);
    };

    using Stage = gfx::ShaderManager::Stage;

    VkShaderModule edgeVS   = compile(m_edgePath,     Stage::Vertex,   "VSMain");
    VkShaderModule edgePS   = compile(m_edgePath,     Stage::Fragment, "PSMain");
    VkShaderModule blendVS  = compile(m_blendPath,    Stage::Vertex,   "VSMain");
    VkShaderModule blendPS  = compile(m_blendPath,    Stage::Fragment, "PSMain");
    VkShaderModule neighborVS = compile(m_neighborPath, Stage::Vertex,   "VSMain");
    VkShaderModule neighborPS = compile(m_neighborPath, Stage::Fragment, "PSMain");

    VkDevice dev = m_device->logical();

    // ---- Edge detection pipeline (output: R8G8_UNORM) ----
    {
        gfx::Pipeline::CreateInfo ci{};
        ci.vertShader            = edgeVS;
        ci.vertEntryPoint        = "VSMain";
        ci.fragShader            = edgePS;
        ci.fragEntryPoint        = "PSMain";
        ci.globalSetLayout       = m_globalSetLayout;
        ci.colorAttachmentFormat = VK_FORMAT_R8G8_UNORM;
        ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        ci.cullMode              = VK_CULL_MODE_NONE;
        ci.blendEnable           = false;
        ci.depthWriteEnable      = false;
        ci.pushConstantSize      = sizeof(SMAAEdgePushBlock);

        delete m_edgePipeline;
        m_edgePipeline = new gfx::Pipeline(*m_device, ci);
    }

    // ---- Blending weight pipeline (output: RGBA8_UNORM) ----
    {
        gfx::Pipeline::CreateInfo ci{};
        ci.vertShader            = blendVS;
        ci.vertEntryPoint        = "VSMain";
        ci.fragShader            = blendPS;
        ci.fragEntryPoint        = "PSMain";
        ci.globalSetLayout       = m_globalSetLayout;
        ci.colorAttachmentFormat = VK_FORMAT_R8G8B8A8_UNORM;
        ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        ci.cullMode              = VK_CULL_MODE_NONE;
        ci.blendEnable           = false;
        ci.depthWriteEnable      = false;
        ci.pushConstantSize      = sizeof(SMAABlendPushBlock);

        delete m_blendPipeline;
        m_blendPipeline = new gfx::Pipeline(*m_device, ci);
    }

    // ---- Neighbourhood blend pipeline (output: ldrFormat / swapchain) ----
    {
        gfx::Pipeline::CreateInfo ci{};
        ci.vertShader            = neighborVS;
        ci.vertEntryPoint        = "VSMain";
        ci.fragShader            = neighborPS;
        ci.fragEntryPoint        = "PSMain";
        ci.globalSetLayout       = m_globalSetLayout;
        ci.colorAttachmentFormat = m_ldrFormat;
        ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        ci.cullMode              = VK_CULL_MODE_NONE;
        ci.blendEnable           = false;
        ci.depthWriteEnable      = false;
        ci.pushConstantSize      = sizeof(SMAANeighborPushBlock);

        delete m_neighborPipeline;
        m_neighborPipeline = new gfx::Pipeline(*m_device, ci);
    }

    vkDestroyShaderModule(dev, edgeVS,     nullptr);
    vkDestroyShaderModule(dev, edgePS,     nullptr);
    vkDestroyShaderModule(dev, blendVS,    nullptr);
    vkDestroyShaderModule(dev, blendPS,    nullptr);
    vkDestroyShaderModule(dev, neighborVS, nullptr);
    vkDestroyShaderModule(dev, neighborPS, nullptr);
}

void SMAAPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    reloader.watchGroup({m_edgePath, m_blendPath, m_neighborPath}, [this]() {
        vkDeviceWaitIdle(m_device->logical());
        rebuildPipelines();
        ENIGMA_LOG_INFO("[SMAAPass] hot-reload: pipelines rebuilt");
    });
}

// ---------------------------------------------------------------------------
// Record functions
// ---------------------------------------------------------------------------

void SMAAPass::recordEdge(VkCommandBuffer cmd,
                           VkDescriptorSet globalSet,
                           VkExtent2D      extent,
                           u32             ldrSampledSlot,
                           u32             samplerSlot) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_edgePipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_edgePipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    VkViewport vp{0.f, 0.f,
                  static_cast<float>(extent.width),
                  static_cast<float>(extent.height),
                  0.f, 1.f};
    VkRect2D   sc{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd,  0, 1, &sc);

    SMAAEdgePushBlock pc{};
    pc.colorSlot   = ldrSampledSlot;
    pc.samplerSlot = samplerSlot;
    pc.width       = extent.width;
    pc.height      = extent.height;
    vkCmdPushConstants(cmd, m_edgePipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle
}

void SMAAPass::recordBlend(VkCommandBuffer cmd,
                            VkDescriptorSet globalSet,
                            VkExtent2D      extent,
                            u32             samplerSlot) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_blendPipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_blendPipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    VkViewport vp{0.f, 0.f,
                  static_cast<float>(extent.width),
                  static_cast<float>(extent.height),
                  0.f, 1.f};
    VkRect2D   sc{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd,  0, 1, &sc);

    SMAABlendPushBlock pc{};
    pc.edgeSlot    = m_edgeSampledSlot;
    pc.samplerSlot = samplerSlot;
    pc.width       = extent.width;
    pc.height      = extent.height;
    vkCmdPushConstants(cmd, m_blendPipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void SMAAPass::recordNeighborhood(VkCommandBuffer cmd,
                                   VkDescriptorSet globalSet,
                                   VkExtent2D      extent,
                                   u32             ldrSampledSlot,
                                   u32             samplerSlot) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_neighborPipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_neighborPipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    VkViewport vp{0.f, 0.f,
                  static_cast<float>(extent.width),
                  static_cast<float>(extent.height),
                  0.f, 1.f};
    VkRect2D   sc{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd,  0, 1, &sc);

    SMAANeighborPushBlock pc{};
    pc.colorSlot   = ldrSampledSlot;
    pc.weightSlot  = m_weightSampledSlot;
    pc.samplerSlot = samplerSlot;
    pc.width       = extent.width;
    pc.height      = extent.height;
    pc._pad        = 0;
    vkCmdPushConstants(cmd, m_neighborPipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace enigma
