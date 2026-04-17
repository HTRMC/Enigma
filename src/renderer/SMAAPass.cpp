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

// Jimenez SMAA precomputed lookup tables — raw byte arrays, ~180 KB total.
// Pasted verbatim from the reference repo; see
// https://github.com/iryoku/smaa (MIT).
#include "renderer/smaa/AreaTex.h"
#include "renderer/smaa/SearchTex.h"

#include <cstring>

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
    u32 areaTexSlot;
    u32 searchTexSlot;
    u32 samplerSlot;   // linear-clamp
    u32 width;
    u32 height;
}; // 24 bytes

struct SMAANeighborPushBlock {
    u32 colorSlot;
    u32 weightSlot;
    u32 samplerSlot;
    u32 width;
    u32 height;
    u32 _pad;
}; // 24 bytes

static_assert(sizeof(SMAAEdgePushBlock)    == 16);
static_assert(sizeof(SMAABlendPushBlock)   == 24);
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
    // Note: caller must have already called free() to destroy per-resize textures.
    ENIGMA_ASSERT(m_edgeImage   == VK_NULL_HANDLE && "SMAAPass destroyed without free()");
    ENIGMA_ASSERT(m_weightImage == VK_NULL_HANDLE && "SMAAPass destroyed without free()");

    // Staging should have been released post-submit; this is a defensive
    // cleanup in case the caller forgot.
    releaseLookupUploadStaging();

    VkDevice     dev = m_device    ? m_device->logical()   : VK_NULL_HANDLE;
    VmaAllocator vma = m_allocator ? m_allocator->handle() : VK_NULL_HANDLE;

    if (m_areaTexView   != VK_NULL_HANDLE) { vkDestroyImageView(dev, m_areaTexView,   nullptr); }
    if (m_areaTexImage  != VK_NULL_HANDLE) { vmaDestroyImage(vma, m_areaTexImage,  m_areaTexAlloc);  }
    if (m_searchTexView != VK_NULL_HANDLE) { vkDestroyImageView(dev, m_searchTexView, nullptr); }
    if (m_searchTexImage!= VK_NULL_HANDLE) { vmaDestroyImage(vma, m_searchTexImage, m_searchTexAlloc); }
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

    ENIGMA_ASSERT(m_areaTexSampledSlot   != UINT32_MAX &&
                  m_searchTexSampledSlot != UINT32_MAX &&
                  "SMAAPass::recordBlend requires uploadLookupTextures() at init");

    SMAABlendPushBlock pc{};
    pc.edgeSlot      = m_edgeSampledSlot;
    pc.areaTexSlot   = m_areaTexSampledSlot;
    pc.searchTexSlot = m_searchTexSampledSlot;
    pc.samplerSlot   = samplerSlot;
    pc.width         = extent.width;
    pc.height        = extent.height;
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

// ---------------------------------------------------------------------------
// Precomputed lookup-texture upload (one-shot at engine init)
// ---------------------------------------------------------------------------

namespace {

// Create a GPU-local sampled image and host-visible staging buffer, memcpy
// the raw bytes, record staging->image copy with correct layout transitions,
// register as bindless sampled image. Returns staging handles for later
// release.
void createAndUploadLookupImage(VkDevice            dev,
                                VmaAllocator        vma,
                                VkCommandBuffer     cmd,
                                gfx::DescriptorAllocator& descriptors,
                                VkFormat            fmt,
                                u32                 width,
                                u32                 height,
                                const u8*           srcBytes,
                                VkDeviceSize        srcByteCount,
                                VkImage&            outImage,
                                VkImageView&        outView,
                                VmaAllocation&      outAlloc,
                                u32&                outSlot,
                                VkBuffer&           outStaging,
                                VmaAllocation&      outStagingAlloc)
{
    // 1. Create GPU-local image.
    VkImageCreateInfo imgCI{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = fmt;
    imgCI.extent        = { width, height, 1 };
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ENIGMA_VK_CHECK(vmaCreateImage(vma, &imgCI, &allocCI, &outImage, &outAlloc, nullptr));

    VkImageViewCreateInfo viewCI{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewCI.image            = outImage;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = fmt;
    viewCI.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    ENIGMA_VK_CHECK(vkCreateImageView(dev, &viewCI, nullptr, &outView));

    // 2. Host-visible staging buffer seeded with srcBytes.
    VkBufferCreateInfo stageCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stageCI.size        = srcByteCount;
    stageCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stageAllocCI{};
    stageAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    stageAllocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                       | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VmaAllocationInfo stagingInfo{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(vma, &stageCI, &stageAllocCI,
                                    &outStaging, &outStagingAlloc, &stagingInfo));
    std::memcpy(stagingInfo.pMappedData, srcBytes, static_cast<size_t>(srcByteCount));
    vmaFlushAllocation(vma, outStagingAlloc, 0, VK_WHOLE_SIZE);

    // 3. UNDEFINED -> TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier2 toDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    toDst.srcStageMask     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    toDst.srcAccessMask    = 0;
    toDst.dstStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT;
    toDst.dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toDst.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.image            = outImage;
    toDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &toDst;
    vkCmdPipelineBarrier2(cmd, &dep);

    // 4. Copy buffer -> image.
    VkBufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0; // tightly packed
    region.bufferImageHeight = 0;
    region.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageOffset       = { 0, 0, 0 };
    region.imageExtent       = { width, height, 1 };
    vkCmdCopyBufferToImage(cmd, outStaging, outImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // 5. TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    VkImageMemoryBarrier2 toRead{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    toRead.srcStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT;
    toRead.srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toRead.dstStageMask     = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toRead.dstAccessMask    = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    toRead.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toRead.newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.image            = outImage;
    toRead.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo dep2{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep2.imageMemoryBarrierCount = 1;
    dep2.pImageMemoryBarriers    = &toRead;
    vkCmdPipelineBarrier2(cmd, &dep2);

    // 6. Register as bindless sampled image.
    outSlot = descriptors.registerSampledImage(outView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace

void SMAAPass::uploadLookupTextures(VkCommandBuffer cmd,
                                    gfx::DescriptorAllocator& descriptors)
{
    ENIGMA_ASSERT(m_areaTexImage   == VK_NULL_HANDLE && "uploadLookupTextures called twice");
    ENIGMA_ASSERT(m_searchTexImage == VK_NULL_HANDLE && "uploadLookupTextures called twice");

    static_assert(sizeof(areaTexBytes)   == AREATEX_SIZE,
                  "AreaTex byte array size mismatch — AreaTex.h corruption?");
    static_assert(sizeof(searchTexBytes) == SEARCHTEX_SIZE,
                  "SearchTex byte array size mismatch — SearchTex.h corruption?");

    VkDevice     dev = m_device->logical();
    VmaAllocator vma = m_allocator->handle();

    createAndUploadLookupImage(dev, vma, cmd, descriptors,
                               VK_FORMAT_R8G8_UNORM,
                               AREATEX_WIDTH, AREATEX_HEIGHT,
                               areaTexBytes, AREATEX_SIZE,
                               m_areaTexImage, m_areaTexView, m_areaTexAlloc,
                               m_areaTexSampledSlot,
                               m_areaTexStaging, m_areaTexStagingAlloc);

    createAndUploadLookupImage(dev, vma, cmd, descriptors,
                               VK_FORMAT_R8_UNORM,
                               SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT,
                               searchTexBytes, SEARCHTEX_SIZE,
                               m_searchTexImage, m_searchTexView, m_searchTexAlloc,
                               m_searchTexSampledSlot,
                               m_searchTexStaging, m_searchTexStagingAlloc);

    ENIGMA_LOG_INFO("[SMAAPass] uploaded AreaTex (slot={}) + SearchTex (slot={})",
                    m_areaTexSampledSlot, m_searchTexSampledSlot);
}

void SMAAPass::releaseLookupUploadStaging() {
    VmaAllocator vma = m_allocator ? m_allocator->handle() : VK_NULL_HANDLE;
    if (vma == VK_NULL_HANDLE) return;

    if (m_areaTexStaging != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vma, m_areaTexStaging, m_areaTexStagingAlloc);
        m_areaTexStaging      = VK_NULL_HANDLE;
        m_areaTexStagingAlloc = nullptr;
    }
    if (m_searchTexStaging != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vma, m_searchTexStaging, m_searchTexStagingAlloc);
        m_searchTexStaging      = VK_NULL_HANDLE;
        m_searchTexStagingAlloc = nullptr;
    }
}

} // namespace enigma
