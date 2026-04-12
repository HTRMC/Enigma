#include "renderer/AtmospherePass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderManager.h"

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

namespace enigma {

// ---------------------------------------------------------------------------
// Push constant blocks — must match the corresponding HLSL PushBlock exactly.
// ---------------------------------------------------------------------------

struct TransmittancePush {
    u32 storageSlot;
    u32 _pad[3];
};
static_assert(sizeof(TransmittancePush) == 16);

struct MultiScatterPush {
    float sunDirX, sunDirY, sunDirZ;
    float sunIntensity;
    u32  transmittanceLutSlot;
    u32  multiScatterStoreSlot;
    u32  samplerSlot;
    u32  _pad;
};
static_assert(sizeof(MultiScatterPush) == 32);

struct SkyViewPush {
    float sunDirX, sunDirY, sunDirZ;
    float sunIntensity;
    float camX, camY, camZ;
    float _pad;
    u32  transmittanceLutSlot;
    u32  multiScatterLutSlot;
    u32  skyViewStoreSlot;
    u32  samplerSlot;
};
static_assert(sizeof(SkyViewPush) == 48);

struct APPush {
    float invViewProj[16];  // row-major 4x4
    float sunDirX, sunDirY, sunDirZ;
    float sunIntensity;
    float camX, camY, camZ;
    float apSliceFar;
    u32  transmittanceLutSlot;
    u32  multiScatterLutSlot;
    u32  samplerSlot;
    u32  _pad;
};
static_assert(sizeof(APPush) == 112);

// ---------------------------------------------------------------------------

void AtmospherePass::init(const InitInfo& info) {
    ENIGMA_ASSERT(info.device && info.allocator && info.descriptorAllocator &&
                  info.shaderManager && info.globalSetLayout != VK_NULL_HANDLE);

    m_device              = info.device;
    m_allocator           = info.allocator;
    m_descriptorAllocator = info.descriptorAllocator;
    m_shaderManager       = info.shaderManager;
    m_globalSetLayout     = info.globalSetLayout;

    VkDevice dev = m_device->logical();

    // -- Allocate LUT images ------------------------------------------------
    createLut2D(kTransmittanceSize, kLutFormat2D,
                m_transmittanceImg, m_transmittanceView, m_transmittanceAlloc,
                m_tlutSampledSlot, m_tlutStorageSlot);

    createLut2D(kMultiScatterSize, kLutFormat2D,
                m_multiScatterImg, m_multiScatterView, m_multiScatterAlloc,
                m_msLutSampledSlot, m_msLutStorageSlot);

    createLut2D(kSkyViewSize, kLutFormat2D,
                m_skyViewImg, m_skyViewView, m_skyViewAlloc,
                m_svLutSampledSlot, m_svLutStorageSlot);

    createLut3D();

    // -- Compute pipelines (2D LUTs) ----------------------------------------
    auto shaderDir = Paths::shaderSourceDir();

    auto makeComputePipe = [&](const std::filesystem::path& path, u32 pushSize) {
        VkShaderModule cs = m_shaderManager->compile(
            path, gfx::ShaderManager::Stage::Compute, "CSMain");
        gfx::Pipeline::CreateInfo ci{};
        ci.globalSetLayout  = m_globalSetLayout;
        ci.pushConstantSize = pushSize;
        ci.computeShader    = cs;
        ci.computeEntryPoint = "CSMain";
        auto* pipe = new gfx::Pipeline(*m_device, ci);
        vkDestroyShaderModule(dev, cs, nullptr);
        return pipe;
    };

    m_transmittancePipe = makeComputePipe(
        shaderDir / "atmosphere_transmittance.hlsl", sizeof(TransmittancePush));
    m_multiScatterPipe  = makeComputePipe(
        shaderDir / "atmosphere_multiscatter.hlsl",  sizeof(MultiScatterPush));
    m_skyViewPipe       = makeComputePipe(
        shaderDir / "atmosphere_skyview.hlsl",       sizeof(SkyViewPush));

    // -- AP compute pipeline (needs set 0 + set 1) --------------------------
    {
        VkDescriptorSetLayout setLayouts[2] = {m_globalSetLayout, m_apWriteSetLayout};
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(APPush);

        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount         = 2;
        layoutCI.pSetLayouts            = setLayouts;
        layoutCI.pushConstantRangeCount = 1;
        layoutCI.pPushConstantRanges    = &pcRange;
        ENIGMA_VK_CHECK(vkCreatePipelineLayout(dev, &layoutCI, nullptr, &m_apPipeLayout));

        VkShaderModule apCS = m_shaderManager->compile(
            shaderDir / "atmosphere_aerial_perspective.hlsl",
            gfx::ShaderManager::Stage::Compute, "CSMain");

        VkPipelineShaderStageCreateInfo stageCI{};
        stageCI.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageCI.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stageCI.module = apCS;
        stageCI.pName  = "CSMain";

        VkComputePipelineCreateInfo pipeCI{};
        pipeCI.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeCI.layout = m_apPipeLayout;
        pipeCI.stage  = stageCI;

        ENIGMA_VK_CHECK(vkCreateComputePipelines(
            dev, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &m_apPipe));
        vkDestroyShaderModule(dev, apCS, nullptr);
    }

    ENIGMA_LOG_INFO("[atmosphere] initialized (T={}, MS={}, SV={}, AP slots)",
                    m_tlutSampledSlot, m_msLutSampledSlot, m_svLutSampledSlot);
}

// ---------------------------------------------------------------------------

void AtmospherePass::bakeStaticLUTs(VkCommandBuffer cmd,
                                     const AtmosphereSettings& settings,
                                     const vec3& sunWorldDir,
                                     u32 samplerSlot) {
    VkDescriptorSet globalSet = m_descriptorAllocator->globalSet();

    // ---- Transmittance LUT ----
    transitionImage(cmd, m_transmittanceImg,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_transmittancePipe->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_transmittancePipe->layout(),
                            0, 1, &globalSet, 0, nullptr);

    TransmittancePush tPush{};
    tPush.storageSlot = m_tlutStorageSlot;
    vkCmdPushConstants(cmd, m_transmittancePipe->layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tPush), &tPush);

    // Dispatch: 256×64 pixels, 8×8 groups → ceil(256/8)=32, ceil(64/8)=8
    vkCmdDispatch(cmd,
                  (kTransmittanceSize.width  + 7) / 8,
                  (kTransmittanceSize.height + 7) / 8,
                  1);

    transitionImage(cmd, m_transmittanceImg,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    // ---- Multi-Scatter LUT ----
    transitionImage(cmd, m_multiScatterImg,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_multiScatterPipe->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_multiScatterPipe->layout(),
                            0, 1, &globalSet, 0, nullptr);

    MultiScatterPush msPush{};
    msPush.sunDirX               = sunWorldDir.x;
    msPush.sunDirY               = sunWorldDir.y;
    msPush.sunDirZ               = sunWorldDir.z;
    msPush.sunIntensity          = settings.sunIntensity;
    msPush.transmittanceLutSlot  = m_tlutSampledSlot;
    msPush.multiScatterStoreSlot = m_msLutStorageSlot;
    msPush.samplerSlot           = samplerSlot;
    vkCmdPushConstants(cmd, m_multiScatterPipe->layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(msPush), &msPush);

    vkCmdDispatch(cmd,
                  (kMultiScatterSize.width  + 7) / 8,
                  (kMultiScatterSize.height + 7) / 8,
                  1);

    transitionImage(cmd, m_multiScatterImg,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    ENIGMA_LOG_INFO("[atmosphere] baked Transmittance + MultiScatter LUTs");
}

// ---------------------------------------------------------------------------

void AtmospherePass::updatePerFrame(VkCommandBuffer cmd,
                                     const AtmosphereSettings& settings,
                                     const vec3& sunWorldDir,
                                     const vec3& cameraWorldPosKm,
                                     const mat4& invViewProj,
                                     u32 samplerSlot) {
    VkDescriptorSet globalSet = m_descriptorAllocator->globalSet();

    // ---- SkyView LUT ----
    transitionImage(cmd, m_skyViewImg,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_skyViewPipe->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_skyViewPipe->layout(),
                            0, 1, &globalSet, 0, nullptr);

    SkyViewPush svPush{};
    svPush.sunDirX              = sunWorldDir.x;
    svPush.sunDirY              = sunWorldDir.y;
    svPush.sunDirZ              = sunWorldDir.z;
    svPush.sunIntensity         = settings.sunIntensity;
    svPush.camX                 = cameraWorldPosKm.x;
    svPush.camY                 = cameraWorldPosKm.y;
    svPush.camZ                 = cameraWorldPosKm.z;
    svPush.transmittanceLutSlot = m_tlutSampledSlot;
    svPush.multiScatterLutSlot  = m_msLutSampledSlot;
    svPush.skyViewStoreSlot     = m_svLutStorageSlot;
    svPush.samplerSlot          = samplerSlot;
    vkCmdPushConstants(cmd, m_skyViewPipe->layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(svPush), &svPush);

    vkCmdDispatch(cmd,
                  (kSkyViewSize.width  + 7) / 8,
                  (kSkyViewSize.height + 7) / 8,
                  1);

    transitionImage(cmd, m_skyViewImg,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    // ---- Aerial Perspective Volume ----
    transitionImage(cmd, m_apImg,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_apPipe);
    VkDescriptorSet apSets[2] = {globalSet, m_apWriteSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_apPipeLayout,
                            0, 2, apSets, 0, nullptr);

    APPush apPush{};
    // Column-major glm mat4 → row-major float[16] for HLSL float4x4
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            apPush.invViewProj[row * 4 + col] = invViewProj[col][row];
    apPush.sunDirX              = sunWorldDir.x;
    apPush.sunDirY              = sunWorldDir.y;
    apPush.sunDirZ              = sunWorldDir.z;
    apPush.sunIntensity         = settings.sunIntensity;
    apPush.camX                 = cameraWorldPosKm.x;
    apPush.camY                 = cameraWorldPosKm.y;
    apPush.camZ                 = cameraWorldPosKm.z;
    apPush.apSliceFar           = 50.0f; // 50 km far slice
    apPush.transmittanceLutSlot = m_tlutSampledSlot;
    apPush.multiScatterLutSlot  = m_msLutSampledSlot;
    apPush.samplerSlot          = samplerSlot;
    vkCmdPushConstants(cmd, m_apPipeLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(apPush), &apPush);

    vkCmdDispatch(cmd,
                  (kAerialPerspectiveSize.width  + 3) / 4,
                  (kAerialPerspectiveSize.height + 3) / 4,
                  (kAerialPerspectiveSize.depth  + 3) / 4);

    transitionImage(cmd, m_apImg,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ---------------------------------------------------------------------------

void AtmospherePass::shutdown() {
    if (!m_device) return;
    VkDevice dev = m_device->logical();

    delete m_transmittancePipe; m_transmittancePipe = nullptr;
    delete m_multiScatterPipe;  m_multiScatterPipe  = nullptr;
    delete m_skyViewPipe;       m_skyViewPipe       = nullptr;

    if (m_apPipe       != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_apPipe, nullptr); m_apPipe = VK_NULL_HANDLE; }
    if (m_apPipeLayout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(dev, m_apPipeLayout, nullptr); m_apPipeLayout = VK_NULL_HANDLE; }

    // Descriptor sets freed automatically when pool is destroyed
    if (m_apPool           != VK_NULL_HANDLE) { vkDestroyDescriptorPool(dev, m_apPool, nullptr); m_apPool = VK_NULL_HANDLE; }
    if (m_apWriteSetLayout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, m_apWriteSetLayout, nullptr); m_apWriteSetLayout = VK_NULL_HANDLE; }
    if (m_apReadSetLayout  != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, m_apReadSetLayout, nullptr);  m_apReadSetLayout  = VK_NULL_HANDLE; }

    destroyLut3D();

    auto destroyLut2D = [&](VkImage& img, VkImageView& view, VmaAllocation& alloc) {
        if (view != VK_NULL_HANDLE) { vkDestroyImageView(dev, view, nullptr); view = VK_NULL_HANDLE; }
        if (img  != VK_NULL_HANDLE) { vmaDestroyImage(m_allocator->handle(), img, alloc); img = VK_NULL_HANDLE; alloc = nullptr; }
    };
    destroyLut2D(m_transmittanceImg, m_transmittanceView, m_transmittanceAlloc);
    destroyLut2D(m_multiScatterImg,  m_multiScatterView,  m_multiScatterAlloc);
    destroyLut2D(m_skyViewImg,       m_skyViewView,       m_skyViewAlloc);

    ENIGMA_LOG_INFO("[atmosphere] shutdown");
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void AtmospherePass::createLut2D(VkExtent2D size, VkFormat fmt,
                                  VkImage& outImg, VkImageView& outView,
                                  VmaAllocation& outAlloc,
                                  u32& outSampledSlot, u32& outStorageSlot) {
    VkDevice dev = m_device->logical();

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = fmt;
    imgCI.extent        = {size.width, size.height, 1};
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
                                   &outImg, &outAlloc, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = outImg;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = fmt;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ENIGMA_VK_CHECK(vkCreateImageView(dev, &viewCI, nullptr, &outView));

    outSampledSlot = m_descriptorAllocator->registerSampledImage(
        outView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    outStorageSlot = m_descriptorAllocator->registerStorageImage(outView);
}

void AtmospherePass::createLut3D() {
    VkDevice dev = m_device->logical();

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_3D;
    imgCI.format        = kLutFormat3D;
    imgCI.extent        = kAerialPerspectiveSize;
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
                                   &m_apImg, &m_apAlloc, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = m_apImg;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_3D;
    viewCI.format           = kLutFormat3D;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ENIGMA_VK_CHECK(vkCreateImageView(dev, &viewCI, nullptr, &m_apView));

    // Descriptor set layout for write (STORAGE_IMAGE)
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutCI{};
        layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCI.bindingCount = 1;
        layoutCI.pBindings    = &binding;
        ENIGMA_VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutCI, nullptr,
                                                    &m_apWriteSetLayout));
    }
    // Descriptor set layout for read (SAMPLED_IMAGE)
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutCI{};
        layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCI.bindingCount = 1;
        layoutCI.pBindings    = &binding;
        ENIGMA_VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutCI, nullptr,
                                                    &m_apReadSetLayout));
    }

    // Descriptor pool for 2 sets (one write, one read)
    {
        VkDescriptorPoolSize sizes[2] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
        };
        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.maxSets       = 2;
        poolCI.poolSizeCount = 2;
        poolCI.pPoolSizes    = sizes;
        ENIGMA_VK_CHECK(vkCreateDescriptorPool(dev, &poolCI, nullptr, &m_apPool));
    }

    // Allocate write set
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = m_apPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &m_apWriteSetLayout;
        ENIGMA_VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &m_apWriteSet));

        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView   = m_apView;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = m_apWriteSet;
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo      = &imgInfo;
        vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
    }

    // Allocate read set
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = m_apPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &m_apReadSetLayout;
        ENIGMA_VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &m_apReadSet));

        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView   = m_apView;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = m_apReadSet;
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo      = &imgInfo;
        vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
    }
}

void AtmospherePass::destroyLut3D() {
    if (!m_device) return;
    VkDevice dev = m_device->logical();
    if (m_apView != VK_NULL_HANDLE) {
        vkDestroyImageView(dev, m_apView, nullptr);
        m_apView = VK_NULL_HANDLE;
    }
    if (m_apImg != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_apImg, m_apAlloc);
        m_apImg   = VK_NULL_HANDLE;
        m_apAlloc = nullptr;
    }
}

void AtmospherePass::transitionImage(VkCommandBuffer cmd, VkImage img,
                                      VkImageLayout from, VkImageLayout to,
                                      VkPipelineStageFlags2 srcStage,
                                      VkAccessFlags2 srcAccess,
                                      VkPipelineStageFlags2 dstStage,
                                      VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStage;
    barrier.srcAccessMask       = srcAccess;
    barrier.dstStageMask        = dstStage;
    barrier.dstAccessMask       = dstAccess;
    barrier.oldLayout           = from;
    barrier.newLayout           = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = img;
    barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace enigma
