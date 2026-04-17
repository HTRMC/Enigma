#include "renderer/Renderer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include <imgui.h>
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/FrameContext.h"
#include "gfx/Instance.h"
#include "gfx/GpuProfiler.h"
#include "gfx/RenderGraph.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "gfx/Swapchain.h"
#include "gfx/Validation.h"
#include "platform/Window.h"
#include "renderer/GBufferFormats.h"
#include "renderer/LightingPass.h"
#include "renderer/MeshPass.h"
#include "renderer/RTReflectionPass.h"
#include "renderer/RTGIPass.h"
#include "renderer/RTShadowPass.h"
#include "renderer/WetRoadPass.h"
#include "renderer/Denoiser.h"
#include "renderer/ClusteredForwardPass.h"
#include "renderer/GpuCullPass.h"
#include "renderer/GpuMeshletBuffer.h"
#include "renderer/GpuSceneBuffer.h"
#include "renderer/HiZPass.h"
#include "renderer/IndirectDrawBuffer.h"
#include "renderer/MaterialEvalPass.h"
#include "renderer/TrianglePass.h"
#include "renderer/UpscalerFactory.h"
#include "renderer/DebugVisualizationPass.h"
#include "renderer/VisibilityBufferPass.h"
#include "scene/Camera.h"
#include "scene/Scene.h"

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

namespace enigma {

namespace {

// Convert spherical sun angles (degrees) to a unit world-space direction
// pointing FROM the surface TO the sun (Y-up world, azimuth clockwise from Z+).
// This is the ONLY place az/el are used — all shaders receive the resulting vec3.
vec3 fromAzimuthElevation(f32 azimuthDeg, f32 elevationDeg) {
    const f32 az = glm::radians(azimuthDeg);
    const f32 el = glm::radians(elevationDeg);
    return glm::normalize(vec3{
        glm::cos(el) * glm::sin(az),
        glm::sin(el),
        glm::cos(el) * glm::cos(az)
    });
}

// Halton sequence value for a given index and base (typically 2 or 3).
// Returns a f32 in [0, 1).
f32 haltonJitter(u32 index, u32 base) {
    f32 result = 0.0f;
    f32 f = 1.0f;
    u32 i = index;
    while (i > 0) {
        f /= static_cast<f32>(base);
        result += f * static_cast<f32>(i % base);
        i /= base;
    }
    return result;
}

} // namespace

Renderer::Renderer(Window& window)
    : m_window(window)
    , m_instance(std::make_unique<gfx::Instance>())
    , m_device(std::make_unique<gfx::Device>(*m_instance))
    , m_allocator(std::make_unique<gfx::Allocator>(*m_instance, *m_device))
    , m_swapchain(std::make_unique<gfx::Swapchain>(*m_instance, *m_device, *m_allocator, m_window))
    , m_frames(std::make_unique<gfx::FrameContextSet>(*m_device))
    , m_descriptorAllocator(std::make_unique<gfx::DescriptorAllocator>(*m_device))
    , m_gpuProfiler(std::make_unique<gfx::GpuProfiler>(*m_device))
    , m_renderGraph(std::make_unique<gfx::RenderGraph>())
    , m_shaderManager(std::make_unique<gfx::ShaderManager>(*m_device))
    , m_shaderHotReload(std::make_unique<gfx::ShaderHotReload>())
    , m_trianglePass(std::make_unique<TrianglePass>(*m_device, *m_allocator, *m_descriptorAllocator))
    , m_meshPass(std::make_unique<MeshPass>(*m_device)) {

    m_trianglePass->buildPipeline(*m_shaderManager,
                                  m_descriptorAllocator->layout(),
                                  m_swapchain->format(),
                                  m_swapchain->depthFormat());
    m_trianglePass->registerHotReload(*m_shaderHotReload);

    m_meshPass->buildPipeline(*m_shaderManager,
                              m_descriptorAllocator->layout(),
                              m_swapchain->format(),
                              m_swapchain->depthFormat());
    m_meshPass->registerHotReload(*m_shaderHotReload);

    // G-buffer images owned directly by Renderer.
    createGBufferImages(m_swapchain->extent());

    // Register G-buffer textures as bindless sampled images (read path:
    // LightingPass, RT passes, Denoiser, atmosphere, post-process).
    m_gbufAlbedoSlot     = m_descriptorAllocator->registerSampledImage(
        m_gbufAlbedo.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufNormalSlot     = m_descriptorAllocator->registerSampledImage(
        m_gbufNormal.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufMetalRoughSlot = m_descriptorAllocator->registerSampledImage(
        m_gbufMetalRough.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufMotionVecSlot  = m_descriptorAllocator->registerSampledImage(
        m_gbufMotionVec.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufDepthSlot      = m_descriptorAllocator->registerSampledImage(
        m_gbufDepth.view,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Register G-buffer colour targets as bindless storage images (write path:
    // MaterialEvalPass compute shader writes PBR evaluation results via
    // imageStore()). Depth cannot be a storage image in Vulkan — no slot needed.
    m_gbufAlbedoStorageSlot     = m_descriptorAllocator->registerStorageImage(
        m_gbufAlbedo.view);
    m_gbufNormalStorageSlot     = m_descriptorAllocator->registerStorageImage(
        m_gbufNormal.view);
    m_gbufMetalRoughStorageSlot = m_descriptorAllocator->registerStorageImage(
        m_gbufMetalRough.view);
    m_gbufMotionVecStorageSlot  = m_descriptorAllocator->registerStorageImage(
        m_gbufMotionVec.view);

    // Nearest-neighbour sampler for G-buffer reads in the lighting pass.
    VkSamplerCreateInfo samplerCI{};
    samplerCI.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter    = VK_FILTER_NEAREST;
    samplerCI.minFilter    = VK_FILTER_NEAREST;
    samplerCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ENIGMA_VK_CHECK(vkCreateSampler(m_device->logical(), &samplerCI, nullptr, &m_gbufferSampler));
    m_gbufferSamplerSlot = m_descriptorAllocator->registerSampler(m_gbufferSampler);

    // Trilinear clamp sampler — used for the AP volume and atmosphere LUTs
    // so that the low-resolution (32³) froxel grid blends smoothly instead of
    // stepping at each froxel boundary and producing horizontal bands.
    VkSamplerCreateInfo linearCI{};
    linearCI.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    linearCI.magFilter    = VK_FILTER_LINEAR;
    linearCI.minFilter    = VK_FILTER_LINEAR;
    linearCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    linearCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    linearCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    linearCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ENIGMA_VK_CHECK(vkCreateSampler(m_device->logical(), &linearCI, nullptr, &m_linearSampler));
    m_linearSamplerSlot = m_descriptorAllocator->registerSampler(m_linearSampler);

    // Deferred lighting pass.
    m_lightingPass = std::make_unique<LightingPass>(*m_device);
    m_lightingPass->buildPipeline(*m_shaderManager,
                                   m_descriptorAllocator->layout(),
                                   VK_FORMAT_R16G16B16A16_SFLOAT);
    m_lightingPass->registerHotReload(*m_shaderHotReload);

    // Sky background pass — renders SkyView LUT into sky pixels after lighting.
    m_skyPass = std::make_unique<SkyBackgroundPass>(*m_device);
    m_skyPass->buildPipeline(*m_shaderManager,
                              m_descriptorAllocator->layout(),
                              VK_FORMAT_R16G16B16A16_SFLOAT);
    m_skyPass->registerHotReload(*m_shaderHotReload);

    // Atmosphere pass — must be initialized before PostProcessPass reads its AP layout.
    m_atmospherePass = std::make_unique<AtmospherePass>();
    {
        AtmospherePass::InitInfo atmInfo{};
        atmInfo.device              = m_device.get();
        atmInfo.allocator           = m_allocator.get();
        atmInfo.descriptorAllocator = m_descriptorAllocator.get();
        atmInfo.shaderManager       = m_shaderManager.get();
        atmInfo.globalSetLayout     = m_descriptorAllocator->layout();
        m_atmospherePass->init(atmInfo);
    }

    // Post-process pass — AP apply + bloom + tonemap → swapchain.
    // Uses the swapchain format since it writes the final display-ready image.
    m_postProcessPass = std::make_unique<PostProcessPass>(*m_device);
    m_postProcessPass->buildPipeline(*m_shaderManager,
                                      m_descriptorAllocator->layout(),
                                      m_atmospherePass->aerialPerspectiveReadSetLayout(),
                                      m_swapchain->format());
    m_postProcessPass->registerHotReload(*m_shaderHotReload);

    // SMAA anti-aliasing pass — allocates edge + weight textures, builds pipelines.
    m_smaaPass = std::make_unique<SMAAPass>(*m_device, *m_allocator);
    m_smaaPass->allocate(m_swapchain->extent(), *m_descriptorAllocator);
    m_smaaPass->buildPipelines(*m_shaderManager,
                                m_descriptorAllocator->layout(),
                                m_swapchain->format());
    m_smaaPass->registerHotReload(*m_shaderHotReload);

    // RT reflection pass.
    m_rtReflectionPass = std::make_unique<RTReflectionPass>(*m_device, *m_allocator);
    m_rtReflectionPass->allocate(m_swapchain->extent());
    m_rtReflectionPass->buildPipeline(*m_shaderManager,
                                       m_descriptorAllocator->layout(),
                                       RTReflectionPass::kOutputFormat);
    m_rtReflectionPass->registerHotReload(*m_shaderHotReload);

    // Register the reflection output as a bindless storage image.
    m_reflectionSlot = m_descriptorAllocator->registerStorageImage(m_rtReflectionPass->outputView());
    m_rtReflectionPass->outputSlot = m_reflectionSlot;

    // RT GI pass.
    m_giPass = std::make_unique<RTGIPass>(*m_device, *m_allocator);
    m_giPass->allocate(m_swapchain->extent());
    m_giPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_giPass->registerHotReload(*m_shaderHotReload);
    m_giSlot = m_descriptorAllocator->registerStorageImage(m_giPass->outputView());
    m_giPass->outputSlot = m_giSlot;

    // RT Shadow pass.
    m_shadowPass = std::make_unique<RTShadowPass>(*m_device, *m_allocator);
    m_shadowPass->allocate(m_swapchain->extent());
    m_shadowPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_shadowPass->registerHotReload(*m_shaderHotReload);
    m_shadowSlot = m_descriptorAllocator->registerStorageImage(m_shadowPass->outputView());
    m_shadowPass->outputSlot = m_shadowSlot;

    // Wet Road pass.
    m_wetRoadPass = std::make_unique<WetRoadPass>(*m_device, *m_allocator);
    m_wetRoadPass->allocate(m_swapchain->extent());
    m_wetRoadPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_wetRoadPass->registerHotReload(*m_shaderHotReload);
    m_wetRoadSlot = m_descriptorAllocator->registerStorageImage(m_wetRoadPass->outputView());
    m_wetRoadPass->outputSlot = m_wetRoadSlot;

    // Denoisers (one per RT effect).
    m_giDenoiser = std::make_unique<Denoiser>(*m_device, *m_allocator);
    m_giDenoiser->allocate(m_swapchain->extent(), RTGIPass::kOutputFormat);
    m_giDenoiser->buildPipelines(*m_shaderManager, m_descriptorAllocator->layout(), RTGIPass::kOutputFormat);
    m_giDenoiser->registerHotReload(*m_shaderHotReload);
    m_giDenoiseSlot = m_descriptorAllocator->registerStorageImage(m_giDenoiser->outputView());
    m_giDenoiser->outputSlot = m_giDenoiseSlot;

    m_shadowDenoiser = std::make_unique<Denoiser>(*m_device, *m_allocator);
    m_shadowDenoiser->allocate(m_swapchain->extent(), RTShadowPass::kOutputFormat);
    m_shadowDenoiser->buildPipelines(*m_shaderManager, m_descriptorAllocator->layout(), RTShadowPass::kOutputFormat);
    m_shadowDenoiser->registerHotReload(*m_shaderHotReload);
    m_shadowDenoiseSlot = m_descriptorAllocator->registerStorageImage(m_shadowDenoiser->outputView());
    m_shadowDenoiser->outputSlot = m_shadowDenoiseSlot;

    m_reflectionDenoiser = std::make_unique<Denoiser>(*m_device, *m_allocator);
    m_reflectionDenoiser->allocate(m_swapchain->extent(), RTReflectionPass::kOutputFormat);
    m_reflectionDenoiser->buildPipelines(*m_shaderManager, m_descriptorAllocator->layout(), RTReflectionPass::kOutputFormat);
    m_reflectionDenoiser->registerHotReload(*m_shaderHotReload);
    m_reflDenoiseSlot = m_descriptorAllocator->registerStorageImage(m_reflectionDenoiser->outputView());
    m_reflectionDenoiser->outputSlot = m_reflDenoiseSlot;

    // Visibility buffer pipeline — GPU-driven mesh-shader path.
    m_gpuScene       = std::make_unique<GpuSceneBuffer>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_gpuMeshlets    = std::make_unique<GpuMeshletBuffer>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_indirectBuffer = std::make_unique<IndirectDrawBuffer>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_indirectBuffer->resize(65536); // reserve for up to 64K meshlets
    m_terrainWireIndirectBuffer = std::make_unique<IndirectDrawBuffer>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_terrainWireIndirectBuffer->resize(65536);

    m_hizPass = std::make_unique<HiZPass>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_hizPass->allocate(m_swapchain->extent());
    m_hizPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_hizPass->registerHotReload(*m_shaderHotReload);

    m_gpuCullPass = std::make_unique<GpuCullPass>(*m_device);
    m_gpuCullPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_gpuCullPass->registerHotReload(*m_shaderHotReload);

    m_visibilityPass = std::make_unique<VisibilityBufferPass>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_visibilityPass->allocate(m_swapchain->extent(), gbuf::kDepthFormat);
    m_visibilityPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_visibilityPass->buildTerrainPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_visibilityPass->registerHotReload(*m_shaderHotReload);

    // Wireframe pipeline (conditional on fillModeNonSolid support).
    if (m_device->fillModeNonSolidSupported() && m_visibilityPass) {
        m_visibilityPass->buildWireframePipeline(*m_shaderManager,
                                                  m_descriptorAllocator->layout(),
                                                  m_swapchain->format());
        m_visibilityPass->buildTerrainWireframePipeline(*m_shaderManager,
                                                         m_descriptorAllocator->layout(),
                                                         m_swapchain->format());
    }

    // Debug visualization pass — fullscreen debug modes.
    m_debugVisPass = std::make_unique<DebugVisualizationPass>(*m_device);
    m_debugVisPass->buildPipelines(*m_shaderManager,
                                    m_descriptorAllocator->layout(),
                                    m_swapchain->format());
    m_debugVisPass->registerHotReload(*m_shaderHotReload);

    m_materialEvalPass = std::make_unique<MaterialEvalPass>(*m_device);
    m_materialEvalPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_materialEvalPass->registerHotReload(*m_shaderHotReload);
    m_materialEvalPass->prepare(
        m_swapchain->extent(),
        m_gbufAlbedo.image,     m_gbufNormal.image,
        m_gbufMetalRough.image, m_gbufMotionVec.image,
        m_gbufDepth.image,
        m_gbufAlbedoStorageSlot,     m_gbufNormalStorageSlot,
        m_gbufMetalRoughStorageSlot, m_gbufMotionVecStorageSlot,
        m_gbufDepthSlot);

    // Clustered forward pass — renders transparent geometry after opaque lighting.
    m_clusteredForwardPass = std::make_unique<ClusteredForwardPass>(*m_device);
    m_clusteredForwardPass->buildPipeline(*m_shaderManager,
                                          m_descriptorAllocator->layout(),
                                          VK_FORMAT_R16G16B16A16_SFLOAT);
    m_clusteredForwardPass->registerHotReload(*m_shaderHotReload);

    // Create per-frame camera SSBOs (host-visible, persistently mapped).
    // 13 float4s = 208 bytes per camera buffer.
    constexpr VkDeviceSize kCameraBufferSize = sizeof(GpuCameraData);
    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size        = kCameraBufferSize;
        bufInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                        | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo allocResult{};
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufInfo, &allocInfo,
                                        &m_cameraBuffers[i].buffer,
                                        &m_cameraBuffers[i].allocation,
                                        &allocResult));
        m_cameraBuffers[i].mapped = allocResult.pMappedData;
        ENIGMA_ASSERT(m_cameraBuffers[i].mapped != nullptr);

        m_cameraBuffers[i].bindlessSlot =
            m_descriptorAllocator->registerStorageBuffer(
                m_cameraBuffers[i].buffer, kCameraBufferSize);
    }

    vkGetPhysicalDeviceMemoryProperties(m_device->physical(), &m_memoryProperties);

    ENIGMA_LOG_INFO("[renderer] constructed (camera slots: {}, {})",
                    m_cameraBuffers[0].bindlessSlot,
                    m_cameraBuffers[1].bindlessSlot);

    // Upscaler — auto-select based on vendor/tier, init at swapchain display extent.
    m_upscaler = UpscalerFactory::create(m_device->properties(), m_device->gpuTier());
    m_upscaler->init(m_device->logical(), m_device->physical(),
                     m_swapchain->extent(), m_upscalerSettings.quality);

    // ImGui overlay — must be created after swapchain (needs format + imageCount).
    m_imguiLayer = std::make_unique<gfx::ImGuiLayer>(
        *m_device,
        m_instance->handle(),
        m_window.handle(),
        m_swapchain->format(),
        m_swapchain->imageCount());

    // Physics debug wireframe renderer.
    PhysicsDebugInitInfo dbgInfo{};
    dbgInfo.device              = m_device.get();
    dbgInfo.allocator           = m_allocator.get();
    dbgInfo.descriptorAllocator = m_descriptorAllocator.get();
    dbgInfo.shaderManager       = m_shaderManager.get();
    dbgInfo.globalSetLayout     = m_descriptorAllocator->layout();
    dbgInfo.colorFormat         = VK_FORMAT_R16G16B16A16_SFLOAT;
    dbgInfo.depthFormat         = gbuf::kDepthFormat;
    m_physicsDebugRenderer.init(dbgInfo);

    createHdrColor(m_swapchain->extent());
    createSmaaLdr(m_swapchain->extent());

    // Bake Transmittance + MultiScatter LUTs on startup (one-shot command buffer).
    {
        VkDevice dev = m_device->logical();

        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCI.queueFamilyIndex = m_device->graphicsQueueFamily();
        ENIGMA_VK_CHECK(vkCreateCommandPool(dev, &poolCI, nullptr, &pool));

        VkCommandBuffer bakeCmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cbInfo{};
        cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbInfo.commandPool        = pool;
        cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbInfo.commandBufferCount = 1;
        ENIGMA_VK_CHECK(vkAllocateCommandBuffers(dev, &cbInfo, &bakeCmd));

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ENIGMA_VK_CHECK(vkBeginCommandBuffer(bakeCmd, &beginInfo));

        m_atmospherePass->bakeStaticLUTs(bakeCmd, m_atmosphereSettings,
                                          m_sunWorldDir, m_gbufferSamplerSlot);

        ENIGMA_VK_CHECK(vkEndCommandBuffer(bakeCmd));

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        ENIGMA_VK_CHECK(vkCreateFence(dev, &fenceCI, nullptr, &fence));

        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &bakeCmd;
        ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, fence));
        ENIGMA_VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(dev, fence, nullptr);
        vkDestroyCommandPool(dev, pool, nullptr); // frees bakeCmd implicitly

        m_sunDirty = false;
        ENIGMA_LOG_INFO("[renderer] initial atmosphere LUT bake complete");
    }
}

Renderer::~Renderer() {
    if (m_device) {
        vkDeviceWaitIdle(m_device->logical());
    }

    // CDLOD terrain + heightmap own VMA buffers/images and bindless slots —
    // destroy before VMA / DescriptorAllocator teardown.
    m_terrain.reset();
    m_heightmapLoader.reset();

    // Shut down ImGui before destroying Vulkan resources.
    m_imguiLayer.reset();

    // Shut down the upscaler before device destruction.
    if (m_upscaler) {
        m_upscaler->shutdown();
    }

    // Clean up samplers (pass objects destroy their own images/pipelines).
    if (m_gbufferSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_gbufferSampler, nullptr);
    }
    if (m_linearSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_linearSampler, nullptr);
    }
    // Renderer-owned runtime-rebuilt material sampler. The original sampler
    // created by GltfLoader still lives in scene.ownedSamplers and is freed
    // by Scene::destroy — we only own the replacement created by
    // applyTextureFilterSettings().
    if (m_materialSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_materialSampler, nullptr);
        m_materialSampler = VK_NULL_HANDLE;
    }

    // Atmosphere pass owns LUT images + pipelines.
    if (m_atmospherePass) {
        m_atmospherePass->shutdown();
    }

    // SMAA intermediate textures — free before SMAAPass is destroyed.
    if (m_smaaPass && m_smaaPass->isAllocated()) {
        m_smaaPass->free(*m_descriptorAllocator);
    }
    destroySmaaLdr();

    // HDR intermediate — destroyed before VMA teardown.
    destroyHdrColor();
    destroyGBufferImages();

    // Physics debug renderer owns a VMA buffer — must be destroyed before VMA teardown.
    m_physicsDebugRenderer.destroy();

    // Clean up camera buffers.
    for (auto& cb : m_cameraBuffers) {
        if (cb.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), cb.buffer, cb.allocation);
        }
    }

    const u32 validationCount = gfx::getValidationCounter();
    ENIGMA_LOG_INFO("[renderer] shutdown g_validationCounter = {}", validationCount);
    ENIGMA_ASSERT(validationCount == 0);

    ENIGMA_LOG_INFO("[renderer] shutdown");
}

void Renderer::uploadCameraData() {
    const auto& cb = m_cameraBuffers[m_frameIndex];
    const auto extent = m_swapchain->extent();
    const f32 aspect = (extent.height > 0)
        ? static_cast<f32>(extent.width) / static_cast<f32>(extent.height)
        : 1.0f;

    GpuCameraData data{};
    if (m_camera != nullptr) {
        data = m_camera->gpuData(aspect);
        // Cache camera basis vectors and FOV tangents for the AP bake shader.
        m_cameraRight   = m_camera->right();
        m_cameraUp      = m_camera->up();
        m_cameraForward = m_camera->forward();
        m_tanHalfFovY   = std::tan(m_camera->fovY * 0.5f);
        m_tanHalfFovX   = m_tanHalfFovY * aspect;
    } else {
        // Identity camera fallback.
        data.view        = mat4{1.0f};
        data.proj        = mat4{1.0f};
        data.viewProj    = mat4{1.0f};
        data.invViewProj = mat4{1.0f};
        data.worldPos    = vec4{0.0f, 0.0f, 0.0f, 1.0f};
        // Basis vectors stay at their initialized defaults.
    }

    // Inject previous frame's viewProj for motion vector computation.
    data.prevViewProj = m_prevViewProj;
    std::memcpy(cb.mapped, &data, sizeof(data));
    vmaFlushAllocation(m_allocator->handle(), cb.allocation, 0, sizeof(data));

    // Save current viewProj so next frame can use it as prevViewProj.
    m_prevViewProj    = data.viewProj;
    m_invViewProj     = data.invViewProj;
    m_cameraWorldPos  = vec3(data.worldPos);
}

void Renderer::drawFrame() {
    {
        const auto fb = m_window.framebufferSize();
        if (fb.width == 0 || fb.height == 0) {
            m_window.waitEvents();
            return;
        }
    }

    // CPU frame time.
    {
        const auto now = std::chrono::steady_clock::now();
        if (m_lastFrameTime.time_since_epoch().count() != 0) {
            m_cpuFrameTimeMs = std::chrono::duration<f32, std::milli>(now - m_lastFrameTime).count();
        }
        m_lastFrameTime = now;
    }

    m_shaderHotReload->poll();

    // Compute canonical sun direction once per frame from UI az/el.
    // This is the single source of truth fanned to all pass push constants.
    {
        const vec3 newDir = fromAzimuthElevation(
            m_atmosphereSettings.sunAzimuth,
            m_atmosphereSettings.sunElevation);
        if (glm::any(glm::notEqual(newDir, m_sunWorldDir))) {
            m_sunWorldDir = newDir;
            m_sunDirty    = true;
        }
    }

    // When the sun direction changes, rebake the static LUTs (Transmittance +
    // MultiScatter). These are slow to compute and only change when the sun
    // moves, so we idle the device and rebuild before opening this frame's
    // command buffer. SkyView + AP are rebuilt every frame in updatePerFrame.
    if (m_sunDirty && m_atmospherePass) {
        vkDeviceWaitIdle(m_device->logical());

        VkCommandPool bakePool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo bakePoolCI{};
        bakePoolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        bakePoolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        bakePoolCI.queueFamilyIndex = m_device->graphicsQueueFamily();
        ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &bakePoolCI, nullptr, &bakePool));

        VkCommandBuffer bakeCmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo bakeCbInfo{};
        bakeCbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        bakeCbInfo.commandPool        = bakePool;
        bakeCbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        bakeCbInfo.commandBufferCount = 1;
        ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &bakeCbInfo, &bakeCmd));

        VkCommandBufferBeginInfo bakeBeginInfo{};
        bakeBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bakeBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ENIGMA_VK_CHECK(vkBeginCommandBuffer(bakeCmd, &bakeBeginInfo));

        m_atmospherePass->bakeStaticLUTs(bakeCmd, m_atmosphereSettings,
                                          m_sunWorldDir, m_gbufferSamplerSlot);

        ENIGMA_VK_CHECK(vkEndCommandBuffer(bakeCmd));

        VkFence bakeFence = VK_NULL_HANDLE;
        VkFenceCreateInfo bakeFenceCI{};
        bakeFenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        ENIGMA_VK_CHECK(vkCreateFence(m_device->logical(), &bakeFenceCI, nullptr, &bakeFence));

        VkSubmitInfo bakeSubmit{};
        bakeSubmit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        bakeSubmit.commandBufferCount = 1;
        bakeSubmit.pCommandBuffers    = &bakeCmd;
        ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &bakeSubmit, bakeFence));
        ENIGMA_VK_CHECK(vkWaitForFences(m_device->logical(), 1, &bakeFence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(m_device->logical(), bakeFence, nullptr);
        vkDestroyCommandPool(m_device->logical(), bakePool, nullptr); // frees bakeCmd

        m_sunDirty = false;
    }

    VkDevice dev = m_device->logical();
    gfx::FrameContext& frame = m_frames->get(m_frameIndex);

    if (frame.frameValue > 0) {
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores    = &frame.inFlight;
        waitInfo.pValues        = &frame.frameValue;
        ENIGMA_VK_CHECK(vkWaitSemaphores(dev, &waitInfo, UINT64_MAX));
    }

    u32 imageIndex = 0;
    {
        const VkResult acquireResult = vkAcquireNextImageKHR(
            dev, m_swapchain->handle(), UINT64_MAX,
            frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            const auto [w, h] = m_window.framebufferSize();
            m_swapchain->recreate(w, h);
            resizeGBuffer(m_swapchain->extent());
            return;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            ENIGMA_VK_CHECK(acquireResult);
        }
    }

    // Upload camera data for this frame.
    uploadCameraData();

    // Process pending deformation: upload deformed vertices and update BLASes.
    if (m_deformationPending && m_scene != nullptr) {
        constexpr VkDeviceSize kVertexStride = 3 * sizeof(vec4); // 48 bytes

        for (u32 i = 0; i < static_cast<u32>(m_scene->primitives.size()); ++i) {
            auto& prim = m_scene->primitives[i];
            if (!prim.blas.has_value()) continue;

            // uploadDeformedPositions no-ops for unregistered primitives.
            m_deformationSystem.uploadDeformedPositions(i, prim.vertexBuffer, *m_device, *m_allocator);

            if (m_deformationSystem.requiresBlasRebuild(i)) {
                prim.blas->rebuild(*m_device, *m_allocator,
                                   prim.vertexBuffer, prim.vertexCount, kVertexStride,
                                   prim.indexBuffer, prim.indexCount);
            } else {
                prim.blas->refit(*m_device, *m_allocator,
                                 prim.vertexBuffer, prim.vertexCount, kVertexStride,
                                 prim.indexBuffer, prim.indexCount);
            }
        }

        m_deformationPending = false;
    }

    ENIGMA_VK_CHECK(vkResetCommandPool(dev, frame.commandPool, 0));

    // Read back GPU timings from the previous frame before resetting the pool.
    // (The previous frame's submissions are guaranteed done by the timeline wait above.)
    if (frame.frameValue > 0) {
        m_lastGpuTimings = m_gpuProfiler->readback();
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo));

    m_gpuProfiler->reset(frame.commandBuffer);

    // Per-frame atmosphere LUT update: SkyView + Aerial Perspective volume.
    // Recorded directly into the main command buffer before the render graph
    // executes so RT passes, sky background, lighting, and post-process all
    // read fresh LUTs this frame. Camera position converted from world-units
    // to km (engine uses metres: 1 world unit = 1 m → / 1000).
    if (m_atmospherePass) {
        m_atmospherePass->updatePerFrame(
            frame.commandBuffer,
            m_atmosphereSettings,
            m_sunWorldDir,
            m_cameraWorldPos * 0.001f, // metres → km
            m_cameraRight,
            m_cameraUp,
            m_cameraForward,
            m_tanHalfFovX,
            m_tanHalfFovY,
            m_gbufferSamplerSlot);
    }

    // Debug mode hotkeys (F1-F6).
    // ImGui::IsKeyPressed works even outside ImGui window scope.
    if (ImGui::IsKeyPressed(ImGuiKey_F1)) m_debugMode = DebugMode::Lit;
    if (ImGui::IsKeyPressed(ImGuiKey_F2)) m_debugMode = DebugMode::Unlit;
    if (ImGui::IsKeyPressed(ImGuiKey_F3) &&
        m_device->fillModeNonSolidSupported() &&
        m_visibilityPass && m_visibilityPass->hasWireframePipeline())
        m_debugMode = DebugMode::Wireframe;
    if (ImGui::IsKeyPressed(ImGuiKey_F4) &&
        m_device->fillModeNonSolidSupported() &&
        m_visibilityPass && m_visibilityPass->hasWireframePipeline())
        m_debugMode = DebugMode::LitWireframe;
    if (ImGui::IsKeyPressed(ImGuiKey_F6)) m_debugMode = DebugMode::DetailLighting;
    if (ImGui::IsKeyPressed(ImGuiKey_F7))
        m_debugMode = DebugMode::Clusters;

    // ImGui new frame -- must be called before any ImGui:: window calls this frame.
    if (m_imguiLayer) {
        m_imguiLayer->newFrame();

        // Query device-local VRAM budget for the perf panel.
        f32 vramUsedMb = 0.f, vramBudgetMb = 0.f;
        {
            std::vector<VmaBudget> budgets(m_memoryProperties.memoryHeapCount);
            vmaGetHeapBudgets(m_allocator->handle(), budgets.data());
            for (u32 i = 0; i < m_memoryProperties.memoryHeapCount; ++i) {
                if (m_memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    vramUsedMb   += static_cast<f32>(budgets[i].usage)  / (1024.f * 1024.f);
                    vramBudgetMb += static_cast<f32>(budgets[i].budget) / (1024.f * 1024.f);
                }
            }
        }

        m_imguiLayer->drawGpuTimings(m_lastGpuTimings, m_cpuFrameTimeMs, vramUsedMb, vramBudgetMb);
        m_imguiLayer->drawSceneInfo(
            m_scene ? static_cast<u32>(m_scene->primitives.size()) : 0u,
            m_tlasSlot > 0 ? 1u : 0u);
        m_imguiLayer->drawUpscalerSettings(m_upscalerSettings);
        m_imguiLayer->drawPhysicsStats(0.0f, 0u); // wired up later when physics is exposed
        m_imguiLayer->drawPhysicsDebugPanel(m_physicsDebugRenderer.enabled,
                                             m_physicsDebugRenderer.depthTestEnabled);

        // Settings panel — sun light + scene knobs.
        ImGui::SetNextWindowPos({310.f, 10.f}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({300.f, 260.f}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Settings")) {
            if (ImGui::CollapsingHeader("Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Sun Azimuth",   &m_atmosphereSettings.sunAzimuth,   0.f,   360.f, "%.1f deg");
                ImGui::SliderFloat("Sun Elevation", &m_atmosphereSettings.sunElevation, -10.f,  90.f, "%.1f deg");
                ImGui::SliderFloat("Sun Intensity", &m_atmosphereSettings.sunIntensity,  0.f,   20.f);
                ImGui::Separator();
                ImGui::SliderFloat("Exposure EV",        &m_atmosphereSettings.exposureEV,      -6.f, 6.f, "%.2f EV");
                ImGui::SliderFloat("Bloom Threshold",    &m_atmosphereSettings.bloomThreshold,   0.f, 4.f);
                ImGui::SliderFloat("Bloom Intensity",    &m_atmosphereSettings.bloomIntensity,   0.f, 2.f);
                ImGui::Checkbox("Bloom",              &m_atmosphereSettings.bloomEnabled);
                ImGui::SameLine();
                ImGui::Checkbox("Aerial Perspective", &m_atmosphereSettings.aerialPerspectiveEnabled);
                if (m_atmosphereSettings.aerialPerspectiveEnabled)
                    ImGui::SliderFloat("AP Strength", &m_atmosphereSettings.aerialPerspectiveStrength, 0.0f, 1.0f);
                const char* tonemapItems[] = {"AgX", "ACES"};
                ImGui::Combo("Tonemap", &m_atmosphereSettings.tonemapMode, tonemapItems, 2);
            }
            if (ImGui::CollapsingHeader("Sun Light")) {
                ImGui::ColorEdit3 ("Color",    &m_light.color.x);
            }
            if (ImGui::CollapsingHeader("Environment")) {
                ImGui::SliderFloat("Wetness", &m_wetnessFactor, 0.f, 1.f);
            }
            if (ImGui::CollapsingHeader("Renderer")) {
                ImGui::Checkbox("SMAA", &m_aaSettings.smaaEnabled);

                ImGui::SeparatorText("Texture Filtering");
                bool tfDirty = false;
                tfDirty |= ImGui::Checkbox("Mipmaps", &m_textureFilterSettings.mipmapsEnabled);
                static const char* kAnisoLabels[] = { "1x (off)", "2x", "4x", "8x", "16x" };
                static const u32   kAnisoValues[] = { 1u, 2u, 4u, 8u, 16u };
                int anisoIdx = 0;
                for (int i = 0; i < 5; ++i) {
                    if (kAnisoValues[i] == m_textureFilterSettings.anisotropy) { anisoIdx = i; break; }
                }
                const f32 devMaxAniso = m_device->properties().limits.maxSamplerAnisotropy;
                const bool anisoSupported = devMaxAniso > 1.0f;
                if (!anisoSupported) ImGui::BeginDisabled();
                if (ImGui::Combo("Anisotropic Filtering", &anisoIdx, kAnisoLabels, 5)) {
                    m_textureFilterSettings.anisotropy = kAnisoValues[anisoIdx];
                    tfDirty = true;
                }
                if (!anisoSupported) ImGui::EndDisabled();
                if (tfDirty) applyTextureFilterSettings();
                ImGui::Separator();
            }
            if (ImGui::CollapsingHeader("Graphics Quality")) {
                const bool rtAvailable = m_device->gpuTier() >= gfx::GpuTier::Recommended;
                if (!rtAvailable) {
                    ImGui::TextDisabled("RT effects require Recommended tier GPU");
                    ImGui::BeginDisabled();
                }
                ImGui::Checkbox("RT Reflections", &m_rtReflectionsEnabled);
                ImGui::Checkbox("RT Global Illumination", &m_rtGIEnabled);
                ImGui::Checkbox("RT Shadows", &m_rtShadowsEnabled);
                ImGui::Checkbox("Wet Road Effect", &m_wetRoadEnabled);
                ImGui::Checkbox("Denoising", &m_denoiseEnabled);
                if (!rtAvailable) {
                    ImGui::EndDisabled();
                }
            }
        }
        ImGui::End();

        // Debug Views panel.
        const bool wireAvail  = m_device->fillModeNonSolidSupported() &&
                                m_visibilityPass && m_visibilityPass->hasWireframePipeline();
        const bool clustAvail = true;

        ImGui::SetNextWindowPos({620.f, 10.f}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({280.f, 140.f}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Debug Views")) {
            static const char* kModeNames[] = {
                "Lit", "Unlit", "Wireframe", "Lit Wireframe", "Detail Lighting", "Clusters"
            };
            int modeInt = static_cast<int>(m_debugMode);
            if (ImGui::Combo("Mode", &modeInt, kModeNames, 6)) {
                DebugMode selected = static_cast<DebugMode>(modeInt);
                if (!wireAvail && (selected == DebugMode::Wireframe || selected == DebugMode::LitWireframe))
                    selected = DebugMode::Lit;
                if (!clustAvail && selected == DebugMode::Clusters)
                    selected = DebugMode::Lit;
                m_debugMode = selected;
            }
            if (m_debugMode == DebugMode::Wireframe || m_debugMode == DebugMode::LitWireframe)
                ImGui::ColorEdit3("Wire Color", &m_wireframeColor.x);
            ImGui::Separator();
            ImGui::TextDisabled("F1=Lit F2=Unlit F3=Wire F4=LitWire F6=Detail F7=Cluster");
            if (!wireAvail)
                ImGui::TextDisabled("Wireframe: requires mesh shaders + fillModeNonSolid");
        }
        ImGui::End();
    }

    VkImage     targetImage = m_swapchain->image(imageIndex);
    VkImageView targetView  = m_swapchain->view(imageIndex);
    VkImage     depthImage  = m_swapchain->depthImage();
    VkImageView depthView   = m_swapchain->depthView();
    const VkExtent2D extent = m_swapchain->extent();

    // Build the render graph for this frame. All barriers and attachment
    // setup are handled by the graph; drawFrame() holds no hand-coded barriers.
    m_renderGraph->reset();

    const auto colorHandle = m_renderGraph->importImage(
        "swapchain_color",
        targetImage, targetView, m_swapchain->format(),
        VK_IMAGE_LAYOUT_UNDEFINED,       // starts undefined each frame
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, // must end ready for present
        VK_IMAGE_ASPECT_COLOR_BIT);

    // HDR intermediate — only needed for Lit and LitWireframe modes.
    const bool needsHdr = (m_debugMode == DebugMode::Lit || m_debugMode == DebugMode::LitWireframe);
    gfx::RGImageHandle hdrColorHandle{};
    if (needsHdr) {
        hdrColorHandle = m_renderGraph->importImage(
            "hdr_color",
            m_hdrColor, m_hdrColorView, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED,                  // re-written every frame
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,   // upscaler reads as SRV
            VK_IMAGE_ASPECT_COLOR_BIT);
    }

    const u32 cameraSlot = m_cameraBuffers[m_frameIndex].bindlessSlot;

    // ---- Visibility buffer pipeline (direct cmd recording, before render graph) ----
    if (m_scene != nullptr && m_visibilityPass && m_materialEvalPass) {
        m_gpuScene->begin_frame();
        for (const auto& node : m_scene->nodes) {
            for (u32 primIdx : node.primitiveIndices) {
                const auto& prim = m_scene->primitives[primIdx];
                if (prim.meshletOffset == UINT32_MAX) { ENIGMA_ASSERT(false && "primitive missing meshlet data — asset pipeline bug"); continue; }
                GpuInstance inst{};
                inst.transform          = node.worldTransform;
                inst.meshlet_offset     = prim.meshletOffset;
                inst.meshlet_count      = static_cast<u32>(prim.meshlets.meshlets.size());
                inst.material_index     = prim.materialIndex < 0 ? 0u : static_cast<u32>(prim.materialIndex);
                inst.vertex_buffer_slot = prim.vertexBufferSlot;
                inst.vertex_base_offset = 0; // Non-terrain meshes index from vertex 0; CDLOD terrain overrides.
                // inst._pad is zero-initialized by GpuInstance inst{}.
                m_gpuScene->add_instance(inst);
            }
        }
        // Remember the split point: scene instances are [0, sceneInstanceCount),
        // terrain instances (appended below) live at [sceneInstanceCount, total).
        // Used by the two-pass cull + VB draw sequence further down so scene
        // meshlets go through visibility_buffer.mesh.hlsl and terrain meshlets
        // go through terrain_cdlod.mesh.hlsl.
        const u32 sceneInstanceCount  = static_cast<u32>(m_gpuScene->instance_count());
        // Use scene_meshlet_count() — frozen at upload()/reserveCapacity() time —
        // not total_meshlet_count(), which grows as terrain patches activate and
        // would pull terrain-meshlet indices into the scene cull range, causing
        // findInstanceAndLocal to fall through to instanceId=0 → corruption.
        const u32 sceneMeshletCount   = m_gpuMeshlets->scene_meshlet_count();

        // Per-frame CDLOD terrain traversal — activates/deactivates patches,
        // emits a GpuInstance per active patch into the scene buffer, and
        // records incremental vertex + meshlet uploads on this frame's cmd.
        if (m_terrain != nullptr && m_terrain->isEnabled()) {
            m_terrain->update(m_cameraWorldPos, frame.commandBuffer,
                              m_frameIndex, *m_gpuScene);
        }
        const u32 totalInstanceCount  = static_cast<u32>(m_gpuScene->instance_count());
        const u32 totalMeshletCount   = m_gpuMeshlets->total_meshlet_count();
        const u32 terrainInstanceCount = totalInstanceCount - sceneInstanceCount;
        const u32 terrainMeshletCount  = totalMeshletCount - sceneMeshletCount;

        m_gpuScene->upload(frame.commandBuffer, m_frameIndex, totalMeshletCount);

        m_indirectBuffer->reset_count(frame.commandBuffer);

        // Barrier: all TRANSFER writes (scene SSBO upload + indirect count reset) →
        //   DRAW_INDIRECT  (count buffer read by vkCmdDrawMeshTasksEXT — unused now but harmless)
        //   COMPUTE_SHADER (cull pass reads scene SSBO)
        //   TASK_SHADER    (task shader reads scene SSBO in findInstanceAndLocal)
        //   MESH_SHADER    (mesh shader reads scene SSBO in loadInstance for transform +
        //                   vertex_buffer_slot; task→mesh ordering only propagates task writes,
        //                   not third-party TRANSFER writes, so explicit coverage is required)
        // Must be unconditional — the reset happens even if there are no meshlets.
        {
            VkMemoryBarrier2 transferBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            transferBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            transferBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            transferBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            transferBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                          | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                          | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                                          | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
            transferBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                                          | VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo transferDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            transferDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            transferDep.memoryBarrierCount = 1;
            transferDep.pMemoryBarriers    = &transferBarrier;
            vkCmdPipelineBarrier2(frame.commandBuffer, &transferDep);
        }

        if (totalMeshletCount > 0) {
            // Helper: compute→task/mesh barrier on the surviving-IDs +
            // count buffer so the task shader sees the cull output.
            auto emitCullToTaskBarrier = [&]() {
                VkMemoryBarrier2 cullBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                cullBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                cullBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                cullBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                cullBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
                cullBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo cullDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                cullDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                cullDep.memoryBarrierCount = 1;
                cullDep.pMemoryBarriers    = &cullBarrier;
                vkCmdPipelineBarrier2(frame.commandBuffer, &cullDep);
            };

            bool terrainRan = false;

            // ---------- Pass B: Terrain meshlets (runs FIRST — clears vis+depth) ----------
            // Terrain is drawn before scene so the scene pass (LOAD) renders on top
            // and always wins at coplanar depth. recordTerrain() CLEARs both images.
            // Skip in pure Wireframe mode so scene survivors remain in the indirect
            // buffer for the wireframe overlay.
            if (m_terrain != nullptr && m_terrain->isEnabled()
                && terrainMeshletCount > 0 && terrainInstanceCount > 0
                && m_visibilityPass->hasTerrainPipeline()
                && m_debugMode != DebugMode::Wireframe)
            {
                // Initial reset_count + transfer barrier above already ran.
                m_gpuProfiler->beginZone(frame.commandBuffer, "CullTerrain");
                m_gpuCullPass->record(frame.commandBuffer,
                                       m_descriptorAllocator->globalSet(),
                                       *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer,
                                       cameraSlot,
                                       sceneInstanceCount, terrainInstanceCount,
                                       sceneMeshletCount,  terrainMeshletCount);
                emitCullToTaskBarrier();
                m_gpuProfiler->endZone(frame.commandBuffer);

                const auto terrainTopo = m_terrain->sharedTopologyHandle();
                m_gpuProfiler->beginZone(frame.commandBuffer, "DrawTerrainVB");
                m_visibilityPass->recordTerrain(
                    frame.commandBuffer, m_descriptorAllocator->globalSet(), extent,
                    m_gbufDepth.view, m_gbufDepth.image,
                    *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer, cameraSlot,
                    terrainTopo.topologyVerticesSlot,
                    terrainTopo.topologyTrianglesSlot,
                    terrainMeshletCount);
                m_gpuProfiler->endZone(frame.commandBuffer);

                terrainRan = true;
            }

            // ---------- Pass A: Scene meshlets (runs SECOND — LOADs vis+depth) ----------
            // When terrainRan==true the images are in SHADER_READ_ONLY_OPTIMAL after
            // recordTerrain()'s post-barriers; record(clearFirst=false) transitions
            // them back to COLOR_ATTACHMENT with LOAD so scene fragments overwrite
            // terrain at equal or greater depth (GREATER_OR_EQUAL, reverse-Z).
            // When terrainRan==false (Wireframe or no terrain), clearFirst=true
            // initialises both images from UNDEFINED with CLEAR.
            if (sceneMeshletCount > 0 && sceneInstanceCount > 0) {
                if (terrainRan) {
                    // Terrain task/mesh reads → transfer+compute writes for the reset.
                    VkMemoryBarrier2 readToWrite{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    readToWrite.srcStageMask  = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                                              | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
                    readToWrite.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    readToWrite.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT
                                              | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    readToWrite.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT
                                              | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers    = &readToWrite;
                    vkCmdPipelineBarrier2(frame.commandBuffer, &dep);

                    m_indirectBuffer->reset_count(frame.commandBuffer);

                    VkMemoryBarrier2 resetBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    resetBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                    resetBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    resetBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    resetBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                               | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo resetDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    resetDep.memoryBarrierCount = 1;
                    resetDep.pMemoryBarriers    = &resetBarrier;
                    vkCmdPipelineBarrier2(frame.commandBuffer, &resetDep);
                }

                m_gpuProfiler->beginZone(frame.commandBuffer, "CullScene");
                m_gpuCullPass->record(frame.commandBuffer,
                                       m_descriptorAllocator->globalSet(),
                                       *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer,
                                       cameraSlot,
                                       /*instanceOffset*/ 0, sceneInstanceCount,
                                       /*meshletOffset*/  0, sceneMeshletCount);

                m_indirectBuffer->record_count_readback(frame.commandBuffer);

                emitCullToTaskBarrier();
                m_gpuProfiler->endZone(frame.commandBuffer);

                m_gpuProfiler->beginZone(frame.commandBuffer, "DrawSceneVB");
                m_visibilityPass->record(frame.commandBuffer, m_descriptorAllocator->globalSet(),
                                          extent,
                                          m_gbufDepth.view, m_gbufDepth.image,
                                          *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer, cameraSlot,
                                          /*clearFirst=*/!terrainRan);
                m_gpuProfiler->endZone(frame.commandBuffer);
            }

            // --- Terrain wireframe cull (Wireframe + LitWireframe debug modes) ---
            // Culls terrain meshlets into m_terrainWireIndirectBuffer so the wireframe
            // overlay pass can draw terrain lines independently of m_indirectBuffer
            // (which at this point holds scene survivors only).
            if ((m_debugMode == DebugMode::Wireframe || m_debugMode == DebugMode::LitWireframe)
                && m_terrain != nullptr && m_terrain->isEnabled()
                && terrainMeshletCount > 0 && terrainInstanceCount > 0
                && m_visibilityPass->hasTerrainWireframePipeline()
                && m_terrainWireIndirectBuffer)
            {
                m_terrainWireIndirectBuffer->reset_count(frame.commandBuffer);

                VkMemoryBarrier2 wireResetBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                wireResetBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                wireResetBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                wireResetBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                wireResetBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkDependencyInfo wireResetDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                wireResetDep.memoryBarrierCount = 1;
                wireResetDep.pMemoryBarriers    = &wireResetBarrier;
                vkCmdPipelineBarrier2(frame.commandBuffer, &wireResetDep);

                m_gpuProfiler->beginZone(frame.commandBuffer, "CullTerrainWire");
                m_gpuCullPass->record(frame.commandBuffer,
                                       m_descriptorAllocator->globalSet(),
                                       *m_gpuScene, *m_gpuMeshlets, *m_terrainWireIndirectBuffer,
                                       cameraSlot,
                                       sceneInstanceCount, terrainInstanceCount,
                                       sceneMeshletCount,  terrainMeshletCount);

                emitCullToTaskBarrier();
                m_gpuProfiler->endZone(frame.commandBuffer);
            }

            // MaterialEvalPass reads vis buffer + depth, writes G-buffer to SHADER_READ_ONLY_OPTIMAL.
            m_gpuProfiler->beginZone(frame.commandBuffer, "MaterialEval");
            m_materialEvalPass->record(frame.commandBuffer, m_descriptorAllocator->globalSet(),
                                        extent, m_visibilityPass->vis_buffer_slot(),
                                        *m_gpuScene, *m_gpuMeshlets,
                                        m_scene->materialBufferSlot, cameraSlot);
            m_gpuProfiler->endZone(frame.commandBuffer);
        }
    }

    if (m_scene != nullptr) {
        // G-buffer images are in SHADER_READ_ONLY_OPTIMAL from MaterialEvalPass.
        const auto gbufAlbedo = m_renderGraph->importImage(
            "gbuf_albedo", m_gbufAlbedo.image, m_gbufAlbedo.view,
            gbuf::kAlbedoFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufNormal = m_renderGraph->importImage(
            "gbuf_normal", m_gbufNormal.image, m_gbufNormal.view,
            gbuf::kNormalFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufMetalRough = m_renderGraph->importImage(
            "gbuf_metalrough", m_gbufMetalRough.image, m_gbufMetalRough.view,
            gbuf::kMetalRoughFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufMotionVec = m_renderGraph->importImage(
            "gbuf_motionvec", m_gbufMotionVec.image, m_gbufMotionVec.view,
            gbuf::kMotionVecFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufDepth = m_renderGraph->importImage(
            "gbuf_depth", m_gbufDepth.image, m_gbufDepth.view,
            gbuf::kDepthFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        // CDLOD terrain is drawn through the visibility-buffer pipeline —
        // per-patch GpuInstances emitted in update() are culled and rasterised
        // alongside scene meshes. No dedicated raster pass is required here.

        // ---- Mode-dependent render graph passes ----
        if (m_debugMode == DebugMode::Lit) {

        // RT reflection pass — runs between G-buffer and lighting.
        // On RT hardware: dispatches vkCmdTraceRaysKHR.
        // On Min tier: no-op (SSR integration deferred to Phase 2).
        if (m_rtReflectionsEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            // RT reflection pass records its own barriers and is not a
            // raster pass, so we inject it as a custom execute callback
            // in a dummy raster pass with no color targets.
            gfx::RenderGraph::RasterPassDesc rtReflectDesc{};
            rtReflectDesc.name    = "RTReflectionPass";
            rtReflectDesc.sampledInputs = {gbufNormal, gbufDepth};
            rtReflectDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "RTReflectionPass");
                m_rtReflectionPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                            m_gbufNormalSlot, m_gbufDepthSlot,
                                            cameraSlot, m_gbufferSamplerSlot,
                                            m_tlasSlot, m_reflectionSlot,
                                            m_atmospherePass->skyViewLutSlot(),
                                            m_atmospherePass->transmittanceLutSlot(),
                                            vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                            vec4{m_cameraWorldPos * 0.001f, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtReflectDesc));
        }

        // RT GI pass — conditional on gpuTier >= Recommended.
        if (m_rtGIEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc rtGIDesc{};
            rtGIDesc.name    = "RTGIPass";
            rtGIDesc.sampledInputs = {gbufNormal, gbufDepth};
            rtGIDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "RTGIPass");
                m_giPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                  m_gbufNormalSlot, m_gbufDepthSlot,
                                  cameraSlot, m_gbufferSamplerSlot, m_tlasSlot, m_giSlot,
                                  m_atmospherePass->skyViewLutSlot(),
                                  m_atmospherePass->transmittanceLutSlot(),
                                  vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                  vec4{m_cameraWorldPos * 0.001f, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtGIDesc));
        }

        // RT Shadow pass — conditional on gpuTier >= Recommended.
        if (m_rtShadowsEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc rtShadowDesc{};
            rtShadowDesc.name    = "RTShadowPass";
            rtShadowDesc.sampledInputs = {gbufNormal, gbufDepth};
            rtShadowDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "RTShadowPass");
                m_shadowPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                      m_gbufNormalSlot, m_gbufDepthSlot,
                                      cameraSlot, m_tlasSlot, m_shadowSlot,
                                      vec4{m_sunWorldDir, 0.02f}); // 0.02 rad cone angle
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtShadowDesc));
        }

        // Wet Road pass — conditional on gpuTier >= Recommended.
        if (m_wetRoadEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc wetRoadDesc{};
            wetRoadDesc.name    = "WetRoadPass";
            wetRoadDesc.sampledInputs = {gbufNormal, gbufDepth};
            wetRoadDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "WetRoadPass");
                m_wetRoadPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                       m_gbufNormalSlot, m_gbufDepthSlot,
                                       cameraSlot, m_gbufferSamplerSlot, m_tlasSlot, m_wetRoadSlot,
                                       m_atmospherePass->skyViewLutSlot(),
                                       m_atmospherePass->transmittanceLutSlot(),
                                       vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                       vec4{m_cameraWorldPos * 0.001f, 0.0f},
                                       m_wetnessFactor);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(wetRoadDesc));
        }

        // Denoise RT effects — conditional on gpuTier >= Recommended.
        if (m_denoiseEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc denoiseDesc{};
            denoiseDesc.name = "DenoisePass";
            denoiseDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DenoisePass");
                m_reflectionDenoiser->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                              m_reflectionSlot, m_gbufMotionVecSlot,
                                              m_reflDenoiseSlot, m_reflDenoiseSlot);
                m_giDenoiser->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                      m_giSlot, m_gbufMotionVecSlot,
                                      m_giDenoiseSlot, m_giDenoiseSlot);
                m_shadowDenoiser->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                          m_shadowSlot, m_gbufMotionVecSlot,
                                          m_shadowDenoiseSlot, m_shadowDenoiseSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(denoiseDesc));
        }

        gfx::RenderGraph::RasterPassDesc lightPassDesc{};
        lightPassDesc.name          = "LightingPass";
        lightPassDesc.colorTargets  = {hdrColorHandle};
        lightPassDesc.sampledInputs = {gbufAlbedo, gbufNormal, gbufMetalRough, gbufDepth};
        lightPassDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
        lightPassDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
            m_gpuProfiler->beginZone(cmd, "LightingPass");
            m_lightingPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                    m_gbufAlbedoSlot, m_gbufNormalSlot,
                                    m_gbufMetalRoughSlot, m_gbufDepthSlot,
                                    cameraSlot, m_gbufferSamplerSlot,
                                    vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                    vec4{m_light.color, 0.0f});
            m_gpuProfiler->endZone(cmd);
        };
        m_renderGraph->addRasterPass(std::move(lightPassDesc));

        // Sky background — overwrites sky pixels (depth==0) with the SkyView LUT.
        // Must run after LightingPass (LOAD_OP_LOAD preserves geometry pixels)
        // and before PhysicsDebug (which re-attaches gbufDepth as depth target).
        // gbufDepth stays in SHADER_READ_ONLY_OPTIMAL from the LightingPass,
        // so we list it in sampledInputs to let the graph verify the layout.
        {
            gfx::RenderGraph::RasterPassDesc skyDesc{};
            skyDesc.name         = "SkyBackgroundPass";
            skyDesc.colorTargets = {hdrColorHandle};
            skyDesc.colorLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            skyDesc.sampledInputs = {gbufDepth};
            skyDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "SkyBackgroundPass");
                m_skyPass->record(cmd,
                    m_descriptorAllocator->globalSet(), ext,
                    cameraSlot,
                    m_gbufDepthSlot,
                    m_atmospherePass->skyViewLutSlot(),
                    m_atmospherePass->transmittanceLutSlot(),
                    m_gbufferSamplerSlot,
                    vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                    vec4{m_cameraWorldPos * 0.001f, 0.0f}); // metres → km
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(skyDesc));
        }

        // Physics debug wireframe overlay — inserted after lighting so it
        // composites on top of the final lit image. Only active when enabled
        // and there are lines to draw (upload() sets the count).
        if (m_physicsDebugRenderer.enabled) {
            m_physicsDebugRenderer.upload();

            gfx::RenderGraph::RasterPassDesc debugDesc{};
            debugDesc.name        = "PhysicsDebugPass";
            debugDesc.colorTargets = {hdrColorHandle};
            debugDesc.colorLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            if (m_physicsDebugRenderer.depthTestEnabled) {
                // Re-attach the G-buffer depth read-only so lines are occluded
                // by scene geometry. depthWriteEnable=false in the pipeline
                // ensures the depth buffer is never modified.
                debugDesc.depthTarget = gbufDepth;
                debugDesc.depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            }
            debugDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "PhysicsDebugPass");
                m_physicsDebugRenderer.drawFrame(cmd, m_descriptorAllocator->globalSet(),
                                                  ext, cameraSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(debugDesc));
        }

        // Clustered forward pass — transparent geometry blended onto HDR after sky/debug.
        if (m_scene != nullptr && m_clusteredForwardPass) {
            gfx::RenderGraph::RasterPassDesc fwdDesc{};
            fwdDesc.name         = "ClusteredForwardPass";
            fwdDesc.colorTargets = {hdrColorHandle};
            fwdDesc.colorLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            fwdDesc.depthTarget  = gbufDepth;
            fwdDesc.depthLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            fwdDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "ClusteredForwardPass");
                m_clusteredForwardPass->record(
                    cmd, m_descriptorAllocator->globalSet(), ext, *m_scene,
                    cameraSlot, m_scene->materialBufferSlot,
                    m_sunWorldDir, vec3(1.0f), m_atmosphereSettings.sunIntensity);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(fwdDesc));
        }

        // PostProcessPass + optional SMAA.
        // When SMAA is on: PostProcess → LDR intermediate, then 3 SMAA passes → swapchain.
        // When SMAA is off: PostProcess → swapchain directly.
        const bool smaaActive = m_aaSettings.smaaEnabled
                             && m_smaaPass
                             && m_smaaPass->isAllocated()
                             && m_smaaLdrImage != VK_NULL_HANDLE;

        if (smaaActive) {
            const auto ldrHandle = m_renderGraph->importImage(
                "smaa_ldr",
                m_smaaLdrImage, m_smaaLdrView, m_swapchain->format(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);
            const auto smaaEdgeHandle = m_renderGraph->importImage(
                "smaa_edge",
                m_smaaPass->edgeImage(), m_smaaPass->edgeView(), VK_FORMAT_R8G8_UNORM,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);
            const auto smaaWeightHandle = m_renderGraph->importImage(
                "smaa_weight",
                m_smaaPass->weightImage(), m_smaaPass->weightView(), VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);

            // PostProcessPass → LDR intermediate.
            {
                gfx::RenderGraph::RasterPassDesc ppDesc{};
                ppDesc.name          = "PostProcessPass";
                ppDesc.colorTargets  = {ldrHandle};
                ppDesc.sampledInputs = {hdrColorHandle, gbufDepth};
                ppDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
                ppDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "PostProcessPass");
                    m_postProcessPass->record(cmd,
                        m_descriptorAllocator->globalSet(),
                        m_atmospherePass->aerialPerspectiveReadSet(),
                        ext,
                        m_hdrColorSampledSlot,
                        m_gbufDepthSlot,
                        cameraSlot,
                        m_gbufferSamplerSlot,
                        m_linearSamplerSlot,
                        m_atmosphereSettings,
                        vec4{m_cameraWorldPos * 0.001f, 0.0f});
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(ppDesc));
            }

            // SMAA Pass 1: luma edge detection — LDR → edge texture.
            {
                gfx::RenderGraph::RasterPassDesc edgeDesc{};
                edgeDesc.name          = "SMAAEdgePass";
                edgeDesc.colorTargets  = {smaaEdgeHandle};
                edgeDesc.sampledInputs = {ldrHandle};
                edgeDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 0.0f}};
                edgeDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "SMAAEdgePass");
                    // Linear sampler for all three SMAA passes: the reference
                    // algorithm relies on sub-texel bilinear interpolation
                    // (edge search taps, area/search tex lookups, neighbourhood
                    // 0.5-texel sampling). Nearest here would break the math.
                    m_smaaPass->recordEdge(cmd, m_descriptorAllocator->globalSet(), ext,
                                            m_smaaLdrSampledSlot, m_linearSamplerSlot);
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(edgeDesc));
            }

            // SMAA Pass 2: blending weights — edge → weight texture.
            {
                gfx::RenderGraph::RasterPassDesc blendDesc{};
                blendDesc.name          = "SMAABlendPass";
                blendDesc.colorTargets  = {smaaWeightHandle};
                blendDesc.sampledInputs = {smaaEdgeHandle};
                blendDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 0.0f}};
                blendDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "SMAABlendPass");
                    m_smaaPass->recordBlend(cmd, m_descriptorAllocator->globalSet(), ext,
                                             m_linearSamplerSlot);
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(blendDesc));
            }

            // SMAA Pass 3: neighbourhood blend — LDR + weights → swapchain.
            {
                gfx::RenderGraph::RasterPassDesc nbrDesc{};
                nbrDesc.name          = "SMAANeighborPass";
                nbrDesc.colorTargets  = {colorHandle};
                nbrDesc.sampledInputs = {ldrHandle, smaaWeightHandle};
                nbrDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
                nbrDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "SMAANeighborPass");
                    // Neighborhood pass MUST use the linear sampler: the shader samples at
                    // uv ± 0.5*texel and relies on bilinear interpolation to produce the
                    // sub-pixel blend between the current pixel and its neighbour. A
                    // nearest sampler snaps back to one of the two pixels, collapsing the
                    // AA effect to a no-op (SMAA appears disabled).
                    m_smaaPass->recordNeighborhood(cmd, m_descriptorAllocator->globalSet(), ext,
                                                    m_smaaLdrSampledSlot, m_linearSamplerSlot);
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(nbrDesc));
            }
        } else {
            // PostProcessPass — final output directly to swapchain.
            gfx::RenderGraph::RasterPassDesc ppDesc{};
            ppDesc.name          = "PostProcessPass";
            ppDesc.colorTargets  = {colorHandle};
            ppDesc.sampledInputs = {hdrColorHandle, gbufDepth};
            ppDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            ppDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "PostProcessPass");
                m_postProcessPass->record(cmd,
                    m_descriptorAllocator->globalSet(),
                    m_atmospherePass->aerialPerspectiveReadSet(),
                    ext,
                    m_hdrColorSampledSlot,
                    m_gbufDepthSlot,
                    cameraSlot,
                    m_gbufferSamplerSlot,
                    m_linearSamplerSlot,
                    m_atmosphereSettings,
                    vec4{m_cameraWorldPos * 0.001f, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(ppDesc));
        }

        } else if (m_debugMode == DebugMode::Unlit) {
            // === UNLIT — read albedo G-buffer, skip lighting ===
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            dbgDesc.name          = "DebugUnlitPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.sampledInputs = {gbufAlbedo, gbufDepth};
            dbgDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
            dbgDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugUnlitPass");
                m_debugVisPass->recordUnlit(cmd, m_descriptorAllocator->globalSet(), ext,
                                            m_gbufAlbedoSlot, m_gbufDepthSlot,
                                            m_gbufferSamplerSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::Wireframe) {
            // === WIREFRAME — hardware line rasterization on black background ===
            gfx::RenderGraph::RasterPassDesc wireDesc{};
            wireDesc.name         = "DebugWireframePass";
            wireDesc.colorTargets = {colorHandle};
            wireDesc.clearColor   = {{0.0f, 0.0f, 0.0f, 1.0f}};
            wireDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugWireframePass");
                if (m_terrain && m_terrain->isEnabled() && m_terrainWireIndirectBuffer
                    && m_visibilityPass->hasTerrainWireframePipeline()) {
                    const u32 tmc  = m_gpuMeshlets->total_meshlet_count() - m_gpuMeshlets->scene_meshlet_count();
                    const auto topo = m_terrain->sharedTopologyHandle();
                    m_visibilityPass->recordTerrainWireframe(cmd, m_descriptorAllocator->globalSet(), ext,
                                                              *m_gpuScene, *m_gpuMeshlets, *m_terrainWireIndirectBuffer,
                                                              cameraSlot,
                                                              topo.topologyVerticesSlot,
                                                              topo.topologyTrianglesSlot,
                                                              tmc, m_wireframeColor);
                }
                m_visibilityPass->recordWireframe(cmd, m_descriptorAllocator->globalSet(), ext,
                                                  *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer,
                                                  cameraSlot, m_wireframeColor);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(wireDesc));

        } else if (m_debugMode == DebugMode::LitWireframe) {
            // === LIT WIREFRAME — lit scene (no post-proc) + wireframe overlay ===
            // Phase 1: lighting -> hdrColorHandle
            gfx::RenderGraph::RasterPassDesc lightPassDesc{};
            lightPassDesc.name          = "LightingPass";
            lightPassDesc.colorTargets  = {hdrColorHandle};
            lightPassDesc.sampledInputs = {gbufAlbedo, gbufNormal, gbufMetalRough, gbufDepth};
            lightPassDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
            lightPassDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "LightingPass");
                m_lightingPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                       m_gbufAlbedoSlot, m_gbufNormalSlot,
                                       m_gbufMetalRoughSlot, m_gbufDepthSlot,
                                       cameraSlot, m_gbufferSamplerSlot,
                                       vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                       vec4{m_light.color, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(lightPassDesc));

            // Phase 1b: SkyBackgroundPass — writes the atmospheric sky into
            // HDR at pixels where depth==0.  Without this the LitWire frame
            // shows black void where the sky should be, which also defeats
            // the visual cue for aerial perspective (fog reads as a gradient
            // *into* the sky). Mirror of the pass that runs in Lit mode.
            {
                gfx::RenderGraph::RasterPassDesc skyDesc{};
                skyDesc.name          = "SkyBackgroundPass";
                skyDesc.colorTargets  = {hdrColorHandle};
                skyDesc.colorLoadOp   = VK_ATTACHMENT_LOAD_OP_LOAD;
                skyDesc.sampledInputs = {gbufDepth};
                skyDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "SkyBackgroundPass");
                    m_skyPass->record(cmd,
                        m_descriptorAllocator->globalSet(), ext,
                        cameraSlot,
                        m_gbufDepthSlot,
                        m_atmospherePass->skyViewLutSlot(),
                        m_atmospherePass->transmittanceLutSlot(),
                        m_gbufferSamplerSlot,
                        vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                        vec4{m_cameraWorldPos * 0.001f, 0.0f});
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(skyDesc));
            }

            // Phase 2: PostProcessPass — identical setup to Lit mode so the
            // wireframe overlay sits on top of a fully-composited beauty
            // frame (aerial perspective, bloom, tonemap, exposure).  The
            // prior DebugBlitPass only ran a Reinhard tonemap and skipped
            // atmosphere, which made LitWire look stylistically detached
            // from the pure Lit mode.
            gfx::RenderGraph::RasterPassDesc ppDesc{};
            ppDesc.name          = "PostProcessPass";
            ppDesc.colorTargets  = {colorHandle};
            ppDesc.sampledInputs = {hdrColorHandle, gbufDepth};
            ppDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            ppDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "PostProcessPass");
                m_postProcessPass->record(cmd,
                    m_descriptorAllocator->globalSet(),
                    m_atmospherePass->aerialPerspectiveReadSet(),
                    ext,
                    m_hdrColorSampledSlot,
                    m_gbufDepthSlot,
                    cameraSlot,
                    m_gbufferSamplerSlot,
                    m_linearSamplerSlot,
                    m_atmosphereSettings,
                    vec4{m_cameraWorldPos * 0.001f, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(ppDesc));

            // Phase 3: wireframe overlay -> swapchain (LOAD preserves lit base)
            gfx::RenderGraph::RasterPassDesc wireOverlayDesc{};
            wireOverlayDesc.name         = "DebugWireframeOverlayPass";
            wireOverlayDesc.colorTargets = {colorHandle};
            wireOverlayDesc.colorLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            wireOverlayDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugWireframeOverlayPass");
                if (m_terrain && m_terrain->isEnabled() && m_terrainWireIndirectBuffer
                    && m_visibilityPass->hasTerrainWireframePipeline()) {
                    const u32 tmc  = m_gpuMeshlets->total_meshlet_count() - m_gpuMeshlets->scene_meshlet_count();
                    const auto topo = m_terrain->sharedTopologyHandle();
                    m_visibilityPass->recordTerrainWireframe(cmd, m_descriptorAllocator->globalSet(), ext,
                                                              *m_gpuScene, *m_gpuMeshlets, *m_terrainWireIndirectBuffer,
                                                              cameraSlot,
                                                              topo.topologyVerticesSlot,
                                                              topo.topologyTrianglesSlot,
                                                              tmc, m_wireframeColor);
                }
                m_visibilityPass->recordWireframe(cmd, m_descriptorAllocator->globalSet(), ext,
                                                  *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer,
                                                  cameraSlot, m_wireframeColor);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(wireOverlayDesc));

        } else if (m_debugMode == DebugMode::DetailLighting) {
            // === DETAIL LIGHTING — Cook-Torrance on white material ===
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            dbgDesc.name          = "DebugDetailLightingPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.sampledInputs = {gbufAlbedo, gbufNormal, gbufMetalRough, gbufDepth};
            dbgDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
            dbgDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugDetailLightingPass");
                m_debugVisPass->recordDetailLighting(cmd, m_descriptorAllocator->globalSet(), ext,
                                                     m_gbufAlbedoSlot, m_gbufNormalSlot,
                                                     m_gbufMetalRoughSlot, m_gbufDepthSlot,
                                                     cameraSlot, m_gbufferSamplerSlot,
                                                     vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                                     vec4{m_light.color, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::Clusters) {
            // === CLUSTERS — meshlet color visualization ===
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            dbgDesc.name          = "DebugClustersPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
            dbgDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugClustersPass");
                if (m_visibilityPass)
                    m_debugVisPass->recordClusters(cmd, m_descriptorAllocator->globalSet(), ext,
                                                   m_visibilityPass->vis_buffer_slot(),
                                                   m_gbufferSamplerSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));
        }

    } else {
        // ---- Forward fallback: TrianglePass ----
        const auto depthHandle = m_renderGraph->importImage(
            "swapchain_depth",
            depthImage, depthView, m_swapchain->depthFormat(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        gfx::RenderGraph::RasterPassDesc triPassDesc{};
        triPassDesc.name         = "TrianglePass";
        triPassDesc.colorTargets = {colorHandle};
        triPassDesc.depthTarget  = depthHandle;
        triPassDesc.clearColor   = {{0.02f, 0.02f, 0.05f, 1.0f}};
        triPassDesc.clearDepth   = {0.0f, 0};
        triPassDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
            m_gpuProfiler->beginZone(cmd, "TrianglePass");
            m_trianglePass->record(cmd, m_descriptorAllocator->globalSet(), ext);
            m_gpuProfiler->endZone(cmd);
        };
        m_renderGraph->addRasterPass(std::move(triPassDesc));
    }

    m_renderGraph->execute(frame.commandBuffer, extent);

    // Upscaler evaluate — after lighting pass, before present.
    // Compute Halton(2,3) jitter for the current frame.
    ++m_jitterIndex;
    const f32 jx = haltonJitter(m_jitterIndex, 2) - 0.5f;
    const f32 jy = haltonJitter(m_jitterIndex, 3) - 0.5f;

    // Track the swapchain image layout so ImGuiLayer can emit the correct
    // pre-barrier. The render graph leaves it in PRESENT_SRC_KHR; the
    // upscaler (when real) writes to it with outputLayout COLOR_ATTACHMENT_OPTIMAL.
    VkImageLayout swapchainLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // PostProcessPass writes the final tonemapped image directly to the swapchain,
    // so the upscaler is skipped when it is active. The upscaler can be
    // re-integrated in a future pass once the HDR → TAA → tonemap ordering is
    // designed with a proper intermediate render target.
    if (m_upscaler && m_scene != nullptr && !m_postProcessPass) {
        m_upscaler->evaluate(frame.commandBuffer,
                             m_hdrColorView,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             depthView,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             m_gbufMotionVec.view,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             m_swapchain->view(imageIndex),
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             extent, extent,
                             jx, jy,
                             m_upscalerSettings.sharpness,
                             false);
    }

    // ImGui render -- after scene, before end command buffer.
    if (m_imguiLayer) {
        m_imguiLayer->render(frame.commandBuffer,
                             targetImage,
                             targetView,
                             extent,
                             swapchainLayout);
    }

    ENIGMA_VK_CHECK(vkEndCommandBuffer(frame.commandBuffer));

    // Submit.
    const u64 signalValue = frame.frameValue + 1;
    const VkSemaphore imageRenderFinished = m_swapchain->renderFinished(imageIndex);

    const VkSemaphore        waitSems[]    = { frame.imageAvailable };
    const VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    const VkSemaphore        signalSems[]  = { imageRenderFinished, frame.inFlight };
    const u64                signalValues[] = { 0, signalValue };

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount   = 0;
    timelineInfo.pWaitSemaphoreValues      = nullptr;
    timelineInfo.signalSemaphoreValueCount = 2;
    timelineInfo.pSignalSemaphoreValues    = signalValues;

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext                = &timelineInfo;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSems;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &frame.commandBuffer;
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores    = signalSems;

    ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    frame.frameValue = signalValue;

    // Present.
    VkSwapchainKHR swapchain = m_swapchain->handle();

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &imageRenderFinished;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain;
    presentInfo.pImageIndices      = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(m_device->graphicsQueue(), &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        const auto [w, h] = m_window.framebufferSize();
        m_swapchain->recreate(w, h);
        resizeGBuffer(m_swapchain->extent());
    } else if (presentResult != VK_SUCCESS) {
        ENIGMA_VK_CHECK(presentResult);
    }

    m_frameIndex = (m_frameIndex + 1) % gfx::MAX_FRAMES_IN_FLIGHT;
}

void Renderer::setScene(Scene* scene) {
    m_scene = scene;
    if (m_scene != nullptr && m_device->gpuTier() >= gfx::GpuTier::Recommended) {
        buildAccelerationStructures();
    }
    if (m_scene != nullptr && m_gpuMeshlets) {
        uploadMeshlets();
    }
}

void Renderer::buildAccelerationStructures() {
    ENIGMA_ASSERT(m_scene != nullptr);

    // Vertex stride: packed as 3 x float4 per vertex (see packVertices in GltfLoader).
    constexpr VkDeviceSize kVertexStride = 3 * sizeof(vec4); // 48 bytes

    for (auto& prim : m_scene->primitives) {
        if (prim.blas.has_value()) continue;
        if (prim.vertexBuffer == VK_NULL_HANDLE) continue;

        prim.blas = gfx::BLAS::build(*m_device, *m_allocator,
                                     prim.vertexBuffer, prim.vertexCount, kVertexStride,
                                     prim.indexBuffer, prim.indexCount);
    }

    // Build TLAS from all scene nodes.
    m_scene->tlas.emplace(*m_device, *m_allocator, 4096);

    for (const auto& node : m_scene->nodes) {
        for (u32 primIdx : node.primitiveIndices) {
            auto& prim = m_scene->primitives[primIdx];
            if (!prim.blas.has_value()) continue;

            const u32 slot = m_scene->tlas->allocateInstanceSlot();

            VkAccelerationStructureInstanceKHR inst{};
            // Copy the 3x4 transform matrix (row-major, column-major GLM needs transpose).
            const mat4& m = node.worldTransform;
            inst.transform.matrix[0][0] = m[0][0]; inst.transform.matrix[0][1] = m[1][0];
            inst.transform.matrix[0][2] = m[2][0]; inst.transform.matrix[0][3] = m[3][0];
            inst.transform.matrix[1][0] = m[0][1]; inst.transform.matrix[1][1] = m[1][1];
            inst.transform.matrix[1][2] = m[2][1]; inst.transform.matrix[1][3] = m[3][1];
            inst.transform.matrix[2][0] = m[0][2]; inst.transform.matrix[2][1] = m[1][2];
            inst.transform.matrix[2][2] = m[2][2]; inst.transform.matrix[2][3] = m[3][2];

            inst.instanceCustomIndex                    = primIdx;
            inst.mask                                   = 0xFF;
            inst.instanceShaderBindingTableRecordOffset  = 0;
            inst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference         = prim.blas->deviceAddress();

            m_scene->tlas->setInstance(slot, inst);
        }
    }

    // Build TLAS on the GPU via immediate command buffer.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = m_device->graphicsQueueFamily();

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &poolCI, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool        = cmdPool;
    cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &cmdAllocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    m_scene->tlas->build(cmd, *m_device, *m_allocator);

    ENIGMA_VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    ENIGMA_VK_CHECK(vkQueueWaitIdle(m_device->graphicsQueue()));

    vkDestroyCommandPool(m_device->logical(), cmdPool, nullptr);

    // Register TLAS in the bindless descriptor set.
    m_tlasSlot = m_descriptorAllocator->registerAccelerationStructure(m_scene->tlas->handle());

    ENIGMA_LOG_INFO("[renderer] acceleration structures built ({} BLASes, TLAS slot={})",
                    m_scene->primitives.size(), m_tlasSlot);
}

void Renderer::uploadMeshlets() {
    ENIGMA_ASSERT(m_scene != nullptr);
    ENIGMA_ASSERT(m_gpuMeshlets != nullptr);

    // Append scene meshlets into the CPU-side accumulation. These must be
    // committed BEFORE reserveCapacity() is called so the reserved GPU buffer
    // is sized large enough to cover scene meshlets + terrain activation
    // ceiling (see GpuMeshletBuffer::reserveCapacity contract).
    bool anyMeshlets = false;
    for (auto& prim : m_scene->primitives) {
        if (prim.meshlets.meshlets.empty()) continue;
        prim.meshletOffset = m_gpuMeshlets->append(prim.meshlets);
        anyMeshlets = true;
    }

    // Even with no mesh primitives we still need the one-shot command buffer
    // for heightmap upload + CDLOD initialize.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = m_device->graphicsQueueFamily();

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &poolCI, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool        = cmdPool;
    cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &cmdAllocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Load heightmap (record staging->image copy + layout transitions).
    m_heightmapLoader = std::make_unique<HeightmapLoader>(
        *m_device, *m_allocator, *m_descriptorAllocator);
    HeightmapDesc hDesc{};
    hDesc.path        = "assets/terrain/height_4k.r32f";
    hDesc.sampleCount = 4097;
    hDesc.worldSize   = 4096.0f;
    hDesc.minHeight   = 0.0f;
    hDesc.maxHeight   = 512.0f;
    (void)m_heightmapLoader->load(hDesc, cmd); // tolerate missing file — zeros

    // Initialize CDLOD terrain. initialize() internally calls
    // GpuMeshletBuffer::reserveCapacity() which uploads the current
    // CPU-side meshlet array to the device-local buffer and pre-reserves
    // room for per-patch incremental appends. This replaces the old
    // m_gpuMeshlets->upload() call on the meshlet-only path.
    m_terrain = std::make_unique<CdlodTerrain>(
        *m_device, *m_allocator, *m_descriptorAllocator);
    CdlodConfig terrainConfig{};
    m_terrain->initialize(terrainConfig, *m_heightmapLoader,
                          *m_gpuMeshlets, *m_indirectBuffer, *m_scene,
                          cmd);

    // Upload the precomputed AreaTex + SearchTex lookup textures that the
    // reference SMAA blend pass consumes. Records on the same init cmd; the
    // waitIdle below guarantees staging buffers are safe to destroy.
    if (m_smaaPass) {
        m_smaaPass->uploadLookupTextures(cmd, *m_descriptorAllocator);
    }

    ENIGMA_VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    ENIGMA_VK_CHECK(vkQueueWaitIdle(m_device->graphicsQueue()));

    // Release heightmap staging + meshlet staging + SMAA lookup-texture
    // staging now that the GPU has finished consuming them.
    m_heightmapLoader->releaseStaging();
    m_gpuMeshlets->flush_staging();
    if (m_smaaPass) m_smaaPass->releaseLookupUploadStaging();

    vkDestroyCommandPool(m_device->logical(), cmdPool, nullptr);

    // Rebuild material SSBO to include the terrain material appended by
    // CdlodTerrain::initialize() — the buffer built by loadGltf was sized
    // before terrain init, so the terrain slot was absent.
    if (m_terrain && m_terrain->isEnabled() && m_scene->materialBufferSlot != 0xFFFFFFFFu) {
        const VkDeviceSize matBufSize = m_scene->materials.size() * sizeof(Material);

        VkBufferCreateInfo bufCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufCI.size        = matBufSize;
        bufCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo devAllocCI{};
        devAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VkBuffer      newBuf   = VK_NULL_HANDLE;
        VmaAllocation newAlloc = nullptr;
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &devAllocCI,
                                        &newBuf, &newAlloc, nullptr));

        VkBufferCreateInfo stagCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        stagCI.size        = matBufSize;
        stagCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo stagAllocCI{};
        stagAllocCI.usage  = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        stagAllocCI.flags  = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        VkBuffer          stagBuf   = VK_NULL_HANDLE;
        VmaAllocation     stagAlloc = nullptr;
        VmaAllocationInfo stagInfo{};
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stagCI, &stagAllocCI,
                                        &stagBuf, &stagAlloc, &stagInfo));
        std::memcpy(stagInfo.pMappedData, m_scene->materials.data(), matBufSize);
        vmaFlushAllocation(m_allocator->handle(), stagAlloc, 0, VK_WHOLE_SIZE);

        VkCommandPoolCreateInfo pool2CI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool2CI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool2CI.queueFamilyIndex = m_device->graphicsQueueFamily();
        VkCommandPool pool2 = VK_NULL_HANDLE;
        ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &pool2CI, nullptr, &pool2));
        VkCommandBufferAllocateInfo cb2AI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cb2AI.commandPool        = pool2;
        cb2AI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb2AI.commandBufferCount = 1;
        VkCommandBuffer cmd2 = VK_NULL_HANDLE;
        ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &cb2AI, &cmd2));
        VkCommandBufferBeginInfo cb2BI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cb2BI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd2, &cb2BI);
        VkBufferCopy region{0, 0, matBufSize};
        vkCmdCopyBuffer(cmd2, stagBuf, newBuf, 1, &region);
        vkEndCommandBuffer(cmd2);
        VkSubmitInfo sub2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub2.commandBufferCount = 1;
        sub2.pCommandBuffers    = &cmd2;
        ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &sub2, VK_NULL_HANDLE));
        ENIGMA_VK_CHECK(vkQueueWaitIdle(m_device->graphicsQueue()));
        vkDestroyCommandPool(m_device->logical(), pool2, nullptr);
        vmaDestroyBuffer(m_allocator->handle(), stagBuf, stagAlloc);

        vmaDestroyBuffer(m_allocator->handle(),
                         m_scene->materialBuffer.buffer,
                         m_scene->materialBuffer.allocation);
        m_scene->materialBuffer.buffer     = newBuf;
        m_scene->materialBuffer.allocation = newAlloc;
        m_scene->materialBufferSlot = m_descriptorAllocator->registerStorageBuffer(
            newBuf, matBufSize);
        ENIGMA_LOG_INFO("[renderer] material SSBO rebuilt with {} materials ({} B)",
                        m_scene->materials.size(), matBufSize);
    }

    ENIGMA_LOG_INFO("[renderer] meshlets reserved ({} scene meshlets), "
                    "cdlod terrain {} (anyMeshlets={})",
                    m_gpuMeshlets->total_meshlet_count(),
                    m_terrain->isEnabled() ? "enabled" : "disabled",
                    anyMeshlets);
}

void Renderer::applyImpact(const ImpactEvent& event) {
    m_deformationSystem.applyImpact(event);
    m_deformationPending = true;
}

void Renderer::setUpscalerSettings(const UpscalerSettings& s) {
    const auto oldBackend = m_upscalerSettings.effectiveBackend(
        m_device->properties(), m_device->gpuTier());
    m_upscalerSettings = s;
    const auto newBackend = m_upscalerSettings.effectiveBackend(
        m_device->properties(), m_device->gpuTier());

    if (newBackend != oldBackend) {
        // Backend changed: destroy old, create new.
        vkDeviceWaitIdle(m_device->logical());
        m_upscaler->shutdown();
        m_upscaler = UpscalerFactory::create(
            m_device->properties(), m_device->gpuTier(), newBackend);
        m_upscaler->init(m_device->logical(), m_device->physical(),
                         m_swapchain->extent(), m_upscalerSettings.quality);
    } else if (m_upscalerSettings.quality != s.quality) {
        // Quality only changed: reinit in place.
        vkDeviceWaitIdle(m_device->logical());
        m_upscaler->reinit(m_swapchain->extent(), m_upscalerSettings.quality);
    }
}

void Renderer::resizeGBuffer(VkExtent2D extent) {
    // createGBufferImages calls vkDeviceWaitIdle before destroying in-use images.
    createGBufferImages(extent);

    // Re-write the bindless descriptor slots to point at the new image views.
    m_descriptorAllocator->updateSampledImage(
        m_gbufAlbedoSlot,     m_gbufAlbedo.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufNormalSlot,     m_gbufNormal.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufMetalRoughSlot, m_gbufMetalRough.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufMotionVecSlot,  m_gbufMotionVec.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufDepthSlot,      m_gbufDepth.view,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Re-allocate RT pass images on resize.
    m_rtReflectionPass->allocate(extent);
    m_descriptorAllocator->updateStorageImage(m_reflectionSlot, m_rtReflectionPass->outputView());

    m_giPass->allocate(extent);
    m_descriptorAllocator->updateStorageImage(m_giSlot, m_giPass->outputView());

    m_shadowPass->allocate(extent);
    m_descriptorAllocator->updateStorageImage(m_shadowSlot, m_shadowPass->outputView());

    m_wetRoadPass->allocate(extent);
    m_descriptorAllocator->updateStorageImage(m_wetRoadSlot, m_wetRoadPass->outputView());

    m_giDenoiser->allocate(extent, RTGIPass::kOutputFormat);
    m_descriptorAllocator->updateStorageImage(m_giDenoiseSlot, m_giDenoiser->outputView());

    m_shadowDenoiser->allocate(extent, RTShadowPass::kOutputFormat);
    m_descriptorAllocator->updateStorageImage(m_shadowDenoiseSlot, m_shadowDenoiser->outputView());

    m_reflectionDenoiser->allocate(extent, RTReflectionPass::kOutputFormat);
    m_descriptorAllocator->updateStorageImage(m_reflDenoiseSlot, m_reflectionDenoiser->outputView());

    // Update storage image slots to point at the new G-buffer views.
    m_descriptorAllocator->updateStorageImage(m_gbufAlbedoStorageSlot,     m_gbufAlbedo.view);
    m_descriptorAllocator->updateStorageImage(m_gbufNormalStorageSlot,     m_gbufNormal.view);
    m_descriptorAllocator->updateStorageImage(m_gbufMetalRoughStorageSlot, m_gbufMetalRough.view);
    m_descriptorAllocator->updateStorageImage(m_gbufMotionVecStorageSlot,  m_gbufMotionVec.view);

    // VB pipeline resize.
    if (m_hizPass)        m_hizPass->allocate(extent);
    if (m_visibilityPass) m_visibilityPass->allocate(extent, gbuf::kDepthFormat);
    if (m_materialEvalPass) {
        m_materialEvalPass->prepare(
            extent,
            m_gbufAlbedo.image,     m_gbufNormal.image,
            m_gbufMetalRough.image, m_gbufMotionVec.image,
            m_gbufDepth.image,
            m_gbufAlbedoStorageSlot,     m_gbufNormalStorageSlot,
            m_gbufMetalRoughStorageSlot, m_gbufMotionVecStorageSlot,
            m_gbufDepthSlot);
    }

    // Recreate HDR intermediate at the new extent and patch bindless slots.
    createHdrColor(extent);

    // Recreate SMAA intermediates (LDR + edge/weight textures) at the new extent.
    createSmaaLdr(extent);
    if (m_smaaPass) {
        if (m_smaaPass->isAllocated()) {
            m_smaaPass->free(*m_descriptorAllocator);
        }
        m_smaaPass->allocate(extent, *m_descriptorAllocator);
    }
}

void Renderer::createGBufferImage(VkFormat format, VkImageUsageFlags usage,
                                   VkImageAspectFlags aspect, VkExtent2D extent, GBufferImage& out) {
    VkImageCreateInfo imageCI{};
    imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType     = VK_IMAGE_TYPE_2D;
    imageCI.format        = format;
    imageCI.extent        = {extent.width, extent.height, 1};
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

void Renderer::destroyGBufferImage(GBufferImage& img) {
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

void Renderer::createGBufferImages(VkExtent2D extent) {
    if (m_gbufAlbedo.image != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroyGBufferImages();
    }
    constexpr VkImageUsageFlags kColorUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    createGBufferImage(gbuf::kAlbedoFormat,     kColorUsage, VK_IMAGE_ASPECT_COLOR_BIT, extent, m_gbufAlbedo);
    createGBufferImage(gbuf::kNormalFormat,     kColorUsage, VK_IMAGE_ASPECT_COLOR_BIT, extent, m_gbufNormal);
    createGBufferImage(gbuf::kMetalRoughFormat, kColorUsage, VK_IMAGE_ASPECT_COLOR_BIT, extent, m_gbufMetalRough);
    createGBufferImage(gbuf::kMotionVecFormat,  kColorUsage, VK_IMAGE_ASPECT_COLOR_BIT, extent, m_gbufMotionVec);
    createGBufferImage(gbuf::kDepthFormat,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT, extent, m_gbufDepth);
    ENIGMA_LOG_INFO("[renderer] G-buffer allocated {}x{}", extent.width, extent.height);
}

void Renderer::destroyGBufferImages() {
    destroyGBufferImage(m_gbufAlbedo);
    destroyGBufferImage(m_gbufNormal);
    destroyGBufferImage(m_gbufMetalRough);
    destroyGBufferImage(m_gbufMotionVec);
    destroyGBufferImage(m_gbufDepth);
}

void Renderer::createHdrColor(VkExtent2D extent) {
    if (m_hdrColor != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroyHdrColor();
    }

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    imgCI.extent        = {extent.width, extent.height, 1};
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT
                        | VK_IMAGE_USAGE_STORAGE_BIT
                        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imgCI, &allocCI,
                                   &m_hdrColor, &m_hdrColorAlloc, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = m_hdrColor;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &m_hdrColorView));

    if (m_hdrColorSampledSlot == UINT32_MAX) {
        m_hdrColorSampledSlot = m_descriptorAllocator->registerSampledImage(
            m_hdrColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        m_descriptorAllocator->updateSampledImage(
            m_hdrColorSampledSlot, m_hdrColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (m_hdrColorStorageSlot == UINT32_MAX) {
        m_hdrColorStorageSlot = m_descriptorAllocator->registerStorageImage(m_hdrColorView);
    } else {
        m_descriptorAllocator->updateStorageImage(m_hdrColorStorageSlot, m_hdrColorView);
    }

    ENIGMA_LOG_INFO("[renderer] HDR intermediate {}x{} created (sampled={}, storage={})",
                    extent.width, extent.height,
                    m_hdrColorSampledSlot, m_hdrColorStorageSlot);
}

void Renderer::destroyHdrColor() {
    if (m_hdrColorView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), m_hdrColorView, nullptr);
        m_hdrColorView = VK_NULL_HANDLE;
    }
    if (m_hdrColor != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_hdrColor, m_hdrColorAlloc);
        m_hdrColor      = VK_NULL_HANDLE;
        m_hdrColorAlloc = nullptr;
    }
}

void Renderer::createSmaaLdr(VkExtent2D extent) {
    if (m_smaaLdrImage != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroySmaaLdr();
    }

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = m_swapchain->format();
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

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imgCI, &allocCI,
                                    &m_smaaLdrImage, &m_smaaLdrAlloc, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = m_smaaLdrImage;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = m_swapchain->format();
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &m_smaaLdrView));

    if (m_smaaLdrSampledSlot == UINT32_MAX) {
        m_smaaLdrSampledSlot = m_descriptorAllocator->registerSampledImage(
            m_smaaLdrView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        m_descriptorAllocator->updateSampledImage(
            m_smaaLdrSampledSlot, m_smaaLdrView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    ENIGMA_LOG_INFO("[renderer] SMAA LDR intermediate {}x{} created (slot={})",
                    extent.width, extent.height, m_smaaLdrSampledSlot);
}

void Renderer::destroySmaaLdr() {
    if (m_smaaLdrView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), m_smaaLdrView, nullptr);
        m_smaaLdrView = VK_NULL_HANDLE;
    }
    if (m_smaaLdrImage != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_smaaLdrImage, m_smaaLdrAlloc);
        m_smaaLdrImage = VK_NULL_HANDLE;
        m_smaaLdrAlloc = nullptr;
    }
}

void Renderer::applyTextureFilterSettings() {
    if (m_scene == nullptr ||
        m_scene->defaultMaterialSamplerSlot == 0xFFFFFFFFu) {
        return;
    }

    vkDeviceWaitIdle(m_device->logical());

    const f32  devMaxAniso   = m_device->properties().limits.maxSamplerAnisotropy;
    const bool anisoSupported = devMaxAniso > 1.0f;
    const u32  requestedAniso = std::clamp<u32>(
        m_textureFilterSettings.anisotropy,
        1u,
        static_cast<u32>(std::floor(devMaxAniso)));
    m_textureFilterSettings.anisotropy = requestedAniso;

    VkSamplerCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter        = VK_FILTER_LINEAR;
    ci.minFilter        = VK_FILTER_LINEAR;
    ci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.anisotropyEnable = (anisoSupported && requestedAniso > 1u) ? VK_TRUE : VK_FALSE;
    ci.maxAnisotropy    = ci.anisotropyEnable ? static_cast<f32>(requestedAniso) : 1.0f;
    ci.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    ci.minLod           = 0.0f;
    ci.maxLod           = m_textureFilterSettings.mipmapsEnabled ? VK_LOD_CLAMP_NONE : 0.25f;

    VkSampler newSampler = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateSampler(m_device->logical(), &ci, nullptr, &newSampler));

    m_descriptorAllocator->updateSampler(
        m_scene->defaultMaterialSamplerSlot, newSampler);

    // Destroy only our previously-owned sampler. The very first call (m_materialSampler==null)
    // leaves the GltfLoader-created default in scene.ownedSamplers; it's no longer
    // referenced by the bindless slot but will be freed by Scene::destroy during shutdown.
    // Not destroying it here avoids double-free paths if the scene is later swapped.
    if (m_materialSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_materialSampler, nullptr);
    }
    m_materialSampler = newSampler;

    ENIGMA_LOG_INFO("[renderer] material sampler rebuilt: mipmaps={}, aniso={}x",
                    m_textureFilterSettings.mipmapsEnabled ? "on" : "off",
                    requestedAniso);
}

} // namespace enigma
