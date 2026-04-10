#include "renderer/Renderer.h"

#include "core/Assert.h"
#include "core/Log.h"
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
#include "renderer/GBufferPass.h"
#include "renderer/LightingPass.h"
#include "renderer/MeshPass.h"
#include "renderer/RTReflectionPass.h"
#include "renderer/RTGIPass.h"
#include "renderer/RTShadowPass.h"
#include "renderer/WetRoadPass.h"
#include "renderer/Denoiser.h"
#include "renderer/TrianglePass.h"
#include "scene/Camera.h"
#include "scene/Scene.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstring>

namespace enigma {

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

    // Deferred G-buffer pass — allocate images at swapchain resolution.
    m_gbufferPass = std::make_unique<GBufferPass>(*m_device, *m_allocator);
    m_gbufferPass->allocate(m_swapchain->extent());
    m_gbufferPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_gbufferPass->registerHotReload(*m_shaderHotReload);

    // Register G-buffer textures as bindless sampled images.
    m_gbufAlbedoSlot     = m_descriptorAllocator->registerSampledImage(
        m_gbufferPass->albedoView(),     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufNormalSlot     = m_descriptorAllocator->registerSampledImage(
        m_gbufferPass->normalView(),     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufMetalRoughSlot = m_descriptorAllocator->registerSampledImage(
        m_gbufferPass->metalRoughView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufMotionVecSlot  = m_descriptorAllocator->registerSampledImage(
        m_gbufferPass->motionVecView(),  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufDepthSlot      = m_descriptorAllocator->registerSampledImage(
        m_gbufferPass->depthView(),      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

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

    // Deferred lighting pass.
    m_lightingPass = std::make_unique<LightingPass>(*m_device);
    m_lightingPass->buildPipeline(*m_shaderManager,
                                   m_descriptorAllocator->layout(),
                                   m_swapchain->format());
    m_lightingPass->registerHotReload(*m_shaderHotReload);

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

    ENIGMA_LOG_INFO("[renderer] constructed (camera slots: {}, {})",
                    m_cameraBuffers[0].bindlessSlot,
                    m_cameraBuffers[1].bindlessSlot);
}

Renderer::~Renderer() {
    if (m_device) {
        vkDeviceWaitIdle(m_device->logical());
    }

    // Clean up G-buffer sampler (pass objects destroy their own images/pipelines).
    if (m_gbufferSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_gbufferSampler, nullptr);
    }

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
    } else {
        // Identity camera fallback.
        data.view        = mat4{1.0f};
        data.proj        = mat4{1.0f};
        data.viewProj    = mat4{1.0f};
        data.invViewProj = mat4{1.0f};
        data.worldPos    = vec4{0.0f, 0.0f, 0.0f, 1.0f};
    }

    // Inject previous frame's viewProj for motion vector computation.
    data.prevViewProj = m_prevViewProj;
    std::memcpy(cb.mapped, &data, sizeof(data));

    // Save current viewProj so next frame can use it as prevViewProj.
    m_prevViewProj = data.viewProj;
}

void Renderer::drawFrame() {
    {
        const auto fb = m_window.framebufferSize();
        if (fb.width == 0 || fb.height == 0) {
            m_window.waitEvents();
            return;
        }
    }

    m_shaderHotReload->poll();

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

    ENIGMA_VK_CHECK(vkResetCommandPool(dev, frame.commandPool, 0));

    // Read back GPU timings from the previous frame before resetting the pool.
    // (The previous frame's submissions are guaranteed done by the timeline wait above.)
    if (frame.frameValue > 0) {
        const auto results = m_gpuProfiler->readback();
        for (const auto& r : results) {
            ENIGMA_LOG_INFO("[gpu] {} = {:.3f} ms", r.name, r.durationMs);
        }
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo));

    m_gpuProfiler->reset(frame.commandBuffer);

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

    const u32 cameraSlot = m_cameraBuffers[m_frameIndex].bindlessSlot;

    if (m_scene != nullptr) {
        // ---- Deferred path: GBufferPass → LightingPass ----
        const auto gbufAlbedo = m_renderGraph->importImage(
            "gbuf_albedo", m_gbufferPass->albedoImage(), m_gbufferPass->albedoView(),
            GBufferPass::kAlbedoFormat,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufNormal = m_renderGraph->importImage(
            "gbuf_normal", m_gbufferPass->normalImage(), m_gbufferPass->normalView(),
            GBufferPass::kNormalFormat,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufMetalRough = m_renderGraph->importImage(
            "gbuf_metalrough", m_gbufferPass->metalRoughImage(), m_gbufferPass->metalRoughView(),
            GBufferPass::kMetalRoughFormat,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufMotionVec = m_renderGraph->importImage(
            "gbuf_motionvec", m_gbufferPass->motionVecImage(), m_gbufferPass->motionVecView(),
            GBufferPass::kMotionVecFormat,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufDepth = m_renderGraph->importImage(
            "gbuf_depth", m_gbufferPass->depthImage(), m_gbufferPass->depthView(),
            GBufferPass::kDepthFormat,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        gfx::RenderGraph::RasterPassDesc gbufPassDesc{};
        gbufPassDesc.name         = "GBufferPass";
        gbufPassDesc.colorTargets = {gbufAlbedo, gbufNormal, gbufMetalRough, gbufMotionVec};
        gbufPassDesc.depthTarget  = gbufDepth;
        gbufPassDesc.clearColor   = {{0.0f, 0.0f, 0.0f, 0.0f}};
        gbufPassDesc.clearDepth   = {0.0f, 0}; // reverse-Z: far = 0
        gbufPassDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
            m_gpuProfiler->beginZone(cmd, "GBufferPass");
            m_gbufferPass->record(cmd, m_descriptorAllocator->globalSet(),
                                   ext, *m_scene, cameraSlot);
            m_gpuProfiler->endZone(cmd);
        };
        m_renderGraph->addRasterPass(std::move(gbufPassDesc));

        // RT reflection pass — runs between G-buffer and lighting.
        // On RT hardware: dispatches vkCmdTraceRaysKHR.
        // On Min tier: no-op (SSR integration deferred to Phase 2).
        if (m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
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
                                            m_tlasSlot, m_reflectionSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtReflectDesc));
        }

        // RT GI pass — conditional on gpuTier >= Recommended.
        if (m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc rtGIDesc{};
            rtGIDesc.name    = "RTGIPass";
            rtGIDesc.sampledInputs = {gbufNormal, gbufDepth};
            rtGIDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "RTGIPass");
                m_giPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                  m_gbufNormalSlot, m_gbufDepthSlot,
                                  cameraSlot, m_tlasSlot, m_giSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtGIDesc));
        }

        // RT Shadow pass — conditional on gpuTier >= Recommended.
        if (m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc rtShadowDesc{};
            rtShadowDesc.name    = "RTShadowPass";
            rtShadowDesc.sampledInputs = {gbufNormal, gbufDepth};
            rtShadowDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "RTShadowPass");
                m_shadowPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                      m_gbufNormalSlot, m_gbufDepthSlot,
                                      cameraSlot, m_tlasSlot, m_shadowSlot,
                                      vec4{m_light.direction, 0.02f}); // 0.02 rad cone angle
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtShadowDesc));
        }

        // Wet Road pass — conditional on gpuTier >= Recommended.
        if (m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc wetRoadDesc{};
            wetRoadDesc.name    = "WetRoadPass";
            wetRoadDesc.sampledInputs = {gbufNormal, gbufDepth};
            wetRoadDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "WetRoadPass");
                m_wetRoadPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                       m_gbufNormalSlot, m_gbufDepthSlot,
                                       cameraSlot, m_tlasSlot, m_wetRoadSlot,
                                       m_wetnessFactor);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(wetRoadDesc));
        }

        // Denoise RT effects — conditional on gpuTier >= Recommended.
        if (m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
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
        lightPassDesc.colorTargets  = {colorHandle};
        lightPassDesc.sampledInputs = {gbufAlbedo, gbufNormal, gbufMetalRough, gbufDepth};
        lightPassDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
        lightPassDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
            m_gpuProfiler->beginZone(cmd, "LightingPass");
            m_lightingPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                    m_gbufAlbedoSlot, m_gbufNormalSlot,
                                    m_gbufMetalRoughSlot, m_gbufDepthSlot,
                                    cameraSlot, m_gbufferSamplerSlot,
                                    vec4{m_light.direction, m_light.intensity},
                                    vec4{m_light.color, 0.0f});
            m_gpuProfiler->endZone(cmd);
        };
        m_renderGraph->addRasterPass(std::move(lightPassDesc));

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

void Renderer::resizeGBuffer(VkExtent2D extent) {
    // GBufferPass::allocate() calls vkDeviceWaitIdle internally before
    // destroying existing images, so no explicit idle needed here.
    m_gbufferPass->allocate(extent);

    // Re-write the bindless descriptor slots to point at the new image views.
    m_descriptorAllocator->updateSampledImage(
        m_gbufAlbedoSlot,     m_gbufferPass->albedoView(),     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufNormalSlot,     m_gbufferPass->normalView(),     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufMetalRoughSlot, m_gbufferPass->metalRoughView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufMotionVecSlot,  m_gbufferPass->motionVecView(),  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufDepthSlot,      m_gbufferPass->depthView(),      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

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
}

} // namespace enigma
