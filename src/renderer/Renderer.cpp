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
#include "renderer/MeshPass.h"
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
        data.view     = mat4{1.0f};
        data.proj     = mat4{1.0f};
        data.viewProj = mat4{1.0f};
        data.worldPos = vec4{0.0f, 0.0f, 0.0f, 1.0f};
    }

    std::memcpy(cb.mapped, &data, sizeof(data));
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

    const auto depthHandle = m_renderGraph->importImage(
        "swapchain_depth",
        depthImage, depthView, m_swapchain->depthFormat(),
        VK_IMAGE_LAYOUT_UNDEFINED,  // re-cleared every frame
        VK_IMAGE_LAYOUT_UNDEFINED,  // don't care about final layout
        VK_IMAGE_ASPECT_DEPTH_BIT);

    const u32 cameraSlot = m_cameraBuffers[m_frameIndex].bindlessSlot;

    gfx::RenderGraph::RasterPassDesc meshPassDesc{};
    meshPassDesc.name         = "MeshPass";
    meshPassDesc.colorTargets = {colorHandle};
    meshPassDesc.depthTarget  = depthHandle;
    meshPassDesc.clearColor   = {{0.02f, 0.02f, 0.05f, 1.0f}};
    meshPassDesc.clearDepth   = {0.0f, 0}; // reverse-Z: far = 0
    meshPassDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
        m_gpuProfiler->beginZone(cmd, "MeshPass");
        if (m_scene != nullptr) {
            m_meshPass->record(cmd,
                               m_descriptorAllocator->globalSet(),
                               ext, *m_scene, cameraSlot,
                               vec4{m_light.direction, m_light.intensity},
                               vec4{m_light.color, 0.0f});
        } else {
            m_trianglePass->record(cmd,
                                   m_descriptorAllocator->globalSet(),
                                   ext);
        }
        m_gpuProfiler->endZone(cmd);
    };
    m_renderGraph->addRasterPass(std::move(meshPassDesc));

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
    } else if (presentResult != VK_SUCCESS) {
        ENIGMA_VK_CHECK(presentResult);
    }

    m_frameIndex = (m_frameIndex + 1) % gfx::MAX_FRAMES_IN_FLIGHT;
}

} // namespace enigma
