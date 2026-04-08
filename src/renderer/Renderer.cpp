#include "renderer/Renderer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/FrameContext.h"
#include "gfx/Instance.h"
#include "gfx/ShaderManager.h"
#include "gfx/Swapchain.h"
#include "platform/Window.h"
#include "renderer/TrianglePass.h"

#include <algorithm>

namespace enigma {

Renderer::Renderer(Window& window)
    : m_window(window)
    , m_instance(std::make_unique<gfx::Instance>())
    , m_device(std::make_unique<gfx::Device>(*m_instance))
    , m_allocator(std::make_unique<gfx::Allocator>(*m_instance, *m_device))
    , m_swapchain(std::make_unique<gfx::Swapchain>(*m_instance, *m_device, m_window))
    , m_frames(std::make_unique<gfx::FrameContextSet>(*m_device))
    , m_descriptorAllocator(std::make_unique<gfx::DescriptorAllocator>(*m_device))
    , m_shaderManager(std::make_unique<gfx::ShaderManager>(*m_device))
    , m_trianglePass(std::make_unique<TrianglePass>(*m_device, *m_allocator, *m_descriptorAllocator)) {

    m_trianglePass->buildPipeline(*m_shaderManager,
                                  m_descriptorAllocator->layout(),
                                  m_swapchain->format());

    ENIGMA_LOG_INFO("[renderer] constructed");
}

Renderer::~Renderer() {
    if (m_device) {
        vkDeviceWaitIdle(m_device->logical());
    }
    ENIGMA_LOG_INFO("[renderer] shutdown");
}

void Renderer::drawFrame() {
    // Minimized window (0x0): block until an event arrives so the
    // frame loop does not spin, and return without submitting any
    // Vulkan work. Zero contribution to the validation counter.
    {
        const auto fb = m_window.framebufferSize();
        if (fb.width == 0 || fb.height == 0) {
            m_window.waitEvents();
            return;
        }
    }

    VkDevice dev = m_device->logical();
    gfx::FrameContext& frame = m_frames->get(m_frameIndex);

    // -------------------------------------------------------------------
    // Timeline semaphore wait: hold off until the FrameContext's last
    // submitted work (frameValue - MAX_FRAMES_IN_FLIGHT + 1) has retired.
    // For the first few frames this clamps to zero which is an immediate
    // no-wait.
    // -------------------------------------------------------------------
    const u64 waitValue = (frame.frameValue >= gfx::MAX_FRAMES_IN_FLIGHT)
                              ? (frame.frameValue - gfx::MAX_FRAMES_IN_FLIGHT + 1)
                              : 0;
    if (waitValue > 0) {
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores    = &frame.inFlight;
        waitInfo.pValues        = &waitValue;
        ENIGMA_VK_CHECK(vkWaitSemaphores(dev, &waitInfo, UINT64_MAX));
    }

    // -------------------------------------------------------------------
    // Acquire next swapchain image. OUT_OF_DATE triggers a rebuild and
    // skips the frame; SUBOPTIMAL is accepted and flagged to rebuild
    // after present.
    // -------------------------------------------------------------------
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

    // -------------------------------------------------------------------
    // Record. Reset the per-frame command pool, begin the command
    // buffer, record the triangle pass, end.
    //
    // NOTE: At step 39 the color attachment is still in UNDEFINED layout
    // and no dynamic-rendering begin/end is wired. Validation will fire
    // at this commit. Step 40 adds the sync2 layout transitions and
    // vkCmdBeginRendering/vkCmdEndRendering to make the frame
    // validation-clean.
    //
    // BISECT WARNING: do not `git bisect` any unrelated regression
    // across the boundary between this commit and step 40 — the engine
    // is intentionally not validation-clean at this intermediate state.
    // -------------------------------------------------------------------
    ENIGMA_VK_CHECK(vkResetCommandPool(dev, frame.commandPool, 0));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo));

    VkImage     targetImage = m_swapchain->image(imageIndex);
    VkImageView targetView  = m_swapchain->view(imageIndex);
    const VkExtent2D extent = m_swapchain->extent();

    // ---- UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL (sync2 barrier) --------
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask    = 0;
        barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image            = targetImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(frame.commandBuffer, &dep);
    }

    // ---- Dynamic rendering begin --------------------------------------
    VkRenderingAttachmentInfo colorAttach{};
    colorAttach.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttach.imageView   = targetView;
    colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttach.clearValue.color = {{0.02f, 0.02f, 0.05f, 1.0f}};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset    = {0, 0};
    renderingInfo.renderArea.extent    = extent;
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttach;
    vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);

    // ---- Triangle draw ------------------------------------------------
    m_trianglePass->record(frame.commandBuffer,
                           m_descriptorAllocator->globalSet(),
                           extent);

    // ---- Dynamic rendering end ----------------------------------------
    vkCmdEndRendering(frame.commandBuffer);

    // ---- COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR ------------------
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask     = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.dstAccessMask    = 0;
        barrier.oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout        = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image            = targetImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(frame.commandBuffer, &dep);
    }

    ENIGMA_VK_CHECK(vkEndCommandBuffer(frame.commandBuffer));

    // -------------------------------------------------------------------
    // Submit. Waits on `imageAvailable` (binary), signals
    // `renderFinished` (binary) for present, plus the timeline
    // `inFlight` at value `frameValue + 1` for CPU/GPU pipelining.
    // -------------------------------------------------------------------
    const u64 signalValue = frame.frameValue + 1;

    const VkSemaphore        waitSems[]   = { frame.imageAvailable };
    const VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    const VkSemaphore        signalSems[] = { frame.renderFinished, frame.inFlight };
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

    // -------------------------------------------------------------------
    // Present.
    // -------------------------------------------------------------------
    VkSwapchainKHR swapchain = m_swapchain->handle();

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &frame.renderFinished;
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
