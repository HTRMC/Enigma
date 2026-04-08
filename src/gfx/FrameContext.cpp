#include "gfx/FrameContext.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"

namespace enigma::gfx {

FrameContextSet::FrameContextSet(Device& device)
    : m_device(&device) {

    VkDevice dev = m_device->logical();

    for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        FrameContext& frame = m_frames[i];

        // Transient command pool, reset-per-frame.
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                                  | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = m_device->graphicsQueueFamily();
        ENIGMA_VK_CHECK(vkCreateCommandPool(dev, &poolInfo, nullptr, &frame.commandPool));

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = frame.commandPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        ENIGMA_VK_CHECK(vkAllocateCommandBuffers(dev, &allocInfo, &frame.commandBuffer));

        // Binary semaphores for acquire/present handshake.
        VkSemaphoreCreateInfo binInfo{};
        binInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        ENIGMA_VK_CHECK(vkCreateSemaphore(dev, &binInfo, nullptr, &frame.imageAvailable));
        ENIGMA_VK_CHECK(vkCreateSemaphore(dev, &binInfo, nullptr, &frame.renderFinished));

        // Timeline semaphore for in-flight gating. Starts at value 0.
        VkSemaphoreTypeCreateInfo typeInfo{};
        typeInfo.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        typeInfo.initialValue  = 0;

        VkSemaphoreCreateInfo timelineInfo{};
        timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        timelineInfo.pNext = &typeInfo;
        ENIGMA_VK_CHECK(vkCreateSemaphore(dev, &timelineInfo, nullptr, &frame.inFlight));

        frame.frameValue = 0;
    }

    ENIGMA_LOG_INFO("[gfx] frame contexts created ({} in flight)", MAX_FRAMES_IN_FLIGHT);
}

FrameContextSet::~FrameContextSet() {
    if (m_device == nullptr || m_device->logical() == VK_NULL_HANDLE) {
        return;
    }
    VkDevice dev = m_device->logical();
    for (FrameContext& frame : m_frames) {
        if (frame.inFlight != VK_NULL_HANDLE) {
            vkDestroySemaphore(dev, frame.inFlight, nullptr);
            frame.inFlight = VK_NULL_HANDLE;
        }
        if (frame.renderFinished != VK_NULL_HANDLE) {
            vkDestroySemaphore(dev, frame.renderFinished, nullptr);
            frame.renderFinished = VK_NULL_HANDLE;
        }
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(dev, frame.imageAvailable, nullptr);
            frame.imageAvailable = VK_NULL_HANDLE;
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev, frame.commandPool, nullptr);
            frame.commandPool   = VK_NULL_HANDLE;
            frame.commandBuffer = VK_NULL_HANDLE;
        }
    }
}

} // namespace enigma::gfx
