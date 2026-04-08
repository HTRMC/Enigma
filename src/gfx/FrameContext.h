#pragma once

#include "core/Types.h"

#include <volk.h>

#include <array>

namespace enigma::gfx {

class Device;

inline constexpr u32 MAX_FRAMES_IN_FLIGHT = 2;

// Per-frame GPU resources. Each FrameContext owns:
//   - a transient command pool (reset each frame)
//   - one primary command buffer allocated from that pool
//   - imageAvailable (binary semaphore, acquire -> render wait)
//   - renderFinished (binary semaphore, render -> present wait)
//   - inFlight (timeline semaphore with monotonic frameValue)
//
// The hybrid sync model (binary for presentation, timeline for CPU/GPU
// pipelining) matches the Vulkan 1.3 idiom for modern engines — see
// the plan §"Option set B" for the full rationale.
struct FrameContext {
    VkCommandPool   commandPool    = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer  = VK_NULL_HANDLE;
    VkSemaphore     imageAvailable = VK_NULL_HANDLE;
    VkSemaphore     renderFinished = VK_NULL_HANDLE;
    VkSemaphore     inFlight       = VK_NULL_HANDLE; // timeline
    u64             frameValue     = 0;              // monotonic, starts at 0
};

// Owns MAX_FRAMES_IN_FLIGHT FrameContexts. Construction creates all
// command pools / buffers / semaphores; destruction destroys them in
// the reverse order after `vkDeviceWaitIdle`.
class FrameContextSet {
public:
    explicit FrameContextSet(Device& device);
    ~FrameContextSet();

    FrameContextSet(const FrameContextSet&)            = delete;
    FrameContextSet& operator=(const FrameContextSet&) = delete;
    FrameContextSet(FrameContextSet&&)                 = delete;
    FrameContextSet& operator=(FrameContextSet&&)      = delete;

    FrameContext& get(u32 index)       { return m_frames[index]; }
    const FrameContext& get(u32 index) const { return m_frames[index]; }

private:
    Device* m_device = nullptr;
    std::array<FrameContext, MAX_FRAMES_IN_FLIGHT> m_frames{};
};

} // namespace enigma::gfx
