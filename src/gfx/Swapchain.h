#pragma once

#include "core/Types.h"

#include <volk.h>

#include <vector>

// Forward-declare the VMA allocation opaque handle so Swapchain can own
// a depth image's allocation without pulling vk_mem_alloc.h into every
// TU that includes this header.
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma {
class Window;
} // namespace enigma

namespace enigma::gfx {

class Instance;
class Device;
class Allocator;

// Owns the VkSurfaceKHR + VkSwapchainKHR + per-image VkImageView list,
// plus a single shared depth image sized to the color extent. Uses
// dynamic rendering exclusively — no VkRenderPass / VkFramebuffer.
//
// Depth image lifetime rationale: one VkImage shared across frames
// instead of one-per-frame-in-flight. Each frame begins with an
// UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL transition + LOAD_OP_CLEAR, so
// the previous frame's contents are explicitly discarded. The timeline
// semaphore wait in `Renderer::drawFrame` is the CPU/GPU sync gate.
// Recreation on resize happens inside `recreate()` together with the
// swapchain rebuild so extent never drifts.
class Swapchain {
public:
    Swapchain(Instance& instance, Device& device, Allocator& allocator, Window& window);
    ~Swapchain();

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&)                 = delete;
    Swapchain& operator=(Swapchain&&)      = delete;

    VkSurfaceKHR      surface()     const { return m_surface;     }
    VkSwapchainKHR    handle()      const { return m_swapchain;   }
    VkFormat          format()      const { return m_format;      }
    VkExtent2D        extent()      const { return m_extent;      }
    u32               imageCount()  const { return static_cast<u32>(m_images.size()); }

    VkImage     image(u32 index) const { return m_images[index]; }
    VkImageView view(u32 index)  const { return m_views[index]; }

    // Depth attachment accessors. Single shared depth image — see the
    // class-level comment for lifetime rationale. The format is picked
    // once at construction; it is `VK_FORMAT_D32_SFLOAT` (no stencil)
    // which is a universally-supported depth format.
    VkFormat    depthFormat() const { return m_depthFormat; }
    VkImage     depthImage()  const { return m_depthImage;  }
    VkImageView depthView()   const { return m_depthView;   }

    // Per-swapchain-image "render finished" binary semaphore. Presentation
    // ties the semaphore lifetime to the IMAGE, not to the frame-in-flight
    // slot, so a single shared semaphore per slot cannot be re-signaled
    // safely while the previous present on another image still holds it.
    // One semaphore per image sidesteps the problem cleanly.
    VkSemaphore renderFinished(u32 index) const { return m_renderFinished[index]; }

    // Tear down and rebuild at the new extent. Called from resize / out-of-
    // date handling. Arrives functionally at step 28.
    void recreate(u32 width, u32 height);

private:
    void create(u32 width, u32 height);
    void destroyImagesAndSwapchain();

    Instance*  m_instance  = nullptr;
    Device*    m_device    = nullptr;
    Allocator* m_allocator = nullptr;

    VkSurfaceKHR     m_surface   = VK_NULL_HANDLE;
    VkSwapchainKHR   m_swapchain = VK_NULL_HANDLE;
    VkFormat         m_format    = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR  m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D       m_extent{};
    std::vector<VkImage>     m_images;
    std::vector<VkImageView> m_views;
    std::vector<VkSemaphore> m_renderFinished;

    // Depth attachment (single shared image).
    VkFormat      m_depthFormat     = VK_FORMAT_D32_SFLOAT;
    VkImage       m_depthImage      = VK_NULL_HANDLE;
    VmaAllocation m_depthAllocation = nullptr;
    VkImageView   m_depthView       = VK_NULL_HANDLE;
};

} // namespace enigma::gfx
