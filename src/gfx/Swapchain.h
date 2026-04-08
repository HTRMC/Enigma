#pragma once

#include "core/Types.h"

#include <volk.h>

#include <vector>

namespace enigma {
class Window;
} // namespace enigma

namespace enigma::gfx {

class Instance;
class Device;

// Owns the VkSurfaceKHR + VkSwapchainKHR + per-image VkImageView list.
// Uses dynamic rendering exclusively — no VkRenderPass / VkFramebuffer.
class Swapchain {
public:
    Swapchain(Instance& instance, Device& device, Window& window);
    ~Swapchain();

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&)                 = delete;
    Swapchain& operator=(Swapchain&&)      = delete;

    VkSurfaceKHR      surface()    const { return m_surface; }
    VkSwapchainKHR    handle()     const { return m_swapchain; }
    VkFormat          format()     const { return m_format; }
    VkExtent2D        extent()     const { return m_extent; }
    u32               imageCount() const { return static_cast<u32>(m_images.size()); }

    VkImage     image(u32 index) const { return m_images[index]; }
    VkImageView view(u32 index)  const { return m_views[index]; }

    // Tear down and rebuild at the new extent. Called from resize / out-of-
    // date handling. Arrives functionally at step 28.
    void recreate(u32 width, u32 height);

private:
    void create(u32 width, u32 height);
    void destroyImagesAndSwapchain();

    Instance* m_instance = nullptr;
    Device*   m_device   = nullptr;

    VkSurfaceKHR     m_surface   = VK_NULL_HANDLE;
    VkSwapchainKHR   m_swapchain = VK_NULL_HANDLE;
    VkFormat         m_format    = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR  m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D       m_extent{};
    std::vector<VkImage>     m_images;
    std::vector<VkImageView> m_views;
};

} // namespace enigma::gfx
