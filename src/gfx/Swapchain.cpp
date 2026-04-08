#include "gfx/Swapchain.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"
#include "gfx/Instance.h"
#include "platform/Window.h"

#include <GLFW/glfw3.h>

// vk_mem_alloc.h is driven via volk — the guards match the existing
// Allocator.cpp usage (function pointers supplied manually, no static
// or dynamic loading by VMA itself).
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <algorithm>

namespace enigma::gfx {

Swapchain::Swapchain(Instance& instance, Device& device, Allocator& allocator, Window& window)
    : m_instance(&instance)
    , m_device(&device)
    , m_allocator(&allocator) {

    // Surface creation is the Swapchain's responsibility — the surface has
    // the same lifetime as the swapchain for all practical purposes.
    ENIGMA_VK_CHECK(glfwCreateWindowSurface(
        m_instance->handle(),
        window.handle(),
        nullptr,
        &m_surface));

    const auto fb = window.framebufferSize();
    create(fb.width, fb.height);
}

Swapchain::~Swapchain() {
    destroyImagesAndSwapchain();
    if (m_surface != VK_NULL_HANDLE && m_instance != nullptr) {
        vkDestroySurfaceKHR(m_instance->handle(), m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }
}

void Swapchain::create(u32 width, u32 height) {
    VkPhysicalDevice phys = m_device->physical();

    // Query surface capabilities.
    VkSurfaceCapabilitiesKHR caps{};
    ENIGMA_VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, m_surface, &caps));

    // Pick format: prefer B8G8R8A8_SRGB / SRGB_NONLINEAR.
    u32 fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, m_surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, m_surface, &fmtCount, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    m_format     = chosen.format;
    m_colorSpace = chosen.colorSpace;

    // Present mode: FIFO is always available and is the spec-guaranteed
    // baseline; stick with it at this milestone.
    const VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;

    // Extent: clamp to the capabilities window. Zero-size framebuffer is a
    // caller error at step 26; step 42 handles minimized windows cleanly.
    VkExtent2D extent{width, height};
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        extent.width  = std::clamp(extent.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    m_extent = extent;

    u32 imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface          = m_surface;
    info.minImageCount    = imageCount;
    info.imageFormat      = m_format;
    info.imageColorSpace  = m_colorSpace;
    info.imageExtent      = m_extent;
    info.imageArrayLayers = 1;
    info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                          | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform     = caps.currentTransform;
    info.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode      = presentMode;
    info.clipped          = VK_TRUE;
    info.oldSwapchain     = VK_NULL_HANDLE;

    ENIGMA_VK_CHECK(vkCreateSwapchainKHR(m_device->logical(), &info, nullptr, &m_swapchain));
    ENIGMA_ASSERT(m_swapchain != VK_NULL_HANDLE);

    // Retrieve the swapchain images and build a view per image for use
    // as a color attachment under dynamic rendering.
    u32 actualImageCount = 0;
    vkGetSwapchainImagesKHR(m_device->logical(), m_swapchain, &actualImageCount, nullptr);
    m_images.resize(actualImageCount);
    vkGetSwapchainImagesKHR(m_device->logical(), m_swapchain, &actualImageCount, m_images.data());

    m_views.resize(actualImageCount, VK_NULL_HANDLE);
    for (u32 i = 0; i < actualImageCount; ++i) {
        VkImageViewCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image                           = m_images[i];
        vi.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        vi.format                          = m_format;
        vi.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        vi.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        vi.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        vi.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        vi.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.baseMipLevel   = 0;
        vi.subresourceRange.levelCount     = 1;
        vi.subresourceRange.baseArrayLayer = 0;
        vi.subresourceRange.layerCount     = 1;
        ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &vi, nullptr, &m_views[i]));
    }

    // One renderFinished binary semaphore per swapchain image — see the
    // Swapchain.h comment for the "present ties semaphore to image, not
    // slot" rationale.
    m_renderFinished.resize(actualImageCount, VK_NULL_HANDLE);
    for (u32 i = 0; i < actualImageCount; ++i) {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        ENIGMA_VK_CHECK(vkCreateSemaphore(m_device->logical(), &si, nullptr, &m_renderFinished[i]));
    }

    // -------------------------------------------------------------------
    // Depth attachment: a single shared VkImage sized to the color
    // extent. Recreated alongside the swapchain so extent stays in
    // sync through resizes. D32_SFLOAT is universally supported and
    // is the format declared in Swapchain.h; no runtime format
    // query needed.
    // -------------------------------------------------------------------
    VkImageCreateInfo depthImgInfo{};
    depthImgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depthImgInfo.imageType     = VK_IMAGE_TYPE_2D;
    depthImgInfo.format        = m_depthFormat;
    depthImgInfo.extent        = { m_extent.width, m_extent.height, 1 };
    depthImgInfo.mipLevels     = 1;
    depthImgInfo.arrayLayers   = 1;
    depthImgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    depthImgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    depthImgInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImgInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    depthImgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo depthAllocInfo{};
    depthAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &depthImgInfo, &depthAllocInfo,
                                   &m_depthImage, &m_depthAllocation, nullptr));

    VkImageViewCreateInfo depthViewInfo{};
    depthViewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthViewInfo.image    = m_depthImage;
    depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.format   = m_depthFormat;
    depthViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &depthViewInfo, nullptr, &m_depthView));

    ENIGMA_LOG_INFO("[gfx] swapchain created ({}x{}, {} images, format {}, depth D32_SFLOAT)",
                    m_extent.width, m_extent.height, actualImageCount,
                    static_cast<int>(m_format));
}

void Swapchain::destroyImagesAndSwapchain() {
    if (m_device == nullptr || m_device->logical() == VK_NULL_HANDLE) {
        return;
    }
    // Depth attachment first (reverse-order cleanup: view before image).
    if (m_depthView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), m_depthView, nullptr);
        m_depthView = VK_NULL_HANDLE;
    }
    if (m_depthImage != VK_NULL_HANDLE && m_allocator != nullptr) {
        vmaDestroyImage(m_allocator->handle(), m_depthImage, m_depthAllocation);
        m_depthImage      = VK_NULL_HANDLE;
        m_depthAllocation = nullptr;
    }
    for (VkSemaphore s : m_renderFinished) {
        if (s != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device->logical(), s, nullptr);
        }
    }
    m_renderFinished.clear();
    for (VkImageView v : m_views) {
        if (v != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device->logical(), v, nullptr);
        }
    }
    m_views.clear();
    m_images.clear();
    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device->logical(), m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

void Swapchain::recreate(u32 width, u32 height) {
    if (m_device == nullptr || m_device->logical() == VK_NULL_HANDLE) {
        return;
    }
    // Serialize with the GPU before tearing down swapchain resources.
    // Dynamic rendering path: no VkRenderPass / VkFramebuffer to rebuild.
    vkDeviceWaitIdle(m_device->logical());
    destroyImagesAndSwapchain();
    create(width, height);
}

} // namespace enigma::gfx
