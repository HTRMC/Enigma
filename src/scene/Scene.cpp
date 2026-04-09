#include "scene/Scene.h"

#include "gfx/Allocator.h"
#include "gfx/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

namespace enigma {

void Scene::destroy(gfx::Device& device, gfx::Allocator& allocator) {
    VkDevice dev = device.logical();

    for (auto& sampler : ownedSamplers) {
        vkDestroySampler(dev, sampler, nullptr);
    }
    ownedSamplers.clear();

    for (auto& img : ownedImages) {
        if (img.view != VK_NULL_HANDLE) {
            vkDestroyImageView(dev, img.view, nullptr);
        }
        if (img.image != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator.handle(), img.image, img.allocation);
        }
    }
    ownedImages.clear();

    for (auto& buf : ownedBuffers) {
        if (buf.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator.handle(), buf.buffer, buf.allocation);
        }
    }
    ownedBuffers.clear();

    primitives.clear();
    nodes.clear();
    materials.clear();
}

} // namespace enigma
