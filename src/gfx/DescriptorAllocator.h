#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma::gfx {

class Device;

// DescriptorAllocator
// ===================
// Owns the single global bindless descriptor set used by every pipeline
// in the engine. The layout has four binding arrays (one per "bindless"
// resource type) so that adding a new resource class at milestone 2 is a
// `vkUpdateDescriptorSets` call, not a layout rewrite.
//
// Layout:
//   binding 0: SAMPLED_IMAGE   (PARTIALLY_BOUND + UPDATE_AFTER_BIND)
//   binding 1: STORAGE_IMAGE   (PARTIALLY_BOUND + UPDATE_AFTER_BIND)
//   binding 2: STORAGE_BUFFER  (PARTIALLY_BOUND + UPDATE_AFTER_BIND)
//   binding 3: SAMPLER         (PARTIALLY_BOUND + UPDATE_AFTER_BIND +
//                               VARIABLE_DESCRIPTOR_COUNT)
//
// Only binding 3 is VARIABLE_DESCRIPTOR_COUNT per Vulkan spec rule that
// only the last binding of a layout may carry that flag. Bindings 0-2
// are fixed-size PARTIALLY_BOUND arrays sized to the device limit
// (clamped to conservative caps).
//
// Second-caller design intent (discharges Principle 6):
//   - At milestone 1 only the storage buffer binding is populated (the
//     triangle vertex SSBO sits in binding 2).
//   - Milestone 2 textures will land in binding 0, compute storage
//     images in binding 1, samplers in binding 3 — with no layout,
//     pool, or set changes required.
class DescriptorAllocator {
public:
    // Per-type caps. Clamped to device limits at construction time.
    struct Caps {
        u32 sampledImages  = 16384;
        u32 storageImages  = 512;
        u32 storageBuffers = 512;
        u32 samplers       = 256;
    };

    explicit DescriptorAllocator(Device& device, Caps caps = {});
    ~DescriptorAllocator();

    DescriptorAllocator(const DescriptorAllocator&)            = delete;
    DescriptorAllocator& operator=(const DescriptorAllocator&) = delete;
    DescriptorAllocator(DescriptorAllocator&&)                 = delete;
    DescriptorAllocator& operator=(DescriptorAllocator&&)      = delete;

    VkDescriptorSetLayout layout()    const { return m_layout; }
    VkDescriptorSet       globalSet() const { return m_globalSet; }
    const Caps&           caps()      const { return m_caps; }

    // Register a storage buffer in binding 2 and return the bindless
    // slot index. At milestone 1 this is the only populated binding.
    // The descriptor is written via `vkUpdateDescriptorSets` with
    // UPDATE_AFTER_BIND semantics so it is legal to call at any time,
    // even after the set has been bound to a command buffer.
    u32 registerStorageBuffer(VkBuffer buffer, VkDeviceSize size);

    // --- stubs for milestone 2 (Principle 6: pin the API shape now) ---
    u32 registerSampledImage(VkImageView view, VkImageLayout layout);
    u32 registerStorageImage(VkImageView view);
    u32 registerSampler(VkSampler sampler);

    // Re-write an existing sampled-image slot without allocating a new one.
    // Used when G-buffer images are reallocated on swapchain resize.
    void updateSampledImage(u32 slot, VkImageView view, VkImageLayout layout);

private:
    Device*               m_device     = nullptr;
    Caps                  m_caps{};
    VkDescriptorSetLayout m_layout     = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       m_globalSet  = VK_NULL_HANDLE;

    u32 m_nextSampledImage  = 0;
    u32 m_nextStorageImage  = 0;
    u32 m_nextStorageBuffer = 0;
    u32 m_nextSampler       = 0;
};

} // namespace enigma::gfx
