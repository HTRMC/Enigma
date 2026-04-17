#pragma once

#include "core/Types.h"

#include <volk.h>

#include <vector>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {

class Device;
class Allocator;

// One-shot GPU upload utility. Batches buffer and image copies into a
// single command buffer, submits with a fence, and blocks until complete.
// All staging memory is freed after the fence signals.
//
// Usage:
//   UploadContext ctx(device, allocator);
//   ctx.uploadBuffer(dst, data, size);
//   ctx.uploadImage(dst, extent, format, pixels, size);
//   ctx.submitAndWait();   // blocks, frees staging
//
// Extracted from the TrianglePass texture upload pattern so it can be
// reused by MeshPass, GltfLoader, and any future upload site.
class UploadContext {
public:
    UploadContext(Device& device, Allocator& allocator);
    ~UploadContext();

    UploadContext(const UploadContext&)            = delete;
    UploadContext& operator=(const UploadContext&) = delete;
    UploadContext(UploadContext&&)                 = delete;
    UploadContext& operator=(UploadContext&&)      = delete;

    // Stage a buffer-to-buffer copy. `dst` must have TRANSFER_DST_BIT.
    void uploadBuffer(VkBuffer dst, const void* data, VkDeviceSize size);

    // Stage a buffer-to-image copy with layout transitions:
    //   UNDEFINED → TRANSFER_DST_OPTIMAL (pre-copy)
    //   TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (post-copy)
    // `dst` must have TRANSFER_DST_BIT | SAMPLED_BIT.
    void uploadImage(VkImage dst, VkExtent3D extent, VkFormat format,
                     const void* pixels, VkDeviceSize size);

    // Stage a buffer-to-image copy of the base mip then generate all
    // subsequent mip levels via vkCmdBlitImage with a linear filter.
    // `dst` must have TRANSFER_SRC_BIT | TRANSFER_DST_BIT | SAMPLED_BIT
    // and have been created with `mipLevels` levels. Ends with every level
    // in SHADER_READ_ONLY_OPTIMAL.
    void uploadImageWithMipchain(VkImage dst, u32 width, u32 height,
                                 VkFormat format, u32 mipLevels,
                                 const void* basePixels, VkDeviceSize baseSize);

    // Submit the command buffer, wait on fence, free all staging resources.
    // Must be called exactly once. The UploadContext is consumed after this.
    void submitAndWait();

private:
    struct StagingEntry {
        VkBuffer      buffer     = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };

    Device*    m_device    = nullptr;
    Allocator* m_allocator = nullptr;

    VkCommandPool   m_commandPool   = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkFence         m_fence         = VK_NULL_HANDLE;
    bool            m_submitted     = false;

    std::vector<StagingEntry> m_stagingBuffers;
};

} // namespace enigma::gfx
