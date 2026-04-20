#include "renderer/micropoly/MicropolyPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"

// VMA: Allocator.cpp stamps VMA_IMPLEMENTATION exactly once. Here we
// just need the types for vmaCreateImage / vmaDestroyImage.
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4100 4127 4189 4324 4505)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

namespace enigma {

namespace {

// Probe whether the device supports STORAGE_IMAGE usage for VK_FORMAT_R64_UINT.
// Atomic-min support on the storage image is a separate gate (checked at
// shader-pipeline build time in M3) — here we only ensure the format is
// usable as a storage image in principle.
bool r64UintStorageSupported(VkPhysicalDevice phys) {
    VkImageFormatProperties props{};
    const VkResult r = vkGetPhysicalDeviceImageFormatProperties(
        phys,
        VK_FORMAT_R64_UINT,
        VK_IMAGE_TYPE_2D,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT,
        0,
        &props);
    return r == VK_SUCCESS;
}

// Per plan §3.M0a format-decision block:
//   - Prefer VK_FORMAT_R64_UINT when the device supports it as a storage image.
//   - Otherwise fall back to VK_FORMAT_R32G32_UINT, which M3 aliases for u64
//     atomic-min via SPV_EXT_shader_image_int64 (hence the imageInt64 gate).
//   - If neither path is available, return VK_FORMAT_UNDEFINED — the pass
//     is known-inactive and M3 will not allocate anyway.
VkFormat chooseVisFormat(gfx::Device& device) {
    if (r64UintStorageSupported(device.physical())) {
        return VK_FORMAT_R64_UINT;
    }
    // NOTE (M3.4): the R32G32_UINT fallback path is selected here but the
    // live M3.3 raster pipeline + M3.4 material_eval shader both declare
    // RWTexture2D<uint64_t>, which requires the native R64_UINT path. The
    // MicropolyRasterPass::create() and Renderer.cpp enableMp gates both
    // check supportsShaderImageInt64() AND the raster path (which in turn
    // needs the same capability), so no pipeline is built against this
    // fallback today. Full R32G32 support (mutable-format view + uint2
    // load path in material_eval) is deferred to M4.
    if (device.supportsShaderImageInt64()) {
        return VK_FORMAT_R32G32_UINT;
    }
    return VK_FORMAT_UNDEFINED;
}

// Compose the plan-mandated single log line. Separate function so the
// format-chosen detail appears on the same line as the row status.
const char* visFormatName(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_R64_UINT:    return "R64_UINT";
        case VK_FORMAT_R32G32_UINT: return "R32G32_UINT (aliased u64)";
        case VK_FORMAT_UNDEFINED:   return "none";
        default:                    return "unknown";
    }
}

} // namespace

MicropolyPass::MicropolyPass(gfx::Device& device, const MicropolyConfig& config)
    : m_device(&device)
    , m_config(config)
    , m_caps(micropolyCaps(device)) {

    // MicropolyConfig contract: forceSW and forceHW are mutually exclusive
    // debug flags. Asserting both true is a programmer error — they pick
    // opposite classification paths and the second would silently win (or
    // neither would, depending on consumer). Fail loud in debug builds.
    ENIGMA_ASSERT(!(m_config.forceSW && m_config.forceHW));

    m_visFormat = chooseVisFormat(device);

    // Single boot-time status line per plan §3.M0a. Shape matches the plan
    // exactly: "micropoly: <status>"; the chosen vis-buffer format is
    // appended so hardware-matrix triage only needs this one line.
    if (!m_config.enabled) {
        ENIGMA_LOG_INFO("[renderer] micropoly: disabled (config.enabled=false; row='{}'; visFormat={})",
                        m_caps.statusString, visFormatName(m_visFormat));
    } else if (m_caps.row == HwMatrixRow::Disabled) {
        // Explain which required feature was missing — the plan demands
        // "disabled — missing <reason>" phrasing for this path.
        const char* reason =
            !m_caps.meshShader   ? "VK_EXT_mesh_shader" :
            !m_caps.atomicInt64  ? "VK_KHR_shader_atomic_int64" :
                                   "unknown";
        ENIGMA_LOG_INFO("[renderer] micropoly: disabled \u2014 missing {} (visFormat={})",
                        reason, visFormatName(m_visFormat));
    } else {
        ENIGMA_LOG_INFO("[renderer] micropoly: {} (visFormat={})",
                        m_caps.statusString, visFormatName(m_visFormat));
    }
}

MicropolyPass::~MicropolyPass() {
    // m_visImage/m_visAlloc ownership: by contract the Renderer tears these
    // down before the pass dies (destroyVisImage). In the rare path where
    // a caller forgot, we'd need an Allocator& + DescriptorAllocator& to
    // clean up — which we can't capture. Assert rather than leak silently.
    ENIGMA_ASSERT(m_visImage == VK_NULL_HANDLE);
    ENIGMA_ASSERT(m_visAlloc == VK_NULL_HANDLE);
    ENIGMA_ASSERT(m_visImageView == VK_NULL_HANDLE);
    ENIGMA_ASSERT(m_visBindlessSlot == UINT32_MAX);
}

bool MicropolyPass::active() const {
    return m_config.enabled && m_caps.row != HwMatrixRow::Disabled;
}

void MicropolyPass::createVisImage(VkExtent2D extent,
                                   gfx::Allocator& allocator,
                                   gfx::DescriptorAllocator& descriptorAllocator) {
    // Principle 1: disabled configs allocate zero GPU resources. Bail
    // before any vkCreate / vmaCreate call so disabled runs stay bit-
    // identical to pre-micropoly (screenshot_diff maxDelta=0 depends on
    // this).
    if (!active()) return;

    // Device may also have rejected both the native R64_UINT path and the
    // R32G32_UINT alias — defensively skip allocation rather than letting
    // VK_FORMAT_UNDEFINED propagate into vmaCreateImage.
    if (m_visFormat == VK_FORMAT_UNDEFINED) return;

    // Tear down the previous allocation when resizing. Renderer's resize
    // path calls vkDeviceWaitIdle before invoking this, so destroying the
    // image here is safe from GPU-side outstanding references.
    if (m_visImage != VK_NULL_HANDLE) {
        destroyVisImage(allocator, descriptorAllocator);
    }

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = m_visFormat;
    imgCI.extent        = {extent.width, extent.height, 1u};
    imgCI.mipLevels     = 1u;
    imgCI.arrayLayers   = 1u;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // STORAGE for the atomic-min writes (M3.3 HW raster, M4 SW raster),
    // TRANSFER_DST for vkCmdClearColorImage at frame start.
    imgCI.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                        | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocCI.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    const VkResult imgRes = vmaCreateImage(allocator.handle(), &imgCI, &allocCI,
                                           &m_visImage, &m_visAlloc, nullptr);
    if (imgRes != VK_SUCCESS) {
        ENIGMA_LOG_ERROR("[micropoly] visImage vmaCreateImage failed: {}",
                         static_cast<int>(imgRes));
        m_visImage = VK_NULL_HANDLE;
        m_visAlloc = nullptr;
        return;
    }

    VkImageViewCreateInfo viewCI{};
    viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image                           = m_visImage;
    viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format                          = m_visFormat;
    viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.baseMipLevel   = 0u;
    viewCI.subresourceRange.levelCount     = 1u;
    viewCI.subresourceRange.baseArrayLayer = 0u;
    viewCI.subresourceRange.layerCount     = 1u;

    const VkResult viewRes = vkCreateImageView(m_device->logical(), &viewCI,
                                               nullptr, &m_visImageView);
    if (viewRes != VK_SUCCESS) {
        ENIGMA_LOG_ERROR("[micropoly] visImage vkCreateImageView failed: {}",
                         static_cast<int>(viewRes));
        vmaDestroyImage(allocator.handle(), m_visImage, m_visAlloc);
        m_visImage = VK_NULL_HANDLE;
        m_visAlloc = nullptr;
        return;
    }

    m_visExtent            = extent;
    m_visLayoutInitialized = false;  // first clearVisImage will transition
    m_visBindlessSlot      = descriptorAllocator.registerStorageImage(m_visImageView);

    ENIGMA_LOG_INFO("[micropoly] visImage allocated {}x{} format={} bindlessSlot={}",
                    extent.width, extent.height,
                    static_cast<int>(m_visFormat), m_visBindlessSlot);
}

void MicropolyPass::destroyVisImage(gfx::Allocator& allocator,
                                    gfx::DescriptorAllocator& descriptorAllocator) {
    if (m_visBindlessSlot != UINT32_MAX) {
        descriptorAllocator.releaseStorageImage(m_visBindlessSlot);
        m_visBindlessSlot = UINT32_MAX;
    }
    if (m_visImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), m_visImageView, nullptr);
        m_visImageView = VK_NULL_HANDLE;
    }
    if (m_visImage != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator.handle(), m_visImage, m_visAlloc);
        m_visImage = VK_NULL_HANDLE;
        m_visAlloc = nullptr;
    }
    m_visExtent            = {0u, 0u};
    m_visLayoutInitialized = false;
}

void MicropolyPass::clearVisImage(VkCommandBuffer cmd) const {
    if (!active()) return;
    if (m_visImage == VK_NULL_HANDLE) return;

    // Transition UNDEFINED -> GENERAL on first use; subsequent clears just
    // write into the already-GENERAL image (vkCmdClearColorImage is legal
    // in GENERAL per the Vulkan spec).
    if (!m_visLayoutInitialized) {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = 0u;
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = m_visImage;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0u,
                             0u, nullptr,
                             0u, nullptr,
                             1u, &b);
        m_visLayoutInitialized = true;
    }

    // kMpVisEmpty = 0 (see shaders/micropoly/mp_vis_pack.hlsl).
    // Under reverse-Z (far=0, near=1) + InterlockedMax semantics the
    // sentinel must be the SMALLEST possible 64-bit value so any real
    // fragment wins the first atomic-max write. Zero satisfies that: a
    // rasterised sample anywhere forward of the far plane has a non-zero
    // depth, which yields a non-zero packed value that strictly beats the
    // cleared slot. MaterialEvalPass treats == 0 as "no mp sample here,
    // fall back to the 32-bit vis path."
    //
    // The image is typed R64_UINT (or R32G32_UINT alias for the
    // capability-deferred fallback path — see Renderer.cpp gating); for
    // R64_UINT vkCmdClearColorImage expects VkClearColorValue::uint32[0..1]
    // to encode the low/high halves of the 64-bit value. Setting all four
    // u32 slots to 0 yields 0 regardless of which format the device picked.
    VkClearColorValue clearVal{};
    clearVal.uint32[0] = 0u;
    clearVal.uint32[1] = 0u;
    clearVal.uint32[2] = 0u;
    clearVal.uint32[3] = 0u;

    VkImageSubresourceRange range{};
    range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel   = 0u;
    range.levelCount     = 1u;
    range.baseArrayLayer = 0u;
    range.layerCount     = 1u;

    vkCmdClearColorImage(cmd, m_visImage, VK_IMAGE_LAYOUT_GENERAL,
                         &clearVal, 1u, &range);
}

void MicropolyPass::record(VkCommandBuffer /*cmd*/) const {
    // Principle 1 invariant: when disabled (or hardware-gated off) this
    // function must issue zero commands. Real work lands in M3/M4.
    if (!active()) {
        return;
    }
    // M3.1 scope ends at vis image allocation + clear helper. Cluster cull
    // (M3.2) + HW raster (M3.3) + MaterialEval merge (M3.4) slot in here.
}

} // namespace enigma
