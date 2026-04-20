#pragma once

#include "core/Types.h"
#include "renderer/micropoly/MicropolyCapability.h"
#include "renderer/micropoly/MicropolyConfig.h"

#include <volk.h>

// Forward declare VMA handles.
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Allocator;
class Device;
class DescriptorAllocator;
}  // namespace enigma::gfx

namespace enigma {

// MicropolyPass
// =============
// Scaffolding pass for the micropolygon geometry subsystem introduced by
// .omc/plans/ralplan-micropolygon.md. In M0a this pass does NOTHING at
// record time and allocates NO GPU resources — it exists only to:
//   - reserve the render-graph slot so later milestones can fill it in,
//   - probe the device once and record the chosen 64-bit vis image format,
//   - own the VkImage handle that M3 will populate.
//
// Principle 1 invariant (plan §0.5.1): when MicropolyConfig::enabled == false
// (the default), this pass must not observably change anything. record() is
// an empty function; no descriptor updates, no pipeline creation, no image
// allocation. DO NOT add side-effecting work without updating the plan.
class MicropolyPass {
public:
    // Construct the pass shell. Performs the device capability probe,
    // chooses the visibility image format per plan §3.M0a's format-decision
    // block, and logs a single "micropoly: <status>" line.
    //
    // The VkImage member is DECLARED but NOT allocated here — M3 owns the
    // allocation. Passing this through now keeps later milestones from
    // retouching Renderer's construction code.
    MicropolyPass(gfx::Device& device, const MicropolyConfig& config);
    ~MicropolyPass();

    MicropolyPass(const MicropolyPass&)            = delete;
    MicropolyPass& operator=(const MicropolyPass&) = delete;

    // M3.1: Allocate the 64-bit visibility image at `extent` and register
    // it as a bindless storage image. No-op when active() is false
    // (Principle 1: disabled configs allocate no GPU resources). Calling
    // this with an already-allocated image is a valid resize — the old
    // image is torn down first (the caller is expected to have issued
    // vkDeviceWaitIdle prior, mirroring resizeGBuffer's contract).
    //
    // Expects `allocator` + `descriptorAllocator` to outlive the pass.
    // The bindless slot is released + re-acquired on resize.
    void createVisImage(VkExtent2D extent,
                        gfx::Allocator& allocator,
                        gfx::DescriptorAllocator& descriptorAllocator);

    // Release the vis image + bindless slot. Safe to call on an already-
    // destroyed instance. The destructor calls this automatically; explicit
    // callers are the resize path + shutdown.
    void destroyVisImage(gfx::Allocator& allocator,
                         gfx::DescriptorAllocator& descriptorAllocator);

    // Clear the 64-bit vis image to `kMpVisEmpty` (0 ≡ "no sample" under
    // reverse-Z + InterlockedMax semantics — any real fragment beats the
    // cleared slot).
    // Recorded into `cmd`; callers must subsequently issue a pipeline barrier
    // to make the write visible to cluster-cull / HW-raster (M3.2-M3.3).
    // No-op when active() is false or the image is not allocated.
    //
    // Current image layout is VK_IMAGE_LAYOUT_GENERAL; no transition needed
    // since both vkCmdClearColorImage and downstream atomic-max stores run
    // in GENERAL.
    void clearVisImage(VkCommandBuffer cmd) const;

    // Record this pass's work into `cmd`. No-op when `enabled == false` or
    // the capability row classifies as Disabled. Kept taking a
    // VkCommandBuffer so M3 can wire it in without a signature change.
    void record(VkCommandBuffer cmd) const;

    // Accessors — useful to tests and to the eventual M3 integration.
    const MicropolyCaps&   caps()        const { return m_caps; }
    const MicropolyConfig& config()      const { return m_config; }
    VkFormat               visFormat()   const { return m_visFormat; }
    VkImage                visImage()    const { return m_visImage; }
    VkImageView            visImageView()const { return m_visImageView; }
    VkExtent2D             visExtent()   const { return m_visExtent; }
    // Bindless storage-image slot registered with DescriptorAllocator.
    // UINT32_MAX when no vis image is allocated.
    u32                    visBindlessSlot() const { return m_visBindlessSlot; }

    // True iff the pass would perform real work in record(). Equivalent to
    // (config.enabled && caps.row != Disabled) — exposed for observability.
    bool active() const;

private:
    gfx::Device*    m_device  = nullptr;
    MicropolyConfig m_config{};
    MicropolyCaps   m_caps{};

    // Visibility-buffer image format chosen at construction time per plan
    // §3.M0a: prefer VK_FORMAT_R64_UINT; fall back to VK_FORMAT_R32G32_UINT
    // (aliased for u64 atomic-min via SPV_EXT_shader_image_int64) when R64
    // is not storage-image-atomic on this device.
    VkFormat m_visFormat = VK_FORMAT_UNDEFINED;

    // M3.1: 64-bit visibility image + bookkeeping.
    //   - m_visImage/Alloc/View : the image resource + allocation + view.
    //   - m_visExtent           : current sized extent (matches the view).
    //   - m_visBindlessSlot     : bindless storage-image slot, UINT32_MAX
    //                             when not registered.
    //   - m_visLayoutInitialized: true once the first createVisImage pass has
    //                             transitioned the image to GENERAL layout.
    //                             Used by clearVisImage to decide whether
    //                             to emit an initial UNDEFINED→GENERAL
    //                             barrier.
    VkImage       m_visImage        = VK_NULL_HANDLE;
    VmaAllocation m_visAlloc        = nullptr;
    VkImageView   m_visImageView    = VK_NULL_HANDLE;
    VkExtent2D    m_visExtent       = {0u, 0u};
    u32           m_visBindlessSlot = UINT32_MAX;
    // Mutable on purpose: clearVisImage() is const (it only records commands
    // into a caller-supplied VkCommandBuffer and does not mutate Vulkan
    // resource ownership), but it needs to remember whether the one-time
    // UNDEFINED->GENERAL transition has been recorded for the current
    // image. The single-threaded-recording invariant on command buffers
    // means concurrent callers can't race on this flag — Renderer records
    // all micropoly work serially on frame.commandBuffer. If that ever
    // changes (multi-thread recording), this flag must migrate to a real
    // atomic or move out of the const path.
    mutable bool  m_visLayoutInitialized = false;
};

} // namespace enigma
