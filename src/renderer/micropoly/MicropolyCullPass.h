#pragma once

// MicropolyCullPass.h
// ====================
// Per-frame compute pass that drives shaders/micropoly/mp_cluster_cull.comp.hlsl.
// Owns:
//   * The compute VkPipeline + layout (bindless + push constants).
//   * A cull-stats RWByteAddressBuffer (7 u32 counters, 32 bytes padded).
//   * An indirect-draw RWByteAddressBuffer (count header + cmd array).
//
// The render-graph slot is: AFTER MicropolyStreaming::beginFrame() and
// BEFORE any micropoly HW raster (M3.3 — currently a no-op). The dispatch
// scales with totalClusterCount / 64 (round-up); a zero-cluster input is a
// legal no-op. Principle 1: when MicropolyConfig::enabled==false this pass
// is NEVER CONSTRUCTED so `screenshot_diff maxDelta=0` holds unconditionally.
//
// Shader caveat (recorded so future editors preserve it): under DXC -Zpc
// -spirv the float4x4(v0..v3) constructor takes COLUMN vectors. The shader
// loads camera matrices via `transpose(float4x4(...))` — don't "simplify"
// that to the direct constructor or the frustum cull silently rotates wrong.
//
// Test-only note: MicropolyCullPass::create() needs a ShaderManager (which
// itself needs a Device). The headless test harness brings up a minimal
// Device via gfx::Device::adopt() — see tests/micropoly_cull_test.cpp for
// the canonical pattern.

#include "core/Types.h"

#include <volk.h>

#include <expected>
#include <filesystem>
#include <string>

// Forward declare VMA.
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Allocator;
class DescriptorAllocator;
class Device;
class Pipeline;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma::renderer::micropoly {

// Cluster cap the indirect-draw buffer is sized for. 65,536 surviving
// clusters is plenty for a single baked asset in M3.2 (the DamagedHelmet
// benchmark produces ~120 leaf clusters); the buffer is a one-time
// allocation so oversizing is cheap.
inline constexpr u32 kMpMaxIndirectDrawClusters = 65536u;

// M4.1 — vis-pack v2 carves a 2-bit rasterClass field out of the former
// 25-bit clusterId space, leaving 23 bits for clusterId. If the cap above
// is ever raised past (1u << 23) the packed vis encoding in
// shaders/micropoly/mp_vis_pack.hlsl needs another bit reshuffle.
// Catching it here at compile time is cheap; catching it in a frame
// capture is not.
static_assert(kMpMaxIndirectDrawClusters <= (1u << 23),
              "kMpMaxIndirectDrawClusters exceeds the 23-bit clusterId "
              "field in shaders/micropoly/mp_vis_pack.hlsl (see M4.1 "
              "vis-pack v2). Widen kMpVisClusterBits / kMpRasterClassBits "
              "before raising this cap.");

// Number of u32 counters exposed by the cull-stats buffer. Layout matches
// the shader's bumpStat() byte offsets in mp_cluster_cull.comp.hlsl:
//   0 totalDispatched | 1 culledLOD       | 2 culledResidency
//   3 culledFrustum   | 4 culledBackface  | 5 culledHiZ    | 6 visible
inline constexpr u32 kMpCullStatsCounterCount = 7u;

// CPU-side snapshot of the cull-stats buffer. Layout matches the shader's
// bumpStat() byte offsets 1:1 — adding a new counter means extending both
// the struct here and the shader (order matters).
struct CullStats {
    u32 totalDispatched{0u};
    u32 culledLOD{0u};
    u32 culledResidency{0u};
    u32 culledFrustum{0u};
    u32 culledBackface{0u};
    u32 culledHiZ{0u};
    u32 visible{0u};
};

enum class MicropolyCullErrorKind {
    PipelineBuildFailed,
    BufferCreationFailed,
    BindlessRegistrationFailed,
};

struct MicropolyCullError {
    MicropolyCullErrorKind kind{};
    std::string            detail;
};

class MicropolyCullPass {
public:
    // Non-throwing factory. Builds the compute pipeline, allocates the
    // cull-stats + indirect-draw buffers, and registers both as bindless
    // UAV RWByteAddressBuffers. On failure the returned expected<> carries
    // a short diagnostic.
    static std::expected<MicropolyCullPass, MicropolyCullError> create(
        gfx::Device&              device,
        gfx::Allocator&           allocator,
        gfx::DescriptorAllocator& descriptors,
        gfx::ShaderManager&       shaderManager);

    ~MicropolyCullPass();

    MicropolyCullPass(const MicropolyCullPass&)            = delete;
    MicropolyCullPass& operator=(const MicropolyCullPass&) = delete;
    MicropolyCullPass(MicropolyCullPass&& other) noexcept;
    MicropolyCullPass& operator=(MicropolyCullPass&& other) noexcept;

    // Per-frame dispatch inputs. See the compute shader's PushBlock for the
    // meaning of each field; the pass forwards all of them as push constants.
    struct DispatchInputs {
        VkCommandBuffer cmd                            = VK_NULL_HANDLE;
        VkDescriptorSet globalSet                      = VK_NULL_HANDLE;
        u32 totalClusterCount                          = 0u;
        u32 dagBufferBindlessIndex                     = UINT32_MAX;
        // Note: residency lookup now routes through pageToSlotBuffer (a
        // pageId-indexed u32 array where UINT32_MAX == not resident),
        // removing the need for a separate residency bitmap SSBO. The
        // former `residencyBitmapBindlessIndex` field was dead weight and
        // has been dropped from the push block.
        u32 requestQueueBindlessIndex                  = UINT32_MAX;
        // Caller can leave indirectBufferBindlessIndex / cullStatsBindlessIndex
        // at UINT32_MAX to route through the pass-owned buffers (registered
        // at create time); any other value is forwarded verbatim so tests
        // can point them at their own bindless slots.
        u32 indirectBufferBindlessIndexOverride        = UINT32_MAX;
        u32 cullStatsBindlessIndexOverride             = UINT32_MAX;
        u32 hiZBindlessIndex                           = UINT32_MAX;
        u32 cameraSlot                                 = UINT32_MAX;
        f32 hiZMipCount                                = 1.0f;
        f32 screenSpaceErrorThreshold                  = 1.0f;
        // Security HIGH-2: total page count for the currently bound .mpa
        // asset. The shader uses it to bounds-check pageId < pageCount
        // before indexing the residency bitmap. Leave at 0 for zero-cluster
        // dispatches; the shader early-outs before any residency query.
        u32 pageCount                                  = 0u;
        // M4.4 dispatcher classifier inputs — the cull pass reads the
        // cluster's triangleCount from the page cache to feed
        // classifyRasterClass(). Without these slots the classifier would
        // have no triangle-count signal; leave them at defaults when
        // unavailable (zero-cluster dispatches short-circuit before the
        // classifier runs).
        u32 pageToSlotBufferBindlessIndex              = UINT32_MAX;
        u32 pageCacheBufferBindlessIndex               = UINT32_MAX;
        u32 pageSlotBytes                              = 0u;
        // M4.4: height of the viewport in pixels — the classifier uses it
        // to convert the cluster's world-space bounding radius into a
        // screen-space pixel-radius estimate.
        u32 screenHeight                               = 0u;
        // M4.5: pageId -> firstDagNodeIdx SSBO bindless slot. Shader reads
        // it to derive page-local cluster index for multi-cluster pages.
        // Leave at UINT32_MAX when the runtime hasn't attached the buffer
        // yet (zero-cluster dispatches short-circuit before the lookup).
        u32 pageFirstDagNodeBufferBindlessIndex        = UINT32_MAX;
    };

    // Record the cull dispatch + a count-buffer reset barrier. The caller
    // is responsible for any producer→cull barrier (the DAG SSBO / residency
    // bitmap must be upload-visible before entry).
    //
    // After the dispatch emits an SSBO writes → DRAW_INDIRECT / TASK / MESH
    // barrier on the indirect-draw buffer + cull-stats buffer so M3.3's HW
    // raster pass sees a coherent indirect-draw buffer.
    void dispatch(const DispatchInputs& inputs);

    // Clear the header word (count = 0) of both owned buffers. Caller
    // records this at frame start — after the previous frame's consumer
    // has retired but before the new frame's producer writes.
    void resetCounters(VkCommandBuffer cmd) const;

    // Register the compute shader with the hot-reload manager so edits to
    // mp_cluster_cull.comp.hlsl rebuild the pipeline in place.
    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Read the 7 cull-stats counters from the persistent-mapped pointer.
    // Returns a zero-initialised CullStats when the buffer hasn't been
    // allocated yet (e.g. micropoly disabled). Atomic writes on a host-
    // visible allocation are legal per the Vulkan spec; this is a lock-
    // free load of the last values the GPU flushed to the HOST_COHERENT
    // heap — individual counters may be 1-2 frames stale depending on
    // driver/queue state, which is acceptable for a debug HUD.
    CullStats readbackStats() const;

    // Accessors — handy for tests and for M3.3's consumer wiring.
    VkBuffer cullStatsBuffer()      const { return m_cullStatsBuffer; }
    u32      cullStatsBindlessSlot()const { return m_cullStatsBindlessSlot; }
    VkBuffer indirectDrawBuffer()   const { return m_indirectDrawBuffer; }
    u32      indirectDrawBindlessSlot() const { return m_indirectDrawBindlessSlot; }
    VkDeviceSize indirectDrawBufferBytes() const { return m_indirectDrawBufferBytes; }
    VkDeviceSize cullStatsBufferBytes()    const { return m_cullStatsBufferBytes; }
    // M4.4 dispatcher classifier buffer — 4 B per drawSlot (u32 rasterClass).
    // Both raster paths read this tag; see shaders/micropoly/mp_vis_pack.hlsl
    // for kMpRasterClassHw (0) / kMpRasterClassSw (1) values.
    VkBuffer rasterClassBuffer()        const { return m_rasterClassBuffer; }
    u32      rasterClassBufferBindlessSlot() const { return m_rasterClassBufferBindlessSlot; }
    VkDeviceSize rasterClassBufferBytes()    const { return m_rasterClassBufferBytes; }

private:
    MicropolyCullPass(gfx::Device& device,
                      gfx::Allocator& allocator,
                      gfx::DescriptorAllocator& descriptors);

    // Release VMA + bindless resources. Idempotent.
    void destroy_();

    // (Re)build the compute pipeline. Called once at create() and on hot
    // reload. Returns true on success.
    bool rebuildPipeline_();

    // Pass-owned device state.
    gfx::Device*              m_device        = nullptr;
    gfx::Allocator*           m_allocator     = nullptr;
    gfx::DescriptorAllocator* m_descriptors   = nullptr;
    gfx::ShaderManager*       m_shaderManager = nullptr;
    gfx::Pipeline*            m_pipeline      = nullptr;

    VkDescriptorSetLayout     m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path     m_shaderPath{};

    // Cull-stats buffer (kMpCullStatsCounterCount u32s, padded). Allocated
    // HOST_VISIBLE + HOST_COHERENT + persistently-mapped so readbackStats()
    // can do a plain load — no explicit copy, no fence wait. Atomic GPU
    // writes on host-visible memory are spec-legal and the per-frame cost
    // of the ~thousands of atomics is sub-microsecond (debug-only buffer).
    VkBuffer      m_cullStatsBuffer          = VK_NULL_HANDLE;
    VmaAllocation m_cullStatsAllocation      = VK_NULL_HANDLE;
    VkDeviceSize  m_cullStatsBufferBytes     = 0ull;
    u32           m_cullStatsBindlessSlot    = UINT32_MAX;
    void*         m_cullStatsMapped          = nullptr;

    // Indirect-draw buffer (16B header + kMpMaxIndirectDrawClusters * 16B cmds).
    VkBuffer      m_indirectDrawBuffer       = VK_NULL_HANDLE;
    VmaAllocation m_indirectDrawAllocation   = VK_NULL_HANDLE;
    VkDeviceSize  m_indirectDrawBufferBytes  = 0ull;
    u32           m_indirectDrawBindlessSlot = UINT32_MAX;

    // M4.4 rasterClassBuffer — u32 per drawSlot
    // (kMpMaxIndirectDrawClusters * 4 B = 256 KiB at the 65,536 slot cap).
    VkBuffer      m_rasterClassBuffer            = VK_NULL_HANDLE;
    VmaAllocation m_rasterClassAllocation        = VK_NULL_HANDLE;
    VkDeviceSize  m_rasterClassBufferBytes       = 0ull;
    u32           m_rasterClassBufferBindlessSlot = UINT32_MAX;
};

const char* micropolyCullErrorKindString(MicropolyCullErrorKind kind);

} // namespace enigma::renderer::micropoly
