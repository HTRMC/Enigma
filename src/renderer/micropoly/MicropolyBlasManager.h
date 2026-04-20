#pragma once

// MicropolyBlasManager.h
// =======================
// M5a — per-asset-proxy BLAS for micropoly shadow casting. One BLAS per
// loaded micropoly asset, built once at asset-load from a fixed mid-DAG
// LOD cut (default: DAG level 3 of ~7). Typical BLAS count: ~10-20 per
// acceptance scene. No per-frame rebuild under steady-state; TLAS is
// re-merged with these instances at asset-residency-add time (rare).
//
// Plan reference: .omc/plans/ralplan-micropolygon.md §3.M5a lines 495-513.
//
// Capability gate: this subsystem only does real work on RT-capable HW.
// When Device::supportsRayTracing() == false (Min tier), `create()`
// returns a stub manager whose `instances()` span is empty. Principle 1
// holds: non-RT devices see zero BLAS work, zero VMA allocations beyond
// the stub object itself.
//
// Source triangles come from the DAG: level-3 clusters are a subset of
// the full DAG. We iterate dagNodes(), filter on dagLodLevel==3, and
// collect the unique pageIds that carry those clusters. Each unique
// page is zstd-decompressed CPU-side (one-shot, not per-frame), the
// cluster triangle list is extracted, positions are scatter-gathered
// through the per-cluster vertex offset, and the aggregate is built
// into a single BLAS via vkCmdBuildAccelerationStructuresKHR on the
// graphics queue with a fence-wait (mirrors gfx::BLAS::build).
//
// Per-page BLAS is an escape hatch only — stubbed out behind
// #ifdef MP_BLAS_PER_PAGE below, not compiled in default builds.

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

// Forward declare VMA.
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::asset {
class MpAssetReader;
}  // namespace enigma::asset

namespace enigma::gfx {
class Allocator;
class Device;
}  // namespace enigma::gfx

namespace enigma::renderer::micropoly {

// Default DAG LOD cut used for the per-asset-proxy BLAS. Plan §0.5.3
// specifies "DAG level 3 of ~7 levels". Callers may override to tune
// shadow-mesh density if shadow-quality regressions appear.
inline constexpr u32 kMpBlasDefaultDagLodLevel = 3u;

enum class MicropolyBlasErrorKind {
    NotSupported,         // non-RT device — factory returns OK stub; used
                          //   only when buildForAsset is called on a stub.
    PageDecompressFailed, // MpAssetReader::fetchPage reported an error.
    BlasBuildFailed,      // Vulkan vkCmd / vmaCreateBuffer / fence wait.
    NoLevel3Clusters,     // asset has no clusters at the requested LOD —
                          //   caller should log and continue.
    ReaderNotOpen,        // buildForAsset called with a non-open reader.
};

struct MicropolyBlasError {
    MicropolyBlasErrorKind kind{};
    std::string            detail;
};

// Stable string for MicropolyBlasErrorKind. Never null.
const char* micropolyBlasErrorKindString(MicropolyBlasErrorKind kind) noexcept;

// Instance descriptor for a built per-asset-proxy BLAS. Renderer merges
// this list into the TLAS build input alongside the existing non-micropoly
// BLAS instances. The shape mirrors the VkAccelerationStructureInstanceKHR
// fields that actually vary at the call site — transform, customIndex,
// mask, and the BLAS address — but leaves SBT record offset / flags at
// defaults the Renderer chooses uniformly.
struct MicropolyBlasInstanceEntry {
    VkAccelerationStructureKHR blas        = VK_NULL_HANDLE;
    VkDeviceAddress            blasAddress = 0ull;
    u32                        customIndex = 0u;     // asset slot; shader lookup.
    u32                        mask        = 0xFFu;  // all rays hit by default.
    mat4                       transform   { 1.0f }; // world-space (M5a: identity).
};

class MicropolyBlasManager {
public:
    // Non-throwing factory. On non-RT devices returns an OK stub whose
    // instances() is empty and whose buildForAsset() returns a
    // NotSupported error (the caller should treat that as "continue
    // without micropoly shadows"). The stub path never calls any RT
    // extension entrypoint — Principle 1 (no-op when disabled).
    static std::expected<std::unique_ptr<MicropolyBlasManager>, MicropolyBlasError>
    create(gfx::Device& device, gfx::Allocator& allocator);

    ~MicropolyBlasManager();

    MicropolyBlasManager(const MicropolyBlasManager&)            = delete;
    MicropolyBlasManager& operator=(const MicropolyBlasManager&) = delete;
    MicropolyBlasManager(MicropolyBlasManager&&)                 = delete;
    MicropolyBlasManager& operator=(MicropolyBlasManager&&)      = delete;

    // Build one per-asset-proxy BLAS from the reader's DAG at
    // `dagLodLevel`. Re-invocation with the same reader is a no-op
    // (cached on reader pointer identity). `NoLevel3Clusters` is a
    // soft error the caller should log and skip.
    //
    // Sync model: builds on the graphics queue via an immediate command
    // buffer + queue-wait-idle, mirroring gfx::BLAS::build's pattern.
    // One call per asset at asset-load, so the wait is acceptable.
    std::expected<void, MicropolyBlasError> buildForAsset(
        asset::MpAssetReader& reader,
        u32                   dagLodLevel = kMpBlasDefaultDagLodLevel);

    // Contiguous view of all built BLAS instance entries. Empty on the
    // stub (non-RT) manager and before any successful buildForAsset().
    std::span<const MicropolyBlasInstanceEntry> instances() const noexcept {
        return {m_instances.data(), m_instances.size()};
    }

    // True iff this instance was constructed with Device::supportsRayTracing().
    // Renderer uses this to gate the TLAS-merge step; redundant with a
    // non-empty instances() span today but kept explicit.
    bool supportsRayTracing() const noexcept { return m_rtCapable; }

#ifdef MP_BLAS_PER_PAGE
    // Escape hatch (plan §3.M5a): per-resident-page BLAS. Activated at
    // build time if the per-asset-proxy approach produces shadow-quality
    // regressions beyond tolerance. Intentionally not compiled in default
    // builds — see plan §3.M5a lines 500-502.
    std::expected<void, MicropolyBlasError> buildPerPage(
        asset::MpAssetReader& reader);
#endif

private:
    MicropolyBlasManager(gfx::Device& device,
                         gfx::Allocator& allocator,
                         bool rtCapable);

    // Release all BLASes + their backing vertex/index buffers. Idempotent.
    void destroy_() noexcept;

    struct BuiltBlas {
        VkAccelerationStructureKHR as             = VK_NULL_HANDLE;
        VkBuffer                   asBuffer       = VK_NULL_HANDLE;
        VmaAllocation              asAllocation   = nullptr;
        VkBuffer                   vertexBuffer   = VK_NULL_HANDLE;
        VmaAllocation              vertexAlloc    = nullptr;
        VkBuffer                   indexBuffer    = VK_NULL_HANDLE;
        VmaAllocation              indexAlloc     = nullptr;
    };

    gfx::Device*                                 m_device     = nullptr;
    gfx::Allocator*                              m_allocator  = nullptr;
    bool                                         m_rtCapable  = false;

    std::vector<BuiltBlas>                       m_built;
    std::vector<MicropolyBlasInstanceEntry>      m_instances;

    // Cache — so buildForAsset(reader) called twice on the same reader
    // is a no-op. Plan note: "rerunning for the same reader is a no-op
    // (cached)." Matches on pointer identity; the Renderer holds the
    // MpAssetReader by unique_ptr so the address is stable for the
    // manager's lifetime.
    std::vector<const asset::MpAssetReader*>     m_alreadyBuilt;
};

} // namespace enigma::renderer::micropoly
