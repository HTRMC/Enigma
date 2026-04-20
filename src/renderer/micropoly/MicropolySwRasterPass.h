#pragma once

// MicropolySwRasterPass.h
// =========================
// M4.2 — SW raster BINNING pass. Consumes the cluster-cull survivor list
// (from MicropolyCullPass / M3.2) and partitions each cluster's projected
// triangles into screen-space 8x8 tile bins. The per-pixel fragment
// rasterisation pass that consumes these bins lands in M4.3 on top of this
// scaffold.
//
// Pass outputs (all GPU-only SSBOs — never host-visible):
//   * tileBinCountBuffer  — u32 * numTiles. Per-tile count of bin entries.
//   * tileBinEntriesBuffer — u32 * numTiles * MP_SW_TILE_BIN_CAP. Packed
//                            triRef = (clusterIdx<<7 | triIdx) matching
//                            vis-pack v2 (mp_vis_pack.hlsl) so M4.3 can feed
//                            each ref straight into PackMpVis64 with class
//                            = kMpRasterClassSw.
//   * spillBuffer         — header {u32 spillCount, u32 spillDroppedCount}
//                            followed by MP_SW_SPILL_CAP * {u32 tileIdx,
//                            u32 triRef}. Receives overflow when a tile's
//                            fixed-size bin saturates.
//   * dispatchIndirectBuffer — 12 B {groupCountX, groupCountY, groupCountZ}
//                              written by sw_raster_bin_prep.comp.hlsl;
//                              consumed by vkCmdDispatchIndirect for the
//                              binning compute.
//
// Principle 1: the pass is NEVER constructed when MicropolyConfig::enabled
// is false OR when the device lacks VK_EXT_mesh_shader / shaderImageInt64
// (the capability umbrella tracks MicropolyRasterPass for now). On
// non-capable devices the Renderer skips construction entirely so
// screenshot_diff maxDelta=0 holds unconditionally.
//
// Render-graph slot (Renderer.cpp):
//   [cluster cull dispatch] -> [HW raster record] -> [SW BIN record (here)]
//     -> [M4.3: SW raster fragment record — not yet wired]
//     -> [MaterialEval merge]
//
// Bin + fragment-rasterisation compute are both live: the bin SSBOs are
// consumed by the fragment-raster dispatch step in the same pass
// (see record() step 8 — "SW raster fragment dispatch"). The two dispatch
// steps share a pipeline barrier so bin writes are visible to the
// fragment reader in the same frame.
//
// Shader caveat (carried from peer passes): under DXC -Zpc -spirv the
// float4x4(v0..v3) constructor takes COLUMN vectors, so the shader loads
// camera matrices via transpose(float4x4(...)). Don't "simplify" that.

#include "core/Types.h"

#include <volk.h>

#include <expected>
#include <filesystem>
#include <memory>
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

// --- Tile + bin sizing -----------------------------------------------------
// Mirror the #defines at the top of sw_raster_bin.comp.hlsl. Any change to
// these MUST be reflected in both places — a mismatched BIN_CAP/SPILL_CAP
// silently corrupts memory. Keep the C++ + HLSL values in lockstep.
inline constexpr u32 kMpSwTileX      = 8u;
inline constexpr u32 kMpSwTileY      = 8u;
// M4.6: 1024 now that overflow routes to a per-tile spill linked list
// (see sw_raster_bin.comp.hlsl binTriangleToTiles). No more non-
// deterministic drops at the BIN_CAP boundary — overflowed triangles
// are preserved and walked at raster time via the spill head chain.
// 1024 * 4B * 14400 tiles ≈ 59 MiB for 1280x720 — fits comfortably on
// any discrete GPU. Must mirror MP_SW_TILE_BIN_CAP in
// sw_raster.comp.hlsl and sw_raster_bin.comp.hlsl.
inline constexpr u32 kMpSwTileBinCap = 1024u;
// Raised to 1M (8 MB) so small-on-screen BMW — where thousands of
// triangles concentrate into a handful of tiles — can spill without
// dropping. Drops produced 8x8-pixel "tile-shaped holes" that worsened
// as the model shrank. Must mirror MP_SW_SPILL_CAP in
// sw_raster.comp.hlsl and sw_raster_bin.comp.hlsl.
// Raised 1M → 16M (128 MB spillBuffer) because 1M was still hitting on
// dense BMW tiles and dropping triangles per-frame non-deterministically
// (atomic-add ordering), showing up as tile-shaped holes that flickered.
// Must mirror MP_SW_SPILL_CAP in sw_raster.comp.hlsl and
// sw_raster_bin.comp.hlsl.
inline constexpr u32 kMpSwSpillCap   = 16777216u;

enum class MicropolySwRasterErrorKind {
    MeshShadersUnsupported,
    PipelineBuildFailed,
    BufferAllocFailed,
    BindlessRegistrationFailed,
    // M4.3: added to distinguish the raster fragment pipeline build
    // failure from the bin/prep pipeline build failure in logs/tests.
    RasterPipelineBuildFailed,
    // Verifier blocker fix: SW raster fragment writes RWTexture2D<uint64_t>
    // via InterlockedMax, which requires VK_EXT_shader_image_atomic_int64.
    // Previously only gated on supportsMeshShaders() — would build the pipeline
    // on mesh-capable-but-Int64-lacking hardware and UB at dispatch.
    Int64ImageUnsupported,
};

struct MicropolySwRasterError {
    MicropolySwRasterErrorKind kind{};
    std::string                detail;
};

class MicropolySwRasterPass {
public:
    // Non-throwing factory. Builds both compute pipelines (sw_raster_bin_prep
    // + sw_raster_bin), allocates the 4 output buffers sized for `extent`,
    // and registers 4 bindless UAV slots. Returns an error when the
    // MicropolyRasterPass-aligned capability umbrella is absent — callers
    // MUST check these before constructing to honour Principle 1.
    //
    // `extent` drives tileBinCount / tileBinEntries sizing. A later resize
    // requires pass destruction + reconstruction (same idiom as
    // MicropolyCullPass's one-shot sizing). 4K peak = (3840/8) * (2160/8) =
    // 129,600 tiles -> tileBinCountBuffer ~ 0.5 MiB,
    // tileBinEntriesBuffer ~ 132 MiB. The caller is free to pass a smaller
    // extent at low resolutions.
    static std::expected<std::unique_ptr<MicropolySwRasterPass>, MicropolySwRasterError>
    create(gfx::Device&              device,
           gfx::Allocator&           allocator,
           gfx::DescriptorAllocator& descriptors,
           gfx::ShaderManager&       shaderManager,
           VkExtent2D                extent);

    ~MicropolySwRasterPass();

    MicropolySwRasterPass(const MicropolySwRasterPass&)            = delete;
    MicropolySwRasterPass& operator=(const MicropolySwRasterPass&) = delete;
    MicropolySwRasterPass(MicropolySwRasterPass&&)                 = delete;
    MicropolySwRasterPass& operator=(MicropolySwRasterPass&&)      = delete;

    // Per-frame dispatch inputs. The caller is responsible for the upstream
    // barrier (cull SSBO writes -> compute SSBO reads) — MicropolyCullPass::
    // dispatch already covers that barrier since this pass's input is the
    // same indirect-draw buffer the HW raster reads.
    struct DispatchInputs {
        VkCommandBuffer cmd                          = VK_NULL_HANDLE;
        VkDescriptorSet globalSet                    = VK_NULL_HANDLE;

        // Cull survivors (shared with MicropolyRasterPass).
        VkBuffer indirectBuffer                      = VK_NULL_HANDLE;
        u32      indirectBufferBindlessIndex         = UINT32_MAX;

        // DAG + page-table + page-cache SSBOs (same bindless slots the HW
        // raster task/mesh shaders see).
        u32      dagBufferBindlessIndex              = UINT32_MAX;
        u32      pageToSlotBufferBindlessIndex       = UINT32_MAX;
        u32      pageCacheBufferBindlessIndex        = UINT32_MAX;
        u32      cameraSlot                          = UINT32_MAX;

        // M4.3: R64_UINT vis image the fragment pipeline writes via
        // InterlockedMax. Same bindless slot the HW raster's PSMain
        // uses — both paths co-exist on one image (reverse-Z atomic-max).
        u32      visImage64Bindless                  = UINT32_MAX;

        // Runtime constants.
        VkExtent2D extent                            = {0u, 0u};
        u32        pageSlotBytes                     = 0u;
        u32        pageCount                         = 0u;
        u32        dagNodeCount                      = 0u;
        // M4.4: per-drawSlot rasterClass tag SSBO — the bin shader's
        // workgroup-thread-0 early-outs the whole group when the cluster
        // is HW-classified (0). Only the bin pipeline consumes this.
        u32        rasterClassBufferBindlessIndex    = UINT32_MAX;
        // M4.5: pageId -> firstDagNodeIdx SSBO bindless slot. Both bin +
        // raster compute paths read it to derive page-local cluster
        // indices for multi-cluster pages.
        u32        pageFirstDagNodeBufferBindlessIndex = UINT32_MAX;
    };

    // Record the binning dispatch. Steps:
    //   1. vkCmdFillBuffer(tileBinCountBuffer, 0u) — zero all tile counts.
    //   2. vkCmdFillBuffer(spillBuffer first 8 B, 0u) — zero both headers.
    //   3. Barrier TRANSFER -> COMPUTE_SHADER read/write.
    //   4. Bind+dispatch the prep compute (1 thread). Writes the
    //      dispatchIndirectBuffer with {count, 1, 1}.
    //   5. Barrier COMPUTE_SHADER write -> DRAW_INDIRECT read on
    //      dispatchIndirectBuffer.
    //   6. Bind the bin compute and vkCmdDispatchIndirect against the
    //      dispatchIndirectBuffer.
    //   7. Barrier COMPUTE_SHADER write -> COMPUTE_SHADER read on the
    //      three bin SSBOs so M4.3 (and future consumers) see coherent
    //      data.
    //
    // No-op guards: extent zero, indirectBuffer null, or dagBufferBindlessIndex
    // == UINT32_MAX make this function return before any GPU work is
    // emitted (matches MicropolyRasterPass::record guards).
    void record(const DispatchInputs& inputs);

    // Register the two compute shaders with hot reload. Mirrors peers.
    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Resize the tile-bin SSBOs to match a new viewport extent. Called by
    // Renderer::resizeGBuffer() after a swapchain recreation. Issues a
    // vkDeviceWaitIdle before destroying old buffers so any in-flight
    // dispatch retires first. Only the three extent-dependent bin SSBOs
    // (tileBinCountBuffer, tileBinEntriesBuffer, spillBuffer) are torn
    // down + re-allocated — the dispatchIndirectBuffer and both compute
    // pipelines are extent-independent and survive the resize. The three
    // bindless slots are released and re-acquired so consumers that
    // cache the old values observe fresh slots.
    //
    // Returns an error on failure (allocation / bindless registration).
    // Failures are recoverable — the caller (Renderer) logs and leaves
    // the SW raster path in a non-functional but crash-safe state until
    // the next resize call.
    std::expected<void, MicropolySwRasterError> resize(VkExtent2D newExtent);

    // Accessors — for tests + M4.3 fragment pass wiring.
    VkBuffer tileBinCountBuffer()         const { return m_tileBinCountBuffer; }
    VkBuffer tileBinEntriesBuffer()       const { return m_tileBinEntriesBuffer; }
    VkBuffer spillBuffer()                const { return m_spillBuffer; }
    VkBuffer dispatchIndirectBuffer()     const { return m_dispatchIndirectBuffer; }
    u32      tileBinCountBindlessSlot()   const { return m_tileBinCountBindlessSlot; }
    u32      tileBinEntriesBindlessSlot() const { return m_tileBinEntriesBindlessSlot; }
    u32      spillBufferBindlessSlot()    const { return m_spillBufferBindlessSlot; }
    u32      dispatchIndirectBindlessSlot() const { return m_dispatchIndirectBindlessSlot; }
    u32      tilesX()                     const { return m_tilesX; }
    u32      tilesY()                     const { return m_tilesY; }
    VkExtent2D extent()                   const { return m_extent; }

    // M4.3: test hooks — returns non-null once create() has built the
    // fragment raster pipeline. Tests assert this is distinct from the
    // bin + prep pipelines (create() builds all three eagerly).
    const gfx::Pipeline* binPipeline()    const { return m_binPipeline; }
    const gfx::Pipeline* prepPipeline()   const { return m_prepPipeline; }
    const gfx::Pipeline* rasterPipeline() const { return m_rasterPipeline; }

private:
    MicropolySwRasterPass(gfx::Device&              device,
                          gfx::Allocator&           allocator,
                          gfx::DescriptorAllocator& descriptors);

    void destroy_();
    bool rebuildPipelines_();

    gfx::Device*              m_device        = nullptr;
    gfx::Allocator*           m_allocator     = nullptr;
    gfx::DescriptorAllocator* m_descriptors   = nullptr;
    gfx::ShaderManager*       m_shaderManager = nullptr;

    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path m_binShaderPath{};
    std::filesystem::path m_prepShaderPath{};
    // M4.3: fragment raster shader source. Consumes the bin SSBOs written
    // by m_binPipeline and writes to the R64_UINT vis image via
    // InterlockedMax.
    std::filesystem::path m_rasterShaderPath{};

    // Three compute pipelines. Owned by gfx::Pipeline (same pattern as the
    // cull pass) to share the wait-idle-on-rebuild behaviour.
    gfx::Pipeline* m_binPipeline    = nullptr;
    gfx::Pipeline* m_prepPipeline   = nullptr;
    gfx::Pipeline* m_rasterPipeline = nullptr;

    // Sizing.
    VkExtent2D m_extent = {0u, 0u};
    u32        m_tilesX = 0u;
    u32        m_tilesY = 0u;
    u32        m_numTiles = 0u;

    // Output buffers + their bindless slots.
    VkBuffer      m_tileBinCountBuffer          = VK_NULL_HANDLE;
    VmaAllocation m_tileBinCountAllocation      = VK_NULL_HANDLE;
    VkDeviceSize  m_tileBinCountBufferBytes     = 0ull;
    u32           m_tileBinCountBindlessSlot    = UINT32_MAX;

    VkBuffer      m_tileBinEntriesBuffer        = VK_NULL_HANDLE;
    VmaAllocation m_tileBinEntriesAllocation    = VK_NULL_HANDLE;
    VkDeviceSize  m_tileBinEntriesBufferBytes   = 0ull;
    u32           m_tileBinEntriesBindlessSlot  = UINT32_MAX;

    VkBuffer      m_spillBuffer                 = VK_NULL_HANDLE;
    VmaAllocation m_spillAllocation             = VK_NULL_HANDLE;
    VkDeviceSize  m_spillBufferBytes            = 0ull;
    u32           m_spillBufferBindlessSlot     = UINT32_MAX;

    // M4.6 per-tile spill linked-list heads. u32 * numTiles. Reset to
    // UINT32_MAX each frame so an empty tile's head walker sees "no
    // chain". Extent-dependent like the other tile-grid buffers.
    VkBuffer      m_spillHeadsBuffer            = VK_NULL_HANDLE;
    VmaAllocation m_spillHeadsAllocation        = VK_NULL_HANDLE;
    VkDeviceSize  m_spillHeadsBufferBytes       = 0ull;
    u32           m_spillHeadsBufferBindlessSlot = UINT32_MAX;

    VkBuffer      m_dispatchIndirectBuffer      = VK_NULL_HANDLE;
    VmaAllocation m_dispatchIndirectAllocation  = VK_NULL_HANDLE;
    VkDeviceSize  m_dispatchIndirectBufferBytes = 0ull;
    u32           m_dispatchIndirectBindlessSlot = UINT32_MAX;
};

const char* micropolySwRasterErrorKindString(MicropolySwRasterErrorKind kind);

} // namespace enigma::renderer::micropoly
