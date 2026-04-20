#pragma once

#include "core/Types.h"

#include <filesystem>

namespace enigma {

// MicropolyConfig
// ===============
// Runtime flags that drive the micropolygon geometry subsystem. Owned by
// Renderer and passed into MicropolyPass at construction. Values here
// correspond to the feature gate + debug toggles described in
// .omc/plans/ralplan-micropolygon.md §3.M0a.
//
// Invariant (Principle 1): when `enabled == false`, the pipeline must be
// bit-identical to pre-micropoly behavior — no resource allocations, no
// descriptor updates, no pipeline creation may occur downstream of this
// flag. MicropolyPass honors this contract; do not bypass it.
//
// TBD(M0a-exec): command-line + ImGui wiring is limited to the Renderer's
// existing Settings panel idiom (see Renderer.cpp's ImGui::Checkbox block);
// a full command-line parser is deferred to M6 per plan §3.M6 (config UI).
struct MicropolyConfig {
    // Master enable. When false, MicropolyPass::record() is a no-op and no
    // micropoly GPU resources are created.
    bool enabled = false;

    // Force the software-rasterization classification path regardless of
    // per-cluster screen-area heuristics. Debug/testing use only.
    //
    // Invariant: `forceSW` and `forceHW` are mutually exclusive. Asserting
    // both true is a programmer error; MicropolyPass's constructor enforces
    // this with an ENIGMA_ASSERT. Setting neither is the production default.
    bool forceSW = false;

    // Force the hardware-rasterization (mesh-shader) path regardless of
    // per-cluster screen-area heuristics. Debug/testing use only.
    // Mutually exclusive with forceSW (see above).
    bool forceHW = false;

    // Upper bound (in MiB) for the streaming page cache that M2 will manage.
    // Held here so M0a can pre-declare the value before M2 consumes it.
    u32  pageCacheMB = 1024;

    // Per-frame LOD scale. Feeds MicropolyCullPass::DispatchInputs::
    // screenSpaceErrorThreshold. 1.0 = reference DAG traversal (one-pixel
    // error budget at bake resolution). >1.0 trades quality for perf
    // (accept coarser LODs earlier); <1.0 forces finer LODs at the cost of
    // more clusters passing cull. Read per-frame — runtime tunable via the
    // Micropoly settings panel.
    f32  lodScale = 1.0f;

    // Surface the per-cluster classification + vis-buffer inspector overlays
    // in the debug UI. Wired in M6; stored here from M0a onward.
    bool debugOverlay = false;

    // Path to the .mpa asset (baked by enigma-mpbake) that MicropolyStreaming
    // opens at Renderer construction when enabled == true. Empty path +
    // enabled == true is a programmer error and surfaces as a streaming
    // init-failure log line. Ignored when enabled == false (Principle 1:
    // no allocations, no file IO, no behavior change).
    std::filesystem::path mpaFilePath;
};

} // namespace enigma
