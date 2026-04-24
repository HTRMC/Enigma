#include "ui/MicropolySettingsPanel.h"

#include "core/Types.h"
#include "gfx/Device.h"
#include "renderer/Renderer.h"                      // DebugMode
#include "renderer/micropoly/MicropolyConfig.h"     // MicropolyConfig
#include "renderer/micropoly/MicropolyCullPass.h"   // CullStats

#include <imgui.h>

namespace enigma {

namespace {

// Page-cache slider bounds. 64 MiB is the lowest value the streaming
// subsystem is wired to handle without immediate eviction pressure on the
// DamagedHelmet reference asset; 4 GiB is a practical upper bound that
// covers the worst-case residency ceiling identified in the M2 cold-cache
// stress test on the reference workstation. Kept in the panel TU so a
// policy change does not drag the header along.
constexpr u32 kMinPageCacheMB = 64u;
constexpr u32 kMaxPageCacheMB = 4096u;

const char* tierName(gfx::GpuTier tier) {
    switch (tier) {
        case gfx::GpuTier::Min:         return "Min";
        case gfx::GpuTier::Recommended: return "Recommended";
        case gfx::GpuTier::Extreme:     return "Extreme";
        case gfx::GpuTier::ExtremeRT:   return "ExtremeRT";
    }
    return "?";
}

} // namespace

void MicropolySettingsPanel::draw(bool* open,
                                  const gfx::Device& dev,
                                  MicropolyConfig& cfg,
                                  DebugMode& mode,
                                  const renderer::micropoly::CullStats* stats) {
    // Caller-controlled visibility (ImGui idiom): skip the whole window
    // when closed. nullptr == always-open.
    if (open != nullptr && !*open) return;

    ImGui::SetNextWindowPos({930.f, 10.f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({320.f, 340.f}, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Micropoly", open)) {
        ImGui::End();
        return;
    }

    // ---------------------------------------------------------------
    // Capability tier row — shows the user what the device is and which
    // optional features are available. This mirrors the HW matrix the
    // capability classifier keys off (plan §3.M0a).
    // ---------------------------------------------------------------
    ImGui::TextDisabled("Device: %s (%s)",
                        dev.properties().deviceName,
                        tierName(dev.gpuTier()));
    ImGui::TextDisabled("Mesh shaders: %s  |  Int64 image: %s",
                        dev.supportsMeshShaders()     ? "yes" : "no",
                        dev.supportsShaderImageInt64() ? "yes" : "no");
    ImGui::TextDisabled("Sparse residency: %s  |  RT: %s",
                        dev.supportsSparseResidency() ? "yes" : "no",
                        dev.supportsRayTracing()      ? "yes" : "no");
    ImGui::Separator();

    // ---------------------------------------------------------------
    // Runtime overlay radio (LIVE — writes back through DebugMode each
    // frame). Ordered before config so the useful runtime-wired control
    // is immediately visible, and so users don't confuse the next-launch
    // config toggles with live switches.
    //
    // Available entries match the shader inventory that ships in M4.6,
    // M6.1, M6.2b, and §3.M6 (BinOverflowHeat).
    //
    // Availability gate matches Renderer's Debug Views panel: the
    // vis-image-sampling overlays (LOD / HW-SW class / Residency) need
    // shaderImageInt64 so the 64-bit vis image has meaningful writes.
    // The Bounds overlay does NOT need Int64 — it iterates the DAG
    // directly — and BinOverflow reads only the SW raster bin SSBOs —
    // but reusing the same gate keeps the UI simple since all five
    // overlays only ship on micropoly-capable builds anyway.
    // ---------------------------------------------------------------
    ImGui::TextDisabled("Debug Overlay (live)");
    const bool overlayOK = dev.supportsShaderImageInt64();
    ImGui::BeginDisabled(!overlayOK);

    int radioIdx = 0;
    if (mode == DebugMode::MicropolyLodHeatmap)            radioIdx = 1;
    else if (mode == DebugMode::MicropolyRasterClass)      radioIdx = 2;
    else if (mode == DebugMode::MicropolyResidencyHeatmap) radioIdx = 3;
    else if (mode == DebugMode::MicropolyBounds)           radioIdx = 4;
    else if (mode == DebugMode::MicropolyBinOverflowHeat)  radioIdx = 5;

    bool changed = false;
    changed |= ImGui::RadioButton("None",         &radioIdx, 0);
    changed |= ImGui::RadioButton("LOD heatmap",  &radioIdx, 1);
    changed |= ImGui::RadioButton("HW/SW class",  &radioIdx, 2);
    changed |= ImGui::RadioButton("Residency",    &radioIdx, 3);
    changed |= ImGui::RadioButton("Bounds",       &radioIdx, 4);
    changed |= ImGui::RadioButton("BinOverflow",  &radioIdx, 5);
    if (changed) {
        switch (radioIdx) {
            case 1:  mode = DebugMode::MicropolyLodHeatmap;       break;
            case 2:  mode = DebugMode::MicropolyRasterClass;      break;
            case 3:  mode = DebugMode::MicropolyResidencyHeatmap; break;
            case 4:  mode = DebugMode::MicropolyBounds;           break;
            case 5:  mode = DebugMode::MicropolyBinOverflowHeat;  break;
            default: mode = DebugMode::Lit;                       break;
        }
    }

    ImGui::EndDisabled();
    if (!overlayOK) {
        ImGui::TextDisabled("(Debug overlays need shaderImageInt64)");
    }

    ImGui::Separator();

    // ---------------------------------------------------------------
    // Runtime LOD scale. Feeds MicropolyCullPass::screenSpaceErrorThreshold
    // per frame — unlike the config-time knobs below, this is LIVE.
    //   <1.0 = finer LODs (more clusters pass cull, higher fidelity, slower)
    //    1.0 = reference (one-pixel screen-space error at bake resolution)
    //   >1.0 = coarser LODs (fewer clusters, lower fidelity, faster)
    // ---------------------------------------------------------------
    ImGui::TextDisabled("LOD scale (live)");
    f32 lodScale = cfg.lodScale;
    if (ImGui::SliderFloat("##lod_scale", &lodScale, 0.25f, 4.0f, "%.2fx")) {
        cfg.lodScale = lodScale;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Screen-space error threshold multiplier. 1.0 = reference.");
    }

    bool disableLOD = cfg.disableLOD;
    if (ImGui::Checkbox("Disable LOD (diagnostic)", &disableLOD)) {
        cfg.disableLOD = disableLOD;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Skip the screen-space-error gate in the cull shader. "
                          "Residency, frustum, cone, and HiZ culls still run. "
                          "Use to isolate LOD bugs from raster/data bugs.");
    }

    ImGui::Separator();

    // ---------------------------------------------------------------
    // Live cull-stats readback. Values come from the HOST_VISIBLE cull-
    // stats buffer the MicropolyCullPass atomically bumps each frame —
    // see MicropolyCullPass::readbackStats(). The section is default-
    // collapsed to keep the panel compact when the user isn't debugging
    // cull behaviour; a 1-2 frame lag is acceptable for a debug HUD.
    //
    // Null stats => cull pass unavailable (micropoly disabled or pre-
    // bring-up). Skip the whole section rather than show misleading
    // zeros that look like "everything got culled".
    // ---------------------------------------------------------------
    if (stats != nullptr) {
        if (ImGui::CollapsingHeader("Cull stats (live)")) {
            const u32 total   = stats->totalDispatched;
            const u32 visible = stats->visible;

            ImGui::Text("Dispatched: %u", total);
            ImGui::Text("Visible:    %u", visible);
            if (total > 0u) {
                const f32 passRate = 100.f * static_cast<f32>(visible)
                                           / static_cast<f32>(total);
                ImGui::Text("Pass rate:  %.2f%%", passRate);
            } else {
                ImGui::TextDisabled("Pass rate:  n/a (nothing dispatched)");
            }

            ImGui::Separator();
            ImGui::TextDisabled("Culled breakdown");
            // Helper lambda: avoids repeating the total==0 guard per row.
            // Captures `total` by value — it's a plain u32 so the cost
            // is nil and the lambda stays self-contained.
            auto cullRow = [total](const char* label, u32 count) {
                if (total > 0u) {
                    const f32 pct = 100.f * static_cast<f32>(count)
                                          / static_cast<f32>(total);
                    ImGui::Text("%-10s %8u (%5.2f%%)", label, count, pct);
                } else {
                    ImGui::Text("%-10s %8u", label, count);
                }
            };
            cullRow("LOD",       stats->culledLOD);
            cullRow("Residency", stats->culledResidency);
            cullRow("Frustum",   stats->culledFrustum);
            cullRow("Backface",  stats->culledBackface);
            cullRow("HiZ",       stats->culledHiZ);
        }

        ImGui::Separator();
    }

    // ---------------------------------------------------------------
    // Construction-time config. MicropolyConfig::enabled / forceHW /
    // forceSW / pageCacheMB are read by Renderer at construction; they
    // do NOT take effect at runtime. Surfacing them here lets a user
    // inspect the current startup state and (via the next-launch
    // rationale) pre-configure before a restart; the controls are
    // intentionally read-only to avoid giving the false impression that
    // toggling lights the pipeline up mid-session.
    // ---------------------------------------------------------------
    if (ImGui::CollapsingHeader("Startup config (read-only)")) {
        ImGui::BeginDisabled(true);

        bool enabled = cfg.enabled;
        ImGui::Checkbox("Enabled", &enabled);

        u32 pageCacheMB = cfg.pageCacheMB;
        ImGui::SliderScalar("Page cache (MiB)", ImGuiDataType_U32,
                            &pageCacheMB,
                            &kMinPageCacheMB, &kMaxPageCacheMB,
                            "%u");

        bool fhw = cfg.forceHW;
        bool fsw = cfg.forceSW;
        ImGui::Checkbox("Force HW raster (debug)", &fhw);
        ImGui::Checkbox("Force SW raster (debug)", &fsw);

        ImGui::EndDisabled();

        // Surface the .mpa asset path when streaming is enabled — this is
        // the value the user passed via --micropoly at launch. Kept outside
        // the BeginDisabled block so the text renders at normal contrast
        // (TextDisabled already styles it as secondary info).
        if (cfg.enabled) {
            ImGui::TextDisabled("Asset: %s", cfg.mpaFilePath.string().c_str());
        }

        ImGui::TextDisabled("(Read-only: applies on next renderer bring-up)");
    }

    ImGui::End();
}

} // namespace enigma
