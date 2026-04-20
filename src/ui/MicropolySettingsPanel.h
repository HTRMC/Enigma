#pragma once

// MicropolySettingsPanel (M6.2a)
// ==============================
// A single ImGui window that exposes the runtime-tunable micropolygon knobs
// in one place:
//   - Enabled toggle   (MicropolyConfig::enabled)
//   - Page cache MiB   (MicropolyConfig::pageCacheMB)
//   - Force HW / SW    (MicropolyConfig::forceHW / forceSW, enforced mutex)
//   - Overlay radio    (writes back through a DebugMode reference)
//
// Deferred to M6.2b: LOD-scale slider, Bounds + BinOverflowHeat overlay
// entries. Those need new config fields / shaders that do not ship yet.
//
// Stateless by design — no per-frame UI state is stashed on the class; the
// caller owns both MicropolyConfig and DebugMode, the panel only reads +
// writes through the references passed into draw(). Call once per frame
// between ImGui::NewFrame() and ImGui::Render().

namespace enigma {

struct MicropolyConfig;
enum class DebugMode;

namespace gfx { class Device; }
namespace renderer::micropoly { struct CullStats; }

class MicropolySettingsPanel {
public:
    MicropolySettingsPanel() = default;

    // Draws the panel for this frame. `open` controls window visibility
    // (ImGui idiom); pass nullptr to always-open. `device` gates which
    // overlay-radio entries are selectable — entries requiring
    // shaderImageInt64 are greyed out when unsupported. `config` and
    // `debugMode` are written in-place on user interaction.
    //
    // `stats` is optional (nullable) — when non-null the panel surfaces
    // the live cull-counter read in a collapsing section. Pass nullptr
    // when the cull pass isn't available (micropoly disabled / pre-
    // bring-up).
    void draw(bool* open,
              const gfx::Device& device,
              MicropolyConfig& config,
              DebugMode& debugMode,
              const renderer::micropoly::CullStats* stats = nullptr);
};

} // namespace enigma
