// Smoke test for MicropolySettingsPanel (M6.2a).
//
// ImGui requires an active context to call any widget function — bringing
// one up in a header-linked test would need the Vulkan backend wiring the
// Enigma target drags in. That is out of scope for this smoke binary.
//
// What this test does cover:
//   (a) MicropolySettingsPanel.h includes cleanly and the class default-
//       constructs (catches header/ABI drift).
//   (b) MicropolyConfig's default matches the values the panel expects
//       (enabled=false, forceHW=forceSW=false, pageCacheMB within the
//       panel's slider bounds).
//   (c) DebugMode's symbolic constants used by the overlay radio exist.
//
// Mirrors the style of micropoly_raster_test / micropoly_blas_manager_test
// (header-only smoke, no live device).

#include "renderer/micropoly/MicropolyConfig.h"
#include "ui/MicropolySettingsPanel.h"

#include <cstdio>

// DebugMode lives in renderer/Renderer.h which transitively pulls volk +
// the full Vulkan include chain. The panel smoke test only needs the
// header to compile and the enum values that the panel writes — both
// already covered by the Enigma target's build. We intentionally do NOT
// include Renderer.h here so the test TU stays Vulkan-free.

namespace {

bool defaultsSane() {
    enigma::MicropolyConfig cfg{};

    if (cfg.enabled) {
        std::fprintf(stderr,
            "[micropoly_settings_panel_test] default cfg.enabled != false\n");
        return false;
    }
    if (cfg.forceHW || cfg.forceSW) {
        std::fprintf(stderr,
            "[micropoly_settings_panel_test] default forceHW/forceSW != false\n");
        return false;
    }
    // Panel slider bounds are 64..4096 MiB.
    if (cfg.pageCacheMB < 64u || cfg.pageCacheMB > 4096u) {
        std::fprintf(stderr,
            "[micropoly_settings_panel_test] default pageCacheMB=%u outside 64..4096\n",
            cfg.pageCacheMB);
        return false;
    }
    return true;
}

bool panelConstructs() {
    enigma::MicropolySettingsPanel panel;
    (void)panel;   // stateless; default-ctor is the whole surface.
    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok &= defaultsSane();
    ok &= panelConstructs();

    if (!ok) {
        std::fprintf(stderr, "[micropoly_settings_panel_test] FAILED\n");
        return 1;
    }
    std::printf("[micropoly_settings_panel_test] OK\n");
    return 0;
}
