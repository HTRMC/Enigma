#pragma once

namespace enigma {

// All atmosphere/post-process tuning knobs exposed via the Settings panel.
// azimuth/elevation are UI-only — no shader ever receives them directly.
// Renderer::drawFrame() converts them to m_sunWorldDir (a vec3) once per
// frame and fans that canonical direction to all consumers.
struct AtmosphereSettings {
    float sunAzimuth   = 135.0f; // degrees, 0=north, 90=east (UI only)
    float sunElevation =  45.0f; // degrees above horizon, -10..90 (UI only)
    float sunIntensity =  10.0f;

    float exposureEV       = 0.0f;  // EV offset applied pre-tone-map
    float bloomThreshold   = 1.0f;  // linear luminance threshold
    float bloomIntensity   = 0.8f;

    int   tonemapMode = 0;  // 0 = AgX, 1 = ACES

    bool bloomEnabled              = true;
    bool aerialPerspectiveEnabled  = true;
};

} // namespace enigma
