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

    float exposureEV       = -3.0f; // EV offset applied pre-tone-map (-3 EV ÷8 brings sunIntensity=10 into AgX sweet spot)
    float bloomThreshold   = 2.0f;  // post-exposure luminance threshold: fires on bright specular/emissives only
    float bloomIntensity   = 0.4f;

    int   tonemapMode = 0;  // 0 = AgX, 1 = ACES

    bool  bloomEnabled             = true;
    bool  aerialPerspectiveEnabled = true;
    float aerialPerspectiveStrength = 0.25f; // 0=off, 1=full physical; 0.25 suits ground-level game scenes
};

} // namespace enigma
