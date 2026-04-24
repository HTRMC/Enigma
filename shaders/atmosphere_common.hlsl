// atmosphere_common.hlsl
// ======================
// Hillaire 2020 physically-based atmosphere — shared constants, density
// models, and helper functions used by all four LUT compute shaders.
//
// Unit convention: all distances in km, extinction values in km⁻¹.
// Reference: Sébastien Hillaire, "A Scalable and Production Ready Sky and
// Atmosphere Rendering Technique", EGSR 2020.

static const float PI = 3.14159265358979f;

// Planetary radii (km)
static const float R_EARTH = 6360.0f;   // planet surface radius
static const float R_TOP   = 6460.0f;   // top of atmosphere radius

// Rayleigh — σ_s at sea level (km⁻¹), scale height 8 km
static const float3 RAYLEIGH_SIGMA_S = float3(5.802e-3f, 13.558e-3f, 33.100e-3f);
static const float  RAYLEIGH_H       = 8.0f;

// Mie — σ_s, σ_a at sea level (km⁻¹), scale height 1.2 km
// Phase asymmetry parameter g = 0.8 (Cornette-Shanks)
static const float  MIE_SIGMA_S = 3.996e-3f;
static const float  MIE_SIGMA_A = 4.400e-3f;
static const float  MIE_H       = 1.2f;
static const float  MIE_G       = 0.8f;

// Ozone absorption (km⁻¹) — tent function centered at 25 km
static const float3 OZONE_SIGMA_A = float3(0.650e-3f, 1.881e-3f, 0.085e-3f);

// Aerial Perspective froxel depth range (km).
// AP_NEAR / AP_FAR must match AtmospherePass::apSliceFar (C++ side) and
// any shader that samples the volume with a log-distributed depth mapping.
//
// AP_NEAR = 1 km — near cutoff for the LUT. Fragments closer than this do
// NOT receive aerial perspective at all (composite skips the blend — see
// post_process.hlsl); the LUT starts its integration budget at 1 km so
// slice resolution is spent on the 1–50 km range where haze is actually
// perceptible on a ground-level scene. At 1 km sea-level Rayleigh optical
// depth ≈ 0.0058 (transmittance ≈ 99.4 %) — the first froxel represents
// the very onset of visible haze. Slice 16/32 ≈ 7 km, slice 24/32 ≈ 19 km,
// slice 31/32 ≈ 50 km.
static const float AP_NEAR = 1.0f;
static const float AP_FAR  = 50.0f;

// ---------------------------------------------------------------------------
// Density profiles
// ---------------------------------------------------------------------------

float rayleighDensity(float h) {
    return exp(-max(0.0f, h) / RAYLEIGH_H);
}

float mieDensity(float h) {
    return exp(-max(0.0f, h) / MIE_H);
}

float ozoneDensity(float h) {
    // Tent: 0 at h<=10, peak 1 at h=25, 0 at h>=40 km
    return max(0.0f, h < 25.0f ? (h - 10.0f) / 15.0f : (40.0f - h) / 15.0f);
}

// ---------------------------------------------------------------------------
// Extinction and scattering at altitude h
// ---------------------------------------------------------------------------

float3 totalExtinction(float h) {
    float3 ext = RAYLEIGH_SIGMA_S  * rayleighDensity(h)
               + (MIE_SIGMA_S + MIE_SIGMA_A) * mieDensity(h)
               + OZONE_SIGMA_A * ozoneDensity(h);
    return ext;
}

void getScatteringCoeffs(float h, out float3 sigmaRayleigh, out float sigmaMie) {
    sigmaRayleigh = RAYLEIGH_SIGMA_S * rayleighDensity(h);
    sigmaMie      = MIE_SIGMA_S      * mieDensity(h);
}

// ---------------------------------------------------------------------------
// Phase functions
// ---------------------------------------------------------------------------

float rayleighPhase(float cosTheta) {
    return (3.0f / (16.0f * PI)) * (1.0f + cosTheta * cosTheta);
}

float miePhase(float cosTheta) {
    float g  = MIE_G;
    float g2 = g * g;
    float num = (1.0f - g2) * (1.0f + cosTheta * cosTheta);
    float den = (2.0f + g2) * pow(max(1e-6f, 1.0f + g2 - 2.0f * g * cosTheta), 1.5f);
    return (3.0f / (8.0f * PI)) * num / max(1e-9f, den);
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

// Ray-sphere intersection. Returns true if hit, populates t0 (entry) and t1 (exit).
// Negative t means behind ray origin.
bool raySphereIntersect(float3 ro, float3 rd, float r, out float t0, out float t1) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float disc = b * b - c;
    if (disc < 0.0f) { t0 = t1 = 0.0f; return false; }
    float sq = sqrt(disc);
    t0 = -b - sq;
    t1 = -b + sq;
    return true;
}

// Altitude above planet surface (km) for a world-space position (km from center)
float getAltitude(float3 posKm) {
    return length(posKm) - R_EARTH;
}

// ---------------------------------------------------------------------------
// Transmittance LUT UV parameterization (Hillaire 2020)
//   u = (cos_zenith + 1) / 2   [zenith angle from vertical, range [-1,1]]
//   v = sqrt(h / H_ATM)        [sqrt mapping for surface precision]
// ---------------------------------------------------------------------------

float2 transmittanceUV(float h, float cosZenith) {
    float u = (cosZenith + 1.0f) * 0.5f;
    float v = sqrt(saturate(h / (R_TOP - R_EARTH)));
    return float2(u, v);
}

void transmittanceUVToParams(float2 uv, out float h, out float cosZenith) {
    h        = (R_TOP - R_EARTH) * uv.y * uv.y;
    cosZenith = 2.0f * uv.x - 1.0f;
}

// Sample transmittance LUT: T(pos at altitude h, toward direction with cosZenith)
float3 sampleTransmittanceLUT(Texture2D lut, SamplerState samp, float h, float cosZenith) {
    float2 uv = transmittanceUV(h, cosZenith);
    return lut.SampleLevel(samp, uv, 0).rgb;
}
