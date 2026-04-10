// gi.rchit.hlsl
// =============
// Closest-hit shader for RT global illumination. At secondary hit:
// evaluate simple Lambertian * sun_light visibility (no recursive RT).

struct GIPayload {
    float3 color;
    float  hitT;
};

[shader("closesthit")]
void ClosestHitMain(inout GIPayload payload,
                    in BuiltInTriangleIntersectionAttributes attribs) {
    // Simple Lambertian evaluation at the hit point.
    // Sun direction (hardcoded for now — Phase 3 will pass via SBT data).
    static const float3 sunDir   = normalize(float3(0.5, 1.0, 0.3));
    static const float3 sunColor = float3(1.0, 1.0, 1.0) * 3.0;

    // Approximate normal from world-space ray direction (no vertex data access).
    float3 hitNormal = float3(0.0, 1.0, 0.0);

    // Lambertian diffuse: NdotL * sun_color / PI.
    float NdotL = saturate(dot(hitNormal, sunDir));
    float3 diffuse = (sunColor * NdotL) / 3.14159265;

    // Modulate by a neutral albedo (0.5 grey).
    payload.color = diffuse * 0.5;
    payload.hitT  = RayTCurrent();
}
