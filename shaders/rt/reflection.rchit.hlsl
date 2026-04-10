// reflection.rchit.hlsl
// =====================
// Closest-hit shader for RT reflections. Phase 1 placeholder: samples a
// simple sky-like colour based on hit normal to give visual feedback.
// Full PBR at the hit point is deferred to Phase 2.

struct ReflectionPayload {
    float3 color;
    float  hitT;
};

[shader("closesthit")]
void ClosestHitMain(inout ReflectionPayload payload,
                    in BuiltInTriangleIntersectionAttributes attribs) {
    // Placeholder: sky colour modulated by a simple directional term.
    // Use the world-space ray direction to derive a gradient.
    float3 hitNormal = float3(0.0, 1.0, 0.0); // approximate
    float3 skyTop    = float3(0.3, 0.5, 0.9);
    float3 skyBottom = float3(0.1, 0.1, 0.2);
    float  t         = saturate(WorldRayDirection().y * 0.5 + 0.5);
    payload.color    = lerp(skyBottom, skyTop, t) * 0.5;
    payload.hitT     = RayTCurrent();
}
