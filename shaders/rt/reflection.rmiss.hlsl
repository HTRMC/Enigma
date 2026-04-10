// reflection.rmiss.hlsl
// =====================
// Miss shader for RT reflections. Returns a sky colour gradient when
// the reflection ray does not hit any geometry.

struct ReflectionPayload {
    float3 color;
    float  hitT;
};

[shader("miss")]
void MissMain(inout ReflectionPayload payload) {
    // Simple sky gradient based on ray direction.
    float3 dir      = normalize(WorldRayDirection());
    float  t        = saturate(dir.y * 0.5 + 0.5);
    float3 skyTop   = float3(0.4, 0.6, 1.0);
    float3 skyBot   = float3(0.1, 0.1, 0.2);
    payload.color   = lerp(skyBot, skyTop, t);
    payload.hitT    = -1.0; // no hit
}
