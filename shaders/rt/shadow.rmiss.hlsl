// shadow.rmiss.hlsl
// =================
// Miss shader for RT shadows. Miss = surface is lit (no occluder found).

struct ShadowPayload {
    float visibility;
};

[shader("miss")]
void MissMain(inout ShadowPayload payload) {
    payload.visibility = 1.0; // fully lit — no occluder
}
