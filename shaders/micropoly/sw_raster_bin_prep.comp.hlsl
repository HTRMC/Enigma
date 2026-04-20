// sw_raster_bin_prep.comp.hlsl
// =============================
// Single-thread compute that copies the cull pass's indirect-draw header
// count into a dispatchIndirect buffer shaped {x, y, z}. The SW raster
// binning compute (sw_raster_bin.comp.hlsl) consumes this via
// vkCmdDispatchIndirect — one workgroup per surviving cluster.
//
// Context: MicropolyCullPass writes its survivor count as the first u32 of
// its indirect-draw buffer (offset 0). That buffer's layout is shaped for
// vkCmdDrawMeshTasksIndirectCountEXT (16-byte header + 16-byte cmds), NOT
// for vkCmdDispatchIndirect which expects {groupCountX, groupCountY,
// groupCountZ} starting at the indirect offset. This trivial compute
// bridges the two by reading the u32 count from offset 0 and writing
// {count, 1, 1} to a dedicated dispatchIndirect buffer.
//
// Principle 1: when MicropolyConfig::enabled==false this shader is never
// dispatched (pass is gated on the same capability row as the HW raster).

#include "../common.hlsl"

// --- Bindless resource arrays ---------------------------------------------
[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// --- Push constants --------------------------------------------------------
// Matches MicropolySwRasterBinPrepPushBlock in MicropolySwRasterPass.cpp.
struct PushBlock {
    uint indirectBufferBindlessIndex;         // input — cull indirect-draw buffer
    uint dispatchIndirectBufferBindlessIndex; // output — {x, y, z} dispatch header
    uint maxDispatchGroups;                   // security clamp — cluster slot cap
    uint _pad0;                               // round push block to 16 B
};
[[vk::push_constant]] PushBlock pc;

[numthreads(1, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x != 0u) return;

    RWByteAddressBuffer src = g_rwBuffers[NonUniformResourceIndex(pc.indirectBufferBindlessIndex)];
    RWByteAddressBuffer dst = g_rwBuffers[NonUniformResourceIndex(pc.dispatchIndirectBufferBindlessIndex)];

    // Read the count header the cull shader emitted via InterlockedAdd.
    // Security: the cull pass's emitDrawCmd InterlockedAdds the counter
    // UNCONDITIONALLY before its own bounds check, so this header can exceed
    // kMpMaxIndirectDrawClusters (64K) under heavy visibility. An unclamped
    // count would feed vkCmdDispatchIndirect a groupCountX past
    // maxComputeWorkGroupCount[0] (VVL failure) AND let workgroups at
    // gid.x >= cap read past the indirect-draw buffer's allocated region
    // (16 B header + cap * 16 B commands). Clamp here so we only dispatch
    // groups the buffer actually has slots for.
    const uint count   = src.Load(0u);
    const uint clamped = min(count, pc.maxDispatchGroups);

    // Write dispatchIndirect shape {x, y, z}. One workgroup per cluster.
    dst.Store3(0u, uint3(clamped, 1u, 1u));
}
