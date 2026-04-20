// page_request_emit.hlsl
// =======================
// Inline helper for compute shaders to emit page-streaming requests to
// the Micropoly RequestQueue (CPU-side: RequestQueue.h/.cpp).
//
// Usage (caller is responsible for binding the buffers):
//
//   RWStructuredBuffer<RequestQueueHeader> g_mpRequestHeader : register(u0);
//   RWStructuredBuffer<uint>               g_mpRequestSlots  : register(u1);
//
//   #include "micropoly/page_request_emit.hlsl"
//
//   [numthreads(64,1,1)]
//   void CSMain(uint3 dtid : SV_DispatchThreadID) {
//       if (needsPage) {
//           EmitPageRequest(g_mpRequestHeader, g_mpRequestSlots, pageId);
//       }
//   }
//
// This file is INCLUDE-ONLY and must NEVER be listed in the spirv_diff
// manifest — it produces no standalone SPIR-V blob.
//
// Layout contract: RequestQueueHeader must match exactly the C++ struct
// enigma::renderer::micropoly::RequestQueueHeader in RequestQueue.h.

#ifndef MICROPOLY_PAGE_REQUEST_EMIT_HLSL
#define MICROPOLY_PAGE_REQUEST_EMIT_HLSL

struct RequestQueueHeader {
    uint count;        // InterlockedAdd'd by GPU; drained + reset by CPU.
    uint capacity;     // written once at CPU-side create(); read-only on GPU.
    uint overflowed;   // InterlockedOr'd by GPU when a slot is rejected.
    uint _pad;
};

// Emit a page request into the queue. Returns true if the request was
// accepted, false if the queue was already full (in which case the
// overflowed flag is atomically set so the CPU drain can observe it).
//
// IMPLEMENTATION NOTE: we bump header.count unconditionally to keep the
// fast path branch-free on the common (non-overflow) case. The CPU drain
// clamps observed count to capacity on its side so the overshoot is
// harmless — uninitialized slots above capacity are never read.
bool EmitPageRequest(RWStructuredBuffer<RequestQueueHeader> header,
                     RWStructuredBuffer<uint>               slots,
                     uint                                   pageId)
{
    uint slot;
    InterlockedAdd(header[0].count, 1u, slot);
    if (slot >= header[0].capacity) {
        InterlockedOr(header[0].overflowed, 1u);
        return false;
    }
    slots[slot] = pageId;
    return true;
}

#endif // MICROPOLY_PAGE_REQUEST_EMIT_HLSL
