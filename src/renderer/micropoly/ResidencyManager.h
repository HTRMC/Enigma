#pragma once

// ResidencyManager.h
// ===================
// CPU-side LRU cache tracking which Micropoly pages are "resident" (have been
// decompressed and, in later milestones, uploaded to the GPU page pool). This
// is the M2.1 scaffold: pure CPU bookkeeping, no Vulkan.
//
// Contract
// --------
// - All public methods are thread-safe. Internally we take a single mutex
//   per call, so fine-grained concurrency is intentionally limited to keep
//   the LRU structure simple.
// - `insert()` is the only mutator. It returns the ordered event log for
//   the call (evictions first, then insert/touch) so callers can drive
//   GPU-side eviction + upload in the proper order in M2.2+.
// - `beginFrame()` is a lightweight monotonic-tick bump; we do not rely on
//   wall-clock time for LRU ordering.
// - Determinism: the LRU runtime order is driven by the access pattern,
//   not required to be bit-identical across runs. The decompressed bytes
//   for a given page must still be identical to what MpAssetReader produces.
//
// M2.2 will extend ResidencyEntry with a VkBuffer slot/offset (indexed into
// a fixed GPU page pool). M2.4 will wire the transfer-queue upload path.
// The present class deliberately stops at CPU-side tracking so M2.1 can
// be built and tested without any Vulkan dependency.

#include "core/Types.h"

#include <cstddef>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace enigma::renderer::micropoly {

// One bookkeeping entry per resident page. The ordering of entries in the
// owning std::list defines the LRU order: front = oldest, back = newest.
struct ResidencyEntry {
    u32 pageId        = 0u;
    u64 lastUsedTick  = 0u;   // monotonic frame counter set by beginFrame()
    u32 sizeBytes     = 0u;   // decompressed page size, byte-accurate
    // M2.2 will add VkBuffer offset / GPU page slot index here.
};

// Manager configuration. Only the capacity is tunable in M2.1.
struct ResidencyManagerOptions {
    u64 capacityBytes = 512ull * 1024ull * 1024ull;  // default 512 MiB
};

// Event kinds emitted by insert() so callers can reconcile GPU state.
enum class ResidencyEventKind {
    Inserted,   // page newly made resident this call
    Evicted,    // page pushed out by LRU to make room for an insert
    Touched,    // page was already resident; timestamp updated only
};

struct ResidencyEvent {
    ResidencyEventKind kind = ResidencyEventKind::Inserted;
    u32                pageId = 0u;
};

// Thread-safe LRU residency tracker. Value-semantics, not copyable.
class ResidencyManager {
public:
    explicit ResidencyManager(const ResidencyManagerOptions& opts);
    ~ResidencyManager() = default;

    ResidencyManager(const ResidencyManager&)            = delete;
    ResidencyManager& operator=(const ResidencyManager&) = delete;
    ResidencyManager(ResidencyManager&&)                 = delete;
    ResidencyManager& operator=(ResidencyManager&&)      = delete;

    // Result returned from insert(). `wasAlreadyResident` tells the caller
    // whether it needs to kick off a new upload; `events` lists every state
    // transition this call caused, in order: evictions first (oldest first),
    // then the Inserted / Touched terminal event. Callers in M2.2+ will
    // walk this list to drive GPU-side evictions before uploads.
    struct InsertResult {
        bool                        wasAlreadyResident = false;
        std::vector<ResidencyEvent> events;
    };

    // Record that `pageId` (with decompressed size `sizeBytes`) is in use.
    // If the page is already resident, bump its LRU timestamp (Touched).
    // Otherwise evict from the front of the LRU list until the new entry
    // fits under the capacity cap, then append it to the back (Inserted).
    //
    // Evictions are capped per call at kMaxEvictionsPerInsert to prevent
    // a single pathological input from driving unbounded work; further
    // evictions will happen naturally on subsequent inserts.
    InsertResult insert(u32 pageId, u32 sizeBytes);

    // Cheap O(1) query. Does not touch the LRU order — use insert() for
    // that so the event log stays coherent.
    bool isResident(u32 pageId) const;

    // Advance the monotonic tick counter. Safe to call from the render
    // thread once per frame. The tick is used to stamp `lastUsedTick`
    // during insert() so tests can reason about relative ages without
    // threading wall-clock time through.
    void beginFrame();

    // Debug / test snapshot. Safe to call from any thread.
    struct Stats {
        u64 residentBytes      = 0u;
        u64 capacityBytes      = 0u;
        u64 insertionCount     = 0u;  // cumulative Inserted events
        u64 evictionCount      = 0u;  // cumulative Evicted events
        u64 touchCount         = 0u;  // cumulative Touched events
        u32 residentPageCount  = 0u;
    };
    Stats stats() const;

    // Exposed for tests so assertions can verify the LRU cap without
    // spawning pathological inputs.
    static constexpr std::size_t kMaxEvictionsPerInsert = 16u;

private:
    // The list holds all resident entries in LRU order. The map indexes
    // into the list in O(1) for hit tests. std::list::iterator is stable
    // under insertions and erasures elsewhere in the list, so the map
    // stays valid across splice / erase of other entries. This is the
    // one place we use unordered_map: runtime LRU order is already
    // non-deterministic by design (access-pattern-driven), so hash
    // ordering inside this class does not introduce new determinism loss.
    using EntryList  = std::list<ResidencyEntry>;
    using EntryIter  = EntryList::iterator;

    // Mutex ordering: never acquire any other lock while holding this one.
    // ResidencyManager does not call any external callbacks while locked,
    // so deadlock is impossible as long as future code preserves that.
    mutable std::mutex                      mutex_;
    EntryList                               order_;
    std::unordered_map<u32, EntryIter>      index_;
    u64                                     residentBytes_   = 0u;
    u64                                     capacityBytes_   = 0u;
    u64                                     currentTick_     = 0u;
    u64                                     insertionCount_  = 0u;
    u64                                     evictionCount_   = 0u;
    u64                                     touchCount_      = 0u;
};

} // namespace enigma::renderer::micropoly
