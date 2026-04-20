// ResidencyManager.cpp
// =====================
// Implementation of the CPU-side LRU residency tracker. See header for the
// contract; this TU contains no Vulkan code.

#include "renderer/micropoly/ResidencyManager.h"

#include <algorithm>
#include <utility>

namespace enigma::renderer::micropoly {

ResidencyManager::ResidencyManager(const ResidencyManagerOptions& opts)
    : capacityBytes_(opts.capacityBytes) {
    // No up-front allocation: the std::list grows on demand, and the
    // unordered_map rehashes lazily. Reserving would prevent the first
    // few pages from fragmenting a separate allocation but also burn
    // memory on clients that never fill the cache; leave it lazy.
}

ResidencyManager::InsertResult
ResidencyManager::insert(u32 pageId, u32 sizeBytes) {
    InsertResult out{};
    std::lock_guard<std::mutex> guard(mutex_);

    // Fast path: already resident. Splice to the back of the list to
    // mark as most-recently-used and emit a Touched event.
    auto it = index_.find(pageId);
    if (it != index_.end()) {
        EntryIter entryIt = it->second;
        entryIt->lastUsedTick = currentTick_;
        // Splice is O(1) — transfers the node into place without
        // invalidating the iterator stored in index_.
        order_.splice(order_.end(), order_, entryIt);
        out.wasAlreadyResident = true;
        out.events.push_back(ResidencyEvent{
            ResidencyEventKind::Touched, pageId,
        });
        ++touchCount_;
        return out;
    }

    // Slow path: we need to insert a new entry. Evict from the front of
    // the list until the new entry fits, up to the per-call eviction cap.
    // The cap guards against a pathological insert (e.g. a single page
    // larger than the whole cache) driving unbounded eviction work.
    const u64 newSize64 = static_cast<u64>(sizeBytes);
    std::size_t evictionsThisCall = 0u;
    while (residentBytes_ + newSize64 > capacityBytes_ &&
           !order_.empty() &&
           evictionsThisCall < kMaxEvictionsPerInsert) {
        // Pop oldest: front of the list.
        ResidencyEntry victim = order_.front();
        index_.erase(victim.pageId);
        order_.pop_front();
        residentBytes_ -= victim.sizeBytes;
        ++evictionCount_;
        ++evictionsThisCall;
        out.events.push_back(ResidencyEvent{
            ResidencyEventKind::Evicted, victim.pageId,
        });
    }

    // Append the new entry regardless of whether we fully achieved
    // capacity — if the new page is itself larger than the cache, the
    // caller is already in trouble and the eviction cap prevented us
    // from spinning. We still honor the insert so subsequent calls can
    // kick it out in turn.
    ResidencyEntry entry{};
    entry.pageId       = pageId;
    entry.lastUsedTick = currentTick_;
    entry.sizeBytes    = sizeBytes;
    order_.push_back(entry);
    // The iterator to the just-inserted back-element is std::prev(end()).
    auto insertedIt = std::prev(order_.end());
    index_.emplace(pageId, insertedIt);
    residentBytes_ += newSize64;
    ++insertionCount_;
    out.events.push_back(ResidencyEvent{
        ResidencyEventKind::Inserted, pageId,
    });
    return out;
}

bool ResidencyManager::isResident(u32 pageId) const {
    std::lock_guard<std::mutex> guard(mutex_);
    return index_.find(pageId) != index_.end();
}

void ResidencyManager::beginFrame() {
    std::lock_guard<std::mutex> guard(mutex_);
    ++currentTick_;
}

ResidencyManager::Stats ResidencyManager::stats() const {
    std::lock_guard<std::mutex> guard(mutex_);
    Stats s{};
    s.residentBytes     = residentBytes_;
    s.capacityBytes     = capacityBytes_;
    s.insertionCount    = insertionCount_;
    s.evictionCount     = evictionCount_;
    s.touchCount        = touchCount_;
    s.residentPageCount = static_cast<u32>(order_.size());
    return s;
}

} // namespace enigma::renderer::micropoly
