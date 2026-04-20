// Unit test for ResidencyManager (M2.1).
// Four cases:
//   1) Basic insertion + hit: insert, then insert-again -> wasAlreadyResident.
//   2) LRU eviction: 1 MB capacity, 5 x 300 KB pages -> 2 evictions.
//   3) beginFrame + touch: touching a page keeps it alive under pressure.
//   4) Stats sanity: cumulative counters match an explicit scenario.
//
// Plain main, printf output, exit 0 on pass. Mirrors M1 test conventions.

#include "renderer/micropoly/ResidencyManager.h"

#include "core/Types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using enigma::u32;
using enigma::renderer::micropoly::ResidencyEvent;
using enigma::renderer::micropoly::ResidencyEventKind;
using enigma::renderer::micropoly::ResidencyManager;
using enigma::renderer::micropoly::ResidencyManagerOptions;

namespace {

// Count events of a given kind in an InsertResult.
std::size_t countKind(const std::vector<ResidencyEvent>& events,
                      ResidencyEventKind kind) {
    std::size_t c = 0u;
    for (const auto& e : events) {
        if (e.kind == kind) ++c;
    }
    return c;
}

const char* kindStr(ResidencyEventKind k) {
    switch (k) {
        case ResidencyEventKind::Inserted: return "Inserted";
        case ResidencyEventKind::Evicted:  return "Evicted";
        case ResidencyEventKind::Touched:  return "Touched";
    }
    return "?";
}

bool testBasic() {
    ResidencyManager mgr(ResidencyManagerOptions{});
    auto r1 = mgr.insert(42u, 1024u);
    if (r1.wasAlreadyResident) {
        std::fprintf(stderr, "[residency_lru_test] case 1 FAIL: wasAlreadyResident=true on first insert\n");
        return false;
    }
    if (countKind(r1.events, ResidencyEventKind::Inserted) != 1u ||
        r1.events.empty() ||
        r1.events.back().kind != ResidencyEventKind::Inserted) {
        std::fprintf(stderr, "[residency_lru_test] case 1 FAIL: expected single Inserted event, got %zu events\n",
                     r1.events.size());
        return false;
    }
    if (!mgr.isResident(42u)) {
        std::fprintf(stderr, "[residency_lru_test] case 1 FAIL: isResident false after insert\n");
        return false;
    }
    auto r2 = mgr.insert(42u, 1024u);
    if (!r2.wasAlreadyResident) {
        std::fprintf(stderr, "[residency_lru_test] case 1 FAIL: wasAlreadyResident=false on re-insert\n");
        return false;
    }
    if (countKind(r2.events, ResidencyEventKind::Touched) != 1u) {
        std::fprintf(stderr, "[residency_lru_test] case 1 FAIL: expected Touched event on re-insert\n");
        return false;
    }
    std::printf("[residency_lru_test] case 1 PASS: basic insert + hit.\n");
    return true;
}

bool testLruEviction() {
    // 1 MB capacity. 5 x 300 KB -> 1500 KB requested, eviction starts once
    // residentBytes_ + newSize > 1024*1024 = 1048576 bytes.
    ResidencyManagerOptions opts;
    opts.capacityBytes = 1u * 1024u * 1024u;
    ResidencyManager mgr(opts);

    constexpr u32 kPageSize = 300u * 1024u;  // 300 KB
    std::size_t totalEvictions = 0u;
    for (u32 i = 0; i < 5u; ++i) {
        auto r = mgr.insert(i, kPageSize);
        totalEvictions += countKind(r.events, ResidencyEventKind::Evicted);
        // Eviction events come before the Inserted event; verify ordering.
        bool sawInserted = false;
        for (const auto& e : r.events) {
            if (e.kind == ResidencyEventKind::Inserted) sawInserted = true;
            else if (sawInserted && e.kind == ResidencyEventKind::Evicted) {
                std::fprintf(stderr,
                    "[residency_lru_test] case 2 FAIL: Evicted event came after Inserted in page %u\n", i);
                return false;
            }
        }
    }

    // 5 pages of 300 KB = 1500 KB requested. Capacity 1024 KB.
    // After page 3 (1200 KB resident), page 4 triggers eviction of page 0
    // (900 KB resident -> fits another 300 KB in 1024 KB cap).
    // After page 4 (1200 KB resident), page 5 triggers eviction of page 1.
    // Total evictions = 2.
    if (totalEvictions != 2u) {
        std::fprintf(stderr,
            "[residency_lru_test] case 2 FAIL: expected 2 evictions, got %zu\n",
            totalEvictions);
        return false;
    }

    // Pages 0 + 1 should be the victims (oldest first).
    if (mgr.isResident(0u)) {
        std::fprintf(stderr, "[residency_lru_test] case 2 FAIL: page 0 still resident (expected evicted)\n");
        return false;
    }
    if (mgr.isResident(1u)) {
        std::fprintf(stderr, "[residency_lru_test] case 2 FAIL: page 1 still resident (expected evicted)\n");
        return false;
    }
    for (u32 i = 2u; i < 5u; ++i) {
        if (!mgr.isResident(i)) {
            std::fprintf(stderr, "[residency_lru_test] case 2 FAIL: page %u evicted (expected resident)\n", i);
            return false;
        }
    }
    std::printf("[residency_lru_test] case 2 PASS: LRU evicted 2 oldest pages under pressure.\n");
    return true;
}

bool testBeginFrameTouch() {
    // Scenario: capacity fits exactly 2 x 300 KB pages + a hair, so the
    // third insert must evict. If we touch page 1 between page 2 and page 3
    // inserts, page 2 (the older untouched entry) should be evicted.
    ResidencyManagerOptions opts;
    opts.capacityBytes = 700u * 1024u;  // fits 2x 300 KB, not 3x
    ResidencyManager mgr(opts);

    constexpr u32 kPageSize = 300u * 1024u;
    mgr.insert(1u, kPageSize);
    mgr.beginFrame();
    mgr.insert(2u, kPageSize);
    // Touch page 1 -> makes it most-recently-used.
    auto touchRes = mgr.insert(1u, kPageSize);
    if (!touchRes.wasAlreadyResident ||
        countKind(touchRes.events, ResidencyEventKind::Touched) != 1u) {
        std::fprintf(stderr, "[residency_lru_test] case 3 FAIL: expected Touched on re-insert of page 1\n");
        return false;
    }
    // Insert page 3 -> should evict page 2 (oldest now), keeping page 1.
    auto r3 = mgr.insert(3u, kPageSize);
    if (countKind(r3.events, ResidencyEventKind::Evicted) != 1u) {
        std::fprintf(stderr,
            "[residency_lru_test] case 3 FAIL: expected 1 eviction inserting page 3, got %zu events\n",
            r3.events.size());
        for (const auto& e : r3.events) {
            std::fprintf(stderr, "    -> %s page=%u\n", kindStr(e.kind), e.pageId);
        }
        return false;
    }
    if (!mgr.isResident(1u)) {
        std::fprintf(stderr, "[residency_lru_test] case 3 FAIL: page 1 not resident (touched but evicted)\n");
        return false;
    }
    if (mgr.isResident(2u)) {
        std::fprintf(stderr, "[residency_lru_test] case 3 FAIL: page 2 still resident (should be evicted)\n");
        return false;
    }
    if (!mgr.isResident(3u)) {
        std::fprintf(stderr, "[residency_lru_test] case 3 FAIL: page 3 not resident\n");
        return false;
    }
    std::printf("[residency_lru_test] case 3 PASS: beginFrame + touch kept page 1 alive through eviction.\n");
    return true;
}

bool testStatsSanity() {
    ResidencyManagerOptions opts;
    opts.capacityBytes = 1024u;  // tight
    ResidencyManager mgr(opts);

    // Insert 3 x 400-byte pages. 3rd must evict page 0.
    mgr.insert(10u, 400u);
    mgr.insert(11u, 400u);
    auto r = mgr.insert(12u, 400u);  // 1200 > 1024 -> evict page 10.
    if (countKind(r.events, ResidencyEventKind::Evicted) != 1u) {
        std::fprintf(stderr, "[residency_lru_test] case 4 FAIL: expected 1 eviction\n");
        return false;
    }
    // Touch page 11.
    mgr.insert(11u, 400u);

    const auto s = mgr.stats();
    if (s.capacityBytes != 1024u) {
        std::fprintf(stderr, "[residency_lru_test] case 4 FAIL: capacityBytes=%llu\n",
                     static_cast<unsigned long long>(s.capacityBytes));
        return false;
    }
    if (s.insertionCount != 3u) {
        std::fprintf(stderr, "[residency_lru_test] case 4 FAIL: insertionCount=%llu (want 3)\n",
                     static_cast<unsigned long long>(s.insertionCount));
        return false;
    }
    if (s.evictionCount != 1u) {
        std::fprintf(stderr, "[residency_lru_test] case 4 FAIL: evictionCount=%llu (want 1)\n",
                     static_cast<unsigned long long>(s.evictionCount));
        return false;
    }
    if (s.touchCount != 1u) {
        std::fprintf(stderr, "[residency_lru_test] case 4 FAIL: touchCount=%llu (want 1)\n",
                     static_cast<unsigned long long>(s.touchCount));
        return false;
    }
    if (s.residentPageCount != 2u) {
        std::fprintf(stderr, "[residency_lru_test] case 4 FAIL: residentPageCount=%u (want 2)\n",
                     s.residentPageCount);
        return false;
    }
    if (s.residentBytes != 800u) {
        std::fprintf(stderr, "[residency_lru_test] case 4 FAIL: residentBytes=%llu (want 800)\n",
                     static_cast<unsigned long long>(s.residentBytes));
        return false;
    }
    std::printf("[residency_lru_test] case 4 PASS: stats counters match expected scenario.\n");
    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok &= testBasic();
    ok &= testLruEviction();
    ok &= testBeginFrameTouch();
    ok &= testStatsSanity();
    if (!ok) {
        std::fprintf(stderr, "[residency_lru_test] FAILED\n");
        return 1;
    }
    std::printf("[residency_lru_test] All cases passed.\n");
    return 0;
}
