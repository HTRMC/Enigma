// Unit test for AsyncIOWorker (M2.1 + M2.4b).
// Five cases:
//   1) Read page from DamagedHelmet .mpa and verify decompressed bytes
//      match MpAssetReader::fetchPage for the same page.
//   2) Enqueue + shutdown: shutdown drains in-flight work, further
//      enqueues return false, no hangs.
//   3) Bad fileOffset: worker surfaces success=false with a detail string.
//   4) M2.4b: 100 concurrent requests across all baked pages — every
//      completion's decompressed bytes round-trip against the reader's
//      fetchPage (order-independent).
//   5) M2.4b: pending() reports the correct in-flight + queued count
//      during a burst enqueue under IOCP load.
//
// Plain main, printf output, exit 0 on pass. Mirrors M1 test conventions.

#include "asset/MpAssetFormat.h"
#include "asset/MpAssetReader.h"
#include "renderer/micropoly/AsyncIOWorker.h"

#include "mpbake/ClusterBuilder.h"
#include "mpbake/DagBuilder.h"
#include "mpbake/GltfIngest.h"
#include "mpbake/PageWriter.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using enigma::u8;
using enigma::u32;
using enigma::asset::MpAssetReader;
using enigma::asset::MpPageEntry;
using enigma::mpbake::ClusterBuildOptions;
using enigma::mpbake::ClusterBuilder;
using enigma::mpbake::ClusterData;
using enigma::mpbake::DagBuildOptions;
using enigma::mpbake::DagBuilder;
using enigma::mpbake::GltfIngest;
using enigma::mpbake::PageWriter;
using enigma::mpbake::PageWriteOptions;
using enigma::renderer::micropoly::AsyncIOWorker;
using enigma::renderer::micropoly::AsyncIOWorkerOptions;
using enigma::renderer::micropoly::PageCompletion;
using enigma::renderer::micropoly::PageRequest;

namespace {

fs::path locateDamagedHelmet(const char* argv0) {
    std::error_code ec;
    fs::path start = argv0 ? fs::absolute(argv0, ec).parent_path() : fs::current_path(ec);
    if (ec) start = fs::current_path(ec);
    for (int i = 0; i < 6 && !start.empty(); ++i) {
        fs::path candidate = start / "assets" / "DamagedHelmet.glb";
        if (fs::exists(candidate, ec)) return candidate;
        if (start == start.parent_path()) break;
        start = start.parent_path();
    }
    fs::path cwd = fs::current_path(ec) / "assets" / "DamagedHelmet.glb";
    if (fs::exists(cwd, ec)) return cwd;
    return {};
}

fs::path tmpPath(const std::string& tag) {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec || base.empty()) return fs::path{"."} / ("async_io_worker_test_" + tag + ".mpa");
    return base / ("async_io_worker_test_" + tag + ".mpa");
}

bool bakeDamagedHelmet(const fs::path& asset, const fs::path& out) {
    GltfIngest ingest;
    auto ingestRes = ingest.load(asset);
    if (!ingestRes.has_value()) return false;
    ClusterBuilder cb;
    auto clusterRes = cb.build(*ingestRes, ClusterBuildOptions{});
    if (!clusterRes.has_value()) return false;
    DagBuilder db;
    auto dagRes = db.build(std::span<const ClusterData>(*clusterRes), DagBuildOptions{});
    if (!dagRes.has_value()) return false;
    PageWriter writer;
    auto wrRes = writer.write(*dagRes, out, PageWriteOptions{});
    return wrRes.has_value();
}

// Shared completion sink used by tests 1 + 3.
struct CompletionSink {
    std::mutex              mu;
    std::condition_variable cv;
    std::vector<PageCompletion> received;

    void push(PageCompletion c) {
        std::lock_guard<std::mutex> g(mu);
        received.push_back(std::move(c));
        cv.notify_all();
    }

    // Wait up to timeoutMs for `count` completions.
    bool waitFor(std::size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, timeout, [&]() {
            return received.size() >= count;
        });
    }
};

bool testReadPage(const fs::path& asset) {
    const fs::path mpa = tmpPath("readpage");
    std::error_code ec;
    fs::remove(mpa, ec);
    if (!bakeDamagedHelmet(asset, mpa)) {
        std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: bake\n");
        return false;
    }

    // Use MpAssetReader to pull both the expected decompressed bytes AND
    // the page-entry metadata we feed to the worker.
    MpAssetReader reader;
    auto openRes = reader.open(mpa);
    if (!openRes.has_value()) {
        std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: reader.open\n");
        fs::remove(mpa, ec);
        return false;
    }
    auto pageTbl = reader.pageTable();
    if (pageTbl.empty()) {
        std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: no pages in baked file\n");
        reader.close();
        fs::remove(mpa, ec);
        return false;
    }
    const MpPageEntry page0 = pageTbl[0];

    std::vector<u8> expected;
    {
        auto viewRes = reader.fetchPage(0u, expected);
        if (!viewRes.has_value()) {
            std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: fetchPage\n");
            reader.close();
            fs::remove(mpa, ec);
            return false;
        }
    }
    // Done with the reader — release the mmap BEFORE the worker opens the
    // same file so the two don't race on the handle.
    reader.close();

    CompletionSink sink;
    AsyncIOWorkerOptions opts;
    opts.mpaFilePath = mpa;
    opts.onComplete  = [&sink](PageCompletion c) { sink.push(std::move(c)); };

    {
        AsyncIOWorker worker(opts);
        PageRequest req;
        req.pageId           = 0u;
        req.fileOffset       = page0.payloadByteOffset;
        req.compressedSize   = page0.compressedSize;
        req.decompressedSize = page0.decompressedSize;
        if (!worker.enqueue(req)) {
            std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: enqueue returned false\n");
            fs::remove(mpa, ec);
            return false;
        }
        if (!sink.waitFor(1u, std::chrono::seconds(10))) {
            std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: timed out waiting for completion\n");
            fs::remove(mpa, ec);
            return false;
        }
    }  // worker destroyed here; joins cleanly.

    if (sink.received.size() != 1u) {
        std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: got %zu completions, expected 1\n",
                     sink.received.size());
        fs::remove(mpa, ec);
        return false;
    }
    const PageCompletion& c = sink.received[0];
    if (!c.success) {
        std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: completion.success=false: %s\n",
                     c.errorDetail.c_str());
        fs::remove(mpa, ec);
        return false;
    }
    if (c.pageId != 0u) {
        std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: pageId=%u\n", c.pageId);
        fs::remove(mpa, ec);
        return false;
    }
    if (c.decompressedData.size() != expected.size()) {
        std::fprintf(stderr,
            "[async_io_worker_test] case 1 FAIL: size mismatch worker=%zu reader=%zu\n",
            c.decompressedData.size(), expected.size());
        fs::remove(mpa, ec);
        return false;
    }
    if (std::memcmp(c.decompressedData.data(), expected.data(), expected.size()) != 0) {
        std::fprintf(stderr, "[async_io_worker_test] case 1 FAIL: byte mismatch\n");
        fs::remove(mpa, ec);
        return false;
    }

    fs::remove(mpa, ec);
    std::printf("[async_io_worker_test] case 1 PASS: worker output matches MpAssetReader::fetchPage (%zu bytes).\n",
                expected.size());
    return true;
}

bool testShutdown(const fs::path& asset) {
    const fs::path mpa = tmpPath("shutdown");
    std::error_code ec;
    fs::remove(mpa, ec);
    if (!bakeDamagedHelmet(asset, mpa)) {
        std::fprintf(stderr, "[async_io_worker_test] case 2 FAIL: bake\n");
        return false;
    }

    // Pull a real page entry so we enqueue a valid request.
    MpPageEntry page0{};
    {
        MpAssetReader reader;
        auto openRes = reader.open(mpa);
        if (!openRes.has_value() || reader.pageTable().empty()) {
            std::fprintf(stderr, "[async_io_worker_test] case 2 FAIL: reader open\n");
            fs::remove(mpa, ec);
            return false;
        }
        page0 = reader.pageTable()[0];
    }

    std::atomic<std::size_t> completed{0u};
    AsyncIOWorkerOptions opts;
    opts.mpaFilePath = mpa;
    opts.onComplete  = [&completed](PageCompletion c) {
        (void)c;
        completed.fetch_add(1u, std::memory_order_relaxed);
    };

    AsyncIOWorker worker(opts);
    // Enqueue a handful of valid requests all pointing at page 0.
    constexpr std::size_t kN = 4u;
    for (std::size_t i = 0; i < kN; ++i) {
        PageRequest req;
        req.pageId           = static_cast<u32>(i);
        req.fileOffset       = page0.payloadByteOffset;
        req.compressedSize   = page0.compressedSize;
        req.decompressedSize = page0.decompressedSize;
        if (!worker.enqueue(req)) {
            std::fprintf(stderr, "[async_io_worker_test] case 2 FAIL: enqueue %zu returned false\n", i);
            fs::remove(mpa, ec);
            return false;
        }
    }
    worker.shutdown();
    // Post-shutdown enqueue must fail.
    PageRequest late;
    late.pageId           = 999u;
    late.fileOffset       = page0.payloadByteOffset;
    late.compressedSize   = page0.compressedSize;
    late.decompressedSize = page0.decompressedSize;
    if (worker.enqueue(late)) {
        std::fprintf(stderr, "[async_io_worker_test] case 2 FAIL: enqueue succeeded after shutdown\n");
        fs::remove(mpa, ec);
        return false;
    }
    // All queued work should have drained (no-hang assertion): pending()==0.
    if (worker.pending() != 0u) {
        std::fprintf(stderr, "[async_io_worker_test] case 2 FAIL: pending()=%zu after shutdown\n",
                     worker.pending());
        fs::remove(mpa, ec);
        return false;
    }
    const std::size_t got = completed.load(std::memory_order_acquire);
    if (got != kN) {
        std::fprintf(stderr,
            "[async_io_worker_test] case 2 FAIL: got %zu completions, expected %zu\n",
            got, kN);
        fs::remove(mpa, ec);
        return false;
    }
    // Second shutdown is a no-op and must not crash.
    worker.shutdown();

    fs::remove(mpa, ec);
    std::printf(
        "[async_io_worker_test] case 2 PASS: shutdown drained %zu in-flight requests, rejected post-shutdown enqueue.\n",
        got);
    return true;
}

bool testBadOffset(const fs::path& asset) {
    const fs::path mpa = tmpPath("badoffset");
    std::error_code ec;
    fs::remove(mpa, ec);
    if (!bakeDamagedHelmet(asset, mpa)) {
        std::fprintf(stderr, "[async_io_worker_test] case 3 FAIL: bake\n");
        return false;
    }
    // Grab a real compressedSize so the worker's sanity checks pass, then
    // point fileOffset way past EOF so ReadFile fails.
    u32 compSize = 0u;
    u32 decompSize = 0u;
    {
        MpAssetReader reader;
        auto openRes = reader.open(mpa);
        if (!openRes.has_value() || reader.pageTable().empty()) {
            std::fprintf(stderr, "[async_io_worker_test] case 3 FAIL: reader open\n");
            fs::remove(mpa, ec);
            return false;
        }
        compSize   = reader.pageTable()[0].compressedSize;
        decompSize = reader.pageTable()[0].decompressedSize;
    }

    CompletionSink sink;
    AsyncIOWorkerOptions opts;
    opts.mpaFilePath = mpa;
    opts.onComplete  = [&sink](PageCompletion c) { sink.push(std::move(c)); };

    {
        AsyncIOWorker worker(opts);
        PageRequest req;
        req.pageId           = 777u;
        req.fileOffset       = 0xFFFFFFFF'FFFFFFFFull;  // obviously out of range
        req.compressedSize   = compSize;
        req.decompressedSize = decompSize;
        if (!worker.enqueue(req)) {
            std::fprintf(stderr, "[async_io_worker_test] case 3 FAIL: enqueue returned false\n");
            fs::remove(mpa, ec);
            return false;
        }
        if (!sink.waitFor(1u, std::chrono::seconds(5))) {
            std::fprintf(stderr, "[async_io_worker_test] case 3 FAIL: timed out waiting for completion\n");
            fs::remove(mpa, ec);
            return false;
        }
    }

    if (sink.received.size() != 1u) {
        std::fprintf(stderr, "[async_io_worker_test] case 3 FAIL: got %zu completions, expected 1\n",
                     sink.received.size());
        fs::remove(mpa, ec);
        return false;
    }
    const PageCompletion& c = sink.received[0];
    if (c.success) {
        std::fprintf(stderr, "[async_io_worker_test] case 3 FAIL: expected failure, got success\n");
        fs::remove(mpa, ec);
        return false;
    }
    if (c.errorDetail.empty()) {
        std::fprintf(stderr, "[async_io_worker_test] case 3 FAIL: errorDetail is empty\n");
        fs::remove(mpa, ec);
        return false;
    }
    if (c.pageId != 777u) {
        std::fprintf(stderr, "[async_io_worker_test] case 3 FAIL: pageId=%u\n", c.pageId);
        fs::remove(mpa, ec);
        return false;
    }

    fs::remove(mpa, ec);
    std::printf("[async_io_worker_test] case 3 PASS: bad offset surfaced as failure: %s\n",
                c.errorDetail.c_str());
    return true;
}

// Case 4 (M2.4b): issue kBurst requests that cycle through every baked page,
// wait for all completions, and verify every completion's bytes match the
// reader's fetchPage for the corresponding pageId. Exercises the IOCP
// in-flight pool saturation path.
bool testConcurrentRequests(const fs::path& asset) {
    const fs::path mpa = tmpPath("concurrent");
    std::error_code ec;
    fs::remove(mpa, ec);
    if (!bakeDamagedHelmet(asset, mpa)) {
        std::fprintf(stderr, "[async_io_worker_test] case 4 FAIL: bake\n");
        return false;
    }

    MpAssetReader reader;
    auto openRes = reader.open(mpa);
    if (!openRes.has_value() || reader.pageTable().empty()) {
        std::fprintf(stderr, "[async_io_worker_test] case 4 FAIL: reader.open\n");
        fs::remove(mpa, ec);
        return false;
    }
    // Copy the page table out of the reader BEFORE close() — pageTable()
    // returns a span into the mmapped region which is invalidated by close().
    const auto pageTblSpan = reader.pageTable();
    std::vector<MpPageEntry> pageTbl(pageTblSpan.begin(), pageTblSpan.end());
    const std::size_t nPages = pageTbl.size();

    // Snapshot the expected decompressed bytes for each page so we can
    // cross-check every completion independent of arrival order.
    std::vector<std::vector<u8>> expectedByPage(nPages);
    for (std::size_t i = 0; i < nPages; ++i) {
        auto viewRes = reader.fetchPage(static_cast<u32>(i), expectedByPage[i]);
        if (!viewRes.has_value()) {
            std::fprintf(stderr, "[async_io_worker_test] case 4 FAIL: fetchPage(%zu)\n", i);
            reader.close();
            fs::remove(mpa, ec);
            return false;
        }
    }
    reader.close();

    constexpr std::size_t kBurst = 100u;

    std::mutex sinkMu;
    std::condition_variable sinkCv;
    std::vector<PageCompletion> received;
    received.reserve(kBurst);

    AsyncIOWorkerOptions opts;
    opts.mpaFilePath         = mpa;
    opts.maxInflightRequests = kBurst;  // permit the full burst to queue
    opts.onComplete = [&](PageCompletion c) {
        std::lock_guard<std::mutex> g(sinkMu);
        received.push_back(std::move(c));
        sinkCv.notify_all();
    };

    {
        AsyncIOWorker worker(opts);
        // Tight enqueue loop — may briefly hit capacity; back off and retry.
        std::size_t enqueued = 0;
        while (enqueued < kBurst) {
            const u32 idx = static_cast<u32>(enqueued % nPages);
            PageRequest req;
            req.pageId           = idx;
            req.fileOffset       = pageTbl[idx].payloadByteOffset;
            req.compressedSize   = pageTbl[idx].compressedSize;
            req.decompressedSize = pageTbl[idx].decompressedSize;
            if (!worker.enqueue(req)) {
                std::this_thread::yield();
                continue;
            }
            ++enqueued;
        }
        // Wait for all completions.
        std::unique_lock<std::mutex> lk(sinkMu);
        const bool allIn = sinkCv.wait_for(lk, std::chrono::seconds(30),
            [&]() { return received.size() >= kBurst; });
        if (!allIn) {
            std::fprintf(stderr,
                "[async_io_worker_test] case 4 FAIL: only %zu/%zu completions in 30s\n",
                received.size(), kBurst);
            fs::remove(mpa, ec);
            return false;
        }
    }  // worker joins here

    // Verify every completion is a successful match for its pageId.
    std::size_t bytesVerified = 0u;
    for (const auto& c : received) {
        if (!c.success) {
            std::fprintf(stderr, "[async_io_worker_test] case 4 FAIL: success=false pageId=%u: %s\n",
                         c.pageId, c.errorDetail.c_str());
            fs::remove(mpa, ec);
            return false;
        }
        if (c.pageId >= nPages) {
            std::fprintf(stderr, "[async_io_worker_test] case 4 FAIL: pageId=%u out of range\n",
                         c.pageId);
            fs::remove(mpa, ec);
            return false;
        }
        const auto& expected = expectedByPage[c.pageId];
        if (c.decompressedData.size() != expected.size() ||
            std::memcmp(c.decompressedData.data(), expected.data(), expected.size()) != 0) {
            std::fprintf(stderr,
                "[async_io_worker_test] case 4 FAIL: bytes mismatch for pageId=%u\n",
                c.pageId);
            fs::remove(mpa, ec);
            return false;
        }
        bytesVerified += expected.size();
    }

    fs::remove(mpa, ec);
    std::printf(
        "[async_io_worker_test] case 4 PASS: %zu concurrent requests completed, %zu bytes verified.\n",
        kBurst, bytesVerified);
    return true;
}

// Case 5 (M2.4b): verify pending() reports queue+in-flight correctly under
// burst load. Before any completion, pending() must equal the number of
// successful enqueues. After shutdown drains, pending() must be zero.
bool testPendingUnderLoad(const fs::path& asset) {
    const fs::path mpa = tmpPath("pending");
    std::error_code ec;
    fs::remove(mpa, ec);
    if (!bakeDamagedHelmet(asset, mpa)) {
        std::fprintf(stderr, "[async_io_worker_test] case 5 FAIL: bake\n");
        return false;
    }

    MpPageEntry page0{};
    {
        MpAssetReader reader;
        auto openRes = reader.open(mpa);
        if (!openRes.has_value() || reader.pageTable().empty()) {
            std::fprintf(stderr, "[async_io_worker_test] case 5 FAIL: reader open\n");
            fs::remove(mpa, ec);
            return false;
        }
        page0 = reader.pageTable()[0];
    }

    constexpr std::size_t kCap = 32u;

    std::atomic<std::size_t> completed{0u};
    AsyncIOWorkerOptions opts;
    opts.mpaFilePath         = mpa;
    opts.maxInflightRequests = kCap;
    opts.onComplete = [&completed](PageCompletion c) {
        (void)c;
        completed.fetch_add(1u, std::memory_order_relaxed);
    };

    AsyncIOWorker worker(opts);

    // Issue kCap requests as fast as possible. Each enqueue() bumps the
    // queue before the worker has a chance to dequeue; pending() must
    // observe a count >= successfully-enqueued - already-completed.
    std::size_t enq = 0;
    for (std::size_t i = 0; i < kCap; ++i) {
        PageRequest req;
        req.pageId           = static_cast<u32>(i);
        req.fileOffset       = page0.payloadByteOffset;
        req.compressedSize   = page0.compressedSize;
        req.decompressedSize = page0.decompressedSize;
        if (worker.enqueue(req)) ++enq;
        else break;  // hit cap early
    }
    if (enq == 0u) {
        std::fprintf(stderr, "[async_io_worker_test] case 5 FAIL: no enqueues accepted\n");
        fs::remove(mpa, ec);
        return false;
    }

    // Sample pending(): must be <= enq (some may have completed already)
    // and >= enq - completed (the rest are in flight + queued).
    const std::size_t snapPending   = worker.pending();
    const std::size_t snapCompleted = completed.load(std::memory_order_acquire);
    if (snapPending > enq) {
        std::fprintf(stderr,
            "[async_io_worker_test] case 5 FAIL: pending()=%zu > enq=%zu\n",
            snapPending, enq);
        fs::remove(mpa, ec);
        return false;
    }
    if (snapPending + snapCompleted < enq) {
        std::fprintf(stderr,
            "[async_io_worker_test] case 5 FAIL: pending=%zu + completed=%zu < enq=%zu (lost a request)\n",
            snapPending, snapCompleted, enq);
        fs::remove(mpa, ec);
        return false;
    }

    worker.shutdown();

    if (worker.pending() != 0u) {
        std::fprintf(stderr, "[async_io_worker_test] case 5 FAIL: pending()=%zu after shutdown\n",
                     worker.pending());
        fs::remove(mpa, ec);
        return false;
    }
    if (completed.load(std::memory_order_acquire) != enq) {
        std::fprintf(stderr,
            "[async_io_worker_test] case 5 FAIL: completed=%zu but enqueued=%zu\n",
            completed.load(std::memory_order_acquire), enq);
        fs::remove(mpa, ec);
        return false;
    }

    fs::remove(mpa, ec);
    std::printf(
        "[async_io_worker_test] case 5 PASS: pending() tracked %zu enqueued requests under load; final pending()==0.\n",
        enq);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr,
            "[async_io_worker_test] FAIL: could not locate assets/DamagedHelmet.glb\n");
        return 1;
    }
    bool ok = true;
    ok &= testReadPage(asset);
    ok &= testShutdown(asset);
    ok &= testBadOffset(asset);
    ok &= testConcurrentRequests(asset);
    ok &= testPendingUnderLoad(asset);
    if (!ok) {
        std::fprintf(stderr, "[async_io_worker_test] FAILED\n");
        return 1;
    }
    std::printf("[async_io_worker_test] All cases passed.\n");
    return 0;
}
