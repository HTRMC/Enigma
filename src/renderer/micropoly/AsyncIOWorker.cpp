// AsyncIOWorker.cpp
// ==================
// Background page-reader thread. See header for the contract.
//
// Platform notes
// --------------
// Windows (M2.4b): CreateFileW(FILE_FLAG_OVERLAPPED) + ReadFile + IOCP. The
//          worker thread keeps up to kMaxInFlightPerWorker reads outstanding
//          simultaneously; completions are drained via
//          GetQueuedCompletionStatus on the same worker thread so callbacks
//          continue to fire from the worker (public-API invariant). A
//          sentinel PostQueuedCompletionStatus wake is used to unblock the
//          loop when new requests are enqueued.
// Non-Windows: POSIX pread() — unchanged from M2.1. The engine is Windows-
//              only per the .omc plan; Linux/macOS compile only exists for
//              CI hygiene and can keep the simple synchronous path.

#include "renderer/micropoly/AsyncIOWorker.h"

#include "asset/MpAssetFormat.h"
#include "asset/MpPathUtils.h"
#include "core/Assert.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

#include <zstd.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace enigma::renderer::micropoly {

namespace {

// Keep IO sizes bounded. The plan caps per-page decompressed size at 64 MiB
// (see MpAssetFormat.h :: kMpMaxPageDecompressedBytes). Compressed pages are
// well under this, but the worker is defensive: requests above this bound
// are rejected instead of driving a multi-GB allocation. Linked to the
// single-source-of-truth cap at the reader level per M2.3 Security MEDIUM-2.
constexpr std::size_t kMaxReasonableCompressedBytes =
    static_cast<std::size_t>(enigma::asset::kMpMaxPageDecompressedBytes);
static_assert(kMaxReasonableCompressedBytes ==
              static_cast<std::size_t>(enigma::asset::kMpMaxPageDecompressedBytes),
              "AsyncIOWorker's size cap must match the MpAssetFormat.h cap so "
              "both layers share a single source of truth.");

#if defined(_WIN32)
// Sentinel completion key distinguishing wake posts from real IO completions.
// Real IOCP IO packets use CompletionKey=0 (set at CreateIoCompletionPort).
// The 0xE17A1A10 mnemonic spells "ENIGMA" (approximately) in hex and is
// never a valid kernel completion key.
constexpr ULONG_PTR kWakeupCompletionKey = 0xE17A1A10u;
#endif

// M3.0 prereq E — defence-in-depth path validation. Renderer::ctor already
// guards MicropolyConfig::mpaFilePath with an identical check, but future
// tests or editor tooling might construct an AsyncIOWorker directly with an
// untrusted path. We share the rule-set with the renderer-side check via
// asset/MpPathUtils.h (M3.2 closeout fix #1) so the two sites can never drift.
//
// The shared helper additionally rejects NT device namespace prefixes
// (\??\) and paths longer than the Windows long-path limit (32k wchars).

} // namespace

#if defined(_WIN32)
// Per-read tracking block. The OVERLAPPED MUST live at offset 0 so we can
// reinterpret-cast between the two via pointer math when GQCS hands us back
// the OVERLAPPED*. We heap-allocate these (unique_ptr) so their addresses
// stay stable while the kernel has a pointer to the embedded OVERLAPPED.
struct AsyncIOWorker::InFlightRead {
    OVERLAPPED      overlapped{};
    PageRequest     request{};
    std::vector<u8> compressedBuf;
};
static_assert(offsetof(AsyncIOWorker::InFlightRead, overlapped) == 0,
    "OVERLAPPED must be at offset 0 for reinterpret_cast round-trip via GQCS");
#endif

AsyncIOWorker::AsyncIOWorker(AsyncIOWorkerOptions opts)
    : opts_(std::move(opts)) {
    thread_ = std::thread([this]() { this->run_(); });
}

AsyncIOWorker::~AsyncIOWorker() {
    shutdown();
}

bool AsyncIOWorker::enqueue(const PageRequest& req) {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (stop_) {
            return false;
        }
        // +inFlight_ because a request currently being processed still
        // occupies a slot from the caller's point of view.
        if (queue_.size() + inFlight_ >= opts_.maxInflightRequests) {
            return false;
        }
        queue_.push_back(req);
    }
    cv_.notify_one();
#if defined(_WIN32)
    // Also nudge the IOCP so a worker blocked on GetQueuedCompletionStatus
    // wakes up and picks up the new request. Atomic acquire load pairs with
    // the release store in run_() so the handle is visible before use.
    if (void* h = iocpHandle_.load(std::memory_order_acquire); h != nullptr) {
        PostQueuedCompletionStatus(static_cast<HANDLE>(h),
                                   0, kWakeupCompletionKey, nullptr);
    }
#endif
    return true;
}

std::size_t AsyncIOWorker::pending() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return queue_.size() + inFlight_;
}

void AsyncIOWorker::shutdown() {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (joined_) return;
        stop_ = true;
    }
    cv_.notify_all();
#if defined(_WIN32)
    // Kick the IOCP so the worker thread, if currently blocked in GQCS with
    // no in-flight reads, wakes up and notices stop_.
    if (void* h = iocpHandle_.load(std::memory_order_acquire); h != nullptr) {
        PostQueuedCompletionStatus(static_cast<HANDLE>(h),
                                   0, kWakeupCompletionKey, nullptr);
    }

    // M3.0 prereq B — abnormal-termination safety net. Normal shutdown is
    // the sentinel-wake + join path above. If that doesn't complete within
    // a bounded wall-clock window (e.g. kernel ReadFile wedged on a flaky
    // FS or a hot-unplugged NVMe), escalate: CancelIoEx + bounded GQCS
    // drain. Mirrors the pattern in tests/throughput_bench.cpp.
    //
    // The normal path does NOT go through this code. The 5-second gate is
    // deliberately long enough that healthy shutdowns with hundreds of
    // in-flight reads finish cleanly; abnormal stalls hit the CancelIoEx
    // fallback and retire the thread instead of hanging the Renderer dtor.
    if (thread_.joinable()) {
        constexpr auto kNormalJoinBudget = std::chrono::seconds(5);
        const auto joinDeadline = std::chrono::steady_clock::now() + kNormalJoinBudget;

        // Poll the worker thread's exit state. std::thread lacks a timed
        // join so we spin on pending() as a proxy: if the worker is healthy
        // it drains + exits; if it's wedged the count stays above zero.
        while (std::chrono::steady_clock::now() < joinDeadline) {
            {
                std::lock_guard<std::mutex> guard(mutex_);
                if (queue_.empty() && inFlight_ == 0u) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        const bool stillBusy = [this]() {
            std::lock_guard<std::mutex> guard(mutex_);
            return !queue_.empty() || inFlight_ != 0u;
        }();

        if (stillBusy) {
            // Abnormal termination path. Cancel any outstanding kernel I/O
            // on this file handle and let the worker's IOCP loop see the
            // completions (as ERROR_OPERATION_ABORTED) so inFlight_ drains
            // and the loop exits via the normal stop_ && empty condition.
            if (osFileHandle_ != nullptr) {
                CancelIoEx(static_cast<HANDLE>(osFileHandle_), nullptr);
            }
            // Re-post the wakeup sentinel so the loop ticks even if no
            // completion arrives in the cancel grace window.
            if (void* h = iocpHandle_.load(std::memory_order_acquire); h != nullptr) {
                PostQueuedCompletionStatus(static_cast<HANDLE>(h),
                                           0, kWakeupCompletionKey, nullptr);
            }
        }

        thread_.join();
    }
#else
    if (thread_.joinable()) {
        thread_.join();
    }
#endif
    std::lock_guard<std::mutex> guard(mutex_);
    joined_ = true;
}

void AsyncIOWorker::run_() {
    // M3.0 prereq E — belt-and-braces path validation inside the worker
    // thread. The Renderer ctor also checks this (with an identical rule),
    // but a direct AsyncIOWorker construction (tests, editor tooling) may
    // bypass that guard. Fail loud and early instead of issuing syscalls
    // against a potentially attacker-controlled path.
    {
        std::string detail;
        if (!enigma::asset::isSafeMpaPath(opts_.mpaFilePath, detail)) {
            std::lock_guard<std::mutex> guard(mutex_);
            osOpenFailed_ = true;
            osOpenError_  = std::string{"AsyncIOWorker: unsafe mpaFilePath: "} + detail;
            // Drain any pending requests as synthetic failures so the caller
            // sees a completion per enqueue rather than waiting forever.
#if defined(_WIN32)
            // Ensure the iocp handle is observable as nullptr so no wakeup
            // sentinel path tries to post to a valid HANDLE.
            iocpHandle_.store(nullptr, std::memory_order_release);
#endif
        }
    }

#if defined(_WIN32)
    if (osOpenFailed_) {
        // Fall through to the POSIX-style drain behavior: pump the queue,
        // fail every request, exit when stop_ && queue_.empty(). Keeps the
        // destructor's join path short.
        for (;;) {
            PageRequest req{};
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                if (queue_.empty()) continue;
                req = queue_.front();
                queue_.pop_front();
                ++inFlight_;
            }
            PageCompletion c{};
            c.pageId      = req.pageId;
            c.success     = false;
            c.errorDetail = osOpenError_;
            {
                std::lock_guard<std::mutex> guard(mutex_);
                --inFlight_;
            }
            cv_.notify_all();
            if (opts_.onComplete) opts_.onComplete(std::move(c));
        }
    }

    // Open the file once for the worker's lifetime. FILE_SHARE_READ so
    // other readers (including a parallel MpAssetReader mmap) can coexist.
    // FILE_FLAG_OVERLAPPED is the required door to IOCP + parallel reads.
    HANDLE hFile = CreateFileW(
        opts_.mpaFilePath.wstring().c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::lock_guard<std::mutex> guard(mutex_);
        osOpenFailed_ = true;
        osOpenError_  = std::string{"CreateFileW failed for "}
                      + opts_.mpaFilePath.string();
    } else {
        osFileHandle_ = hFile;
        // Create a fresh IOCP and associate the file handle with it. We
        // pass CompletionKey=0 here — the per-read InFlightRead* is keyed
        // via the OVERLAPPED* instead. A non-zero key is reserved for the
        // sentinel wakeup from enqueue()/shutdown().
        HANDLE hIocp = CreateIoCompletionPort(hFile, nullptr, 0, 1);
        if (hIocp == nullptr) {
            std::lock_guard<std::mutex> guard(mutex_);
            osOpenFailed_ = true;
            osOpenError_  = std::string{"CreateIoCompletionPort failed, GLE="}
                          + std::to_string(GetLastError());
            CloseHandle(hFile);
            osFileHandle_ = nullptr;
        } else {
            iocpHandle_.store(hIocp, std::memory_order_release);
        }
    }

    // The IOCP worker loop. Invariants:
    //   - Reads in progress: inFlightReads_.size() (heap-owned); each has
    //     its OVERLAPPED registered with the kernel.
    //   - inFlight_ (from mutex_) mirrors inFlightReads_.size() + "being
    //     handled after GQCS returned" so pending() stays consistent.
    //   - We drain as many completions as the kernel hands back per tick
    //     (tight GQCS loop with 0 timeout), then refill the in-flight
    //     pool, then block on GQCS with a small timeout so we sleep when
    //     idle and wake fast when completions arrive or enqueue() posts a
    //     wakeup sentinel.
    run_iocp_loop_();

    // Cleanup. Worker thread is the only writer; plain load is sufficient
    // here since the thread is about to exit, but we use relaxed for clarity.
    if (void* h = iocpHandle_.load(std::memory_order_relaxed); h != nullptr) {
        CloseHandle(static_cast<HANDLE>(h));
        iocpHandle_.store(nullptr, std::memory_order_relaxed);
    }
    if (osFileHandle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(osFileHandle_));
        osFileHandle_ = nullptr;
    }
#else
    const int fd = ::open(opts_.mpaFilePath.string().c_str(), O_RDONLY);
    if (fd < 0) {
        std::lock_guard<std::mutex> guard(mutex_);
        osOpenFailed_ = true;
        osOpenError_  = std::string{"open() failed for "}
                      + opts_.mpaFilePath.string();
    } else {
        // Shove the fd through a void* so header stays platform-agnostic.
        osFileHandle_ = reinterpret_cast<void*>(static_cast<std::intptr_t>(fd));
    }

    // POSIX fallback: synchronous pread loop, same as M2.1. The engine is
    // Windows-only; this exists for CI compile.
    std::vector<u8> compressedScratch;
    for (;;) {
        PageRequest req{};
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return stop_ || !queue_.empty();
            });
            if (stop_ && queue_.empty()) {
                break;
            }
            req = queue_.front();
            queue_.pop_front();
            ++inFlight_;
        }

        PageCompletion completion = handle_sync_(req, compressedScratch);

        {
            std::lock_guard<std::mutex> guard(mutex_);
            --inFlight_;
        }
        cv_.notify_all();

        if (opts_.onComplete) {
            opts_.onComplete(std::move(completion));
        }
    }

    if (osFileHandle_ != nullptr) {
        ::close(static_cast<int>(reinterpret_cast<std::intptr_t>(osFileHandle_)));
        osFileHandle_ = nullptr;
    }
#endif
}

#if defined(_WIN32)

void AsyncIOWorker::run_iocp_loop_() {
    // Cap concurrent reads at the caller's maxInflightRequests. 64 is the
    // streaming subsystem's default; 32+ is enough to saturate NVMe queue
    // depth on consumer hardware.
    const std::size_t kMaxInFlightPerWorker = opts_.maxInflightRequests;

    // M3.0 prereq D — pre-allocate the full InFlightRead pool up front. The
    // earlier implementation called std::make_unique<InFlightRead>() on the
    // hot path; under sustained memory pressure that throws std::bad_alloc,
    // the exception escapes the worker thread, and the process falls off
    // the std::terminate cliff. Pre-allocating removes that failure mode —
    // the only allocation cost in the loop is the compressedBuf resize per
    // in-flight read, which is already bounded by req.compressedSize.
    //
    // Free-list: `freeSlots` is a stack of indices into `pool`. Allocation
    // pops; deallocation pushes. Invariant: `freeSlots.size() + busy reads
    // == kMaxInFlightPerWorker`.
    std::vector<std::unique_ptr<InFlightRead>> pool;
    pool.reserve(kMaxInFlightPerWorker);
    std::vector<std::size_t> freeSlots;
    freeSlots.reserve(kMaxInFlightPerWorker);
    for (std::size_t i = 0; i < kMaxInFlightPerWorker; ++i) {
        // Allocation is done here, BEFORE we start accepting completions.
        // If this throws bad_alloc at worker startup the caller's ctor
        // unwinds cleanly (std::thread's ctor already propagates system
        // errors); no completions have been promised yet.
        try {
            pool.emplace_back(std::make_unique<InFlightRead>());
        } catch (const std::bad_alloc&) {
            // Shrink the pool to what we got and move on. Worst case a
            // smaller effective kMaxInFlight — preferable to terminating.
            break;
        }
        freeSlots.push_back(pool.size() - 1u);
    }
    const std::size_t kEffectiveInFlightCap = pool.size();

    // Per-slot busy flag. Parallel array to `pool` — true when the kernel
    // holds a pointer to pool[i]->overlapped. Replaces the old erase/
    // push_back dance on a vector of owning pointers.
    std::vector<u8> slotBusy(kEffectiveInFlightCap, 0u);

    const HANDLE hIocp = static_cast<HANDLE>(iocpHandle_.load(std::memory_order_relaxed));
    const HANDLE hFile = static_cast<HANDLE>(osFileHandle_);
    const bool fileOpenOk = (hIocp != nullptr && hFile != nullptr);

    // ---- Helpers. Defined up front so the loop body is linear. ----

    // Find the slot whose OVERLAPPED the kernel handed back. Linear scan;
    // the pool is tiny (64 default) so this is cache-line fast.
    auto findSlotForOverlapped = [&](OVERLAPPED* ov) -> std::size_t {
        for (std::size_t i = 0; i < pool.size(); ++i) {
            if (slotBusy[i] && &pool[i]->overlapped == ov) return i;
        }
        return pool.size();  // sentinel "not found"
    };

    // Return a slot to the free-list. Clears the OVERLAPPED for hygiene.
    auto releaseSlot = [&](std::size_t idx) {
        ENIGMA_ASSERT(idx < pool.size());
        ENIGMA_ASSERT(slotBusy[idx]);
        slotBusy[idx] = 0u;
        pool[idx]->overlapped = OVERLAPPED{};
        pool[idx]->request    = PageRequest{};
        // compressedBuf is shrunk on next allocate() via resize; don't
        // free memory here to avoid churn (the buffer is upper-bounded
        // by kMaxReasonableCompressedBytes anyway).
        freeSlots.push_back(idx);
    };

    // Retire one InFlightRead whose OVERLAPPED the kernel just handed back,
    // decompress, and fire the callback. Called from the drain path.
    auto completeOne = [&](OVERLAPPED* ov, DWORD bytes, BOOL ok) {
        const std::size_t idx = findSlotForOverlapped(ov);
        if (idx == pool.size()) return;  // defensive — stray completion.

        InFlightRead& ifr = *pool[idx];

        PageCompletion c{};
        c.pageId = ifr.request.pageId;

        if (!ok) {
            // M3.2 closeout fix #2: use OVERLAPPED::Internal for per-op
            // status. GetLastError() is thread-local and may reflect a
            // later syscall than the completion that failed — the kernel
            // stashes the per-I/O NTSTATUS in OVERLAPPED::Internal, which
            // is the canonical source for IOCP error reporting.
            const DWORD err = static_cast<DWORD>(ifr.overlapped.Internal);
            c.success     = false;
            c.errorDetail = std::string{"ReadFile (async) failed, NTSTATUS=0x"};
            // Hex-format the NTSTATUS for readability in logs.
            {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "%08x", err);
                c.errorDetail += buf;
            }
        } else if (bytes != ifr.request.compressedSize) {
            c.success     = false;
            c.errorDetail = std::string{"short read (async): got "}
                          + std::to_string(bytes)
                          + " expected "
                          + std::to_string(ifr.request.compressedSize);
        } else {
            c.decompressedData.resize(ifr.request.decompressedSize);
            const std::size_t got = ZSTD_decompress(
                c.decompressedData.data(), c.decompressedData.size(),
                ifr.compressedBuf.data(), ifr.compressedBuf.size());
            if (ZSTD_isError(got)) {
                c.success     = false;
                c.errorDetail = std::string{"ZSTD_decompress failed: "}
                              + ZSTD_getErrorName(got);
                c.decompressedData.clear();
            } else if (got != ifr.request.decompressedSize) {
                c.success     = false;
                c.errorDetail = std::string{"decompressed size mismatch: got "}
                              + std::to_string(got)
                              + " expected "
                              + std::to_string(ifr.request.decompressedSize);
                c.decompressedData.clear();
            } else {
                c.success = true;
            }
        }

        // Release the in-flight slot BEFORE firing the callback so a caller
        // that enqueues from inside onComplete_ sees capacity.
        releaseSlot(idx);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            --inFlight_;
        }
        cv_.notify_all();

        if (opts_.onComplete) {
            opts_.onComplete(std::move(c));
        }
    };

    // Fire a synthetic failure completion for a request we never actually
    // dispatched to the kernel (sanity-check rejection, ReadFile hard fail,
    // file open failure).
    auto fireFailure = [&](const PageRequest& req, std::string detail) {
        PageCompletion c{};
        c.pageId      = req.pageId;
        c.success     = false;
        c.errorDetail = std::move(detail);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            --inFlight_;
        }
        cv_.notify_all();
        if (opts_.onComplete) {
            opts_.onComplete(std::move(c));
        }
    };

    // Convenience: how many reads are currently in-flight (busy pool slots).
    auto busyCount = [&]() -> std::size_t {
        return kEffectiveInFlightCap - freeSlots.size();
    };

    // ---- Main loop. ----
    for (;;) {
        // 1) Issue new reads while we have capacity and queued work.
        while (!freeSlots.empty()) {
            PageRequest req{};
            {
                std::lock_guard<std::mutex> guard(mutex_);
                if (queue_.empty()) break;
                req = queue_.front();
                queue_.pop_front();
                ++inFlight_;
            }

            if (!fileOpenOk) {
                fireFailure(req, osOpenError_.empty() ? std::string{"file not open"}
                                                      : osOpenError_);
                continue;
            }
            if (req.compressedSize == 0u ||
                req.compressedSize > kMaxReasonableCompressedBytes) {
                fireFailure(req, std::string{"compressedSize out of range: "}
                                 + std::to_string(req.compressedSize));
                continue;
            }
            if (req.decompressedSize == 0u ||
                req.decompressedSize > kMaxReasonableCompressedBytes) {
                fireFailure(req, std::string{"decompressedSize out of range: "}
                                 + std::to_string(req.decompressedSize));
                continue;
            }

            // M3.0 prereq D — pull from pre-allocated pool. No bad_alloc
            // on the hot path regardless of memory pressure.
            const std::size_t slotIdx = freeSlots.back();
            freeSlots.pop_back();
            InFlightRead& ifr = *pool[slotIdx];
            ifr.request = req;
            // M3.2 closeout fix #3: compressedBuf grows to req.compressedSize
            // which is bounded by kMaxReasonableCompressedBytes but can still
            // trigger bad_alloc under extreme memory pressure. Catching
            // locally converts the failure into a synthetic completion
            // instead of unwinding through the worker thread's catch-none
            // boundary and hitting std::terminate.
            try {
                ifr.compressedBuf.resize(req.compressedSize);
            } catch (const std::bad_alloc&) {
                // Return the slot BEFORE firing the failure so a caller
                // that re-enqueues from inside onComplete_ sees capacity.
                releaseSlot(slotIdx);
                fireFailure(req, std::string{"compressedBuf.resize OOM for size="}
                                 + std::to_string(req.compressedSize));
                continue;
            }
            ifr.overlapped = OVERLAPPED{};
            ifr.overlapped.Offset =
                static_cast<DWORD>(req.fileOffset & 0xFFFFFFFFull);
            ifr.overlapped.OffsetHigh =
                static_cast<DWORD>(req.fileOffset >> 32);
            slotBusy[slotIdx] = 1u;

            DWORD bytesImmediate = 0;
            const BOOL issued = ReadFile(
                hFile,
                ifr.compressedBuf.data(),
                static_cast<DWORD>(req.compressedSize),
                &bytesImmediate,
                &ifr.overlapped);
            const DWORD gle = issued ? 0u : GetLastError();
            if (!issued && gle != ERROR_IO_PENDING) {
                // Immediate hard failure: OVERLAPPED is not registered, no
                // completion will arrive. Fire a failure here AND return
                // the slot so we don't leak capacity.
                releaseSlot(slotIdx);
                fireFailure(req, std::string{"ReadFile (async) failed, GLE="}
                                 + std::to_string(gle));
                continue;
            }
            // Either synchronous success or ERROR_IO_PENDING — Windows
            // delivers the completion through IOCP in both cases since the
            // handle is associated with it.
        }

        // 2) Decide whether to exit. Exit once stop_ is set and there is
        //    no more work anywhere in the system.
        //    If the file failed to open there can never be in-flight IOCP
        //    reads, so once the queue is empty all pending callers have been
        //    notified and we can exit without blocking on a null IOCP handle.
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (stop_ && queue_.empty() && busyCount() == 0u) {
                break;
            }
            if (!fileOpenOk && queue_.empty()) {
                break;
            }
        }

        // 3) Block until SOMETHING happens:
        //    (Only reached when fileOpenOk, so hIocp is valid.)
        //    - a read completes (GQCS returns ok or ov != null)
        //    - enqueue() or shutdown() posts a sentinel wakeup
        //    - the short timeout elapses (so we re-check stop_ periodically)
        //    We pick 10ms as the idle poll — cheap, and shutdown doesn't
        //    depend on it because shutdown() always posts a sentinel.
        const DWORD kPollMs = (busyCount() == 0u) ? 50u : 10u;
        DWORD bytes = 0;
        ULONG_PTR key = 0;
        OVERLAPPED* ov = nullptr;
        const BOOL ok = GetQueuedCompletionStatus(hIocp, &bytes, &key,
                                                  &ov, kPollMs);
        if (!ok && ov == nullptr) {
            // Timeout (or IOCP closed). Nothing to do; re-check loop
            // invariants at the top.
            continue;
        }
        if (key == kWakeupCompletionKey) {
            // Sentinel; ov==nullptr. Re-check loop invariants.
            continue;
        }
        // Real IO completion (success or failure) — retire it.
        completeOne(ov, bytes, ok);
    }

    // We only exit when the pool is fully idle, so no cancellation is
    // required here. The destructor closes hFile and hIocp, which would
    // cancel any stragglers anyway as a safety net. shutdown()'s abnormal-
    // termination branch (M3.0 prereq B) uses CancelIoEx when the normal
    // drain budget is exceeded.
}

#else  // !_WIN32

PageCompletion
AsyncIOWorker::handle_sync_(const PageRequest& req,
                            std::vector<u8>& compressedScratch) {
    PageCompletion out{};
    out.pageId = req.pageId;

    if (osOpenFailed_ || osFileHandle_ == nullptr) {
        out.success     = false;
        out.errorDetail = osOpenError_.empty()
            ? std::string{"file not open"}
            : osOpenError_;
        return out;
    }

    if (req.compressedSize == 0u ||
        req.compressedSize > kMaxReasonableCompressedBytes) {
        out.success     = false;
        out.errorDetail = std::string{"compressedSize out of range: "}
                        + std::to_string(req.compressedSize);
        return out;
    }
    if (req.decompressedSize == 0u ||
        req.decompressedSize > kMaxReasonableCompressedBytes) {
        out.success     = false;
        out.errorDetail = std::string{"decompressedSize out of range: "}
                        + std::to_string(req.decompressedSize);
        return out;
    }

    // Mirrors the Windows IOCP try/catch around compressedBuf.resize() above:
    // converts a bad_alloc under extreme memory pressure into a synthetic
    // failure completion instead of escaping the worker thread's catch-none
    // boundary and hitting std::terminate.
    try {
        compressedScratch.resize(req.compressedSize);
    } catch (const std::bad_alloc&) {
        out.success     = false;
        out.errorDetail = std::string{"compressedScratch.resize OOM for size="}
                        + std::to_string(req.compressedSize);
        return out;
    }

    const int fd = static_cast<int>(reinterpret_cast<std::intptr_t>(osFileHandle_));
    const ssize_t n = ::pread(fd, compressedScratch.data(),
                              static_cast<size_t>(req.compressedSize),
                              static_cast<off_t>(req.fileOffset));
    if (n < 0) {
        out.success     = false;
        out.errorDetail = std::string{"pread failed: errno="}
                        + std::to_string(errno);
        return out;
    }
    if (static_cast<u32>(n) != req.compressedSize) {
        out.success     = false;
        out.errorDetail = std::string{"short read: got "}
                        + std::to_string(n)
                        + " expected " + std::to_string(req.compressedSize);
        return out;
    }

    out.decompressedData.resize(req.decompressedSize);
    const std::size_t got = ZSTD_decompress(
        out.decompressedData.data(), out.decompressedData.size(),
        compressedScratch.data(), compressedScratch.size());
    if (ZSTD_isError(got)) {
        out.success     = false;
        out.errorDetail = std::string{"ZSTD_decompress failed: "}
                        + ZSTD_getErrorName(got);
        out.decompressedData.clear();
        return out;
    }
    if (got != req.decompressedSize) {
        out.success     = false;
        out.errorDetail = std::string{"decompressed size mismatch: got "}
                        + std::to_string(got)
                        + " expected " + std::to_string(req.decompressedSize);
        out.decompressedData.clear();
        return out;
    }

    out.success = true;
    return out;
}

#endif  // _WIN32

} // namespace enigma::renderer::micropoly
