#pragma once

// AsyncIOWorker.h
// ================
// Background page-reader thread for the Micropoly streaming subsystem.
// A single worker thread pulls PageRequest entries from a bounded queue,
// reads compressed bytes off disk, zstd-decompresses them, and invokes a
// completion callback with the result.
//
// Contract
// --------
// - One worker = one OS thread = one .mpa file. Multiple workers can run
//   against different files concurrently.
// - The completion callback runs on the worker thread; it must be thread-
//   safe and non-blocking (e.g. push a completion onto another lock-free
//   queue consumed by the renderer).
// - enqueue() is thread-safe. Returns false when the queue is full so the
//   caller can back-pressure.
// - shutdown() stops accepting new requests, drains the in-flight request,
//   and joins the worker. The destructor calls shutdown() if not already.
// - No Vulkan. Pure CPU IO + zstd. M2.2 pairs this with the upload queue.
//
// IO strategy (M2.4b)
// -------------------
// Windows: CreateFileW(FILE_FLAG_OVERLAPPED) + ReadFile + IOCP. The worker
// keeps up to `maxInflightRequests` reads outstanding simultaneously;
// completions are drained via GetQueuedCompletionStatus on the worker
// thread so callbacks continue to fire from the worker (public-API
// invariant). The earlier M2.1 synchronous-ReadFile path has been retired.
//
// Non-Windows (CI-compile only): POSIX pread(). The engine proper is
// Windows-only per the .omc plan; Linux/macOS keeps the simple synchronous
// path because no runtime path exercises it.

#include "core/Types.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace enigma::renderer::micropoly {

// One request for a single page. Mirrors the fields the caller would pull
// from an MpPageEntry. Duplicating them here keeps the worker decoupled
// from MpAssetReader's header so unit tests can drive it with synthesized
// inputs.
struct PageRequest {
    u32 pageId            = 0u;   // logical page index (passed back in completion)
    u64 fileOffset        = 0u;   // byte offset inside the .mpa
    u32 compressedSize    = 0u;   // number of compressed bytes to read
    u32 decompressedSize  = 0u;   // expected size after zstd decompress
};

// Completion delivered to the caller's callback. The `decompressedData`
// vector is move-owned — the worker hands it off and does not retain a
// reference.
struct PageCompletion {
    u32               pageId          = 0u;
    std::vector<u8>   decompressedData;
    bool              success         = false;
    std::string       errorDetail;     // empty on success
};

// Completion callback type. Invoked on the worker thread.
//
// Executor contract (M3.0 prereq C — document-only; not enforced in code):
//   * The callback MUST NOT block. Specifically:
//       - No GPU fence waits (vkWaitForFences, vkWaitSemaphores, etc.) —
//         these can stall indefinitely on GPU hangs and the worker thread
//         is the only thread processing IO for this file handle; blocking
//         it starves the whole per-file stream.
//       - No mutex acquisition that the caller's shutdown() path can hold.
//         The classic deadlock: caller thread holds mutex_X, calls
//         AsyncIOWorker::shutdown() which joins the worker; the worker
//         callback tries to acquire mutex_X and deadlocks on the join.
//       - No unbounded allocations or file I/O. Short CPU work only.
//   * Typical well-formed callback: push the PageCompletion onto a
//     lock-free or mutex-protected queue consumed by the renderer's
//     beginFrame() pump. See MicropolyStreaming::onWorkerComplete_ for
//     the canonical pattern.
//   * The callback runs on the dedicated worker thread, so it MUST be
//     thread-safe with respect to the rest of the caller's state.
using CompletionCallback = std::function<void(PageCompletion)>;

// Configuration for constructing a worker.
struct AsyncIOWorkerOptions {
    std::filesystem::path mpaFilePath;
    CompletionCallback    onComplete;
    // Bounded queue length AND the ceiling on IOCP in-flight reads. The
    // M2.4b worker issues up to this many ReadFile(OVERLAPPED) calls in
    // parallel; a value of 32-64 comfortably saturates consumer NVMe
    // queue depth.
    std::size_t           maxInflightRequests = 64u;
};

// Single-thread page-reader. Not copyable or movable (the worker thread
// captures `this`, so the address must be stable for the object's lifetime).
class AsyncIOWorker {
public:
    explicit AsyncIOWorker(AsyncIOWorkerOptions opts);
    ~AsyncIOWorker();

    AsyncIOWorker(const AsyncIOWorker&)            = delete;
    AsyncIOWorker& operator=(const AsyncIOWorker&) = delete;
    AsyncIOWorker(AsyncIOWorker&&)                 = delete;
    AsyncIOWorker& operator=(AsyncIOWorker&&)      = delete;

    // Enqueue one request. Returns false if the queue is full OR if
    // shutdown() has already been called. Thread-safe.
    bool enqueue(const PageRequest& req);

    // How many requests are currently queued or being processed. Thread-
    // safe; useful for back-pressure in unit tests and driver code.
    std::size_t pending() const;

    // Stop accepting new requests, drain in-flight, join the worker
    // thread. Idempotent. The destructor calls this if the user did
    // not — we never silently drop work.
    void shutdown();

private:
    // Worker loop body. Runs on its own thread.
    void run_();

#if defined(_WIN32)
    // Windows IOCP-driven loop (M2.4b). Owns the in-flight pool,
    // pumps completions, refills pending requests, and sleeps idle.
    void run_iocp_loop_();

    // Per-read control block for the IOCP path. OVERLAPPED is first
    // so pointer-math round-trips with the kernel's handback are safe.
    struct InFlightRead;
#else
    // POSIX synchronous pread path (CI compile only).
    PageCompletion handle_sync_(const PageRequest& req,
                                std::vector<u8>& compressedScratch);
#endif

    // Options copy. Held by value because mpaFilePath + callback must
    // outlive the worker thread.
    AsyncIOWorkerOptions opts_;

    // OS file handle opened inside run_() (worker thread owns it so
    // failures surface via a completion, not a constructor exception).
    // Stored as void* to avoid leaking <windows.h> into the header.
    void* osFileHandle_ = nullptr;
    // Windows I/O completion port handle. Stored as atomic<void*> because
    // the worker thread writes it and enqueue()/shutdown() read it from
    // the calling thread. Unused on POSIX (the sync path has no IOCP).
    std::atomic<void*> iocpHandle_  = nullptr;
    bool  osOpenFailed_ = false;
    std::string osOpenError_;

    // Request queue. Lock order for all methods: mutex_ then never any
    // other lock — the worker never calls external code while holding
    // mutex_ (callbacks run after release).
    mutable std::mutex              mutex_;
    std::condition_variable         cv_;
    std::deque<PageRequest>         queue_;
    // Number of requests dequeued but not yet completed. Included in
    // pending() so shutdown() can wait for true drainage. On the IOCP
    // path this also tracks reads currently in the kernel's queue.
    std::size_t                     inFlight_ = 0u;
    // Set once; stop flag consulted inside the worker loop.
    bool                            stop_     = false;
    // True once shutdown() has observed the worker as fully joined.
    // Makes shutdown() safely idempotent.
    bool                            joined_   = false;

    // The worker thread itself. Launched in the constructor; joined
    // in shutdown().
    std::thread                     thread_;
};

} // namespace enigma::renderer::micropoly
