#pragma once

// MpPathUtils.h
// ==============
// Shared path-validation helpers for .mpa-file-opening sites across the
// micropoly subsystem. Consolidates the defence-in-depth checks previously
// duplicated in Renderer.cpp (renderer construction) and AsyncIOWorker.cpp
// (worker thread open).
//
// Rejection rules (Windows-inclusive):
//   * Empty path.
//   * Non-absolute path — cwd-relative resolution is not a stable asset-
//     integrity guarantee.
//   * UNC path prefix (\\server\share, \\?\, \\.\) — escapes sandboxing /
//     redirection and may reach attacker-controlled shares.
//   * NT device namespace prefix (\??\) — the Win32 NT-object root; raw
//     device access bypasses all Win32 path-parsing safety nets.
//   * Path length exceeding the modern Windows long-path limit
//     (kMaxWindowsLongPath = 32767 wchar_t, matching MAX_PATH extended per
//     MSDN "Naming Files, Paths, and Namespaces").
//
// On return true, `detail` is left empty. On false, `detail` carries a
// short human-readable diagnostic string suitable for logging.
//
// Pure header; no Vulkan / no engine state. Safe to include from runtime
// renderer code and from background worker threads.

#include <filesystem>
#include <string>

namespace enigma::asset {

// Upper bound on a safe mpa path length. Matches the Windows extended long
// path limit (32,767 wchar_t). Non-Windows hosts also honor the same cap
// for consistency — real cross-platform builds will never hit it.
inline constexpr std::size_t kMaxWindowsLongPath = 32767u;

// Returns true iff `p` is safe to pass to CreateFileW / open(). When it
// returns false, `detail` carries a short diagnostic suitable for logging.
inline bool isSafeMpaPath(const std::filesystem::path& p, std::string& detail) {
    if (p.empty()) {
        detail = "mpaFilePath is empty";
        return false;
    }
    if (!p.is_absolute()) {
        detail = std::string{"mpaFilePath is not absolute: "} + p.string();
        return false;
    }
#if defined(_WIN32)
    const auto& s = p.native();
    // UNC: \\server\share, \\?\extended, \\.\device
    if (s.size() >= 2u && s[0] == L'\\' && s[1] == L'\\') {
        detail = std::string{"UNC paths are rejected: "} + p.string();
        return false;
    }
    // NT device namespace: \??\ prefix reaches the raw NT object root.
    if (s.size() >= 4u && s[0] == L'\\' && s[1] == L'?' && s[2] == L'?'
        && s[3] == L'\\') {
        detail = std::string{"NT device namespace paths are rejected: "} + p.string();
        return false;
    }
    if (s.size() > kMaxWindowsLongPath) {
        detail = std::string{"path length exceeds Windows long-path limit: "}
               + std::to_string(s.size());
        return false;
    }
#else
    const auto& s = p.native();
    if (s.size() > kMaxWindowsLongPath) {
        detail = std::string{"path length exceeds long-path limit: "}
               + std::to_string(s.size());
        return false;
    }
#endif
    return true;
}

// Overload for callers that don't need the diagnostic string. Returns the
// same boolean but swallows the detail.
inline bool isSafeMpaPath(const std::filesystem::path& p) {
    std::string unused;
    return isSafeMpaPath(p, unused);
}

} // namespace enigma::asset
