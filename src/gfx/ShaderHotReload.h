#pragma once

#include <filesystem>
#include <functional>
#include <vector>

namespace enigma::gfx {

// ShaderHotReload
// ===============
// Polling-based hot-reload registry for shader source files. Owners
// of pipelines register *groups* of shader files together with a
// rebuild callback. Once per frame, the Renderer calls `poll()`,
// which stats every watched file and — if any file's last-modified
// time advanced since the previous poll — invokes that group's
// callback exactly once.
//
// Design contracts:
//   - Polling, not native filesystem events. std::filesystem on
//     every supported platform exposes last_write_time with
//     second-level granularity, which is more than enough for
//     human-scale edit cadence. No new dependencies.
//   - Single-threaded: poll() runs on the render thread; callbacks
//     run synchronously from poll().
//   - Exception-safe around mid-save atomic-rename gaps: if
//     last_write_time fails (file temporarily absent during editor
//     save) the file is skipped for that tick and retried next
//     frame.
//   - Group semantics: a vertex+fragment pair that shares a
//     pipeline is registered once; either file touching disk
//     triggers a single rebuild, not two.
//
// Second-caller design intent (Principle 6): a future compute pass
// or post-process pipeline registers its own group with the same
// API — no rewrite of this class required.
class ShaderHotReload {
public:
    using ReloadCallback = std::function<void()>;

    ShaderHotReload()  = default;
    ~ShaderHotReload() = default;

    ShaderHotReload(const ShaderHotReload&)            = delete;
    ShaderHotReload& operator=(const ShaderHotReload&) = delete;
    ShaderHotReload(ShaderHotReload&&)                 = delete;
    ShaderHotReload& operator=(ShaderHotReload&&)      = delete;

    // Register a group of shader files with a single rebuild
    // callback. If any file in the group changes on disk
    // (last_write_time advances past its stored snapshot),
    // `onChange` is invoked once and the new mtimes are recorded.
    // The initial mtimes are sampled at registration time, so an
    // edit that pre-dates registration does not trigger a spurious
    // reload.
    void watchGroup(std::vector<std::filesystem::path> paths,
                    ReloadCallback onChange);

    // Stat every watched file and fire callbacks for groups whose
    // mtimes advanced. Returns the number of callbacks invoked.
    int poll();

private:
    struct Entry {
        std::vector<std::filesystem::path>           paths;
        std::vector<std::filesystem::file_time_type> mtimes;
        ReloadCallback                               onChange;
    };

    std::vector<Entry> m_entries;
};

} // namespace enigma::gfx
