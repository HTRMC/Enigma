#include "gfx/ShaderHotReload.h"

#include "core/Assert.h"
#include "core/Log.h"

#include <system_error>
#include <utility>

namespace enigma::gfx {

namespace {

// last_write_time can fail during an editor's atomic save (file
// temporarily absent between unlink + rename). Wrap it so the poll
// loop can continue without propagating errors through the render
// thread.
bool safeLastWriteTime(const std::filesystem::path& path,
                       std::filesystem::file_time_type& out) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return false;
    }
    out = t;
    return true;
}

} // namespace

void ShaderHotReload::watchGroup(std::vector<std::filesystem::path> paths,
                                 ReloadCallback onChange) {
    ENIGMA_ASSERT(!paths.empty() && "ShaderHotReload::watchGroup requires at least one path");
    ENIGMA_ASSERT(static_cast<bool>(onChange) && "ShaderHotReload::watchGroup requires a callback");

    Entry entry;
    entry.paths    = std::move(paths);
    entry.onChange = std::move(onChange);
    entry.mtimes.reserve(entry.paths.size());
    for (const auto& p : entry.paths) {
        std::filesystem::file_time_type t{};
        if (!safeLastWriteTime(p, t)) {
            ENIGMA_LOG_WARN("[hot-reload] cannot stat shader at registration: {}",
                            p.string());
        }
        entry.mtimes.push_back(t);
    }

    const size_t count = entry.paths.size();
    m_entries.push_back(std::move(entry));
    ENIGMA_LOG_INFO("[hot-reload] watching {} file(s) for group", count);
}

int ShaderHotReload::poll() {
    int fired = 0;
    for (auto& entry : m_entries) {
        bool changed = false;
        for (size_t i = 0; i < entry.paths.size(); ++i) {
            std::filesystem::file_time_type current{};
            if (!safeLastWriteTime(entry.paths[i], current)) {
                // Mid-save race or file temporarily missing; retry
                // on the next poll tick.
                continue;
            }
            if (current != entry.mtimes[i]) {
                entry.mtimes[i] = current;
                changed         = true;
            }
        }
        if (changed) {
            ENIGMA_LOG_INFO("[hot-reload] change detected, invoking rebuild callback");
            entry.onChange();
            ++fired;
        }
    }
    return fired;
}

} // namespace enigma::gfx
