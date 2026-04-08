#include "core/Paths.h"

#include <system_error>

namespace enigma::Paths {

namespace {

std::filesystem::path& executablePathStorage() {
    static std::filesystem::path s_path;
    return s_path;
}

} // namespace

void init(const char* argv0) {
    auto& stored = executablePathStorage();
    if (argv0 == nullptr) {
        stored.clear();
        return;
    }

    std::error_code ec;
    auto absolute = std::filesystem::absolute(std::filesystem::path(argv0), ec);
    if (ec) {
        stored = std::filesystem::path(argv0);
        return;
    }
    stored = std::move(absolute);
}

const std::filesystem::path& executablePath() {
    return executablePathStorage();
}

std::filesystem::path shaderDir() {
    const auto& exe = executablePathStorage();
    if (exe.empty()) {
        return std::filesystem::path("shaders");
    }
    return exe.parent_path() / "shaders";
}

std::filesystem::path shaderSourceDir() {
#ifdef ENIGMA_SHADER_SOURCE_DIR
    std::filesystem::path source = ENIGMA_SHADER_SOURCE_DIR;
    std::error_code ec;
    if (std::filesystem::is_directory(source, ec)) {
        return source;
    }
#endif
    return shaderDir();
}

} // namespace enigma::Paths
