#pragma once

#include <filesystem>

namespace enigma::Paths {

// Record argv[0] and resolve the absolute executable path. Must be called
// once from main/Application::run before any other Paths accessor is used.
// Calling init() more than once is permitted; the most recent argv0 wins.
void init(const char* argv0);

// Absolute path of the running executable, as resolved at init() time.
// If init() has not been called yet the returned path is empty.
const std::filesystem::path& executablePath();

// <exe dir>/shaders — the directory CMake copies shaders/*.{vert,frag} into
// on every build via the POST_BUILD custom command in CMakeLists.txt.
std::filesystem::path shaderDir();

// <source tree>/shaders — the directory where shader sources live in
// the repository, baked in at build time via the
// ENIGMA_SHADER_SOURCE_DIR macro from CMakeLists.txt. Hot reload uses
// this so edits to `shaders/*.{vert,frag}` in the source tree are
// picked up without a rebuild. Returns `shaderDir()` as a fallback if
// the source directory is not present on disk (e.g. the executable
// was moved to a machine without the source tree).
std::filesystem::path shaderSourceDir();

} // namespace enigma::Paths
