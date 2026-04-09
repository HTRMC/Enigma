#pragma once

#include <filesystem>
#include <optional>

namespace enigma {
struct Scene;
} // namespace enigma

namespace enigma::gfx {
class Allocator;
class DescriptorAllocator;
class Device;
} // namespace enigma::gfx

namespace enigma {

// Load a glTF 2.0 file (.gltf or .glb) into a Scene. Uploads all mesh
// vertex data (as SSBOs in binding 2), index buffers, textures (binding 0),
// and samplers (binding 3) to the GPU via staging transfers.
//
// Returns std::nullopt on any load failure (file not found, parse error,
// unsupported features). Caller owns the returned Scene and must call
// Scene::destroy() before Device/Allocator teardown.
std::optional<Scene> loadGltf(const std::filesystem::path& path,
                              gfx::Device& device,
                              gfx::Allocator& allocator,
                              gfx::DescriptorAllocator& descriptorAllocator);

} // namespace enigma
