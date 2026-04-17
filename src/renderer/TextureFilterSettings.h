#pragma once

#include "core/Types.h"

namespace enigma {

// User-adjustable texture filtering controls. A change to any field must be
// followed by a Renderer::applyTextureFilterSettings() call so the material
// sampler is rebuilt and the bindless slot is updated in place.
struct TextureFilterSettings {
    // When true, the material sampler is unclamped (`maxLod = NONE`) so
    // trilinear mipmap sampling is used end-to-end. When false, `maxLod` is
    // clamped to 0.25 forcing base-mip sampling — useful for A/B comparing
    // the shimmer a mipchain-less setup produces.
    bool mipmapsEnabled = true;

    // Max anisotropy taps. 1 disables anisotropic filtering; higher values
    // reduce crunch on oblique surfaces at a modest GPU cost. Clamped at
    // init time to the device's maxSamplerAnisotropy.
    u32 anisotropy = 16;
};

} // namespace enigma
