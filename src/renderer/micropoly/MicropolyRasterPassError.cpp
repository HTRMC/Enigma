// MicropolyRasterPassError.cpp
// =============================
// Out-of-line definition of micropolyRasterErrorKindString. Split from
// MicropolyRasterPass.cpp so headless smoke tests
// (tests/micropoly_raster_test.cpp) can link the stringifier without
// pulling ShaderManager + Paths + Log + gfx::Device into their TU graph.

#include "renderer/micropoly/MicropolyRasterPass.h"

namespace enigma::renderer::micropoly {

const char* micropolyRasterErrorKindString(MicropolyRasterErrorKind kind) {
    switch (kind) {
        case MicropolyRasterErrorKind::MeshShadersUnsupported: return "MeshShadersUnsupported";
        case MicropolyRasterErrorKind::Int64ImageUnsupported:  return "Int64ImageUnsupported";
        case MicropolyRasterErrorKind::PipelineBuildFailed:    return "PipelineBuildFailed";
        case MicropolyRasterErrorKind::InvalidVisImage:        return "InvalidVisImage";
    }
    return "?";
}

} // namespace enigma::renderer::micropoly
