// MicropolySwRasterPassError.cpp
// ================================
// Out-of-line definition of micropolySwRasterErrorKindString. Split from
// MicropolySwRasterPass.cpp so headless smoke tests
// (tests/micropoly_sw_raster_bin_test.cpp) can link the stringifier without
// pulling ShaderManager + Paths + Log + gfx::Allocator into their TU graph.
// Matches the peer pattern established by MicropolyRasterPassError.cpp.

#include "renderer/micropoly/MicropolySwRasterPass.h"

namespace enigma::renderer::micropoly {

const char* micropolySwRasterErrorKindString(MicropolySwRasterErrorKind kind) {
    switch (kind) {
        case MicropolySwRasterErrorKind::MeshShadersUnsupported:     return "MeshShadersUnsupported";
        case MicropolySwRasterErrorKind::PipelineBuildFailed:        return "PipelineBuildFailed";
        case MicropolySwRasterErrorKind::BufferAllocFailed:          return "BufferAllocFailed";
        case MicropolySwRasterErrorKind::BindlessRegistrationFailed: return "BindlessRegistrationFailed";
        case MicropolySwRasterErrorKind::RasterPipelineBuildFailed:  return "RasterPipelineBuildFailed";
        case MicropolySwRasterErrorKind::Int64ImageUnsupported:      return "Int64ImageUnsupported";
    }
    return "?";
}

} // namespace enigma::renderer::micropoly
