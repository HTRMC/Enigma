// MicropolyBlasManagerError.cpp
// ==============================
// Out-of-line definition of micropolyBlasErrorKindString. Split from
// MicropolyBlasManager.cpp so the headless smoke test
// (tests/micropoly_blas_manager_test.cpp) can link the stringifier
// without pulling gfx::Device, Allocator, and MpAssetReader into its
// TU graph. Peer pattern: MicropolyRasterPassError.cpp.

#include "renderer/micropoly/MicropolyBlasManager.h"

namespace enigma::renderer::micropoly {

const char* micropolyBlasErrorKindString(MicropolyBlasErrorKind kind) noexcept {
    switch (kind) {
        case MicropolyBlasErrorKind::NotSupported:         return "NotSupported";
        case MicropolyBlasErrorKind::PageDecompressFailed: return "PageDecompressFailed";
        case MicropolyBlasErrorKind::BlasBuildFailed:      return "BlasBuildFailed";
        case MicropolyBlasErrorKind::NoLevel3Clusters:     return "NoLevel3Clusters";
        case MicropolyBlasErrorKind::ReaderNotOpen:        return "ReaderNotOpen";
    }
    return "?";
}

} // namespace enigma::renderer::micropoly
