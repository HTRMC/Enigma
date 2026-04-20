#include "renderer/micropoly/MicropolyCapability.h"

#include "gfx/Device.h"

namespace enigma {

HwMatrixRow classifyMicropolyRow(bool meshShader, bool atomicInt64,
                                  bool imageInt64, bool sparseResidency,
                                  bool rayTracing) {
    // Row (f): hard-gate off. Either mesh shaders or 64-bit atomics missing
    // means neither the SW raster atomic-min nor the HW mesh-shader path
    // can run — micropoly is globally disabled.
    if (!meshShader || !atomicInt64) {
        return HwMatrixRow::Disabled;
    }

    // Row (e): image_int64 missing. SW raster falls back to HW; the HW
    // path alone is viable. sparse/RT state is irrelevant in this row.
    if (!imageInt64) {
        return HwMatrixRow::HwOnly;
    }

    // Full geometry available from here on. Distinguish shadow capability.
    if (sparseResidency && rayTracing) {
        return HwMatrixRow::FullFeatureSet;        // (a)
    }
    if (rayTracing) {
        return HwMatrixRow::RtShadowOnly;          // (b)
    }
    if (sparseResidency) {
        return HwMatrixRow::VsmShadow;             // (c)
    }
    return HwMatrixRow::GeometryNoShadows;         // (d)
}

const char* statusStringFor(HwMatrixRow row) {
    switch (row) {
        case HwMatrixRow::FullFeatureSet:    return "full";
        case HwMatrixRow::RtShadowOnly:      return "RT shadow only";
        case HwMatrixRow::VsmShadow:         return "VSM shadow";
        case HwMatrixRow::GeometryNoShadows: return "geometry but no shadows";
        case HwMatrixRow::HwOnly:            return "HW-only micropoly";
        case HwMatrixRow::Disabled:          return "disabled";
    }
    // Unreachable; C++23 lets the compiler see this, but silence /W4 noise.
    return "disabled";
}

MicropolyCaps micropolyCaps(gfx::Device& device) {
    MicropolyCaps caps{};
    caps.meshShader      = device.supportsMeshShaders();
    caps.atomicInt64     = device.supportsShaderAtomicInt64();
    caps.imageInt64      = device.supportsShaderImageInt64();
    caps.sparseResidency = device.supportsSparseResidency();
    caps.rayTracing      = device.supportsRayTracing();
    caps.row             = classifyMicropolyRow(caps.meshShader,
                                                 caps.atomicInt64,
                                                 caps.imageInt64,
                                                 caps.sparseResidency,
                                                 caps.rayTracing);
    caps.statusString    = statusStringFor(caps.row);
    return caps;
}

} // namespace enigma
