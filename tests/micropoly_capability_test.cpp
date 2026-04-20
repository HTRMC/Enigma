// Unit test for the micropoly HW capability classifier (M0a).
// Synthetically masks each probed feature bit and verifies the classifier
// maps to the correct HW matrix row per plan §3.M0a.
//
// Build target registered in CMakeLists.txt (micropoly_capability_test).
// Run under /W4 /WX — same strictness as the Enigma target.

#include "renderer/micropoly/MicropolyCapability.h"

#include <cassert>
#include <cstdio>
#include <cstring>

using enigma::classifyMicropolyRow;
using enigma::statusStringFor;
using enigma::HwMatrixRow;

namespace {

struct Case {
    // Inputs.
    bool meshShader;
    bool atomicInt64;
    bool imageInt64;
    bool sparseResidency;
    bool rayTracing;
    // Expected row.
    HwMatrixRow expected;
    // Expected boot status string (exact match).
    const char* expectedStatus;
};

constexpr Case kCases[] = {
    // Row (a): full feature set.
    { true,  true,  true,  true,  true,  HwMatrixRow::FullFeatureSet,    "full"                    },
    // Row (b): RT but no sparse.
    { true,  true,  true,  false, true,  HwMatrixRow::RtShadowOnly,      "RT shadow only"          },
    // Row (c): sparse but no RT.
    { true,  true,  true,  true,  false, HwMatrixRow::VsmShadow,         "VSM shadow"              },
    // Row (d): geometry only (no shadows).
    { true,  true,  true,  false, false, HwMatrixRow::GeometryNoShadows, "geometry but no shadows" },
    // Row (e): image_int64 missing — HW-only path; sparse/RT irrelevant.
    { true,  true,  false, true,  true,  HwMatrixRow::HwOnly,            "HW-only micropoly"       },
    { true,  true,  false, false, false, HwMatrixRow::HwOnly,            "HW-only micropoly"       },
    // Row (f): hard gate off — mesh shader missing.
    { false, true,  true,  true,  true,  HwMatrixRow::Disabled,          "disabled"                },
    // Row (f): hard gate off — atomic_int64 missing.
    { true,  false, true,  true,  true,  HwMatrixRow::Disabled,          "disabled"                },
    // Row (f): both missing.
    { false, false, false, false, false, HwMatrixRow::Disabled,          "disabled"                },
};

} // namespace

int main() {
    bool allPassed = true;

    for (std::size_t i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i) {
        const Case& c = kCases[i];
        const HwMatrixRow row = classifyMicropolyRow(
            c.meshShader, c.atomicInt64, c.imageInt64,
            c.sparseResidency, c.rayTracing);
        const char* status = statusStringFor(row);

        const bool rowOk    = (row == c.expected);
        const bool statusOk = (std::strcmp(status, c.expectedStatus) == 0);

        if (!rowOk || !statusOk) {
            std::fprintf(stderr,
                "[micropoly_capability_test] case %zu FAIL: "
                "mesh=%d atomic=%d imageI64=%d sparse=%d rt=%d -> "
                "row=%d (expected %d), status='%s' (expected '%s')\n",
                i,
                c.meshShader ? 1 : 0,
                c.atomicInt64 ? 1 : 0,
                c.imageInt64 ? 1 : 0,
                c.sparseResidency ? 1 : 0,
                c.rayTracing ? 1 : 0,
                static_cast<int>(row),
                static_cast<int>(c.expected),
                status,
                c.expectedStatus);
            allPassed = false;
        }
    }

    // All 6 matrix rows must have been hit by at least one case (exit
    // criterion "All 6 HW matrix rows selectable in unit test").
    bool rowHit[6] = {false, false, false, false, false, false};
    for (const Case& c : kCases) {
        const HwMatrixRow row = classifyMicropolyRow(
            c.meshShader, c.atomicInt64, c.imageInt64,
            c.sparseResidency, c.rayTracing);
        rowHit[static_cast<std::size_t>(row)] = true;
    }
    for (std::size_t i = 0; i < 6; ++i) {
        if (!rowHit[i]) {
            std::fprintf(stderr,
                "[micropoly_capability_test] matrix row %zu never exercised\n", i);
            allPassed = false;
        }
    }

    // -----------------------------------------------------------------
    // M3.1 smoke: exercise the MicropolyPass scaffolding without bringing
    // up a VkDevice. Classification-only: the test verifies that certain
    // observable invariants hold for disabled configs (Principle 1 gate).
    // End-to-end visImage creation against a real device is exercised by
    // the Renderer's ctor in the integration tests (micropoly_streaming_test
    // + cold_cache_stress_test) and at runtime.
    //
    // We only test the disabled-row classifier here because any active()
    // row needs a real VkDevice via micropolyCaps() — out of scope for
    // this header-linked test binary. The important gate: classifier
    // returns Disabled for row (f), which is enough for this file.
    // -----------------------------------------------------------------
    {
        // Constructor invariants: disabled classifier row → no-op pass
        // would return false from active(). We cannot instantiate the
        // MicropolyPass here (would need a VkDevice), but we can assert
        // the cap-to-active behavior via the pure classifier.
        const enigma::HwMatrixRow disabledRow =
            enigma::classifyMicropolyRow(false, true, true, true, true);
        if (disabledRow != enigma::HwMatrixRow::Disabled) {
            std::fprintf(stderr,
                "[micropoly_capability_test] M3.1 smoke FAIL: "
                "disabled classifier did not return Disabled\n");
            allPassed = false;
        }

        // visFormat fallback chain: if the device reports neither R64 nor
        // imageInt64 then visFormat must be VK_FORMAT_UNDEFINED in the
        // MicropolyPass. We re-exercise the classifier for the relevant
        // row (e: image_int64 missing) as a guardrail; the visFormat
        // decision itself is probed against the real VkPhysicalDevice in
        // MicropolyPass::MicropolyPass, out of scope here.
        const enigma::HwMatrixRow hwOnly =
            enigma::classifyMicropolyRow(true, true, false, false, false);
        if (hwOnly != enigma::HwMatrixRow::HwOnly) {
            std::fprintf(stderr,
                "[micropoly_capability_test] M3.1 smoke FAIL: "
                "image_int64-absent row did not classify as HwOnly\n");
            allPassed = false;
        }
    }

    if (!allPassed) {
        std::fprintf(stderr, "[micropoly_capability_test] FAILED\n");
        return 1;
    }

    std::printf("[micropoly_capability_test] All %zu cases passed; all 6 rows covered; M3.1 smoke OK.\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
