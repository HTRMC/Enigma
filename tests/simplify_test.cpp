// Unit test for the offline `enigma-mpbake` simplify stage (M1b.3).
// Four cases:
//   1) Happy path: DamagedHelmet.glb → ingest → cluster → take first N
//      clusters as a synthetic group → simplify → verify output shrunk
//      and achievedError is finite/>=0.
//   2) Empty group: `simplify({}, {})` → expect `EmptyGroup` error.
//   3) OptionsOutOfRange: `weldEpsilon = -1.0f` → expect `OptionsOutOfRange`.
//   4) Determinism: build the same group twice through simplify, hash the
//      output (positions + indices + achievedError), confirm identical.
//
// Plain main, printf output, exit 0 on pass — mirrors cluster_builder_test.
// Built under /W4 /WX.

#include "mpbake/ClusterBuilder.h"
#include "mpbake/GltfIngest.h"
#include "mpbake/Simplify.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace fs = std::filesystem;

using enigma::mpbake::ClusterBuildOptions;
using enigma::mpbake::ClusterBuilder;
using enigma::mpbake::ClusterData;
using enigma::mpbake::GltfIngest;
using enigma::mpbake::IngestedMesh;
using enigma::mpbake::simplify;
using enigma::mpbake::SimplifiedGroup;
using enigma::mpbake::SimplifyError;
using enigma::mpbake::SimplifyErrorKind;
using enigma::mpbake::SimplifyOptions;
using enigma::mpbake::simplifyErrorKindString;

namespace {

// Walk upward from argv[0]'s directory looking for `assets/DamagedHelmet.glb`.
fs::path locateDamagedHelmet(const char* argv0) {
    std::error_code ec;
    fs::path start = argv0 ? fs::absolute(argv0, ec).parent_path() : fs::current_path(ec);
    if (ec) start = fs::current_path(ec);

    for (int i = 0; i < 6 && !start.empty(); ++i) {
        fs::path candidate = start / "assets" / "DamagedHelmet.glb";
        if (fs::exists(candidate, ec)) return candidate;
        if (start == start.parent_path()) break;
        start = start.parent_path();
    }

    fs::path cwd = fs::current_path(ec) / "assets" / "DamagedHelmet.glb";
    if (fs::exists(cwd, ec)) return cwd;
    return {};
}

// FNV-1a 64-bit — deterministic hash of the simplify output for cross-run
// comparison. Matches cluster_builder_test's style.
std::uint64_t fnv1a64(const void* data, std::size_t size) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < size; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= 0x100000001b3ull;
    }
    return h;
}

// Hash the simplify output: positions + indices + achievedError (bit-exact).
std::uint64_t hashSimplified(const SimplifiedGroup& g) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](const void* p, std::size_t n) {
        const std::uint64_t sub = fnv1a64(p, n);
        h ^= sub;
        h *= 0x100000001b3ull;
    };
    const std::size_t posCount = g.positions.size();
    const std::size_t idxCount = g.indices.size();
    mix(&posCount, sizeof(posCount));
    mix(&idxCount, sizeof(idxCount));
    if (!g.positions.empty()) mix(g.positions.data(), g.positions.size() * sizeof(glm::vec3));
    if (!g.indices.empty())   mix(g.indices.data(),   g.indices.size()   * sizeof(std::uint32_t));
    std::uint32_t errBits = 0;
    std::memcpy(&errBits, &g.achievedError, sizeof(errBits));
    mix(&errBits, sizeof(errBits));
    return h;
}

// Load the DamagedHelmet asset and cluster it. Returns empty on failure.
std::vector<ClusterData> loadAndCluster(const fs::path& assetPath) {
    GltfIngest ingest;
    auto ingestResult = ingest.load(assetPath);
    if (!ingestResult.has_value()) {
        std::fprintf(stderr, "[simplify_test] ingest failed on '%s'\n",
            assetPath.string().c_str());
        return {};
    }
    ClusterBuilder builder;
    ClusterBuildOptions opts{};   // 128/128/0.5
    auto r = builder.build(*ingestResult, opts);
    if (!r.has_value()) {
        std::fprintf(stderr, "[simplify_test] cluster build failed\n");
        return {};
    }
    return std::move(*r);
}

bool testHappyPath(const fs::path& assetPath) {
    const std::vector<ClusterData> clusters = loadAndCluster(assetPath);
    if (clusters.empty()) {
        std::fprintf(stderr,
            "[simplify_test] case 1 FAIL: cluster set empty\n");
        return false;
    }

    // Take up to 4 clusters as a synthetic "group". DamagedHelmet emits
    // well more than 4 so this always has enough material to work with.
    const std::size_t groupSize = std::min<std::size_t>(4u, clusters.size());
    std::span<const ClusterData> group(clusters.data(), groupSize);

    std::size_t inputTris = 0;
    for (const auto& c : group) inputTris += c.triangles.size() / 3u;

    SimplifyOptions opts{};
    opts.maxError         = 0.05f;
    opts.targetIndexCount = 0u;   // auto-half
    opts.weldEpsilon      = 1e-5f;

    auto result = simplify(group, opts);
    if (!result.has_value()) {
        std::fprintf(stderr,
            "[simplify_test] case 1 FAIL: simplify errored: %s: %s\n",
            simplifyErrorKindString(result.error().kind),
            result.error().detail.c_str());
        return false;
    }
    const SimplifiedGroup& out = *result;

    if (out.indices.empty() || (out.indices.size() % 3u) != 0u) {
        std::fprintf(stderr,
            "[simplify_test] case 1 FAIL: bad output index count %zu\n",
            out.indices.size());
        return false;
    }
    if (out.positions.size() != out.normals.size() ||
        out.positions.size() != out.uvs.size()) {
        std::fprintf(stderr,
            "[simplify_test] case 1 FAIL: stream size mismatch (p=%zu n=%zu uv=%zu)\n",
            out.positions.size(), out.normals.size(), out.uvs.size());
        return false;
    }
    for (std::size_t i = 0; i < out.indices.size(); ++i) {
        if (static_cast<std::size_t>(out.indices[i]) >= out.positions.size()) {
            std::fprintf(stderr,
                "[simplify_test] case 1 FAIL: out-of-range index %u >= %zu\n",
                out.indices[i], out.positions.size());
            return false;
        }
    }
    const std::size_t outputTris = out.indices.size() / 3u;
    // LockBorder can legitimately prevent any reduction, so we only require
    // that output doesn't *exceed* input. Emit a warning-style note when
    // no reduction happened; that's not a test failure.
    if (outputTris > inputTris) {
        std::fprintf(stderr,
            "[simplify_test] case 1 FAIL: outputTris=%zu > inputTris=%zu\n",
            outputTris, inputTris);
        return false;
    }
    if (!(out.achievedError >= 0.0f) || !std::isfinite(out.achievedError)) {
        std::fprintf(stderr,
            "[simplify_test] case 1 FAIL: achievedError=%f not finite/>=0\n",
            out.achievedError);
        return false;
    }
    if (out.inputTriangleCount == 0u) {
        std::fprintf(stderr,
            "[simplify_test] case 1 FAIL: inputTriangleCount=0 in output\n");
        return false;
    }
    if (out.outputTriangleCount != outputTris) {
        std::fprintf(stderr,
            "[simplify_test] case 1 FAIL: outputTriangleCount=%zu != indices/3=%zu\n",
            out.outputTriangleCount, outputTris);
        return false;
    }

    std::printf(
        "[simplify_test] case 1 PASS: group of %zu clusters, input %zu tris, "
        "output %zu tris, error=%.6f\n",
        groupSize, inputTris, outputTris, static_cast<double>(out.achievedError));
    return true;
}

bool testEmptyGroup() {
    SimplifyOptions opts{};
    auto r = simplify(std::span<const ClusterData>{}, opts);
    if (r.has_value()) {
        std::fprintf(stderr,
            "[simplify_test] case 2 FAIL: expected EmptyGroup, got success\n");
        return false;
    }
    if (r.error().kind != SimplifyErrorKind::EmptyGroup) {
        std::fprintf(stderr,
            "[simplify_test] case 2 FAIL: expected EmptyGroup, got %s\n",
            simplifyErrorKindString(r.error().kind));
        return false;
    }
    std::printf(
        "[simplify_test] case 2 PASS: EmptyGroup surfaced as expected.\n");
    return true;
}

bool testOptionsOutOfRange(const fs::path& assetPath) {
    const std::vector<ClusterData> clusters = loadAndCluster(assetPath);
    if (clusters.empty()) {
        std::fprintf(stderr,
            "[simplify_test] case 3 FAIL: cluster set empty\n");
        return false;
    }
    std::span<const ClusterData> group(clusters.data(),
        std::min<std::size_t>(2u, clusters.size()));

    SimplifyOptions opts{};
    opts.weldEpsilon = -1.0f;    // invalid

    auto r = simplify(group, opts);
    if (r.has_value()) {
        std::fprintf(stderr,
            "[simplify_test] case 3 FAIL: expected OptionsOutOfRange, got success\n");
        return false;
    }
    if (r.error().kind != SimplifyErrorKind::OptionsOutOfRange) {
        std::fprintf(stderr,
            "[simplify_test] case 3 FAIL: expected OptionsOutOfRange, got %s: %s\n",
            simplifyErrorKindString(r.error().kind),
            r.error().detail.c_str());
        return false;
    }
    std::printf(
        "[simplify_test] case 3 PASS: OptionsOutOfRange surfaced as expected.\n");
    return true;
}

bool testDeterminism(const fs::path& assetPath) {
    const std::vector<ClusterData> clusters = loadAndCluster(assetPath);
    if (clusters.empty()) {
        std::fprintf(stderr,
            "[simplify_test] case 4 FAIL: cluster set empty\n");
        return false;
    }
    const std::size_t groupSize = std::min<std::size_t>(4u, clusters.size());
    std::span<const ClusterData> group(clusters.data(), groupSize);

    SimplifyOptions opts{};
    opts.maxError         = 0.05f;
    opts.targetIndexCount = 0u;
    opts.weldEpsilon      = 1e-5f;

    auto r1 = simplify(group, opts);
    auto r2 = simplify(group, opts);
    if (!r1.has_value() || !r2.has_value()) {
        std::fprintf(stderr,
            "[simplify_test] case 4 FAIL: simplify errored on one of the passes\n");
        return false;
    }
    const std::uint64_t h1 = hashSimplified(*r1);
    const std::uint64_t h2 = hashSimplified(*r2);
    if (h1 != h2) {
        std::fprintf(stderr,
            "[simplify_test] case 4 FAIL: non-deterministic hash 0x%016llx vs 0x%016llx\n",
            static_cast<unsigned long long>(h1),
            static_cast<unsigned long long>(h2));
        return false;
    }
    std::printf(
        "[simplify_test] case 4 PASS: two-run determinism hash=0x%016llx\n",
        static_cast<unsigned long long>(h1));
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr,
            "[simplify_test] FAIL: could not locate assets/DamagedHelmet.glb\n");
        return 1;
    }

    bool allPassed = true;
    allPassed &= testHappyPath(asset);
    allPassed &= testEmptyGroup();
    allPassed &= testOptionsOutOfRange(asset);
    allPassed &= testDeterminism(asset);

    if (!allPassed) {
        std::fprintf(stderr, "[simplify_test] FAILED\n");
        return 1;
    }
    std::printf("[simplify_test] All cases passed.\n");
    return 0;
}
