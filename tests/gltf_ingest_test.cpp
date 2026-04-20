// Unit test for the offline `enigma-mpbake` glTF ingest stage (M1b.1).
// Exercises the three main branches of GltfIngest::load:
//   1) success on assets/DamagedHelmet.glb — expect populated mesh POD.
//   2) FileNotFound on a bogus path.
//   3) GltfParseFailed on an invalid glTF blob written to a temp file.
//
// Plain main, printf output, exit 0 on pass — mirrors M0a's
// `micropoly_capability_test.cpp` style. Built under /W4 /WX.

#include "mpbake/GltfIngest.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

using enigma::mpbake::GltfIngest;
using enigma::mpbake::IngestError;
using enigma::mpbake::IngestErrorKind;
using enigma::mpbake::IngestedMesh;
using enigma::mpbake::errorKindString;

namespace {

// Walk upward from argv[0]'s directory looking for `assets/DamagedHelmet.glb`,
// then try cwd as a fallback. Returns empty path if not found.
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

bool testDamagedHelmet(const fs::path& assetPath) {
    GltfIngest ingest;
    auto r = ingest.load(assetPath);
    if (!r.has_value()) {
        std::fprintf(stderr,
            "[gltf_ingest_test] case 1 FAIL: expected success on '%s', got %s: %s\n",
            assetPath.string().c_str(),
            errorKindString(r.error().kind),
            r.error().detail.c_str());
        return false;
    }
    const IngestedMesh& m = *r;
    if (m.positions.empty()) {
        std::fprintf(stderr, "[gltf_ingest_test] case 1 FAIL: positions empty\n");
        return false;
    }
    if (m.normals.size() != m.positions.size()) {
        std::fprintf(stderr,
            "[gltf_ingest_test] case 1 FAIL: normals (%zu) != positions (%zu)\n",
            m.normals.size(), m.positions.size());
        return false;
    }
    if (m.uvs.size() != m.positions.size()) {
        std::fprintf(stderr,
            "[gltf_ingest_test] case 1 FAIL: uvs (%zu) != positions (%zu)\n",
            m.uvs.size(), m.positions.size());
        return false;
    }
    if (m.indices.empty() || (m.indices.size() % 3u) != 0u) {
        std::fprintf(stderr,
            "[gltf_ingest_test] case 1 FAIL: indices.size() = %zu (expected > 0, mod 3 == 0)\n",
            m.indices.size());
        return false;
    }
    for (std::size_t i = 0; i < m.indices.size(); ++i) {
        if (m.indices[i] >= m.positions.size()) {
            std::fprintf(stderr,
                "[gltf_ingest_test] case 1 FAIL: indices[%zu]=%u >= positions.size()=%zu\n",
                i, m.indices[i], m.positions.size());
            return false;
        }
    }
    std::printf(
        "[gltf_ingest_test] case 1 PASS: DamagedHelmet.glb -> %zu vertices, %zu triangles\n",
        m.positions.size(), m.indices.size() / 3u);
    return true;
}

bool testFileNotFound() {
    GltfIngest ingest;
    fs::path bogus{ "C:/definitely/does/not/exist_enigma_mpbake_test.gltf" };
    auto r = ingest.load(bogus);
    if (r.has_value()) {
        std::fprintf(stderr,
            "[gltf_ingest_test] case 2 FAIL: expected FileNotFound, got success\n");
        return false;
    }
    if (r.error().kind != IngestErrorKind::FileNotFound) {
        std::fprintf(stderr,
            "[gltf_ingest_test] case 2 FAIL: expected FileNotFound, got %s\n",
            errorKindString(r.error().kind));
        return false;
    }
    std::printf("[gltf_ingest_test] case 2 PASS: FileNotFound surfaced as expected.\n");
    return true;
}

// Write some bytes that look like a .gltf (JSON) but are syntactically
// invalid — tests the GltfParseFailed branch without needing a real asset.
bool testParseFailure() {
    std::error_code ec;
    fs::path tmp = fs::temp_directory_path(ec);
    if (ec) {
        std::fprintf(stderr, "[gltf_ingest_test] case 3 SKIP: no temp dir\n");
        return true;
    }
    tmp /= "enigma_mpbake_bogus.gltf";
    {
        std::ofstream f{ tmp, std::ios::binary | std::ios::trunc };
        if (!f) {
            std::fprintf(stderr, "[gltf_ingest_test] case 3 SKIP: cannot write temp file\n");
            return true;
        }
        // Not valid glTF JSON — parser should reject.
        const char junk[] = "this is not a glTF file {{{{";
        f.write(junk, sizeof(junk) - 1);
    }

    GltfIngest ingest;
    auto r = ingest.load(tmp);
    fs::remove(tmp, ec);

    if (r.has_value()) {
        std::fprintf(stderr,
            "[gltf_ingest_test] case 3 FAIL: expected parse failure, got success\n");
        return false;
    }
    // Any of GltfParseFailed / AccessorTypeMismatch would also be acceptable,
    // but fastgltf surfaces this as GltfParseFailed via the Error enum.
    if (r.error().kind != IngestErrorKind::GltfParseFailed) {
        std::fprintf(stderr,
            "[gltf_ingest_test] case 3 FAIL: expected GltfParseFailed, got %s: %s\n",
            errorKindString(r.error().kind),
            r.error().detail.c_str());
        return false;
    }
    std::printf("[gltf_ingest_test] case 3 PASS: GltfParseFailed surfaced as expected.\n");
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr,
            "[gltf_ingest_test] FAIL: could not locate assets/DamagedHelmet.glb\n");
        return 1;
    }

    bool allPassed = true;
    allPassed &= testDamagedHelmet(asset);
    allPassed &= testFileNotFound();
    allPassed &= testParseFailure();

    if (!allPassed) {
        std::fprintf(stderr, "[gltf_ingest_test] FAILED\n");
        return 1;
    }
    std::printf("[gltf_ingest_test] All cases passed.\n");
    return 0;
}
