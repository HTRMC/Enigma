// enigma-mpbake
// =============
// Command-line driver for the offline Micropoly Asset bake tool.
//
// Usage
// -----
//   enigma-mpbake <input.gltf> -o <out.mpa>
//                 [--target-cluster-tris N]
//                 [--deterministic-seed 0xHHHH]
//                 [-h | --help]
//
// M1b.4 scope: full GltfIngest + ClusterBuilder + DagBuilder pipeline.
// PageWriter is still a stub; this driver runs ingest + clustering + DAG
// build and prints summaries for all three stages.
//
// Exit codes:
//   0  — full bake pipeline succeeded (M1c: ingest + cluster + DAG + page write)
//   2  — argument parse error (missing -o, unknown flag, etc.)
//   3  — GltfIngest::load failed (missing file, unsupported content, etc.)
//   4  — ClusterBuilder::build failed (empty input, overflow, meshopt fail)
//   6  — DagBuilder::build failed (adjacency, METIS, simplify, re-cluster)
//   7  — PageWriter::write failed (zstd, io, invariant)

#include "ClusterBuilder.h"
#include "DagBuilder.h"
#include "GltfIngest.h"
#include "PageWriter.h"

#include "asset/MpAssetFormat.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace {

// Parsed command-line state. Zero-initialized defaults match plan §3.M1.
struct CliOptions {
    std::filesystem::path inputGltf;
    std::filesystem::path outputMpa;
    std::uint32_t         targetClusterTris    = 128;
    std::uint32_t         deterministicSeed    = 0xBEEFu;
    // DagBuildOptions::rootClusterThreshold default is 4 (plan §3.M1). Real
    // assets with disconnected mesh components or strong boundary locks can
    // stall at N roots > 4; raising the threshold lets the bake accept those
    // as roots rather than failing with MaxDepthExceeded.
    std::uint32_t         rootClusterThreshold = 4u;
    bool                  showHelp             = false;
};

void print_usage(std::FILE* out) {
    std::fprintf(out,
        "enigma-mpbake — offline Micropoly Asset (.mpa) bake tool\n"
        "\n"
        "Usage:\n"
        "  enigma-mpbake <input.gltf> -o <out.mpa> [options]\n"
        "\n"
        "Options:\n"
        "  -o <path>                       Output .mpa path (required)\n"
        "  --target-cluster-tris <N>       Triangles per leaf cluster (default 128)\n"
        "  --deterministic-seed <0xHHHH>   RNG seed for METIS (default 0xBEEF)\n"
        "  --root-cluster-threshold <N>    Stop DAG reduction at N or fewer roots (default 4).\n"
        "                                  Raise for multi-component assets that stall at a\n"
        "                                  higher residual count.\n"
        "  -h, --help                      Show this message and exit\n"
        "\n"
        "Scaffold build — M1a. Baking is not yet implemented; run on any input\n"
        "and the tool will print a scaffold notice and exit with status 0.\n");
}

// Parse a base-10 or base-16 (`0x...`) unsigned integer. Rejects negatives,
// overflow, trailing garbage. Returns true on success.
bool parse_u32(std::string_view s, std::uint32_t& out) {
    if (s.empty()) return false;
    int base = 10;
    std::size_t offset = 0;
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        base = 16;
        offset = 2;
    }
    std::uint64_t acc = 0;
    for (std::size_t i = offset; i < s.size(); ++i) {
        const char c = s[i];
        int digit = -1;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (base == 16 && c >= 'a' && c <= 'f') digit = 10 + (c - 'a');
        else if (base == 16 && c >= 'A' && c <= 'F') digit = 10 + (c - 'A');
        else return false;
        acc = acc * static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(digit);
        if (acc > 0xFFFFFFFFull) return false;
    }
    out = static_cast<std::uint32_t>(acc);
    return true;
}

// Walk argv. Returns 0 on success (caller continues), 2 on parse error,
// and sets `opts.showHelp = true` when -h / --help is requested.
int parse_args(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string_view arg{ argv[i] };
        if (arg == "-h" || arg == "--help") {
            opts.showHelp = true;
            continue;
        }
        if (arg == "-o") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: -o requires a path argument\n");
                return 2;
            }
            opts.outputMpa = argv[++i];
            continue;
        }
        if (arg == "--target-cluster-tris") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --target-cluster-tris requires a value\n");
                return 2;
            }
            if (!parse_u32(argv[++i], opts.targetClusterTris) ||
                opts.targetClusterTris == 0u) {
                std::fprintf(stderr,
                    "error: --target-cluster-tris expects a positive integer\n");
                return 2;
            }
            continue;
        }
        if (arg == "--deterministic-seed") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --deterministic-seed requires a value\n");
                return 2;
            }
            if (!parse_u32(argv[++i], opts.deterministicSeed)) {
                std::fprintf(stderr,
                    "error: --deterministic-seed expects a u32 (dec or 0x-hex)\n");
                return 2;
            }
            continue;
        }
        if (arg == "--root-cluster-threshold") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --root-cluster-threshold requires a value\n");
                return 2;
            }
            if (!parse_u32(argv[++i], opts.rootClusterThreshold) ||
                opts.rootClusterThreshold == 0u) {
                std::fprintf(stderr,
                    "error: --root-cluster-threshold expects a positive integer\n");
                return 2;
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "error: unknown flag '%.*s'\n",
                static_cast<int>(arg.size()), arg.data());
            return 2;
        }
        // Positional: first non-flag is the input path. Second+ are errors.
        if (opts.inputGltf.empty()) {
            opts.inputGltf = std::string(arg);
        } else {
            std::fprintf(stderr, "error: unexpected positional argument '%.*s'\n",
                static_cast<int>(arg.size()), arg.data());
            return 2;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    CliOptions opts;
    if (int rc = parse_args(argc, argv, opts); rc != 0) {
        print_usage(stderr);
        return rc;
    }
    if (opts.showHelp) {
        print_usage(stdout);
        return 0;
    }

    // Echo the effective config so users can see the tool is wired up.
    std::fprintf(stdout,
        "enigma-mpbake (M1c):\n"
        "  input              : %s\n"
        "  output             : %s\n"
        "  target-cluster-tris: %u\n"
        "  deterministic-seed : 0x%08X\n"
        "  asset format       : MPA%c (version %u, header %zu B, dag node %zu B)\n",
        opts.inputGltf.string().c_str(),
        opts.outputMpa.string().c_str(),
        opts.targetClusterTris,
        opts.deterministicSeed,
        enigma::asset::kMpAssetMagic[3],
        enigma::asset::kMpAssetVersion,
        sizeof(enigma::asset::MpAssetHeader),
        sizeof(enigma::asset::MpDagNode));

    // --- Stage 1: GltfIngest ---
    if (opts.inputGltf.empty()) {
        std::fprintf(stderr, "enigma-mpbake: error: missing input path\n");
        print_usage(stderr);
        return 2;
    }

    enigma::mpbake::GltfIngest ingest;
    auto loadResult = ingest.load(opts.inputGltf);
    if (!loadResult.has_value()) {
        const auto& err = loadResult.error();
        std::fprintf(stderr, "enigma-mpbake: %s: %s (%s)\n",
            enigma::mpbake::errorKindString(err.kind),
            err.detail.c_str(),
            err.path.string().c_str());
        return 3;
    }
    const enigma::mpbake::IngestedMesh& mesh = *loadResult;
    const std::size_t triangleCount = mesh.indices.size() / 3u;
    std::fprintf(stdout,
        "  ingest             : %zu vertices, %zu triangles (consolidated)\n",
        mesh.positions.size(), triangleCount);

    // --- Stage 2: ClusterBuilder (M1b.2) ---
    enigma::mpbake::ClusterBuilder clusterer;
    enigma::mpbake::ClusterBuildOptions clusterOpts{};
    clusterOpts.targetMaxTriangles = opts.targetClusterTris;
    // `targetMaxVertices` + `coneWeight` keep their defaults; only tri count
    // is exposed on the CLI today (plan §3.M1).

    auto clusterResult = clusterer.build(mesh, clusterOpts);
    if (!clusterResult.has_value()) {
        const auto& err = clusterResult.error();
        std::fprintf(stderr, "enigma-mpbake: ClusterBuild error: %s: %s\n",
            enigma::mpbake::clusterBuildErrorKindString(err.kind),
            err.detail.c_str());
        return 4;
    }
    const std::vector<enigma::mpbake::ClusterData>& clusters = *clusterResult;
    std::size_t totalClusterTris = 0;
    for (const auto& c : clusters) {
        totalClusterTris += c.triangles.size() / 3u;
    }
    const double avgClusterTris = clusters.empty() ? 0.0
        : static_cast<double>(totalClusterTris) / static_cast<double>(clusters.size());
    std::fprintf(stdout,
        "  cluster build     : %zu clusters, avg %.1f tris/cluster, %zu tris total\n",
        clusters.size(), avgClusterTris, totalClusterTris);

    // --- Stage 3 (M1b.4): DagBuilder — group -> simplify -> re-cluster loop. ---
    enigma::mpbake::DagBuilder dagBuilder;
    enigma::mpbake::DagBuildOptions dagOpts{};
    dagOpts.deterministicSeed = static_cast<std::int32_t>(opts.deterministicSeed);
    dagOpts.clusterOpts.targetMaxTriangles = opts.targetClusterTris;
    dagOpts.rootClusterThreshold = opts.rootClusterThreshold;

    auto dagResult = dagBuilder.build(
        std::span<const enigma::mpbake::ClusterData>(clusters), dagOpts);
    if (!dagResult.has_value()) {
        const auto& err = dagResult.error();
        std::fprintf(stderr, "enigma-mpbake: DagBuild error (level %u): %s: %s\n",
            err.lodLevel,
            enigma::mpbake::dagBuildErrorKindString(err.kind),
            err.detail.c_str());
        return 6;
    }
    const enigma::mpbake::DagResult& dag = *dagResult;
    std::fprintf(stdout,
        "  dag                : %u levels, %u leaves, %u roots, %zu clusters total\n",
        dag.maxLodLevel + 1u, dag.leafCount, dag.rootCount, dag.clusters.size());

    // --- Stage 4: PageWriter (M1c) — zstd compress + emit .mpa file. ---
    enigma::mpbake::PageWriter writer;
    enigma::mpbake::PageWriteOptions writeOpts{};
    auto writeResult = writer.write(dag, opts.outputMpa, writeOpts);
    if (!writeResult.has_value()) {
        const auto& err = writeResult.error();
        std::fprintf(stderr,
            "enigma-mpbake: PageWrite error: %s: %s\n",
            enigma::mpbake::pageWriteErrorKindString(err.kind),
            err.detail.c_str());
        return 7;
    }
    const auto& stats = *writeResult;
    const double ratio = stats.totalDecompressedBytes > 0u
        ? static_cast<double>(stats.totalCompressedBytes)
          / static_cast<double>(stats.totalDecompressedBytes)
        : 0.0;
    std::fprintf(stdout,
        "  page write        : %u pages, %llu bytes on disk, zstd ratio %.2f\n",
        stats.pageCount,
        static_cast<unsigned long long>(stats.fileBytes),
        ratio);

    std::fprintf(stdout,
        "enigma-mpbake: baked %s successfully (M1c complete).\n",
        opts.outputMpa.string().c_str());
    return 0;
}
