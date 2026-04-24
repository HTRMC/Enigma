// MpAssetReader.cpp
// =================
// Runtime implementation of MpAssetReader. Memory-maps the `.mpa` file
// via Win32 (`CreateFileW` + `CreateFileMappingW` + `MapViewOfFile`),
// validates the header, exposes DAG + page-table spans, and decompresses
// per-page payloads on demand via zstd.
//
// No Vulkan here. Pure CPU data-access.

#include "asset/MpAssetReader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#include <zstd.h>

#include "core/Log.h"

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
// Non-Windows platforms would use mmap + open. M1 targets Windows only;
// this stub keeps the file compilable if someone drops it into a Linux
// build without first porting the mapping path.
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace enigma::asset {

const char* mpReadErrorKindString(MpReadErrorKind kind) noexcept {
    switch (kind) {
        case MpReadErrorKind::FileNotFound:           return "FileNotFound";
        case MpReadErrorKind::IoError:                return "IoError";
        case MpReadErrorKind::FileTooSmall:           return "FileTooSmall";
        case MpReadErrorKind::ValidateMagicFailed:    return "ValidateMagicFailed";
        case MpReadErrorKind::ValidateVersionFailed:  return "ValidateVersionFailed";
        case MpReadErrorKind::InvalidSectionOffsets:  return "InvalidSectionOffsets";
        case MpReadErrorKind::PageIndexOutOfRange:    return "PageIndexOutOfRange";
        case MpReadErrorKind::BufferTooSmall:         return "BufferTooSmall";
        case MpReadErrorKind::ZstdError:              return "ZstdError";
        case MpReadErrorKind::InvalidPayload:         return "InvalidPayload";
        case MpReadErrorKind::NotOpen:                return "NotOpen";
    }
    return "Unknown";
}

MpAssetReader::~MpAssetReader() {
    close();
}

MpAssetReader::MpAssetReader(MpAssetReader&& other) noexcept {
    *this = std::move(other);
}

MpAssetReader& MpAssetReader::operator=(MpAssetReader&& other) noexcept {
    if (this != &other) {
        close();
        header_             = other.header_;
        mappedBase_         = other.mappedBase_;
        mappedSize_         = other.mappedSize_;
        osFileHandle_       = other.osFileHandle_;
        osMappingHandle_    = other.osMappingHandle_;
        other.header_           = MpAssetHeader{};
        other.mappedBase_       = nullptr;
        other.mappedSize_       = 0u;
        other.osFileHandle_     = nullptr;
        other.osMappingHandle_  = nullptr;
    }
    return *this;
}

void MpAssetReader::reset_() noexcept {
    header_             = MpAssetHeader{};
    mappedBase_         = nullptr;
    mappedSize_         = 0u;
    osFileHandle_       = nullptr;
    osMappingHandle_    = nullptr;
}

void MpAssetReader::close() noexcept {
#if defined(_WIN32)
    if (mappedBase_ != nullptr) {
        UnmapViewOfFile(const_cast<u8*>(mappedBase_));
    }
    if (osMappingHandle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(osMappingHandle_));
    }
    if (osFileHandle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(osFileHandle_));
    }
#else
    if (mappedBase_ != nullptr && mappedSize_ > 0u) {
        ::munmap(const_cast<u8*>(mappedBase_), mappedSize_);
    }
    if (osFileHandle_ != nullptr) {
        ::close(static_cast<int>(reinterpret_cast<std::intptr_t>(osFileHandle_)));
    }
    (void)osMappingHandle_;
#endif
    reset_();
}

std::expected<void, MpReadError>
MpAssetReader::open(const std::filesystem::path& path) {
    // Release any previous mapping so calling open() on an already-open
    // reader does the right thing.
    close();

    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::FileNotFound,
            std::string{"file not found: "} + path.string(),
        });
    }

#if defined(_WIN32)
    HANDLE hFile = CreateFileW(
        path.wstring().c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::IoError,
            std::string{"CreateFileW failed for "} + path.string(),
        });
    }

    LARGE_INTEGER size{};
    if (!GetFileSizeEx(hFile, &size)) {
        CloseHandle(hFile);
        return std::unexpected(MpReadError{
            MpReadErrorKind::IoError,
            std::string{"GetFileSizeEx failed for "} + path.string(),
        });
    }
    const u64 fileSize = static_cast<u64>(size.QuadPart);
    if (fileSize < sizeof(MpAssetHeader)) {
        CloseHandle(hFile);
        return std::unexpected(MpReadError{
            MpReadErrorKind::FileTooSmall,
            std::string{"file size "} + std::to_string(fileSize)
                + " < MpAssetHeader size "
                + std::to_string(sizeof(MpAssetHeader)),
        });
    }

    HANDLE hMap = CreateFileMappingW(
        hFile,
        nullptr,
        PAGE_READONLY,
        0, 0,
        nullptr);
    if (hMap == nullptr) {
        CloseHandle(hFile);
        return std::unexpected(MpReadError{
            MpReadErrorKind::IoError,
            std::string{"CreateFileMappingW failed"},
        });
    }

    const void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        return std::unexpected(MpReadError{
            MpReadErrorKind::IoError,
            std::string{"MapViewOfFile failed"},
        });
    }

    mappedBase_      = static_cast<const u8*>(view);
    mappedSize_      = fileSize;
    osFileHandle_    = hFile;
    osMappingHandle_ = hMap;
#else
    const int fd = ::open(path.string().c_str(), O_RDONLY);
    if (fd < 0) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::IoError,
            std::string{"open() failed for "} + path.string(),
        });
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return std::unexpected(MpReadError{
            MpReadErrorKind::IoError,
            std::string{"fstat failed"},
        });
    }
    const u64 fileSize = static_cast<u64>(st.st_size);
    if (fileSize < sizeof(MpAssetHeader)) {
        ::close(fd);
        return std::unexpected(MpReadError{
            MpReadErrorKind::FileTooSmall,
            std::string{"file size too small"},
        });
    }
    void* view = ::mmap(nullptr, static_cast<size_t>(fileSize), PROT_READ,
        MAP_PRIVATE, fd, 0);
    if (view == MAP_FAILED) {
        ::close(fd);
        return std::unexpected(MpReadError{
            MpReadErrorKind::IoError,
            std::string{"mmap failed"},
        });
    }
    mappedBase_      = static_cast<const u8*>(view);
    mappedSize_      = fileSize;
    osFileHandle_    = reinterpret_cast<void*>(static_cast<std::intptr_t>(fd));
    osMappingHandle_ = nullptr;
#endif

    // Copy out the header so the caller can query header() without
    // needing to trust the mapping is still live.
    std::memcpy(&header_, mappedBase_, sizeof(MpAssetHeader));

    // Magic.
    for (std::size_t i = 0; i < sizeof(kMpAssetMagic); ++i) {
        if (header_.magic[i] != kMpAssetMagic[i]) {
            close();
            return std::unexpected(MpReadError{
                MpReadErrorKind::ValidateMagicFailed,
                std::string{"magic mismatch"},
            });
        }
    }
    // Version.
    if (header_.version != kMpAssetVersion) {
        close();
        return std::unexpected(MpReadError{
            MpReadErrorKind::ValidateVersionFailed,
            std::string{"version "} + std::to_string(header_.version)
                + " != " + std::to_string(kMpAssetVersion),
        });
    }
    // Section offsets must fit within the file. Check raw offsets against
    // mappedSize_ BEFORE computing derived ends — otherwise a crafted
    // header with an offset near UINT64_MAX plus a small count wraps the
    // u64 addition back to a legal-looking value, bypassing the end-of-
    // section checks and letting dagNodes() / pageTable() read at
    // (mappedBase_ + huge_offset). Attacker-supplied .mpa files must not
    // dereference arbitrary host memory.
    if (header_.dagByteOffset    > mappedSize_ ||
        header_.pagesByteOffset  > mappedSize_ ||
        header_.boundsByteOffset > mappedSize_) {
        close();
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidSectionOffsets,
            std::string{"section offset > file size: dagOff="}
                + std::to_string(header_.dagByteOffset)
                + " pagesOff=" + std::to_string(header_.pagesByteOffset)
                + " boundsOff=" + std::to_string(header_.boundsByteOffset)
                + " fileSize=" + std::to_string(mappedSize_),
        });
    }
    const u64 dagEnd = header_.dagByteOffset +
        static_cast<u64>(header_.dagNodeCount) * kMpDagNodeSize;
    const u64 pagesEnd = header_.pagesByteOffset +
        static_cast<u64>(header_.pageCount) * kMpPageEntrySize;
    if (header_.dagByteOffset < sizeof(MpAssetHeader)    ||
        header_.pagesByteOffset < dagEnd                 ||
        header_.boundsByteOffset < pagesEnd              ||
        dagEnd   > mappedSize_                           ||
        pagesEnd > mappedSize_                           ||
        header_.boundsByteOffset > mappedSize_) {
        close();
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidSectionOffsets,
            std::string{"section offsets outside file: dagEnd="}
                + std::to_string(dagEnd)
                + " pagesEnd=" + std::to_string(pagesEnd)
                + " boundsOffset=" + std::to_string(header_.boundsByteOffset)
                + " fileSize=" + std::to_string(mappedSize_),
        });
    }

    // Validate each page entry's payload offset + size fits inside the file,
    // and that its firstDagNodeIdx is a valid index into the DAG node array.
    // The index check (M1c fold-in: security-reviewer Phase-4 HIGH #3) is
    // required because M2's streaming request queue uses firstDagNodeIdx
    // directly as an array index — an out-of-range value would otherwise
    // read past the mapped DAG section.
    const auto* pageTbl = reinterpret_cast<const MpPageEntry*>(
        mappedBase_ + header_.pagesByteOffset);
    for (u32 i = 0; i < header_.pageCount; ++i) {
        // Copy the fields we need into locals BEFORE any error path calls
        // close(), which unmaps the underlying memory. Reading `pageTbl[i]`
        // after close() is a use-after-unmap → access violation. This
        // pattern protects both the payload-range check and the
        // firstDagNodeIdx bounds check.
        const u64 payloadOffset = pageTbl[i].payloadByteOffset;
        const u32 compressedSz  = pageTbl[i].compressedSize;
        const u32 firstDagIdx   = pageTbl[i].firstDagNodeIdx;
        const u32 dagCount      = header_.dagNodeCount;
        const u64 boundsOff     = header_.boundsByteOffset;
        const u64 fileSz        = mappedSize_;

        // Guard against u64 overflow on the end computation — a crafted
        // entry with payloadOffset near UINT64_MAX could wrap `end` back
        // inside the file and bypass the range check below, yielding an
        // OOB read at (mappedBase_ + payloadOffset) in fetchPage().
        const u64 end = payloadOffset + static_cast<u64>(compressedSz);
        if (payloadOffset > fileSz ||
            payloadOffset < boundsOff ||
            end > fileSz ||
            end < payloadOffset ||
            compressedSz == 0u) {
            close();
            return std::unexpected(MpReadError{
                MpReadErrorKind::InvalidSectionOffsets,
                std::string{"page "} + std::to_string(i)
                    + " payload offset/size out of range: offset="
                    + std::to_string(payloadOffset)
                    + " size=" + std::to_string(compressedSz)
                    + " fileSize=" + std::to_string(fileSz),
            });
        }
        if (firstDagIdx >= dagCount) {
            close();
            return std::unexpected(MpReadError{
                MpReadErrorKind::InvalidSectionOffsets,
                std::string{"page "} + std::to_string(i)
                    + " firstDagNodeIdx "
                    + std::to_string(firstDagIdx)
                    + " >= dagNodeCount "
                    + std::to_string(dagCount),
            });
        }
    }

    return {};
}

bool MpAssetReader::validate() const noexcept {
    if (mappedBase_ == nullptr) return false;
    for (std::size_t i = 0; i < sizeof(kMpAssetMagic); ++i) {
        if (header_.magic[i] != kMpAssetMagic[i]) {
            return false;
        }
    }
    return header_.version == kMpAssetVersion;
}

std::span<const MpDagNode> MpAssetReader::dagNodes() const noexcept {
    if (!isOpen()) return {};
    return std::span<const MpDagNode>(
        reinterpret_cast<const MpDagNode*>(
            mappedBase_ + header_.dagByteOffset),
        header_.dagNodeCount);
}

std::span<const MpPageEntry> MpAssetReader::pageTable() const noexcept {
    if (!isOpen()) return {};
    return std::span<const MpPageEntry>(
        reinterpret_cast<const MpPageEntry*>(
            mappedBase_ + header_.pagesByteOffset),
        header_.pageCount);
}

std::vector<u32> MpAssetReader::firstDagNodeIndices() const {
    // M4.5: harvest `firstDagNodeIdx` per page for the streaming subsystem's
    // GPU upload. See the header comment for why we copy rather than span
    // directly over the mmap (MpPageEntry stride is 32 B, not 4 B).
    if (!isOpen()) return {};
    const auto pages = pageTable();
    std::vector<u32> out;
    out.reserve(pages.size());
    for (const auto& p : pages) {
        out.push_back(p.firstDagNodeIdx);
    }
    return out;
}

std::expected<std::vector<MpAssetReader::RuntimeDagNode>, MpReadError>
MpAssetReader::assembleRuntimeDagNodes() const {
    if (!isOpen()) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::NotOpen,
            std::string{"reader is not open"},
        });
    }
    const auto pages = pageTable();
    const auto dag   = dagNodes();
    if (dag.empty() || pages.empty()) {
        return std::vector<RuntimeDagNode>{};
    }

    // Output vector sized for every DAG node. Zero-initialized so any gap
    // from corrupt inputs stays benign (shader treats all-zero as a degenerate
    // cluster — radius=0 frustum-culls, coneCutoff=0 cone-culls). PageWriter's
    // invariant is that sum(page.clusterCount) == dagNodeCount, so under a
    // healthy bake every slot gets overwritten below.
    std::vector<RuntimeDagNode> out(dag.size(), RuntimeDagNode{});

    // Reuse a single scratch buffer across pages (typical page ~25-80 KiB;
    // one allocation vs. N keeps the cold-cache cost minimal).
    std::vector<u8> scratch;

    // Defence-in-depth counters — fire only on corrupt/non-canonical .mpa
    // inputs. We still clamp locally so runtime behaviour stays sane, but
    // the summary warning below surfaces the occurrence so asset issues
    // are not silently masked.
    u64 parentMonotonicClampCount = 0;
    u64 corruptParentIndexCount   = 0;
    u64 nonFiniteParentCenterCount = 0;
    u32 firstMonoChildIdx  = UINT32_MAX;
    u32 firstMonoParentIdx = UINT32_MAX;
    f32 firstMonoChildErr  = 0.0f;
    f32 firstMonoParentErr = 0.0f;
    u32 firstMonoChildLod  = 0u;
    u64 perLodViolations[16] = {};
    // Pairing-integrity check: page's cluster[i] is paired with dag[firstDagIdx+i]
    // by positional correspondence. If PageWriter produced non-contiguous
    // nodeIndices for a page, this pairing is broken and we'd render one
    // cluster's triangles at another cluster's bounds. Detect by comparing
    // boundsSphere which PageWriter writes into BOTH on-disk structures
    // from the same source cluster.
    u64 boundsMismatchCount = 0;
    u32 firstMismatchPage   = UINT32_MAX;
    u32 firstMismatchI      = UINT32_MAX;

    for (u32 pageId = 0; pageId < pages.size(); ++pageId) {
        auto view = fetchPage(pageId, scratch);
        if (!view.has_value()) {
            // Propagate the specific failure — mirrors the reader's other
            // error paths; the caller logs + continues without the buffer.
            return std::unexpected(view.error());
        }

        const MpPageEntry& entry = pages[pageId];
        const u32 firstDagIdx    = entry.firstDagNodeIdx;
        const u32 clusterCount   = view->clusterCount;

        // Bounds check: firstDagIdx + clusterCount must fit inside dagNodeCount.
        // open() already validated firstDagIdx < dagNodeCount; guard the tail
        // as defense-in-depth against a crafted `clusterCount`.
        if (static_cast<u64>(firstDagIdx) + static_cast<u64>(clusterCount) >
            static_cast<u64>(dag.size())) {
            return std::unexpected(MpReadError{
                MpReadErrorKind::InvalidPayload,
                std::string{"page "} + std::to_string(pageId)
                    + " firstDagIdx+clusterCount overruns DAG: "
                    + std::to_string(firstDagIdx) + "+"
                    + std::to_string(clusterCount) + " > "
                    + std::to_string(dag.size()),
            });
        }

        for (u32 i = 0; i < clusterCount; ++i) {
            const ClusterOnDisk& cd = view->clusters[i];
            const u32 globalIdx     = firstDagIdx + i;
            RuntimeDagNode& n       = out[globalIdx];

            // m0: center.xyz + radius. Source: ClusterOnDisk.boundsSphere.
            // open() already validated finite-ness on ClusterOnDisk floats,
            // so no NaN sanitize needed here.
            n.m0[0] = cd.boundsSphere[0];
            n.m0[1] = cd.boundsSphere[1];
            n.m0[2] = cd.boundsSphere[2];
            n.m0[3] = cd.boundsSphere[3];
            // m1: coneApex.xyz + coneCutoff.
            n.m1[0] = cd.coneApex[0];
            n.m1[1] = cd.coneApex[1];
            n.m1[2] = cd.coneApex[2];
            n.m1[3] = cd.coneCutoff;
            // m2: coneAxis.xyz + asfloat(packed pageId|lodLevel).
            n.m2[0] = cd.coneAxis[0];
            n.m2[1] = cd.coneAxis[1];
            n.m2[2] = cd.coneAxis[2];
            // Pack: low 24 bits pageId, high 8 bits lodLevel. Matches the
            // unpack in mp_cluster_cull.comp.hlsl::loadDagNode.
            const u32 packed = (pageId & 0x00FFFFFFu)
                             | ((cd.dagLodLevel & 0xFFu) << 24u);
            std::memcpy(&n.m2[3], &packed, sizeof(u32));

            // m3: screen-space-error traversal metadata.
            //   m3[0] = this cluster's maxSimplificationError (world units)
            //   m3[1] = parent cluster's maxSimplificationError, or FLT_MAX
            //           for roots so the cull shader's "errParent > threshold"
            //           test always passes for root clusters. Looked up via
            //           the on-disk DAG's parentGroupId (see DagBuilder.cpp:
            //           parentGroupId encodes the global DAG index of the
            //           parent cluster — not a METIS group id — via
            //           groupFirstNewCluster[gid] = firstNewIdx).
            const f32 selfError = cd.maxSimplificationError;
            n.m3[0] = selfError;
            const MpDagNode& onDisk = dag[globalIdx];

            // Pairing integrity: ClusterOnDisk and MpDagNode are written from
            // the SAME source cluster by PageWriter; their bounds must match.
            // A mismatch means this page's i-th cluster did NOT correspond to
            // dag[firstDagIdx + i] — page nodeIndices were non-contiguous and
            // the reader is silently pairing the wrong triangles with the
            // wrong bounds/parent. This manifests visually as "meshes
            // connected to the wrong things" at runtime.
            {
                const f32 eps = 1e-5f;
                const bool boundsMatch =
                    std::fabs(cd.boundsSphere[0] - onDisk.boundsSphere[0]) < eps &&
                    std::fabs(cd.boundsSphere[1] - onDisk.boundsSphere[1]) < eps &&
                    std::fabs(cd.boundsSphere[2] - onDisk.boundsSphere[2]) < eps &&
                    std::fabs(cd.boundsSphere[3] - onDisk.boundsSphere[3]) < eps;
                if (!boundsMatch) {
                    if (boundsMismatchCount == 0u) {
                        firstMismatchPage = pageId;
                        firstMismatchI    = i;
                    }
                    ++boundsMismatchCount;
                }
            }
            f32 parentError;
            // M4-fix: parentCenter is the group-coherent LOD anchor. Every
            // child in a DAG group shares the same parent, so projecting the
            // SSE test at the parent's bounds centre guarantees siblings flip
            // LOD together — eliminates cracks + per-frame blink-in/out.
            // Per-disk MpDagNode carries `center[3]` (float) for the cluster's
            // own bounds, so the parent lookup reuses the same dag[] lookup
            // already used for parentError.
            f32 parentCx = cd.boundsSphere[0];
            f32 parentCy = cd.boundsSphere[1];
            f32 parentCz = cd.boundsSphere[2];
            if (onDisk.parentGroupId == UINT32_MAX) {
                parentError = std::numeric_limits<f32>::max();
                // Root: no coarser parent. parentCenter stays at own centre
                // so the SSE projection stays finite; the cull shader's
                // `errParent > threshold` half of the rule is trivially
                // satisfied by +inf, so the root emits iff errSelf <=
                // threshold.
            } else if (onDisk.parentGroupId < dag.size()) {
                parentError = dag[onDisk.parentGroupId].maxError;
                // Monotonic-up-the-DAG invariant (DagBuilder is supposed to
                // enforce `parent.maxError >= every child.maxError`). If
                // the immediate parent violates it, walk up the DAG until
                // we find an ancestor whose error IS strictly greater than
                // selfError — that ancestor is the correct "next coarser"
                // cluster for the SSE rule
                //     accept = (errSelf <= thr) && (errParent > thr)
                // to have a real acceptance window at some camera distance.
                // If no such ancestor exists (chain ends at a root before
                // the invariant recovers), promote to root (+inf) so the
                // cluster emits under the standard Nanite root rule (iff
                // errSelf <= threshold).
                if (!std::isfinite(parentError) || parentError < selfError) {
                    if (parentMonotonicClampCount == 0u) {
                        firstMonoChildIdx  = globalIdx;
                        firstMonoParentIdx = onDisk.parentGroupId;
                        firstMonoChildErr  = selfError;
                        firstMonoParentErr = parentError;
                        firstMonoChildLod  = cd.dagLodLevel;
                    }
                    if (cd.dagLodLevel < 16u) {
                        ++perLodViolations[cd.dagLodLevel];
                    }
                    ++parentMonotonicClampCount;

                    u32 ancestorIdx = onDisk.parentGroupId;
                    f32 recoveredErr = std::numeric_limits<f32>::max();
                    for (u32 hop = 0; hop < 32u; ++hop) {
                        const MpDagNode& a = dag[ancestorIdx];
                        if (std::isfinite(a.maxError) && a.maxError > selfError) {
                            recoveredErr = a.maxError;
                            break;
                        }
                        if (a.parentGroupId == UINT32_MAX ||
                            a.parentGroupId >= dag.size()) {
                            // Hit a root without recovering the invariant.
                            // Emit under the root rule.
                            recoveredErr = std::numeric_limits<f32>::max();
                            break;
                        }
                        ancestorIdx = a.parentGroupId;
                    }
                    parentError = recoveredErr;
                }
                // Parent bounds centre lookup. On-disk MpDagNode stores
                // centre as boundsSphere[0..2] (boundsSphere[3] = radius).
                const MpDagNode& parentNode = dag[onDisk.parentGroupId];
                const f32 pcx = parentNode.boundsSphere[0];
                const f32 pcy = parentNode.boundsSphere[1];
                const f32 pcz = parentNode.boundsSphere[2];
                if (std::isfinite(pcx) && std::isfinite(pcy) && std::isfinite(pcz)) {
                    parentCx = pcx;
                    parentCy = pcy;
                    parentCz = pcz;
                } else {
                    // Non-finite parent centre (crafted / corrupt asset):
                    // keep parentCenter at the cluster's own centre so the
                    // projection stays finite, and record for the summary.
                    ++nonFiniteParentCenterCount;
                }
            } else {
                // parentGroupId out of range — corrupt asset. Treat as a
                // root (FLT_MAX) so the cluster still passes the cull
                // shader's SSE rule when errSelf <= threshold, and record
                // for the summary warning.
                parentError = std::numeric_limits<f32>::max();
                ++corruptParentIndexCount;
            }
            n.m3[1] = parentError;
            n.m3[2] = 0.0f;
            n.m3[3] = 0.0f;

            // m4: parentCenter.xyz — group-coherent LOD anchor (M4-fix).
            n.m4[0] = parentCx;
            n.m4[1] = parentCy;
            n.m4[2] = parentCz;
            n.m4[3] = 0.0f;
        }
    }

    if (parentMonotonicClampCount != 0u ||
        corruptParentIndexCount != 0u ||
        nonFiniteParentCenterCount != 0u) {
        ENIGMA_LOG_WARN(
            "[mpa] assembleRuntimeDagNodes: defence-in-depth fallbacks fired "
            "(parent-monotonic-clamp={}, corrupt-parent-index={}, "
            "non-finite-parent-center={}) — .mpa asset may be corrupt or "
            "from an older baker",
            parentMonotonicClampCount,
            corruptParentIndexCount,
            nonFiniteParentCenterCount);
    }
    if (parentMonotonicClampCount != 0u) {
        ENIGMA_LOG_WARN(
            "[mpa] first monotonic violation: child_idx={}, parent_idx={}, "
            "child_lod={}, child_err={}, parent_err={}. Per-lod violations: "
            "L0={} L1={} L2={} L3={} L4={} L5={} L6={} L7={} L8={} L9={}",
            firstMonoChildIdx, firstMonoParentIdx, firstMonoChildLod,
            firstMonoChildErr, firstMonoParentErr,
            perLodViolations[0], perLodViolations[1], perLodViolations[2],
            perLodViolations[3], perLodViolations[4], perLodViolations[5],
            perLodViolations[6], perLodViolations[7], perLodViolations[8],
            perLodViolations[9]);
    }
    if (boundsMismatchCount != 0u) {
        ENIGMA_LOG_ERROR(
            "[mpa] cluster/dag pairing broken: boundsSphere mismatched on {} "
            "clusters (first page={}, local-idx={}). PageWriter produced "
            "non-contiguous nodeIndices; triangles are being rendered with "
            "the wrong cluster's bounds/parent. Re-bake against a fixed "
            "PageWriter that sorts each page's nodeIndices contiguously, or "
            "adds a per-cluster global index to ClusterOnDisk.",
            boundsMismatchCount,
            firstMismatchPage,
            firstMismatchI);
    }

    return out;
}

std::expected<PageView, MpReadError>
MpAssetReader::fetchPage(u32 pageId, std::vector<u8>& outDecompressedBuffer) const {
    if (!isOpen()) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::NotOpen,
            std::string{"reader is not open"},
        });
    }
    if (pageId >= header_.pageCount) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::PageIndexOutOfRange,
            std::string{"page id "} + std::to_string(pageId)
                + " >= pageCount " + std::to_string(header_.pageCount),
        });
    }

    // Fix E: named pointer matching the pattern used in the open() validation
    // loop, avoids an unparenthesized subscript on a cast expression.
    const MpPageEntry* pageTbl = reinterpret_cast<const MpPageEntry*>(
        mappedBase_ + header_.pagesByteOffset);
    const MpPageEntry& entry = pageTbl[pageId];

    // Fix A: guard against memory-exhaustion from disk corruption or version
    // skew. A u32 decompressedSize of 0xFFFFFFFF would cause a 4 GB resize.
    if (entry.decompressedSize == 0u || entry.decompressedSize > kMpMaxPageDecompressedBytes) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidPayload,
            std::string{"decompressedSize "} + std::to_string(entry.decompressedSize)
                + " exceeds cap " + std::to_string(kMpMaxPageDecompressedBytes),
        });
    }
    outDecompressedBuffer.resize(entry.decompressedSize);

    const std::size_t decompressedSz = ZSTD_decompress(
        outDecompressedBuffer.data(), outDecompressedBuffer.size(),
        mappedBase_ + entry.payloadByteOffset, entry.compressedSize);
    if (ZSTD_isError(decompressedSz)) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::ZstdError,
            std::string{"ZSTD_decompress failed: "}
                + ZSTD_getErrorName(decompressedSz),
        });
    }
    if (decompressedSz != entry.decompressedSize) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::ZstdError,
            std::string{"decompressed size mismatch: got "}
                + std::to_string(decompressedSz)
                + " expected " + std::to_string(entry.decompressedSize),
        });
    }

    // Validate payload structure.
    if (outDecompressedBuffer.size() < sizeof(PagePayloadHeader)) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidPayload,
            std::string{"page "} + std::to_string(pageId)
                + " payload smaller than header",
        });
    }
    PagePayloadHeader ppHdr{};
    std::memcpy(&ppHdr, outDecompressedBuffer.data(), sizeof(PagePayloadHeader));
    if (ppHdr.version != kMpPagePayloadVersion) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidPayload,
            std::string{"payload version "} + std::to_string(ppHdr.version)
                + " != " + std::to_string(kMpPagePayloadVersion),
        });
    }
    if (ppHdr.clusterCount != entry.clusterCount) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidPayload,
            std::string{"payload clusterCount "} + std::to_string(ppHdr.clusterCount)
                + " != entry " + std::to_string(entry.clusterCount),
        });
    }

    // Fix F: guard against multiplication overflow before computing arrayBytes.
    // kMaxClustersPerPage is generous (typical is ~8); the u32 product
    // ppHdr.clusterCount * kClusterOnDiskSize would overflow at ~56M clusters.
    static constexpr u32 kMaxClustersPerPage = 4096u;
    if (ppHdr.clusterCount > kMaxClustersPerPage) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidPayload,
            std::string{"clusterCount "} + std::to_string(ppHdr.clusterCount)
                + " exceeds per-page cap " + std::to_string(kMaxClustersPerPage),
        });
    }
    const std::size_t arrayBytes = static_cast<std::size_t>(ppHdr.clusterCount) * kClusterOnDiskSize;
    if (sizeof(PagePayloadHeader) + arrayBytes > outDecompressedBuffer.size()) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidPayload,
            std::string{"ClusterOnDisk array overruns payload"},
        });
    }

    const auto* clusterArray = reinterpret_cast<const ClusterOnDisk*>(
        outDecompressedBuffer.data() + sizeof(PagePayloadHeader));

    // Derive vertex / triangle blob extents. The vertex blob starts right
    // after the cluster descriptor array; the triangle blob starts after
    // the concatenated vertex bytes. Compute by summing the per-cluster
    // vertex / triangle counts from the ClusterOnDisk descriptors, and
    // enforce per-cluster caps + aggregate caps.
    //
    // Per-cluster bounds validation (M1c fold-in: security-reviewer Phase-4
    // HIGH #2 + MEDIUM #4). The aggregate-blob check alone is insufficient
    // for adversarial inputs: a crafted file can keep aggregate bytes sane
    // while pointing an individual cluster's vertexOffset / triangleOffset
    // far outside the blob, causing out-of-range reads in consumers. We
    // also cap vertex/triangle counts at 128 to match the baker's per-
    // cluster limits (M1 plan §3.M1) so a bogus count field cannot force
    // a huge allocation via totalVertexBytes.
    static constexpr u32 kMpMaxVertsPerCluster    = 128u;
    static constexpr u32 kMpMaxTrianglesPerCluster = 128u;
    // Security MEDIUM-1: reject NaN/Inf in cluster bounds/cone floats. These
    // feed the GPU cull shader's frustum + backface tests where non-finite
    // values produce undefined comparisons — residency requests or draws can
    // leak through and destabilise the streaming queue. The baker never
    // emits non-finite values; this is defence-in-depth against a crafted or
    // corrupted .mpa.
    auto isFiniteF = [](f32 v) { return std::isfinite(v); };
    std::size_t totalVertexBytes   = 0u;
    std::size_t totalTriangleBytes = 0u;
    for (u32 i = 0; i < ppHdr.clusterCount; ++i) {
        const auto& c = clusterArray[i];
        if (c.vertexCount > kMpMaxVertsPerCluster ||
            c.triangleCount > kMpMaxTrianglesPerCluster) {
            return std::unexpected(MpReadError{
                MpReadErrorKind::InvalidPayload,
                std::string{"cluster "} + std::to_string(i)
                    + " exceeds per-cluster caps: vertexCount="
                    + std::to_string(c.vertexCount)
                    + " triangleCount=" + std::to_string(c.triangleCount),
            });
        }
        if (!isFiniteF(c.boundsSphere[0]) || !isFiniteF(c.boundsSphere[1]) ||
            !isFiniteF(c.boundsSphere[2]) || !isFiniteF(c.boundsSphere[3]) ||
            !isFiniteF(c.coneApex[0])     || !isFiniteF(c.coneApex[1])    ||
            !isFiniteF(c.coneApex[2])     ||
            !isFiniteF(c.coneAxis[0])     || !isFiniteF(c.coneAxis[1])    ||
            !isFiniteF(c.coneAxis[2])     ||
            !isFiniteF(c.coneCutoff)) {
            return std::unexpected(MpReadError{
                MpReadErrorKind::InvalidPayload,
                std::string{"cluster "} + std::to_string(i)
                    + " contains NaN/Inf in bounds/cone floats",
            });
        }
        totalVertexBytes   += static_cast<std::size_t>(c.vertexCount) * kMpVertexStride;
        totalTriangleBytes += static_cast<std::size_t>(c.triangleCount) * 3u;
    }

    const std::size_t vertexBlobStart   = sizeof(PagePayloadHeader) + arrayBytes;
    const std::size_t triangleBlobStart = vertexBlobStart + totalVertexBytes;
    const std::size_t triangleBlobEnd   = triangleBlobStart + totalTriangleBytes;
    if (triangleBlobEnd > outDecompressedBuffer.size()) {
        return std::unexpected(MpReadError{
            MpReadErrorKind::InvalidPayload,
            std::string{"vertex/triangle blobs overrun payload: v="}
                + std::to_string(totalVertexBytes)
                + " t=" + std::to_string(totalTriangleBytes)
                + " buffer=" + std::to_string(outDecompressedBuffer.size()),
        });
    }

    // Per-cluster offset+extent validation. Use u64 arithmetic to avoid
    // wraparound when a crafted offset is near UINT32_MAX. The aggregate
    // totals above already proved the blobs fit inside the decompressed
    // buffer; here we prove each cluster's slice fits inside its blob.
    const u64 totalVertexBytes64   = static_cast<u64>(totalVertexBytes);
    const u64 totalTriangleBytes64 = static_cast<u64>(totalTriangleBytes);
    for (u32 i = 0; i < ppHdr.clusterCount; ++i) {
        const auto& c = clusterArray[i];
        const u64 vEnd = static_cast<u64>(c.vertexOffset)
                       + static_cast<u64>(c.vertexCount) * kMpVertexStride;
        const u64 tEnd = static_cast<u64>(c.triangleOffset)
                       + static_cast<u64>(c.triangleCount) * 3ull;
        if (vEnd > totalVertexBytes64 || tEnd > totalTriangleBytes64) {
            return std::unexpected(MpReadError{
                MpReadErrorKind::InvalidPayload,
                std::string{"cluster "} + std::to_string(i)
                    + " slice out of range: vEnd=" + std::to_string(vEnd)
                    + "/" + std::to_string(totalVertexBytes64)
                    + " tEnd=" + std::to_string(tEnd)
                    + "/" + std::to_string(totalTriangleBytes64),
            });
        }
    }

    PageView view{};
    view.clusters     = std::span<const ClusterOnDisk>(clusterArray, ppHdr.clusterCount);
    view.vertexBlob   = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(outDecompressedBuffer.data() + vertexBlobStart),
        totalVertexBytes);
    view.triangleBlob = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(outDecompressedBuffer.data() + triangleBlobStart),
        totalTriangleBytes);
    view.clusterCount = ppHdr.clusterCount;
    view.pageId       = pageId;
    return view;
}

} // namespace enigma::asset
