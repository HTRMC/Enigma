# METIS (vendored)

Graph-partitioning library. Consumed by `enigma-mpbake` (tools/mpbake) for DAG
clustering of meshlet adjacency graphs via `METIS_PartGraphKway`. Not linked
into the runtime `Enigma.exe` binary.

## Provenance

- **Vendored from:** https://github.com/scivision/METIS
- **Commit:** `777472ae3cd15a8e6d1e5b7d6c347d21947e3ab2`
  (branch `main`, the sole scivision mirror branch; resolved via
  `git ls-remote https://github.com/scivision/METIS HEAD`)
- **Date imported:** 2026-04-18
- **Upstream project:** https://github.com/KarypisLab/METIS — METIS version
  5.2.1.3. The scivision mirror modernizes the CMake plumbing (drops the
  upstream pre-build shell step, adds clean `IDXTYPEWIDTH`/`REALTYPEWIDTH`
  compile-definition knobs, and FetchContent-pulls a scivision-maintained
  GKlib fork in place of the upstream submodule).

## Why vendored

Previously pinned via `FetchContent_Declare(... GIT_TAG <40-char-sha>)` at the
scivision mirror's `main` head. The scivision repo has **no tagged releases**:
the only way to name a specific commit was a branch-tip SHA pin. If the mirror
maintainer force-pushed `main` to a different SHA, a fresh checkout would
silently fetch substitute source. METIS runs on attacker-controllable
glTF-derived adjacency graphs in `enigma-mpbake`, so that trust boundary was
unacceptable. Vendoring the source tree into the Enigma repo pins it
permanently to the code we audited — a mirror force-push can no longer change
what we compile.

## What was stripped

From the scivision upstream snapshot, the following directories/files were
**omitted** when copying into `vendor/metis/`:

- `.git/` — version history (re-tracked by Enigma's own git).
- `.github/` — CI workflows (scivision's GitHub Actions config).
- `CMakePresets.json` — developer-workflow presets, unused when consumed as
  an `add_subdirectory()` sub-project.
- `programs/` — the six CLI front-ends (`gpmetis`, `ndmetis`, `mpmetis`,
  `m2gmetis`, `graphchk`, `cmpfillin`). Enigma only calls the library API,
  never the CLI binaries. The corresponding `add_subdirectory(programs)`
  call in the top-level `CMakeLists.txt` has been removed; the edit is
  commented inline.

The scivision upstream `README.md` is preserved as `README.scivision.md`
alongside this file so the original build instructions stay discoverable.

## What was kept

- `CMakeLists.txt` (edited — see the "Enigma vendor edit" comment inline)
- `LICENSE` (Apache 2.0, inherited from METIS upstream)
- `README.scivision.md` (the original scivision README)
- `cmake/` (install + config helpers used by the METIS CMake)
- `conf/gkbuild.cmake` (GKlib compile-definition detection) and
  `conf/check_thread_storage.c` (MSVC TLS probe)
- `include/metis.h` (public header)
- `libmetis/` (all 39 C sources + internal headers — the actual library)

Total on-disk footprint after strip: ~655 KB.

## GKlib dependency

The top-level `CMakeLists.txt` still `FetchContent`-pulls GKlib from the
scivision mirror:

```
https://github.com/scivision/GKlib/archive/b4c45d104d421d3adb85fb900ca240eba781c95f.tar.gz
```

GKlib was **intentionally NOT vendored** in this pass — it is maintained
under the same KarypisLab umbrella as METIS, the scivision fork just
embeds it for build convenience, and the attack surface is secondary
(GKlib is pulled only when METIS builds, and METIS is offline-tool-only).
If a future audit wants to close the last mirror trust boundary, vendoring
GKlib under `vendor/gklib/` is the natural next step.

## How to re-sync to a newer scivision snapshot

This is intentionally manual (no automation): each resync is a deliberate
code-review decision, not a drive-by dependency bump.

```sh
# 1. Fetch the snapshot you want to import.
git clone https://github.com/scivision/METIS.git /tmp/metis-new
cd /tmp/metis-new
git checkout <NEW-SHA>

# 2. Nuke the current vendor tree (so stale files can't linger).
rm -rf <enigma-repo>/vendor/metis/{CMakeLists.txt,LICENSE,README.scivision.md,cmake,conf,include,libmetis}

# 3. Copy the same subset we kept (see "What was kept" above).
cp -r CMakeLists.txt LICENSE cmake conf include libmetis <enigma-repo>/vendor/metis/
mv <enigma-repo>/vendor/metis/README.md <enigma-repo>/vendor/metis/README.scivision.md

# 4. Re-apply the vendor edit to vendor/metis/CMakeLists.txt:
#       delete the line:    add_subdirectory(programs)
#    (replace with the "Enigma vendor edit ..." comment — see git blame
#    on this file for the exact wording).

# 5. Update this README's "Commit" and "Date imported" fields.
# 6. Run build_enigma.bat and the mpbake test suite; fix any diffs.
# 7. Commit with a message that calls out the upstream SHA delta.
```

Do **not** add this resync to CI. Each bump should be manually reviewed.

## License

METIS is licensed under Apache 2.0 (see `LICENSE`). Copyright notice:

> Copyright 1995-2013, Regents of the University of Minnesota
