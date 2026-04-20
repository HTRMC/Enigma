# run_spirv_diff.ps1
# =================
# CI gate for the SPIR-V golden tree under tests/infra/spirv_diff/golden/.
# Invoked by PR CI — fails the job on any golden drift.
#
# The script locates the already-built spirv_diff_test.exe under any
# build*/ tree (CLion users may produce cmake-build-debug/, build_enigma.bat
# produces build/, the release track produces build_release/). On a
# clean checkout with no build tree, this exits 2 and the CI job fails
# loudly so the owner adds a build step upstream.

$ErrorActionPreference = 'Stop'

# Resolve candidate build roots relative to this script's parent (the
# repo root). PowerShell's -Path <wildcard> + -Recurse combo does not
# descend INTO wildcard-matched dirs reliably on all hosts, so we
# enumerate the build roots first, then recurse inside each.
$repoRoot = Split-Path -Parent $PSScriptRoot
$candidateRoots = Get-ChildItem -Path $repoRoot -Directory `
    -ErrorAction SilentlyContinue `
    | Where-Object { $_.Name -like 'build*' -or $_.Name -like 'cmake-build-*' }

$exe = $null
foreach ($root in $candidateRoots) {
    $hit = Get-ChildItem -Path $root.FullName -Recurse -Filter 'spirv_diff_test.exe' `
        -ErrorAction SilentlyContinue `
        | Select-Object -First 1
    if ($hit) { $exe = $hit; break }
}

if (-not $exe) {
    Write-Error 'spirv_diff_test.exe not built (expected under build*/ or cmake-build-*)'
    exit 2
}

Write-Host "[run_spirv_diff] using $($exe.FullName)"
& $exe.FullName
exit $LASTEXITCODE
