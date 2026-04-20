# run_screenshot_diff.ps1
# =======================
# CI gate for the screenshot golden PNG under
# tests/infra/screenshot_diff/reference/. Invoked by PR CI — fails the
# job on any visual drift (pixel-exact by default; tolerance is set
# inside the test binary).
#
# The script locates the already-built screenshot_diff_test.exe under
# any build*/ tree. On a clean checkout with no build tree this exits
# 2 so the CI job fails loudly.

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
    $hit = Get-ChildItem -Path $root.FullName -Recurse -Filter 'screenshot_diff_test.exe' `
        -ErrorAction SilentlyContinue `
        | Select-Object -First 1
    if ($hit) { $exe = $hit; break }
}

if (-not $exe) {
    Write-Error 'screenshot_diff_test.exe not built (expected under build*/ or cmake-build-*)'
    exit 2
}

Write-Host "[run_screenshot_diff] using $($exe.FullName)"
& $exe.FullName
exit $LASTEXITCODE
