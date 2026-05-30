#requires -Version 5.1
<#
.SYNOPSIS
  v4 variant selector. Same as select_hash_kws_variant.ps1 but operates on the
  ISOLATED v4 firmware tree (code/firmware_v4/) so the WORKED_ENSEMBLE baseline
  under code/firmware/ is never touched.

.DESCRIPTION
  Copies the chosen v4 ensemble variant's model payload on top of the active
  v4 runtime code/firmware_v4/hash_kws_runtime/, which the v4 node sketch
  (code/firmware_v4/micro_speech_v4/micro_speech/) compiles via ../../hash_kws_runtime/.
  The runtime's hash_kws_runner.cpp already carries the v4 wire-fix (true logits
  on the wire); this script only swaps the model files, never the runner.

.PARAMETER Variant
  ens_a (node 1), ens_b (node 2), or ens_c (node 3).

.EXAMPLE
  pwsh -File code\scripts\select_hash_kws_variant_v4.ps1 -Variant ens_b
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('ens_a','ens_b','ens_c')]
    [string] $Variant,

    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$source = Join-Path $repoRoot ("code\firmware_v4\hash_kws_runtime_$Variant")
$target = Join-Path $repoRoot 'code\firmware_v4\hash_kws_runtime'

if (-not (Test-Path $source)) {
    throw "Variant directory not found: $source"
}
if (-not (Test-Path $target)) {
    throw "Target runtime directory missing: $target"
}

$payload = @(
    'hash_model_data.cpp',
    'hash_model_data.h',
    'hash_model_settings.cpp',
    'hash_model_settings.h',
    'hash_model_types.h',
    'hash_model_export_metadata.json'
)

foreach ($file in $payload) {
    $src = Join-Path $source $file
    $dst = Join-Path $target $file
    if (-not (Test-Path $src)) {
        Write-Warning "Missing: $src (skipping)"
        continue
    }
    if ($DryRun) {
        Write-Host "DRY: copy $src -> $dst"
    } else {
        Copy-Item -Path $src -Destination $dst -Force
        Write-Host "copied $file"
    }
}

Write-Host ""
Write-Host "Active v4 hash_kws_runtime now reflects variant: $Variant"
Write-Host "Runner wire-fix (true logits) is already in place. Reflash with Arduino IDE."
