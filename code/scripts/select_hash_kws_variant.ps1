#requires -Version 5.1
<#
.SYNOPSIS
  Switches the active hash_kws_runtime model bundle to one of the ensemble
  variants (ens_a / ens_b / ens_c) before flashing a board.

.DESCRIPTION
  The training notebook writes one firmware bundle per ensemble variant into
  code/firmware/hash_kws_runtime_<variant>/. The Arduino sketch always
  compiles `code/firmware/hash_kws_runtime/`. This script copies the chosen
  variant's hash_model_data.{cpp,h} + hash_model_settings.{cpp,h} +
  hash_model_export_metadata.json + hash_model_types.h on top of the active
  runtime, so the next IDE build picks it up without further edits.

.PARAMETER Variant
  ens_a, ens_b, or ens_c.

.PARAMETER DryRun
  Print the planned copies without writing anything.

.EXAMPLE
  pwsh -File code\scripts\select_hash_kws_variant.ps1 -Variant ens_b
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
$source = Join-Path $repoRoot ("code\firmware\hash_kws_runtime_$Variant")
$target = Join-Path $repoRoot 'code\firmware\hash_kws_runtime'

if (-not (Test-Path $source)) {
    throw "Variant directory not found: $source. Run the ensemble notebook first."
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
Write-Host "Active hash_kws_runtime now reflects variant: $Variant"
Write-Host "Reflash with Arduino IDE (Tools menu unchanged)."
