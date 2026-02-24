$ErrorActionPreference = "SilentlyContinue"

$repo = "C:\Users\Danil\diploma_esp32_distributed_nn"
$pidPath = Join-Path $repo "notes\Journal\telegram\daemon_pids.json"

if (Test-Path $pidPath) {
    $pids = Get-Content $pidPath | ConvertFrom-Json
    foreach ($p in $pids) {
        try { Stop-Process -Id $p.Id -Force } catch {}
    }
}

Write-Host "Stopped bridge/relay/autosend (if running)."
