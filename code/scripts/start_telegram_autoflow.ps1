$ErrorActionPreference = "Stop"

$repo = "C:\Users\Danil\diploma_esp32_distributed_nn"
$python = "C:\Users\Danil\AppData\Local\Programs\Python\Python310\python.exe"
$logdir = Join-Path $repo "notes\Journal\telegram"
if (!(Test-Path $logdir)) { New-Item -ItemType Directory -Force -Path $logdir | Out-Null }

$bridge = Join-Path $repo "code\scripts\telegram_bridge.py"
$relay = Join-Path $repo "code\scripts\telegram_relay.py"
$autosend = Join-Path $repo "code\scripts\telegram_autosend.ps1"

$bridgeLog = Join-Path $logdir "bridge.log"
$relayLog = Join-Path $logdir "relay.log"
$autosendLog = Join-Path $logdir "autosend.log"
$bridgeErr = Join-Path $logdir "bridge.err"
$relayErr = Join-Path $logdir "relay.err"
$autosendErr = Join-Path $logdir "autosend.err"

@($bridgeLog, $relayLog, $autosendLog, $bridgeErr, $relayErr, $autosendErr) | ForEach-Object {
    if (Test-Path $_) { Clear-Content -Path $_ -Force }
}

$p1 = Start-Process -FilePath $python -ArgumentList "`"$bridge`" watch --interval 5" -WindowStyle Minimized -RedirectStandardOutput $bridgeLog -RedirectStandardError $bridgeErr -PassThru
$p2 = Start-Process -FilePath $python -ArgumentList "`"$relay`" --print watch --interval 5" -WindowStyle Minimized -RedirectStandardOutput $relayLog -RedirectStandardError $relayErr -PassThru
$p3 = Start-Process -FilePath "powershell" -ArgumentList "-NoProfile -STA -ExecutionPolicy Bypass -File `"$autosend`"" -WindowStyle Minimized -PassThru

$pidPath = Join-Path $logdir "daemon_pids.json"
@(
    @{ Id = $p1.Id; ProcessName = $p1.ProcessName; Role = "bridge" },
    @{ Id = $p2.Id; ProcessName = $p2.ProcessName; Role = "relay" },
    @{ Id = $p3.Id; ProcessName = $p3.ProcessName; Role = "autosend" }
) | ConvertTo-Json | Set-Content -Path $pidPath -Encoding UTF8

Write-Host "Started bridge/relay/autosend."
