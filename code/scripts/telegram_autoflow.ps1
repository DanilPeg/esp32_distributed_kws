$ErrorActionPreference = "Stop"

$repo = "C:\Users\Danil\diploma_esp32_distributed_nn"
$errLog = Join-Path $repo "notes\Journal\telegram\autoflow.err"
$startLog = Join-Path $repo "notes\Journal\telegram\autoflow.start"

function Log-Err {
    param([string]$Msg)
    $ts = (Get-Date).ToString("s")
    try { Add-Content -Path $errLog -Value ("[$ts] $Msg") -Encoding UTF8 } catch {}
}

function Load-EnvFile {
    param([string]$Path)
    $map = @{}
    if (!(Test-Path $Path)) { return $map }
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line.Length -eq 0) { return }
        if ($line.StartsWith("#")) { return }
        if ($line -notmatch "=") { return }
        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        $val = $parts[1].Trim().Trim("'").Trim('"')
        $map[$key] = $val
    }
    return $map
}

try { Add-Type -AssemblyName System.Windows.Forms | Out-Null } catch { Log-Err "Add-Type System.Windows.Forms failed: $($_.Exception.Message)"; exit 1 }
try { Add-Type -AssemblyName Microsoft.VisualBasic | Out-Null } catch { Log-Err "Add-Type Microsoft.VisualBasic failed: $($_.Exception.Message)"; exit 1 }
if (-not ("Win32" -as [type])) {
    try {
        Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
}
"@
    } catch {
        Log-Err "Add-Type Win32 failed: $($_.Exception.Message)"
        exit 1
    }
}

$ErrorActionPreference = "SilentlyContinue"

$python = "C:\Users\Danil\AppData\Local\Programs\Python\Python310\python.exe"
$bridge = Join-Path $repo "code\scripts\telegram_bridge.py"
$relay = Join-Path $repo "code\scripts\telegram_relay.py"
$envFile = Join-Path $repo "code\scripts\telegram_autosend.env"
$config = Load-EnvFile $envFile

$titleSub = $config["WINDOW_TITLE_SUBSTRING"]
if ([string]::IsNullOrWhiteSpace($titleSub)) {
    Log-Err "WINDOW_TITLE_SUBSTRING is not set in $envFile"
    exit 1
}

$pendingPath = $config["PENDING_FILE"]
if ([string]::IsNullOrWhiteSpace($pendingPath)) {
    $pendingPath = Join-Path $repo "notes\Journal\telegram\pending.md"
}

$sentLog = Join-Path $repo "notes\Journal\telegram\sent.log"
$sendMode = $config["SEND_MODE"]
if ([string]::IsNullOrWhiteSpace($sendMode)) { $sendMode = "auto" }
$pasteMode = $config["PASTE_MODE"]
if ([string]::IsNullOrWhiteSpace($pasteMode)) { $pasteMode = "ctrl_shift_v" }
$restoreClipboard = $config["RESTORE_CLIPBOARD"]
if ([string]::IsNullOrWhiteSpace($restoreClipboard)) { $restoreClipboard = "1" }
$runOnce = $config["RUN_ONCE"]
if ([string]::IsNullOrWhiteSpace($runOnce)) { $runOnce = $env:RUN_ONCE }
$runOnce = ($runOnce -eq "1")
$heartbeat = Join-Path $repo "notes\Journal\telegram\autoflow_heartbeat.txt"
$interval = $config["POLL_INTERVAL_SEC"]
if ([string]::IsNullOrWhiteSpace($interval)) { $interval = 5 }
$interval = [int]$interval

Write-Host "[autoflow] interval $interval s; window: $titleSub"
Write-Host "[autoflow] paste mode: $pasteMode | send mode: $sendMode"
try { Add-Content -Path $startLog -Value ("[{0}] start" -f (Get-Date).ToString("s")) -Encoding UTF8 } catch {}

while ($true) {
    try {
        try { Set-Content -Path $heartbeat -Value (Get-Date).ToString("s") -Encoding UTF8 } catch {}

        try { & $python $bridge pull | Out-Null } catch {}
        try { & $python $relay once | Out-Null } catch {}

        if (Test-Path $pendingPath) {
            $text = Get-Content -Raw -Encoding UTF8 -Path $pendingPath
            $text = $text.Trim([char]0xFEFF).Trim()
            if (-not [string]::IsNullOrWhiteSpace($text)) {
                $proc = Get-Process | Where-Object { $_.MainWindowTitle -like "*$titleSub*" } | Select-Object -First 1
                if ($null -ne $proc) {
                    $activated = $false
                    try { [Win32]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null } catch {}
                    try { $activated = [Microsoft.VisualBasic.Interaction]::AppActivate($proc.Id) } catch {}
                    if (-not $activated) {
                        $ts = (Get-Date).ToString("s")
                        Add-Content -Path $sentLog -Value ("[$ts] activate_failed title=$($proc.MainWindowTitle)") -Encoding UTF8
                    }
                    Start-Sleep -Milliseconds 150

                    $prev = $null
                    if ($restoreClipboard -ne "0") {
                        try { $prev = Get-Clipboard -Raw } catch {}
                    }
                    $setOk = $true
                    try { Set-Clipboard -Value $text } catch { $setOk = $false }
                    if (-not $setOk) {
                        $ts = (Get-Date).ToString("s")
                        Add-Content -Path $sentLog -Value ("[$ts] clipboard_failed") -Encoding UTF8
                        Start-Sleep -Seconds $interval
                        continue
                    }

                    Start-Sleep -Milliseconds 120
                    switch ($pasteMode) {
                        "ctrl_v"       { [System.Windows.Forms.SendKeys]::SendWait("^{V}") }
                        "shift_insert" { [System.Windows.Forms.SendKeys]::SendWait("+{INSERT}") }
                        "ctrl_shift_v" { [System.Windows.Forms.SendKeys]::SendWait("^+{V}") }
                        default        { [System.Windows.Forms.SendKeys]::SendWait("^+{V}") }
                    }
                    Start-Sleep -Milliseconds 180

                    $effectiveMode = $sendMode
                    if ($sendMode -eq "auto") {
                        if ($text -match "`n") { $effectiveMode = "ctrl_enter" } else { $effectiveMode = "enter" }
                    }
                    switch ($effectiveMode) {
                        "ctrl_enter" { [System.Windows.Forms.SendKeys]::SendWait("^{ENTER}") }
                        "alt_enter"  { [System.Windows.Forms.SendKeys]::SendWait("%{ENTER}") }
                        "ctrl_d"     { [System.Windows.Forms.SendKeys]::SendWait("^{D}") }
                        default      { [System.Windows.Forms.SendKeys]::SendWait("{ENTER}") }
                    }

                    $ts = (Get-Date).ToString("s")
                    Add-Content -Path $sentLog -Value ("[$ts] sent paste=$pasteMode mode=$effectiveMode title=$($proc.MainWindowTitle)") -Encoding UTF8
                    Set-Content -Path $pendingPath -Value "" -Encoding UTF8

                    if (($restoreClipboard -ne "0") -and ($null -ne $prev)) {
                        try { Set-Clipboard -Value $prev } catch {}
                    }
                } else {
                    $ts = (Get-Date).ToString("s")
                    Add-Content -Path $sentLog -Value ("[$ts] window_not_found") -Encoding UTF8
                }
            }
        }
    } catch {
        Log-Err $_.Exception.Message
    }

    if ($runOnce) { break }
    Start-Sleep -Seconds $interval
}
