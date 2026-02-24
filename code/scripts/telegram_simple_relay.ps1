$ErrorActionPreference = "Stop"

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

function Ensure-UiAssemblies {
    if (-not ("Win32" -as [type])) {
        Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll", CharSet=CharSet.Auto)]
    public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder text, int count);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
"@
    }
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    Add-Type -AssemblyName Microsoft.VisualBasic | Out-Null
}

$ErrorActionPreference = "SilentlyContinue"

$repo = "C:\Users\Danil\diploma_esp32_distributed_nn"
$python = "C:\Users\Danil\AppData\Local\Programs\Python\Python310\python.exe"
$bridge = Join-Path $repo "code\scripts\telegram_bridge.py"
$envFile = Join-Path $repo "code\scripts\telegram_simple_relay.env"
$config = Load-EnvFile $envFile

$titleSub = $config["WINDOW_TITLE_SUBSTRING"]
if ([string]::IsNullOrWhiteSpace($titleSub)) { $titleSub = "Test" }

$interval = $config["POLL_INTERVAL_SEC"]
if ([string]::IsNullOrWhiteSpace($interval)) { $interval = 2 }
$interval = [int]$interval

$triggerText = $config["TRIGGER_TEXT"]
if ([string]::IsNullOrWhiteSpace($triggerText)) { $triggerText = "Telegram: new message. Check inbox.jsonl." }

$triggerCommandsRaw = $config["TRIGGER_COMMANDS"]
if ([string]::IsNullOrWhiteSpace($triggerCommandsRaw)) { $triggerCommandsRaw = "" }
$triggerCommands = @{}
if ($triggerCommandsRaw.Length -gt 0) {
    $triggerCommandsRaw.Split(@(",", ";", " "), [System.StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object {
        $cmd = $_.Trim().ToLower()
        if ($cmd.Length -gt 0) { $triggerCommands[$cmd] = $true }
    }
}

$plainGroup = $config["TRIGGER_PLAIN_TEXT_GROUP"]
if ([string]::IsNullOrWhiteSpace($plainGroup)) { $plainGroup = "1" }

$plainDm = $config["TRIGGER_PLAIN_TEXT_DM"]
if ([string]::IsNullOrWhiteSpace($plainDm)) { $plainDm = "0" }

$pasteMode = $config["PASTE_MODE"]
if ([string]::IsNullOrWhiteSpace($pasteMode)) { $pasteMode = "ctrl_shift_v" }

$sendMode = $config["SEND_MODE"]
if ([string]::IsNullOrWhiteSpace($sendMode)) { $sendMode = "enter" }

$restoreClipboard = $config["RESTORE_CLIPBOARD"]
if ([string]::IsNullOrWhiteSpace($restoreClipboard)) { $restoreClipboard = "1" }

$postPasteDelay = $config["POST_PASTE_DELAY_MS"]
if ([string]::IsNullOrWhiteSpace($postPasteDelay)) { $postPasteDelay = 300 }
$postPasteDelay = [int]$postPasteDelay

$enterDelay = $config["ENTER_DELAY_MS"]
if ([string]::IsNullOrWhiteSpace($enterDelay)) { $enterDelay = 200 }
$enterDelay = [int]$enterDelay

$focusRetries = $config["FOCUS_RETRY_COUNT"]
if ([string]::IsNullOrWhiteSpace($focusRetries)) { $focusRetries = 3 }
$focusRetries = [int]$focusRetries

$focusDelay = $config["FOCUS_RETRY_DELAY_MS"]
if ([string]::IsNullOrWhiteSpace($focusDelay)) { $focusDelay = 150 }
$focusDelay = [int]$focusDelay

$enableCommands = $config["ENABLE_COMMANDS"]
if ([string]::IsNullOrWhiteSpace($enableCommands)) { $enableCommands = "0" }
$commandsScript = $config["COMMANDS_SCRIPT"]
if ([string]::IsNullOrWhiteSpace($commandsScript)) {
    $commandsScript = Join-Path $repo "code\scripts\telegram_command_daemon.py"
}

$statePath = Join-Path $repo "notes\Journal\telegram\simple_relay_state.json"
$bridgeStatePath = Join-Path $repo "notes\Journal\telegram\state.json"
$inboxPath = Join-Path $repo "notes\Journal\telegram\inbox.jsonl"
$logPath = Join-Path $repo "notes\Journal\telegram\simple_relay.log"
$errPath = Join-Path $repo "notes\Journal\telegram\simple_relay.err"
$heartbeatPath = Join-Path $repo "notes\Journal\telegram\simple_relay_heartbeat.txt"
$lastTriggerPath = Join-Path $repo "notes\Journal\telegram\last_trigger_chat.json"

function Load-State {
    if (!(Test-Path $statePath)) { return @{ last_update_id = 0 } }
    return Get-Content $statePath | ConvertFrom-Json
}

function Save-State($state) {
    $state | ConvertTo-Json | Set-Content -Encoding UTF8 $statePath
}

function Read-Inbox {
    if (!(Test-Path $inboxPath)) { return @() }
    $lines = Get-Content -Path $inboxPath -Encoding UTF8
    $out = @()
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try { $out += ($line | ConvertFrom-Json) } catch {}
    }
    return $out
}

function Log-Err([string]$msg) {
    $ts = (Get-Date).ToString("s")
    Add-Content -Path $errPath -Value ("[$ts] $msg") -Encoding UTF8
}

function Get-ActiveWindowTitle {
    Ensure-UiAssemblies
    try {
        $hwnd = [Win32]::GetForegroundWindow()
        if ($hwnd -eq [IntPtr]::Zero) { return "" }
        $sb = New-Object System.Text.StringBuilder 1024
        [Win32]::GetWindowText($hwnd, $sb, $sb.Capacity) | Out-Null
        return $sb.ToString()
    } catch { return "" }
}

function Send-Trigger([string]$text) {
    Ensure-UiAssemblies
    $proc = Get-Process powershell, pwsh, WindowsTerminal -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -like "*$titleSub*" } | Select-Object -First 1
    if ($null -eq $proc) {
        $proc = Get-Process | Where-Object { $_.MainWindowTitle -like "*$titleSub*" } | Select-Object -First 1
    }
    if ($null -eq $proc) {
        $ts = (Get-Date).ToString("s")
        Add-Content -Path $logPath -Value ("[$ts] window_not_found") -Encoding UTF8
        return
    }
    $focused = $false
    $activeTitle = ""
    for ($i = 0; $i -lt $focusRetries; $i++) {
        try { [Win32]::ShowWindow($proc.MainWindowHandle, 9) | Out-Null } catch {} # SW_RESTORE
        try { [Win32]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null } catch {}
        try { [Microsoft.VisualBasic.Interaction]::AppActivate($proc.Id) | Out-Null } catch {}
        Start-Sleep -Milliseconds $focusDelay
        $activeTitle = Get-ActiveWindowTitle
        if ($activeTitle -like "*$titleSub*") {
            $focused = $true
            break
        }
    }
    if (-not $focused) {
        $ts = (Get-Date).ToString("s")
        Add-Content -Path $logPath -Value ("[$ts] focus_failed: $activeTitle") -Encoding UTF8
        return
    }

    $prev = $null
    if ($restoreClipboard -ne "0") {
        try { $prev = Get-Clipboard -Raw } catch {}
    }
    try { Set-Clipboard -Value $text } catch { Log-Err "clipboard_failed"; return }

    Start-Sleep -Milliseconds 120
    switch ($pasteMode) {
        "ctrl_v"       { [System.Windows.Forms.SendKeys]::SendWait("^{V}") }
        "shift_insert" { [System.Windows.Forms.SendKeys]::SendWait("+{INSERT}") }
        "ctrl_shift_v" { [System.Windows.Forms.SendKeys]::SendWait("^+{V}") }
        default        { [System.Windows.Forms.SendKeys]::SendWait("^+{V}") }
    }
    Start-Sleep -Milliseconds $postPasteDelay

    switch ($sendMode) {
        "ctrl_enter"   { [System.Windows.Forms.SendKeys]::SendWait("^{ENTER}") }
        "alt_enter"    { [System.Windows.Forms.SendKeys]::SendWait("%{ENTER}") }
        "ctrl_d"       { [System.Windows.Forms.SendKeys]::SendWait("^{D}") }
        "enter_twice"  {
            [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
            Start-Sleep -Milliseconds $enterDelay
            [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
        }
        default        { [System.Windows.Forms.SendKeys]::SendWait("{ENTER}") }
    }

    $ts = (Get-Date).ToString("s")
    Add-Content -Path $logPath -Value ("[$ts] sent") -Encoding UTF8

    if (($restoreClipboard -ne "0") -and ($null -ne $prev)) {
        try { Set-Clipboard -Value $prev } catch {}
    }
}

while ($true) {
    try {
        try { Set-Content -Path $heartbeatPath -Value (Get-Date).ToString("s") -Encoding UTF8 } catch {}
        & $python $bridge pull | Out-Null
        if ($enableCommands -eq "1") {
            try { & $python $commandsScript once | Out-Null } catch { Log-Err "commands: $($_.Exception.Message)" }
        }
        if (!(Test-Path $bridgeStatePath)) { Start-Sleep -Seconds $interval; continue }
        $bridgeState = Get-Content $bridgeStatePath | ConvertFrom-Json
        $bridgeLast = [int]$bridgeState.last_update_id
        $state = Load-State
        $last = [int]$state.last_update_id
        if ($bridgeLast -gt $last) {
            $msgs = Read-Inbox | Where-Object { [int]$_.update_id -gt $last } | Sort-Object { [int]$_.update_id }
            $shouldTrigger = $false
            $triggerMsg = $null
            foreach ($m in $msgs) {
                $txt = [string]$m.text
                if (-not [string]::IsNullOrWhiteSpace($txt)) {
                    $chatId = 0
                    try { $chatId = [int64]$m.chat_id } catch {}
                    $isGroup = $chatId -lt 0
                    $trim = $txt.Trim()
                    if ($trim.StartsWith("/")) {
                        $cmd = [regex]::Split($trim.Substring(1), "\s+")[0]
                        if ($cmd -like "*@*") { $cmd = $cmd.Split("@")[0] }
                        $cmd = $cmd.ToLower()
                        if ($triggerCommands.ContainsKey($cmd)) {
                            $shouldTrigger = $true
                            $triggerMsg = $m
                            break
                        }
                    } else {
                        if ($isGroup -and ($plainGroup -ne "0")) {
                            $shouldTrigger = $true
                            $triggerMsg = $m
                            break
                        }
                        if ((-not $isGroup) -and ($plainDm -ne "0")) {
                            $shouldTrigger = $true
                            $triggerMsg = $m
                            break
                        }
                    }
                }
            }
            if ($shouldTrigger) {
                try {
                    if ($null -ne $triggerMsg) {
                        $obj = @{
                            chat_id = $triggerMsg.chat_id
                            update_id = $triggerMsg.update_id
                            timestamp = $triggerMsg.timestamp
                        }
                        $obj | ConvertTo-Json | Set-Content -Encoding UTF8 $lastTriggerPath
                    }
                } catch {}
                Send-Trigger $triggerText
            }
            $state.last_update_id = $bridgeLast
            Save-State $state
        }
    } catch {
        Log-Err $_.Exception.Message
    }
    Start-Sleep -Seconds $interval
}
