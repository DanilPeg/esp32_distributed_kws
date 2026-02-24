param(
    [string]$PromptFile = "",
    [string]$TaskId = "",
    [string]$OutFile = "",
    [string]$LogFile = ""
)

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

function Escape-PSString {
    param([string]$Value)
    if ($null -eq $Value) { return "" }
    return $Value.Replace("'", "''")
}

$repo = "C:\Users\Danil\diploma_esp32_distributed_nn"
$envFile = Join-Path $PSScriptRoot "telegram_commands.env"
$cfg = Load-EnvFile $envFile

$title = $cfg["CODEX_WINDOW_TITLE"]
if ([string]::IsNullOrWhiteSpace($title)) { $title = "Test" }

$codexCmdRaw = $cfg["CODEX_COMMAND"]
if ([string]::IsNullOrWhiteSpace($codexCmdRaw)) { $codexCmdRaw = "codex" }

$codexArgs = $cfg["CODEX_ARGS"]
if ([string]::IsNullOrWhiteSpace($codexArgs)) { $codexArgs = "" }

# Allow "codex --flag" in CODEX_COMMAND if CODEX_ARGS not set
$codexCmd = $codexCmdRaw
if ($codexCmdRaw -match "\s" -and [string]::IsNullOrWhiteSpace($codexArgs)) {
    $parts = $codexCmdRaw -split "\s+"
    if ($parts.Length -gt 0) {
        $codexCmd = $parts[0]
        if ($parts.Length -gt 1) {
            $codexArgs = ($parts | Select-Object -Skip 1) -join " "
        }
    }
}

$cmdCheck = Get-Command $codexCmd -ErrorAction SilentlyContinue
if ($null -eq $cmdCheck) {
    Write-Host "error: codex command not found: $codexCmd"
    exit 1
}

# Session cap (total windows with title prefix)
$maxSessionsRaw = $cfg["CODEX_MAX_SESSIONS"]
$maxSessions = 3
if (-not [string]::IsNullOrWhiteSpace($maxSessionsRaw)) {
    [int]$maxSessions = $maxSessionsRaw
}
$titlePrefix = $cfg["CODEX_WINDOW_TITLE_PREFIX"]
if ([string]::IsNullOrWhiteSpace($titlePrefix)) { $titlePrefix = $title }

$existingAll = Get-Process | Where-Object { $_.MainWindowTitle -like "*$titlePrefix*" }
if ($existingAll.Count -ge $maxSessions) {
    Write-Host "limit_reached: $($existingAll.Count)/$maxSessions"
    exit 0
}

# Avoid starting duplicate exact title (add suffix if needed)
$existingExact = Get-Process | Where-Object { $_.MainWindowTitle -eq $title } | Select-Object -First 1
if ($null -ne $existingExact) {
    $title = "$title #$($existingAll.Count + 1)"
}
if (-not [string]::IsNullOrWhiteSpace($TaskId)) {
    $title = "$title [$TaskId]"
}

$noExit = $true
if (-not [string]::IsNullOrWhiteSpace($PromptFile)) {
    $noExit = $false
}

$script = @'
$Host.UI.RawUI.WindowTitle = '__TITLE__'
Set-Location '__REPO__'
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$promptPath = '__PROMPT__'
$logPath = '__LOG__'
$outPath = '__OUT__'
$taskId = '__TASK__'
$codexCmd = '__CODEX__'
$codexArgs = '__ARGS__'
if ([string]::IsNullOrWhiteSpace($promptPath)) {
    & $codexCmd $codexArgs
} else {
    if (-not [string]::IsNullOrWhiteSpace($logPath)) {
        "codex exec start: $(Get-Date -Format s)" | Set-Content -Path $logPath -Encoding UTF8
        "prompt=$promptPath" | Add-Content -Path $logPath -Encoding UTF8
        "out=$outPath" | Add-Content -Path $logPath -Encoding UTF8
        "args=$codexArgs" | Add-Content -Path $logPath -Encoding UTF8
    }
    $argsList = @('exec')
    if (-not [string]::IsNullOrWhiteSpace($codexArgs)) {
        $argsList += ($codexArgs -split '\s+' | Where-Object { $_ -ne '' })
    }
    $argsList += @('-C', '__REPO__')
    if (-not [string]::IsNullOrWhiteSpace($outPath)) {
        $argsList += @('--output-last-message', $outPath)
    }
    $argsList += '-'
    if (-not [string]::IsNullOrWhiteSpace($logPath)) {
        Get-Content -Raw -Path $promptPath | & $codexCmd @argsList 2>&1 | Tee-Object -FilePath $logPath | Out-Default
    } else {
        Get-Content -Raw -Path $promptPath | & $codexCmd @argsList 2>&1 | Out-Default
    }
    if (-not [string]::IsNullOrWhiteSpace($taskId)) {
        python code\agent\executor.py complete $taskId --status auto_closed
    }
}
'@

$script = $script.Replace("__TITLE__", (Escape-PSString $title))
$script = $script.Replace("__REPO__", (Escape-PSString $repo))
$script = $script.Replace("__PROMPT__", (Escape-PSString $PromptFile))
$script = $script.Replace("__LOG__", (Escape-PSString $LogFile))
$script = $script.Replace("__OUT__", (Escape-PSString $OutFile))
$script = $script.Replace("__TASK__", (Escape-PSString $TaskId))
$script = $script.Replace("__CODEX__", (Escape-PSString $codexCmd))
$script = $script.Replace("__ARGS__", (Escape-PSString $codexArgs))

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$encoded = [Convert]::ToBase64String($bytes)

$args = @()
if ($noExit) { $args += '-NoExit' }
$args += @('-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand',$encoded)
Start-Process -FilePath powershell -ArgumentList $args | Out-Null
Write-Host "started"
