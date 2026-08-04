<#
.SYNOPSIS
    Publishes rsi_kernel_sd to GitHub as a SINGLE commit, erasing all previous history.
.USAGE
    .\Publish.ps1
    .\Publish.ps1 -Message "describe the change"
#>

param(
    [string]$Message = "Bybit RSI Reversal Bot"
)

$ErrorActionPreference = "Continue"
$src    = $PSScriptRoot
$remote = "https://github.com/ProperProperProper/bybit-rsi-reversal-bot.git"

# Stage files in a temp directory so we never touch the stf_suite git repo
$tmp = "$env:TEMP\rsi_reversal_pub_$(Get-Random)"
New-Item -ItemType Directory $tmp | Out-Null

# Copy everything except build cache, runtime data (.keys), executables, and pycache
$exclude = @("build", "dist", "data", "__pycache__",
             "bybit_rsi_reversal_bt.exe",
             "bybit_rsi_reversal_live_trader.exe",
             "bybit_rsi_reversal_paper_trader.exe")

Get-ChildItem $src | Where-Object {
    $_.Name -notin $exclude -and $_.Name -notmatch '\.exe(\.old)?$'
} | Copy-Item -Destination $tmp -Recurse

Set-Location $tmp
git init -b main 2>$null
git config user.email "ProperProperProper@users.noreply.github.com"
git config user.name "ProperProperProper"
git config core.autocrlf false
git add -A 2>$null

# Verify .keys never slipped in
$keys = Get-ChildItem -Path $tmp -Filter ".keys" -Recurse -Force
if ($keys) {
    Remove-Item $tmp -Recurse -Force
    throw "ABORT: .keys file found in publish tree - remove it and retry"
}

git commit -m $Message 2>$null
$token = (& gh auth token 2>$null)
if ($token) {
    git remote add origin "https://ProperProperProper:$token@github.com/ProperProperProper/bybit-rsi-reversal-bot.git"
} else {
    git remote add origin $remote
}
git push --force origin main 2>&1 | Where-Object { $_ -notmatch "^remote: warning" }

Set-Location $src
Remove-Item $tmp -Recurse -Force

Write-Host ""
Write-Host "Published to bybit-rsi-reversal-bot. Single commit: $Message" -ForegroundColor Green
