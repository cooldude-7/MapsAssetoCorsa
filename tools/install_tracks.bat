<# : batch part (PowerShell skips this block as a comment)
@echo off
rem Installs every drift-maps*.zip next to this file into Assetto Corsa's content\tracks.
rem Double-click it. It finds AC through Steam, or asks for the folder if it can't.
setlocal
set "DM_HERE=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-Expression ([System.IO.File]::ReadAllText('%~f0'))"
echo.
pause
exit /b
#>

$ErrorActionPreference = 'Stop'
$here = $env:DM_HERE

$zips = @(Get-ChildItem -LiteralPath $here -Filter 'drift-maps*.zip' | Sort-Object Name)
if ($zips.Count -eq 0) {
    Write-Host "No drift-maps*.zip found in $here"
    Write-Host 'Put this file in the same folder as the zip you downloaded, then run it again.'
    return
}

# Steam install folder, then every Steam library listed in libraryfolders.vdf.
$libraries = @()
foreach ($key in 'HKCU:\Software\Valve\Steam', 'HKLM:\SOFTWARE\WOW6432Node\Valve\Steam') {
    $props = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
    if ($props.SteamPath) { $libraries += $props.SteamPath }
    if ($props.InstallPath) { $libraries += $props.InstallPath }
}
foreach ($steam in @($libraries)) {
    $vdf = [IO.Path]::Combine($steam, 'steamapps', 'libraryfolders.vdf')
    if (Test-Path -LiteralPath $vdf) {
        $found = Select-String -LiteralPath $vdf -Pattern 'path\x22\s+\x22([^\x22]+)\x22' -AllMatches
        foreach ($m in $found.Matches) { $libraries += ($m.Groups[1].Value -replace '\\\\', '\') }
    }
}
$libraries += 'C:\Program Files (x86)\Steam', 'C:\Program Files\Steam', 'D:\SteamLibrary', 'E:\SteamLibrary'

$ac = $null
foreach ($lib in ($libraries | Select-Object -Unique)) {
    $candidate = [IO.Path]::Combine($lib, 'steamapps', 'common', 'assettocorsa')
    if (Test-Path -LiteralPath ([IO.Path]::Combine($candidate, 'content', 'tracks'))) { $ac = $candidate; break }
}
if (-not $ac) {
    Write-Host 'Could not find Assetto Corsa automatically.'
    Write-Host 'In Content Manager: Settings > Assetto Corsa > the folder path at the top. Copy it.'
    $ac = (Read-Host 'Paste your assettocorsa folder path here').Trim('"', ' ')
}
$tracks = [IO.Path]::Combine($ac, 'content', 'tracks')
if (-not (Test-Path -LiteralPath $tracks)) {
    Write-Host "That folder has no content\tracks inside: $ac"
    return
}

foreach ($zip in $zips) {
    Write-Host "Installing $($zip.Name) into $tracks ..."
    Expand-Archive -LiteralPath $zip.FullName -DestinationPath $tracks -Force
}
$installed = @(Get-ChildItem -LiteralPath $tracks -Directory -Filter 'dm_*' | ForEach-Object Name)
Write-Host ''
Write-Host "Done. $($installed.Count) drift-maps tracks are installed:"
$installed | ForEach-Object { Write-Host "  $_" }
Write-Host ''
Write-Host 'Restart Content Manager, then look for tracks named "DM T1 Minimal" and so on.'
