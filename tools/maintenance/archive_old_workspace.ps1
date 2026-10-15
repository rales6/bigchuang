param([switch]$Apply)
$ErrorActionPreference = 'Stop'

# Recoverable, same-volume moves only. Never recurse-delete a project.
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot = Split-Path -Parent $projectRoot
$workspacePrefix = $workspaceRoot.TrimEnd('\') + '\'
if (!(Test-Path -LiteralPath (Join-Path $projectRoot 'docs/maintenance/source_manifest.json'))) {
    throw 'Complete and verify source collection before archiving.'
}
$names = @('car1.0', 'car1.0 - 副本', 'car', 'car_sim', 'RobotCompanionAndroid', 'CAD', 'test',
           'windows_network_proxy_scripts_fixed', 'arm_teach_model_sjtu.py', 'ble_direct_test_pi.py',
           'train_arm_mlp_sjtu.py', 'web_app_pi.py', 'git_daily_flow.txt', 'git_upload.txt')
$archiveRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot ('_archive/' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_before_monorepo')))
if (!$archiveRoot.StartsWith($workspacePrefix,[StringComparison]::OrdinalIgnoreCase) -or
    $archiveRoot.StartsWith($projectRoot + '\',[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Archive target must be inside workspace and outside the new Git repository.'
}
$plan = @()
foreach ($name in $names) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $name))
    $destination = [IO.Path]::GetFullPath((Join-Path $archiveRoot $name))
    if (!$source.StartsWith($workspacePrefix,[StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path -Parent $source) -ne $workspaceRoot -or
        !$destination.StartsWith($archiveRoot + '\',[StringComparison]::OrdinalIgnoreCase) -or
        $source -eq $projectRoot) { throw 'Unsafe source or destination.' }
    if (Test-Path -LiteralPath $source) {
        $item = Get-Item -LiteralPath $source -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Do not move reparse point: $name" }
        $plan += [pscustomobject]@{source=$source; destination=$destination; moved=$false; retained_in_place=$false; error=$null}
    }
}
if (!$Apply) { $plan | ConvertTo-Json -Depth 4; return }
if (Test-Path -LiteralPath $archiveRoot) { throw 'Archive already exists; refuse to overwrite.' }
New-Item -ItemType Directory -Path $archiveRoot | Out-Null
$journal = Join-Path $archiveRoot 'move_journal.json'
$plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $journal -Encoding utf8
foreach ($row in $plan) {
    if (Test-Path -LiteralPath $row.destination) { throw 'Archive destination collision.' }
    try {
        Move-Item -LiteralPath $row.source -Destination $row.destination
        $row.moved = $true
    } catch {
        # Do not kill editors/agents or force-delete a locked working tree.
        $row.retained_in_place = Test-Path -LiteralPath $row.source
        $row.error = $_.Exception.Message
        Write-Warning ('Retained occupied path: ' + $row.source)
    }
    $plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $journal -Encoding utf8
}
$report = [ordered]@{archive_root=$archiveRoot; permanent_deletions=$false; moves=$plan;
    original_git_history_preserved=$true; remote_data_changed=$false;
    restore='Move the desired item from its exact destination back to its original source; never overwrite a newer item.'}
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $projectRoot 'docs/maintenance/workspace_archive.json') -Encoding utf8
$report | ConvertTo-Json -Depth 6
