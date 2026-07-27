[CmdletBinding()]
param(
    [string]$SshHost = "202.121.181.124",
    [string]$SshUser = "sjtu",
    [int]$SshPort = 2020,
    [string]$IdentityFile = "F:/SSHKeys/sjtu_ed25519",
    [int]$LocalPort = 8000,
    [string]$RemoteHost = "127.0.0.1",
    [int]$RemotePort = 8000
)

$ErrorActionPreference = "Stop"

function Test-LocalTcpPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutMs = 700
    )

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        if (-not $task.Wait($TimeoutMs)) {
            return $false
        }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

Write-Host ""
Write-Host "Windows 127.0.0.1:$LocalPort -> SSH -> 3090 ${RemoteHost}:$RemotePort" -ForegroundColor Cyan
Write-Host "Start the 3090 server first:"
Write-Host "python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port $RemotePort"
Write-Host ""

if (Test-LocalTcpPort -HostName "127.0.0.1" -Port $LocalPort) {
    Write-Warning "Windows local 127.0.0.1:$LocalPort is already in use. If it is not an old tunnel, use -LocalPort 8001."
}

$sshArgs = @(
    "-N", "-T",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-p", $SshPort.ToString(),
    "-L", "127.0.0.1:${LocalPort}:${RemoteHost}:${RemotePort}"
)

if ($IdentityFile -and (Test-Path $IdentityFile)) {
    $sshArgs += @(
        "-i", $IdentityFile,
        "-o", "IdentitiesOnly=yes"
    )
}
elseif ($IdentityFile) {
    Write-Warning "Identity file not found: $IdentityFile. Falling back to SSH config or password login."
}

$sshArgs += "${SshUser}@${SshHost}"

Write-Host "Opening SSH tunnel. Keep this window open. Press Ctrl+C to stop." -ForegroundColor Green
Write-Host "After the tunnel is ready, use ws://127.0.0.1:$LocalPort/ws in the Windows web console." -ForegroundColor Green
Write-Host ""

& ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "SSH tunnel exited with code $LASTEXITCODE. Check SSH host, port, identity file, and the remote 3090 service."
}
