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
Write-Host "请先在 3090 上启动：python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port $RemotePort"
Write-Host ""

if (Test-LocalTcpPort -HostName "127.0.0.1" -Port $LocalPort) {
    Write-Warning "Windows 本地 127.0.0.1:$LocalPort 已被占用。若它不是旧隧道，请改用 -LocalPort 8001。"
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
    Write-Warning "没有找到私钥 $IdentityFile，将回退到 SSH 配置或密码登录。"
}

$sshArgs += "${SshUser}@${SshHost}"

Write-Host "正在建立隧道；请保持本窗口开启。按 Ctrl+C 停止。" -ForegroundColor Green
Write-Host "隧道建立后，Windows 网页端使用：ws://127.0.0.1:$LocalPort/ws" -ForegroundColor Green
Write-Host ""

& ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "SSH 隧道退出，代码：$LASTEXITCODE。请检查 SSH 主机、端口、私钥和远程 3090 服务。"
}
