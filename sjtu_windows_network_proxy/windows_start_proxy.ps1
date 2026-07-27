[CmdletBinding()]
param(
    [string]$SshHost = "202.121.181.124",
    [string]$SshUser = "sjtu",
    [int]$SshPort = 2020,
    [string]$IdentityFile = "F:/SSHKeys/sjtu_ed25519",
    [int]$LocalProxyPort = 8899,
    [int]$RemotePortStart = 8899,
    [int]$RemotePortEnd = 8999,
    [string]$ProxyUser = "raleproxy"
)

$ErrorActionPreference = "Stop"

function Get-PythonCommand {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @{ Exe = $py.Source; Prefix = @("-3") }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{ Exe = $python.Source; Prefix = @() }
    }

    throw "未找到 Windows Python。请先安装 Python 3，并确认 py 或 python 命令可用。"
}

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

function Get-SshBaseArgs {
    $args = @(
        "-T",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-p", $SshPort.ToString()
    )

    if ($IdentityFile -and (Test-Path $IdentityFile)) {
        $args += @("-i", $IdentityFile, "-o", "IdentitiesOnly=yes")
    }
    elseif ($IdentityFile) {
        Write-Warning "没有找到私钥 $IdentityFile，将回退到 SSH 配置或密码登录。"
    }

    return $args
}

function Find-FreeRemotePort {
    param(
        [int]$StartPort,
        [int]$EndPort
    )

    if ($StartPort -lt 1024 -or $EndPort -gt 65535 -or $StartPort -gt $EndPort) {
        throw "远程端口范围无效：$StartPort-$EndPort"
    }

    $remotePython = @"
import socket
start = $StartPort
end = $EndPort

for port in range(start, end + 1):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        continue
    else:
        sock.close()
        print(port)
        raise SystemExit(0)

raise SystemExit(2)
"@

    $sshArgs = Get-SshBaseArgs
    $target = "${SshUser}@${SshHost}"

    $result = $remotePython | & ssh @sshArgs $target "python3 -"
    if ($LASTEXITCODE -ne 0) {
        throw "无法在远程节点检查空闲端口。请先确认 SSH 登录正常。"
    }

    $candidate = ($result | Select-Object -Last 1).Trim()
    $port = 0
    if (-not [int]::TryParse($candidate, [ref]$port)) {
        throw "远程节点没有返回有效端口。返回内容：$result"
    }

    return $port
}

Write-Host ""
Write-Host "Windows 网络出口 -> SSH 反向隧道 -> 远程节点（自动端口版）" -ForegroundColor Cyan
Write-Host ""

if ($ProxyUser -notmatch '^[A-Za-z0-9._-]{1,32}$') {
    throw "ProxyUser 只能包含英文字母、数字、点、下划线和短横线。"
}

if (Test-LocalTcpPort -HostName "127.0.0.1" -Port $LocalProxyPort) {
    throw "Windows 本地端口 127.0.0.1:$LocalProxyPort 已被占用。请关闭旧代理，或使用 -LocalProxyPort 指定其他端口。"
}

$securePassword = Read-Host "请设置临时代理密码（8-64 位，仅字母、数字、点、下划线、短横线）" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $ProxyPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

if ($ProxyPassword -notmatch '^[A-Za-z0-9._-]{8,64}$') {
    throw "代理密码格式不符合要求。"
}

$python = Get-PythonCommand

Write-Host "检查 Windows Python 和 pip..." -ForegroundColor Yellow
& $python.Exe @($python.Prefix) -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Windows Python 没有可用的 pip。请先执行：py -m ensurepip --upgrade"
}

& $python.Exe @($python.Prefix) -m pip show pproxy *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "首次运行，正在安装 pproxy..." -ForegroundColor Yellow
    & $python.Exe @($python.Prefix) -m pip install --user pproxy
    if ($LASTEXITCODE -ne 0) {
        throw "pproxy 安装失败。"
    }
}

Write-Host "正在远程节点选择空闲端口 $RemotePortStart-$RemotePortEnd..." -ForegroundColor Yellow
$RemoteProxyPort = Find-FreeRemotePort -StartPort $RemotePortStart -EndPort $RemotePortEnd
Write-Host "已选择远程端口：127.0.0.1:$RemoteProxyPort" -ForegroundColor Green

$sshBaseArgs = Get-SshBaseArgs
$target = "${SshUser}@${SshHost}"

# 把自动选择的端口写入远程用户目录，node_proxy_v2.sh 会自动读取。
& ssh @sshBaseArgs $target "umask 077; printf '%s\n' '$RemoteProxyPort' > ~/.windows_proxy_port"
if ($LASTEXITCODE -ne 0) {
    throw "无法把远程代理端口写入 ~/.windows_proxy_port。"
}

$logDir = Join-Path $env:LOCALAPPDATA "SjtuWindowsProxy"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdoutLog = Join-Path $logDir "pproxy.stdout.log"
$stderrLog = Join-Path $logDir "pproxy.stderr.log"
Remove-Item $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue

$listenUri = "http://127.0.0.1:${LocalProxyPort}/#${ProxyUser}:$ProxyPassword"
$proxyArgs = @($python.Prefix) + @("-m", "pproxy", "-l", $listenUri, "-vv")

Write-Host "启动 Windows 本地代理 127.0.0.1:$LocalProxyPort..." -ForegroundColor Yellow
$proxyProcess = Start-Process `
    -FilePath $python.Exe `
    -ArgumentList $proxyArgs `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

try {
    $ready = $false
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 300
        if ($proxyProcess.HasExited) {
            break
        }
        if (Test-LocalTcpPort -HostName "127.0.0.1" -Port $LocalProxyPort) {
            $ready = $true
            break
        }
    }

    if (-not $ready) {
        $errorText = ""
        if (Test-Path $stderrLog) {
            $errorText = Get-Content $stderrLog -Raw -ErrorAction SilentlyContinue
        }
        throw "Windows 本地代理未成功启动。日志：$stderrLog`n$errorText"
    }

    $tunnelArgs = Get-SshBaseArgs
    $tunnelArgs += @(
        "-N",
        "-o", "ExitOnForwardFailure=yes",
        "-R", "127.0.0.1:${RemoteProxyPort}:127.0.0.1:${LocalProxyPort}",
        $target
    )

    Write-Host ""
    Write-Host "Windows 本地代理已启动：127.0.0.1:$LocalProxyPort" -ForegroundColor Green
    Write-Host "远程节点代理端口：127.0.0.1:$RemoteProxyPort" -ForegroundColor Green
    Write-Host "远程端口已经写入：~/.windows_proxy_port" -ForegroundColor Green
    Write-Host "请保持本窗口开启；在节点执行 source ~/node_proxy_v2.sh。" -ForegroundColor Green
    Write-Host "按 Ctrl+C 可停止隧道和 Windows 本地代理。" -ForegroundColor DarkYellow
    Write-Host ""

    & ssh @tunnelArgs
    if ($LASTEXITCODE -ne 0) {
        throw @"
SSH 隧道退出，代码：$LASTEXITCODE。

若仍出现 remote port forwarding failed：
1. 在节点执行：ss -lnt | grep ':$RemoteProxyPort'
2. 在 Windows 检查旧 SSH 进程：
   Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" |
     Where-Object { `$_.CommandLine -match '\-R' } |
     Select-Object ProcessId, CommandLine
3. 若多个端口都失败，可能是服务器禁止远程转发，需要管理员允许 AllowTcpForwarding。
"@
    }
}
finally {
    if ($proxyProcess -and -not $proxyProcess.HasExited) {
        Stop-Process -Id $proxyProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Windows 本地代理已停止。日志目录：$logDir" -ForegroundColor DarkYellow
}
