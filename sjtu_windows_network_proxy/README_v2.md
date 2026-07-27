# Windows 网络代理脚本 v2

本版本专门处理 `remote port forwarding failed for listen port 8899`。旧版把 Windows 本地代理端口和远程监听端口都固定为 `8899`；一旦旧 SSH 隧道尚未退出，或者远程节点已有程序占用该端口，新隧道就会失败。

v2 的改进是：

1. Windows 本地代理默认仍使用 `127.0.0.1:8899`。
2. 启动前自动到远程节点的 `8899-8999` 范围寻找空闲端口。
3. 自动把选中的端口写入节点的 `~/.windows_proxy_port`。
4. `node_proxy_v2.sh` 自动读取该文件，不再需要手工同步修改端口。
5. 代理和反向转发仍只绑定 `127.0.0.1`。

## 使用

Windows PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\windows_start_proxy.ps1
```

保持窗口开启。另一个 PowerShell 上传节点脚本：

```powershell
scp -P 2020 .\node_proxy_v2.sh sjtu@202.121.181.124:~/
```

节点端：

```bash
chmod 700 ~/node_proxy_v2.sh
source ~/node_proxy_v2.sh
proxy_status
proxy_test
```

更新并安装：

```bash
apt_update_via_windows
apt_install_via_windows python3-pip python3.10-venv
```

## 立即判断当前错误原因

节点端检查固定端口是否被占用：

```bash
ss -lnt | grep ':8899'
```

Windows 检查可能遗留的 SSH 隧道进程：

```powershell
Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" |
  Where-Object { $_.CommandLine -match '\-R' } |
  Select-Object ProcessId, CommandLine
```

只停止确认属于旧代理隧道的进程：

```powershell
Stop-Process -Id <进程号>
```

不要一次终止全部 `ssh.exe`，否则 VS Code Remote-SSH 也可能断开。

若 v2 在多个空闲端口上仍然全部报 `remote port forwarding failed`，问题通常不是端口冲突，而是 SSH 服务端禁止远程转发。需要管理员检查 SSH 服务配置中的：

```text
AllowTcpForwarding
DisableForwarding
PermitListen
```

远程转发至少需要允许 TCP forwarding，并允许在 `127.0.0.1` 上监听用户端口。
