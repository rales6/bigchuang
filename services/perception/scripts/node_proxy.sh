#!/usr/bin/env bash
# 用法：
#   source ~/node_proxy.sh
#
# Windows 端必须先运行：
#   F:\SSHKeys\windows_network_proxy_scripts_fixed\windows_start_proxy.ps1
#
# 这个脚本只负责让 3090 通过 Windows 访问外网，例如 pip、apt、模型下载。
# 如果要从 Windows 浏览器访问 3090 网页，请使用 SSH -L 本地隧道。

PROXY_HOST="127.0.0.1"
PROXY_PORT="8899"
PROXY_USER="raleproxy"

_proxy_port_open() {
    (exec 3<>"/dev/tcp/${PROXY_HOST}/${PROXY_PORT}") >/dev/null 2>&1
}

_proxy_require_enabled() {
    if [[ -z "${http_proxy:-}" || -z "${https_proxy:-}" ]]; then
        echo "代理尚未启用。请先运行：proxy_on"
        return 1
    fi
}

proxy_on() {
    if ! _proxy_port_open; then
        echo "节点上的 ${PROXY_HOST}:${PROXY_PORT} 尚未监听。"
        echo "请先在 Windows 运行 windows_start_proxy.ps1，并保持 PowerShell 窗口开启。"
        return 1
    fi

    local proxy_password
    read -r -s -p "请输入与 Windows 脚本中相同的临时代理密码：" proxy_password
    echo

    if [[ ! "${proxy_password}" =~ ^[A-Za-z0-9._-]{8,64}$ ]]; then
        echo "密码格式不符合要求。请只使用字母、数字、点、下划线和短横线。"
        unset proxy_password
        return 1
    fi

    local proxy_url="http://${PROXY_USER}:${proxy_password}@${PROXY_HOST}:${PROXY_PORT}"

    export http_proxy="${proxy_url}"
    export https_proxy="${proxy_url}"
    export HTTP_PROXY="${proxy_url}"
    export HTTPS_PROXY="${proxy_url}"
    export no_proxy="localhost,127.0.0.1,::1,192.168.55.0/24"
    export NO_PROXY="${no_proxy}"

    unset proxy_password
    unset proxy_url

    echo "当前终端已启用 Windows 网络代理：${PROXY_HOST}:${PROXY_PORT}"
}

proxy_off() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
    echo "当前终端的代理环境变量已清除。"
}

proxy_status() {
    if [[ -n "${http_proxy:-}" ]]; then
        echo "代理环境变量：已启用"
        echo "代理地址：${PROXY_HOST}:${PROXY_PORT}"
    else
        echo "代理环境变量：未启用"
    fi

    if _proxy_port_open; then
        echo "SSH 反向端口：正在监听"
    else
        echo "SSH 反向端口：未监听"
    fi
}

proxy_test() {
    _proxy_require_enabled || return 1
    echo "正在通过 Windows 测试 HTTPS..."
    wget -4 --spider --timeout=15 --tries=1 https://www.baidu.com
}

pip_install_via_windows() {
    _proxy_require_enabled || return 1
    if [[ $# -eq 0 ]]; then
        echo "用法：pip_install_via_windows Python包1 [Python包2 ...]"
        return 2
    fi
    python3 -m pip install "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "不要直接执行 bash node_proxy.sh。请使用：source ~/node_proxy.sh"
    exit 1
fi

case "${1:-on}" in
    on) proxy_on ;;
    off) proxy_off ;;
    status) proxy_status ;;
    *)
        echo "可用方式："
        echo "  source ~/node_proxy.sh"
        echo "  source ~/node_proxy.sh off"
        echo "  source ~/node_proxy.sh status"
        return 2
        ;;
esac
