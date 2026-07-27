"""电脑通过 USB-TTL 直接手动控制 ESP32，无需树莓派。

接线：
    USB-TTL TX -> ESP32 GPIO21（UART1 RX）
    USB-TTL RX <- ESP32 GPIO22（UART1 TX）
    USB-TTL GND -- ESP32 GND

ESP32 的原生 USB 口可继续用于 REPL/烧录；本工具使用独立 USB-TTL 的串口。

用法：
    python tools/manual_command_console.py COM7
    python3 tools/manual_command_console.py /dev/ttyUSB0
"""

# 从项目根目录运行（包括 Thonny 的本地 Python 解释器）时，必须带 tools 包名。
# 直接在 tools 目录运行时保留第二种导入方式。
try:
    from tools.pi_command_console import main
except ImportError:
    try:
        from pi_command_console import main
    except ImportError:
        raise ImportError(
            "电脑端工具无法加载。若 Thonny 使用 MicroPython (ESP32) 解释器，"
            "请改用 ESP32 根目录下的 thonny_manual_test.py"
        )


if __name__ == "__main__":
    main()
