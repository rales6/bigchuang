"""临时探针: 直连 ESP32 BLE NUS, 订阅 TX 通知, 发 #ARM 指令并打印固件回复。

用法: python probe_ble.py '#ARMD1!' '#PING!' ...
默认序列: 微小方向测试 + 前伸阶梯 + HOME。必须先停 web_app 释放 BLE 连接。
"""
import asyncio
import sys
import time

ADDR = "64:B7:08:69:9A:E6"
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"


def _log(tag, data):
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: {data!r}", flush=True)


async def main():
    cmds = sys.argv[1:] or [
        "#PING!",
        "#ARMD1!", "#ARMB1!", "#ARMF1!", "#ARML1!", "#ARMR1!", "#ARMU1!",
        "#HOME!",
        "#ARMD30!", "#ARMD70!", "#ARMD100!", "#ARMD150!", "#ARMD200!",
        "#HOME!",
    ]
    client = BleakClient(ADDR, timeout=10)
    await client.connect()
    print(f"connected: {client.is_connected}", flush=True)

    def cb(_handle, data):
        _log("RX", bytes(data))

    await client.start_notify(NUS_TX, cb)
    for cmd in cmds:
        _log("TX", cmd)
        try:
            await client.write_gatt_char(
                NUS_RX, (cmd + "\n").encode(), response=True)
        except Exception as exc:
            _log("TX-ERR", repr(exc))
        await asyncio.sleep(2.2)
    await asyncio.sleep(2.0)
    await client.stop_notify(NUS_TX)
    await client.disconnect()
    print("done", flush=True)


if __name__ == "__main__":
    from bleak import BleakClient  # 延迟导入, 便于单测其它部分
    asyncio.run(main())
