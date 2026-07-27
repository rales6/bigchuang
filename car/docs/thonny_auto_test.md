# Thonny 自动化测试流程

## 卡死原因

旧版 `thonny_manual_test.py` 使用 `_thread` 在后台推进控制，并在 Thonny 已返回
提示符后输出 `accepted:`。Thonny 与 MicroPython 之间使用 raw-REPL 控制字符，
后台异步输出可能插入该控制流，所以会看到 `>`、提示符错乱或下一条命令无响应。

新版改成单线程同步测试：Shell 调用尚未结束时持续执行 `app.step()`，测试结束后
才输出结果并返回提示符。生产固件 `main.py` 仍然使用正常的实时主循环，不受影响。

## 使用步骤

1. 将 `car/` 中的内容上传到 ESP32 根目录，并保留 `core/`、`drivers/`、
   `protocol/`、`transport/` 子目录。
2. 在 Thonny 中停止自动运行的 `main.py`。
3. 在 Shell 输入：

   ```python
   import thonny_manual_test as test
   test.run_auto_test("io")
   ```

4. IO 测试正常后，让四轮悬空，运行：

   ```python
   test.run_auto_test("drive")
   ```

5. 清空机械臂活动范围，运行：

   ```python
   test.run_auto_test("arm")
   ```

6. 全部单项正常后可以运行：

   ```python
   test.run_auto_test("all")
   ```

7. 测试结束释放 UART：

   ```python
   test.stop()
   ```

## 直接 UART 硬件隔离测试

如果自动测试显示驱动层已写出但硬件仍不动，可绕过所有速度/机械臂控制逻辑，
直接发送与原程序相同的字符串：

```python
test.raw_motor_test(150, 400)       # 四轮悬空
test.raw_servo_test(0, 1700, 500)   # 机械臂周围无障碍
```

Shell 会打印真实发送内容，例如：

```text
raw TX: b'#006P1650T0400!#007P1350T0400!#008P1650T0400!#009P1350T0400!'
raw TX: b'#000P1700T0500!'
```

若这两项仍无动作，问题已经不在 V2 协议、运动学或命令分发，应检查 UART2 TX17
到控制板 RX、115200 8-N-1、共地、控制板/电机/舵机电源以及通道 ID。

如果需要强制中止，按 Thonny Stop/Ctrl+C。测试函数捕获 KeyboardInterrupt 后会
先取消底盘、机械臂、蜂鸣器和灯光，再把中断返回给 Thonny。
