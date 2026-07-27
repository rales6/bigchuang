"""
兼容 MicroPython 与桌面测试的单调时钟。

该模块提供了跨平台的时间函数接口，使得同一份代码可以在：
1. MicroPython 环境（如 ESP32、树莓派 Pico）中运行
2. 桌面 Python 环境（如 PC 上的单元测试、仿真）中运行

核心思想：优先使用 MicroPython 的原生时间函数（硬件时钟），
若不存在则回退到 CPython 的标准库实现。
"""

import time


def ticks_ms():
    """
    获取当前系统运行的毫秒数（单调递增时钟）
    
    返回值：
        从系统启动或某个固定参考点开始的毫秒数（无符号整数）
    
    特性：
        - 单调递增：不受系统时间调整影响
        - 循环回绕：MicroPython 中约 49.7 天后会回绕到 0
        - 跨平台兼容：优先使用 MicroPython 原生实现
        
    注意：
        CPython 的 time.monotonic() 返回浮点数（秒），
        乘以 1000 转换为毫秒并截断为整数。
    """
    if hasattr(time, "ticks_ms"):
        # MicroPython 原生实现：硬件级单调时钟
        return time.ticks_ms()
    # CPython 回退方案：使用 monotonic 保证单调性
    return int(time.monotonic() * 1000)


def ticks_diff(new, old):
    """
    计算两个时间戳之间的差值（带回绕保护）
    
    参数：
        new: 较新的时间戳（ticks_ms 的返回值）
        old: 较旧的时间戳（ticks_ms 的返回值）
    
    返回值：
        时间差（毫秒），始终为有符号整数
        
    重要特性：
        - MicroPython 原生实现会自动处理时间戳回绕问题
        - CPython 回退方案使用简单减法（无回绕考虑）
        
    使用场景：
        elapsed_ms = ticks_diff(now, start_time)
        if elapsed_ms > TIMEOUT_MS:
            # 超时处理
            pass
    
    注意：
        在 MicroPython 中，如果 new 和 old 的间隔超过约 24.8 天，
        由于整数回绕，差值可能不准确。建议在短时间内使用。
    """
    if hasattr(time, "ticks_diff"):
        # MicroPython 原生实现：正确处理回绕
        return time.ticks_diff(new, old)
    # CPython 回退方案：直接相减（无回绕保护，但 CPython 整数无限大）
    return new - old


def ticks_add(base, delta):
    """
    在基础时间戳上增加指定的偏移量（带回绕保护）
    
    参数：
        base: 基础时间戳（ticks_ms 的返回值）
        delta: 要增加的毫秒数（正数表示未来，负数表示过去）
    
    返回值：
        新的时间戳（与 base 同类型，可能发生回绕）
    
    使用场景：
        future_time = ticks_add(now, TIMEOUT_MS)
        while ticks_diff(ticks_ms(), future_time) < 0:
            # 等待超时...
            pass
    
    注意：
        - MicroPython 原生实现会正确处理回绕边界
        - CPython 回退方案使用普通加法（无回绕概念）
        - delta 不能太大，否则在 MicroPython 中可能导致意外回绕
    """
    if hasattr(time, "ticks_add"):
        # MicroPython 原生实现：正确处理回绕
        return time.ticks_add(base, delta)
    # CPython 回退方案：直接加法
    return base + delta


def sleep_ms(value):
    """
    使当前线程休眠指定的毫秒数（阻塞式延迟）
    
    参数：
        value: 休眠时间（毫秒）
    
    使用场景：
        sleep_ms(1000)  # 等待 1 秒
        sleep_ms(50)    # 等待 50 毫秒
    
    注意：
        - 非精确延迟，实际延迟可能略长（受系统调度影响）
        - 在嵌入式系统中，长时间阻塞会浪费 CPU，应谨慎使用
        - 在 MicroPython 中，内部使用硬件定时器实现
        - 在 CPython 中，内部调用 time.sleep(秒)
    """
    if hasattr(time, "sleep_ms"):
        # MicroPython 原生实现：毫秒级延迟
        time.sleep_ms(value)
    else:
        # CPython 回退方案：转换为秒（浮点数除法）
        time.sleep(value / 1000.0)