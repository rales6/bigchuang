import asyncio
from bleak import BleakScanner, BleakClient
ADDR = '64:B7:08:69:9A:E6'
NUS_RX = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'

async def test():
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        if d.name and ('ESP32' in d.name or 'Robot' in d.name):
            print(f'SCAN: {d.name} -> {d.address}')
    try:
        async with BleakClient(ADDR, timeout=8.0) as c:
            if c.is_connected:
                print('DIRECT CONNECT OK')
                await c.write_gatt_char(NUS_RX, b'#ARMC90!\n', response=True)
                print('CLAW OK')
    except Exception as e:
        print(f'FAIL: {e}')

asyncio.run(test())
