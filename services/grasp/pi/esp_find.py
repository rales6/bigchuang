import asyncio
from bleak import BleakScanner, BleakClient
NUS_RX = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'

async def s():
    devices = await BleakScanner.discover(timeout=4)
    for d in devices:
        name = d.name or ''
        if 'ESP' in name or 'Robot' in name or 'Car' in name:
            print(f'CANDIDATE: {name} -> {d.address}')
            try:
                async with BleakClient(d.address, timeout=6) as c:
                    if c.is_connected:
                        print('  CONNECTED!')
                        cmd = '#ARMC90!\n'
                        await c.write_gatt_char(NUS_RX, cmd.encode(), response=True)
                        print('  CLAW TEST OK')
            except Exception as e:
                print(f'  connect failed: {e}')
    prefixes = ['A4:CF','24:0A','30:AE','64:B7','3C:71','7C:9E','AC:67','BC:DD','FC:F5','C8:F0']
    for d in devices:
        addr = d.address.upper()
        for p in prefixes:
            if addr.startswith(p):
                print(f'ESP32-LIKE: {d.name or "?"} -> {d.address}')
                break

asyncio.run(s())
