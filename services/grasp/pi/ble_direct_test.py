import asyncio
from bleak import BleakScanner, BleakClient

NUS_RX = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'
ADDR = '64:B7:08:69:9A:E6'

async def test():
    # Scan first
    print('Scanning...')
    devices = await BleakScanner.discover(timeout=5.0)
    found = None
    for d in devices:
        if d.name and ('ESP32' in d.name or 'Robot' in d.name):
            found = d
            print(f'Found: {d.name} -> {d.address}')
            break
        if d.address.upper() == ADDR.upper():
            found = d
            print(f'Found by addr: {d.name or "?"} -> {d.address}')
            break

    # Try direct connect if scan missed
    if not found:
        print('Scan missed, trying direct connect...')
        try:
            async with BleakClient(ADDR, timeout=8.0) as client:
                if client.is_connected:
                    print(f'Direct connect OK!')
        except Exception as e:
            print(f'Direct connect also failed: {e}')
            print('ESP32 is NOT reachable. Check power.')
            return

    # Connect
    print('Connecting...')
    try:
        async with BleakClient(ADDR, timeout=10.0) as client:
            print(f'Connected: {client.is_connected}')

            # Test claw
            print('Testing claw OPEN...')
            await client.write_gatt_char(NUS_RX, b'#ARMC90!\n', response=True)
            await asyncio.sleep(0.8)

            print('Testing claw CLOSE...')
            await client.write_gatt_char(NUS_RX, b'#ARMC10!\n', response=True)
            await asyncio.sleep(0.8)

            print('Testing claw OPEN...')
            await client.write_gatt_char(NUS_RX, b'#ARMC90!\n', response=True)
            await asyncio.sleep(0.8)
            print('Claw test OK!')

            # Test arm
            print('Testing arm deploy...')
            await client.write_gatt_char(NUS_RX, b'#ARMF15D15!\n', response=True)
            await asyncio.sleep(1.2)
            print('Testing arm retract up...')
            await client.write_gatt_char(NUS_RX, b'#ARMF0U15!\n', response=True)
            await asyncio.sleep(1.2)
            print('Testing arm retract back...')
            await client.write_gatt_char(NUS_RX, b'#ARMB15!\n', response=True)
            await asyncio.sleep(1.2)
            print('Arm test OK!')

            print('=== ALL TESTS PASSED ===')
    except Exception as e:
        print(f'Connection/write failed: {e}')

asyncio.run(test())
