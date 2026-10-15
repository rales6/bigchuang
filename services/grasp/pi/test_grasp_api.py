import urllib.request, json, time

# Start grasp
data = json.dumps({'target': '红色方块'}).encode()
req = urllib.request.Request('http://127.0.0.1:8080/api/grasp/start',
    data=data, headers={'Content-Type': 'application/json'}, method='POST')
resp = urllib.request.urlopen(req)
print('Start:', resp.read().decode())

# Monitor
for i in range(35):
    time.sleep(2)
    s = json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/status').read())
    g = s.get('grasp', {})
    cmd = g.get('arm_cmd', {})
    t = g.get('tracking', {})
    print('[%2ds] active=%s phase=%-14s action=%-8s claw=%s tgt=%s | %s' % (
        i*2, g.get('active'), g.get('phase','?'), cmd.get('action','-'),
        t.get('claw_ok','-'), t.get('target_ok','-'), cmd.get('reason','')[:60]))
    if g.get('error'):
        print('  ERROR:', g['error'])
    if not g.get('active') and g.get('phase') != 'idle':
        break

print()
print('=== Last log entries ===')
s = json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/status').read())
for l in s.get('grasp', {}).get('log', [])[-12:]:
    print(l)
