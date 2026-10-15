
import json, websocket
ws = websocket.create_connection("ws://127.0.0.1:8021/annotations", timeout=5)
hello = json.loads(ws.recv())
print("hello:", hello)
ws.settimeout(3)
types = {}
for _ in range(30):
    try:
        m = json.loads(ws.recv())
    except Exception as e:
        break
    t = m.get("type")
    types[t] = types.get(t, 0) + 1
    if t == "frame_result":
        tr = m.get("track") or {}
        print("frame_result: status=%s bbox=%s yolo_box=%s source=%s" % (
            tr.get("status"), tr.get("bbox"), m.get("yolo_box"), tr.get("source")))
        break
print("types seen:", types)
ws.close()

