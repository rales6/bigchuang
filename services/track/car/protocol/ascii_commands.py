"""Readable Raspberry Pi command frames such as #ARMF100U050L000!."""


CMD_ARM_DELTA = "arm_delta"
CMD_MOVE_DELTA = "move_delta"
CMD_LED = "led"
CMD_BEEP = "beep"
CMD_STOP = "stop"
CMD_PING = "ping"


def parse_ascii_command(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("ascii")
    raw = raw.strip()
    if not raw.startswith("#") or not raw.endswith("!"):
        raise ValueError("ASCII command must be wrapped by # and !")
    body = raw[1:-1].upper()
    if body == "PING":
        return CMD_PING, {}
    if body in ("STOP", "CANCEL"):
        return CMD_STOP, {}
    if body.startswith("ARM"):
        values = _parse_signed_fields(body[3:])
        return CMD_ARM_DELTA, {
            "forward_mm": values.get("F", 0) - values.get("B", 0),
            "up_mm": values.get("U", 0) - values.get("D", 0),
            "left_deg": values.get("L", 0) - values.get("R", 0),
            "claw_open": values.get("C", None),
        }
    if body.startswith("MOV"):
        values = _parse_signed_fields(body[3:])
        return CMD_MOVE_DELTA, {
            "forward_mm": values.get("F", 0) - values.get("B", 0),
            "left_deg": values.get("L", 0) - values.get("R", 0), 
            "speed_level": values.get("S", None),
        }
    if body == "LEDON":
        return CMD_LED, {"mode": 1, "period_ms": 0}
    if body == "LEDOFF":
        return CMD_LED, {"mode": 0, "period_ms": 0}
    if body.startswith("LEDB"):
        return CMD_LED, {"mode": 2, "period_ms": _to_int(body[4:] or "500")}
    if body.startswith("BEEP"):
        return CMD_BEEP, {"repeat": _to_int(body[4:] or "1")}
    raise ValueError("unsupported ASCII command")


def _parse_signed_fields(text):
    result = {}
    index = 0
    while index < len(text):
        key = text[index]
        if key not in "FBUDLRCS":
            raise ValueError("bad field key")
        index += 1
        start = index
        while index < len(text) and "0" <= text[index] <= "9":
            index += 1
        if start == index:
            raise ValueError("field without numeric value")
        result[key] = _to_int(text[start:index])
    return result


def _to_int(text):
    value = int(text)
    if value < 0 or value > 9999:
        raise ValueError("numeric value out of range")
    return value
