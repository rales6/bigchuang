"""Manage explicit Pi consent for external DeepSeek semantic planning; no robot action."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from deploy_track_web_pick import HOSTS, ROOT, _connect, _read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable", action="store_true")
    group.add_argument("--disable", action="store_true")
    args = parser.parse_args()
    spec = HOSTS["pi"]
    client = _connect(spec)
    target = spec["root"] + "/final_work/data/deepseek_semantic_upload.allowed"
    adapter = spec["root"] + "/final_work/pi/map_navigation_adapter.py"
    desired_hash = hashlib.sha256(
        (ROOT / "pi/map_navigation_adapter.py").read_bytes()).hexdigest()
    try:
        with client.open_sftp() as sftp:
            if hashlib.sha256(_read(sftp, adapter)).hexdigest() != desired_hash:
                raise RuntimeError("远端规划适配层与已审查版本不一致，拒绝更改授权")
            try:
                previous = _read(sftp, target)
            except FileNotFoundError:
                previous = None
            if not (args.enable or args.disable):
                print("enabled" if previous == b"1\n" else "disabled")
                return
            value = b"1\n" if args.enable else b"0\n"
            if previous == value:
                print("enabled" if args.enable else "disabled")
                return
            if previous is not None:
                backup = target + ".backup_" + time.strftime("%Y%m%d_%H%M%S")
                with sftp.open(backup, "wb") as stream:
                    stream.write(previous)
                sftp.chmod(backup, 0o600)
            staged = target + ".staged_" + str(int(time.time() * 1000))
            with sftp.open(staged, "wb") as stream:
                stream.write(value)
            sftp.chmod(staged, 0o600)
            if _read(sftp, staged) != value:
                raise RuntimeError("授权暂存文件校验失败")
            sftp.posix_rename(staged, target)
            if _read(sftp, target) != value:
                raise RuntimeError("授权文件写入后校验失败")
            print("enabled" if args.enable else "disabled")
    finally:
        client.close()


if __name__ == "__main__":
    main()
