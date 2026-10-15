"""Upload the semantic release to a new Raspberry Pi directory and verify it.

The destination must not already exist. Existing robot maps and runtime files
are never replaced or activated by this uploader.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

import paramiko


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "maps" / "semantic_release_20260919"
DEFAULT_DESTINATION = "/home/dachuang/semantic_map_packages/ble_scene_17_relationship_20260919"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(client: paramiko.SSHClient, command: str) -> str:
    _stdin, stdout, stderr = client.exec_command(command, timeout=60)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    if code:
        raise RuntimeError(error or output or f"remote command failed: {code}")
    return output


def upload(source: Path, destination: str, host: str, user: str, password: str,
           update_existing: bool = False, port: int = 22,
           include: list[str] | None = None) -> dict:
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        and (not include or path.relative_to(source).as_posix() in include)
    )
    if not files:
        raise ValueError("release directory is empty")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=30,
                   auth_timeout=30, banner_timeout=30)
    timestamp = int(time.time())
    staging = destination + f".uploading_{timestamp}"
    backup = destination + f".backup_{timestamp}" if update_existing else None
    try:
        quoted_destination = json.dumps(destination)
        quoted_staging = json.dumps(staging)
        if update_existing:
            run(client, f"test -d {quoted_destination} && test ! -e {quoted_staging} && "
                        f"test ! -e {json.dumps(backup)}")
        else:
            run(client, f"test ! -e {quoted_destination} && test ! -e {quoted_staging}")
        run(client, f"mkdir -p {quoted_staging}")
        hashes = {}
        with client.open_sftp() as sftp:
            made = {staging}
            for local in files:
                relative = local.relative_to(source).as_posix()
                remote = str(PurePosixPath(staging) / relative)
                parent = str(PurePosixPath(remote).parent)
                if parent not in made:
                    run(client, "mkdir -p " + json.dumps(parent))
                    made.add(parent)
                sftp.put(str(local), remote)
                if local.suffix == ".sh":
                    sftp.chmod(remote, 0o755)
                local_hash = digest_bytes(local.read_bytes())
                with sftp.open(remote, "rb") as stream:
                    remote_hash = digest_bytes(stream.read())
                if remote_hash != local_hash:
                    raise RuntimeError(f"hash mismatch after upload: {relative}")
                hashes[relative] = local_hash
            upload_record = {
                "source": str(source),
                "destination": destination,
                "file_count": len(files),
                "sha256": hashes,
                "activated": False,
                "existing_robot_files_modified": False,
                "previous_package_backup": backup,
            }
            record_path = str(PurePosixPath(staging) / "UPLOAD_RECORD.json")
            with sftp.open(record_path, "wb") as stream:
                stream.write((json.dumps(upload_record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
            if update_existing:
                run(client, f"mv {quoted_destination} {json.dumps(backup)} && "
                            f"mv {quoted_staging} {quoted_destination}")
            else:
                sftp.rename(staging, destination)
        listing = run(client, f"find {quoted_destination} -type f -printf '%P\\n' | sort")
        return {**upload_record, "remote_files": listing.splitlines()}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    parser.add_argument("--host", default="dachuang.local")
    parser.add_argument("--user", default="dachuang")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--update-existing", action="store_true",
                        help="Atomically replace this independent package and retain a sibling backup")
    parser.add_argument("--include", action="append", default=None,
                        help="Upload only this source-relative file; repeat as needed")
    args = parser.parse_args()
    password = os.environ.get("PI_SSH_PASSWORD") or getpass.getpass("Pi SSH password: ")
    print(json.dumps(upload(args.source, args.destination, args.host, args.user, password,
                            update_existing=args.update_existing, port=args.port,
                            include=args.include),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
