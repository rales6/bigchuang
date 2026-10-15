"""Read-only download of the latest Pi inspection evidence."""
import getpass
import json
from pathlib import Path
import time

import paramiko


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "simulation_runs" / ("latest_inspection_" + time.strftime("%Y%m%d_%H%M%S"))
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("dachuang.local", username="dachuang", password=getpass.getpass("Pi password: "), timeout=20)
    try:
        stdin, stdout, stderr = client.exec_command(
            "cd /home/dachuang/dachuang/car1.0 && "
            "latest=$(find inspections/ble_scene_17 -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\\n' | sort -nr | head -1 | cut -d' ' -f2-) && "
            "printf '%s\\n' \"$latest\" && "
            "latest_log=$(ls -1t inspection_logs/inspect_map_*.log 2>/dev/null | head -1) && printf '%s\\n' \"$latest_log\"",
            timeout=20,
        )
        lines = stdout.read().decode().splitlines()
        error = stderr.read().decode()
        if stdout.channel.recv_exit_status() or len(lines) < 2:
            raise RuntimeError(error or "latest inspection not found")
        remote_directory, remote_log = lines[-2:]
        output.mkdir(parents=True, exist_ok=False)
        names = ("navigation_diagnostics.jsonl", "localization_diagnostics.jsonl",
                 "inspection_manifest.json", "inspection_plan.png")
        with client.open_sftp() as sftp:
            for name in names:
                try:
                    sftp.get("/home/dachuang/dachuang/car1.0/" + remote_directory + "/" + name, str(output / name))
                except FileNotFoundError:
                    pass
            sftp.get("/home/dachuang/dachuang/car1.0/" + remote_log, str(output / Path(remote_log).name))
            sftp.get("/home/dachuang/dachuang/car1.0/maps/ble_scene_17_semantics_qwen.json",
                     str(output / "ble_scene_17_semantics_qwen.json"))
            manifest_path = output / "inspection_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                names = {name for row in manifest.get("captures", [])
                         for name in (row.get("image"), row.get("camera_2_image"))
                         if name and Path(name).name == name}
                for name in sorted(names):
                    try:
                        sftp.get("/home/dachuang/dachuang/car1.0/" + remote_directory + "/" + name,
                                 str(output / name))
                    except FileNotFoundError:
                        pass
        (output / "source.txt").write_text(remote_directory + "\n" + remote_log + "\n", encoding="utf-8")
        print("SOURCE=" + remote_directory)
        print("OUTPUT=" + str(output))
        for path in output.iterdir():
            print(path.name + "=" + str(path.stat().st_size))
    finally:
        client.close()


if __name__ == "__main__":
    main()
