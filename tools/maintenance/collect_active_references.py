"""Read only the photograph bundle CURRENT localisation actually selects.

Legacy runs stay on the Pi. Do not call the camera API or rebuild annotations.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import posixpath

from tools.maintenance.assemble_workspace import Assembly, ROOT, connect, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-host", required=True)
    parser.add_argument("--reference", required=True, help="Verified active inspection directory on Pi")
    args = parser.parse_args()
    base = args.reference.rstrip("/")
    allowed_root = "/home/dachuang/dachuang/car1.0/inspections/ble_scene_17/"
    if not base.startswith(allowed_root) or ".." in PurePosixPath(base).parts:
        raise ValueError("Reference is outside the verified scene")
    manifest_path = ROOT/"docs/maintenance/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assembly = Assembly()
    assembly.files = manifest["files"]
    c = connect("pi", args.pi_host)
    images = set()
    try:
        with c.open_sftp() as s:
            source = base+"/inspection_manifest.json"
            with s.open(source,"rb") as f: raw = f.read()
            payload = json.loads(raw)
            for record in payload.get("captures",[]):
                for field in ("image", "camera_2_image"):
                    name = record.get(field)
                    if not name:
                        continue
                    remote = posixpath.normpath(posixpath.join(base,str(name)))
                    if not remote.startswith(base+"/") or PurePosixPath(remote).suffix.lower() not in {".png",".jpg",".jpeg"}:
                        raise ValueError("Unexpected reference image path")
                    relative = remote[len(base)+1:]
                    if remote not in images:
                        info=s.stat(remote)
                        if info.st_size>10*1024*1024:raise ValueError("Unexpectedly large reference")
                        with s.open(remote,"rb") as f:
                            assembly.put(remote,Path("maps/current/references")/relative,f.read(),"pi")
                        images.add(remote)
                    record[field]=relative
            assembly.put(source,"maps/current/references/inspection_manifest.remote.json",raw,"pi")
            local=(json.dumps(payload,ensure_ascii=False,indent=2)+"\n").encode()
            assembly.put(source,"maps/current/references/inspection_manifest.json",local,"pi")
            assembly.files["maps/current/references/inspection_manifest.json"]["transform"]="relative_image_paths"
            # Python helpers used by the active scene are code, not map data.
            scene="/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919"
            for entry in s.listdir_attr(scene):
                if Path(entry.filename).suffix not in {".py",".sh",".conf"}:continue
                with s.open(scene+"/"+entry.filename,"rb") as f:
                    assembly.put(scene+"/"+entry.filename,Path("robot/scene_runtime")/entry.filename,f.read(),"pi")
    finally:c.close()
    manifest["files"]=assembly.files
    manifest["active_visual_reference"]={"source":base,"images":len(images),"local":"maps/current/references"}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest["active_visual_reference"],ensure_ascii=False))


if __name__=="__main__":main()
