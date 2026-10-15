"""Read-only source completion for runtime server entrypoints and calibration."""
import json
from pathlib import Path

from tools.maintenance.assemble_workspace import Assembly, ROOT, connect


def main():
    path=ROOT/"docs/maintenance/source_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    assembly=Assembly()
    assembly.files=manifest["files"]
    for role in ("pi","gpu"):
        c=connect(role, "172.18.12.242" if role=="pi" else None)
        try:
            with c.open_sftp() as s:
                if role=="gpu":
                    assembly.remote_tree(s,"/home/sjtu/track","services/track","gpu",recurse=False)
                    assembly.remote_tree(s,"/home/sjtu/track/configs","services/track/configs","gpu")
                else:
                    assembly.remote_tree(s,"/home/dachuang/dachuang/car1.0/tools","robot/tools","pi")
        finally:c.close()
    manifest["files"]=assembly.files
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("source_manifest_entries="+str(len(assembly.files)))


if __name__=="__main__":main()
