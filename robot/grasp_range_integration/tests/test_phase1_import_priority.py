"""Real subprocess import-order regression; no lidar/BLE/camera operations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ImportPriority(unittest.TestCase):
    def test_camera_track_path_cannot_shadow_canonical_navigation_core(self):
        root = Path(__file__).resolve().parents[2]
        car = Path(os.environ.get('CAR_ROOT', root)).resolve()
        entry = root/'grasp_range_integration/phase1/relationship_find_only_entry.py'
        with tempfile.TemporaryDirectory() as tmp:
            track = Path(tmp)
            old = track/'raspberry_pi';old.mkdir()
            (old/'__init__.py').write_text('', encoding='utf-8')
            (old/'known_map_navigation.py').write_text('raise RuntimeError("OLD_TRACK_CORE_IMPORTED")\n', encoding='utf-8')
            (old/'object_search_mission.py').write_text('raise RuntimeError("OLD_TRACK_MISSION_IMPORTED")\n', encoding='utf-8')
            env = dict(os.environ, CAR_ROOT=str(car), PYTHONPATH=os.pathsep.join((str(track),str(root),str(car))))
            # Exercise exactly the entry's top-level import before main() -- no
            # model, motor or camera initialization. Child starts from track/.
            code = '''import inspect,json,runpy,sys
runpy.run_path(sys.argv[1],run_name='import_contract_test')
from raspberry_pi import known_map_navigation,object_search_mission
print(json.dumps(dict(core=known_map_navigation.__file__,mission=object_search_mission.__file__,
    deadline_supported='deadline_s' in inspect.signature(known_map_navigation.ScanMapLocalizer._rank).parameters)))
'''
            result = subprocess.run([sys.executable,'-c',code,str(entry)],cwd=track,env=env,
                capture_output=True,text=True,encoding='utf-8',timeout=20)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            data = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(Path(data['core']).resolve(),car/'raspberry_pi/known_map_navigation.py')
            self.assertEqual(Path(data['mission']).resolve(),car/'raspberry_pi/object_search_mission.py')
            self.assertTrue(data['deadline_supported'])

    def test_shared_camera_shell_prioritizes_car_before_track(self):
        shell = (Path(__file__).resolve().parents[1]/'phase1/run_phase1_find_only.sh').read_text(encoding='utf-8')
        self.assertIn('camera_bootstrap:$car_root:$python_root:$track_root:$PYTHONPATH',shell)


if __name__ == '__main__':unittest.main()
