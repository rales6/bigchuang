"""Local isolated software regressions only; no SSH, API or hardware calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
MODULES=(
    'test_phase1_import_priority',
    'test_phase1_navigation_rotation_support','test_phase1_bounded_navigation',
    'test_task_video','test_phase1_startup_route_candidates',
    'test_pursuit_loss_recovery','test_robot_skills',
    'test_track_web_phase1_pick','test_phase1_pursuit_entry_contract',
    'test_phase1_live_observation','test_phase1_recovery_glass_sweep',
    'test_phase1_search_handoff','test_phase1_midrun_view_change',
    'test_phase1_green_target_review','test_phase1_verified_search',
    'test_phase1_observation_recovery','test_phase1_locked_target_recovery',
    'test_phase1_arrival_visibility','test_phase1_approach_navigation',
    'test_phase1_startup_active_recovery','test_phase1_startup_route_recovery',
    'test_visibility_route_planner','test_phase1_rotation_guard','test_phase1_body_occlusion',
    'test_phase1_glass_boundary_guard','test_phase1_map_margin_egress',
    'test_phase1_turn_policy','test_phase1_fresh_motion_feedback',
    'test_phase1_navigation_avoidance','test_phase1_adaptive_turn',
)
PROTECTED=(
    'raspberry_pi/object_search_mission.py','raspberry_pi/known_map_navigation.py',
    'raspberry_pi/map_camera_inspection.py','raspberry_pi/esp32/client.py',
    'raspberry_pi/mapping/adaptive_speed.py','raspberry_pi/mapping/scan_matcher.py',
)

def fingerprint():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PROTECTED}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path)
    args=p.parse_args();before=fingerprint();rows=[];total=0
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    # Adapters deliberately patch process-local classes. Isolate each module
    # so one test's patched frozen executor cannot contaminate another module.
    for name in MODULES:
        result=subprocess.run([sys.executable,'-m','unittest',
            'grasp_range_integration.tests.'+name,'-q'],cwd=ROOT,env=env,
            capture_output=True,encoding='utf-8')
        found=re.search(r'Ran (\d+) tests?',result.stderr)
        count=int(found.group(1)) if found else 0;total+=count
        rows.append(dict(module=name,passed=result.returncode==0,tests=count))
        print(name, 'OK' if result.returncode==0 else 'FAILED', count)
        if result.returncode:print(result.stdout,result.stderr)
    record=dict(hardware_started=False,external_api_called=False,
        tests=total,passed=all(r['passed'] for r in rows),
        protected_unchanged_during_tests=before==fingerprint(),protected_sha256=before,modules=rows)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))
    return 0 if record['passed'] and record['protected_unchanged_during_tests'] else 1

if __name__=='__main__':sys.exit(main())
