"""Pi-only audited navigation adapter deployment. Never starts robot motion."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

from .deploy_track_web_pick import HOSTS, _connect, _mkdir, _read, _remote
from .deploy_carry_handoff import idle, optional, SHANGHAI, RELEASE, TRACK, CAR
from .deploy_phase1_observation import protected_hashes

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    'phase1/relationship_find_only_entry.py',
    'phase1/navigation_rotation_support.py',
    'tests/test_phase1_navigation_rotation_support.py',
    'tests/test_phase1_bounded_navigation.py',
    'tests/fixtures/navigation_body_blind_20261002_210504.json',
)
TESTS = ('test_phase1_navigation_rotation_support', 'test_phase1_bounded_navigation',
         'test_phase1_rotation_guard', 'test_phase1_body_occlusion',
         'test_phase1_map_margin_egress', 'test_phase1_midrun_view_change',
         'test_phase1_glass_boundary_guard', 'test_phase1_fresh_motion_feedback',
         'test_phase1_live_observation', 'test_phase1_observation_recovery',
         'test_phase1_recovery_glass_sweep', 'test_phase1_startup_active_recovery')


def sha(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    c = _connect(HOSTS['pi'])
    backup = TRACK+'/final_work/deployment_backups/estimated_rotation_'+datetime.now(SHANGHAI).strftime('%Y%m%d_%H%M%S')
    prepared, activated = [], []
    stopped = False
    previous = json.loads((ROOT/'diagnostics/latest_task_repair_deployment.json').read_text(encoding='utf-8'))
    try:
        idle(c)
        with c.open_sftp() as s:
            protected = protected_hashes(s)
            for name in FILES:
                target = RELEASE+'/grasp_range_integration/'+name
                old = optional(s, target)
                new = (ROOT/'grasp_range_integration'/name).read_bytes()
                if name == 'phase1/relationship_find_only_entry.py' and sha(old) not in {previous['files'][target], sha(new)}:
                    raise RuntimeError('Unreviewed remote entry version: '+target)
                if name in {'phase1/navigation_rotation_support.py', 'tests/fixtures/navigation_body_blind_20261002_210504.json'} and old is not None and old != new:
                    raise RuntimeError('New-file name collision: '+target)
                prepared.append((target, old, new))
            target = RELEASE+'/grasp_range_integration/README.md'
            old = optional(s, target)
            local = (ROOT/'grasp_range_integration/README.md').read_text(encoding='utf-8')
            section = local.split('## 2026-10-02：估计导航与车身盲区转向衔接', 1)[1].split('## 最新扩展', 1)[0]
            addition = ('## 2026-10-02：估计导航与车身盲区转向衔接'+section).encode('utf-8')
            # Preserve other students' remote documentation verbatim.
            new = old if old and addition in old else addition+(old or b'')
            prepared.append((target, old, new))
        if not args.deploy:
            print(json.dumps(dict(audited=len(prepared), hardware_started=False)))
            return
        with c.open_sftp() as s:
            _mkdir(s, backup)
            for i, (target, old, new) in enumerate(prepared):
                if old is not None:
                    with s.open(backup+f'/{i}.original', 'wb') as f:
                        f.write(old)
                with s.open(backup+f'/{i}.staged', 'wb') as f:
                    f.write(new)
                if _read(s, backup+f'/{i}.staged') != new:
                    raise RuntimeError('Staging mismatch')
        staged = [shlex.quote(backup+f'/{i}.staged') for i, (p, _, _) in enumerate(prepared) if p.endswith('.py')]
        _remote(c, shlex.quote(HOSTS['pi']['python'])+' -m py_compile '+' '.join(staged))
        idle(c)
        _remote(c, 'sudo -n systemctl stop final-work-pi-bridge.service')
        stopped = True
        with c.open_sftp() as s:
            for i, (target, old, new) in enumerate(prepared):
                if optional(s, target) != old:
                    raise RuntimeError('Concurrent remote edit: '+target)
                _mkdir(s, str(PurePosixPath(target).parent))
                s.posix_rename(backup+f'/{i}.staged', target)
                activated.append((i, target, old))
                if _read(s, target) != new:
                    raise RuntimeError('Activated hash mismatch')
        prefix = 'cd '+shlex.quote(CAR)+' && PYTHONPATH='+shlex.quote(CAR+':'+RELEASE+':'+TRACK)+' '+shlex.quote(HOSTS['pi']['python'])
        results = []
        for name in TESTS:
            output = _remote(c, prefix+' -m unittest grasp_range_integration.tests.'+name+' -q')
            print(name, output.strip(), flush=True)
            results.append(dict(module=name, output=output.strip(), passed=True))
        with c.open_sftp() as s:
            if protected_hashes(s) != protected:
                raise RuntimeError('Protected core changed')
        _remote(c, 'sudo -n systemctl start final-work-pi-bridge.service')
        _remote(c, 'sudo -n systemctl start final-work-pi-tunnel.service')
        _remote(c, 'systemctl is-active final-work-pi-bridge.service final-work-pi-tunnel.service')
        _remote(c, 'curl --retry 6 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8780/health >/dev/null')
        idle(c)
        report = dict(backup=backup, hardware_started=False, services_reloaded=True,
            protected_unchanged=True, software_tests=results,
            files={p: sha(new) for p, _, new in prepared})
        with c.open_sftp() as s:
            with s.open(backup+'/deployment.json', 'w') as f:
                f.write(json.dumps(report, ensure_ascii=False, indent=2))
        (ROOT/'diagnostics/estimated_rotation_deployment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        stopped = False
        print(json.dumps(dict(files=len(prepared), backup=backup, hardware_started=False, services_reloaded=True)))
    except BaseException:
        # Services stay stopped while restoring every activated adapter.
        if activated:
            _remote(c, 'sudo -n systemctl stop final-work-pi-bridge.service')
        with c.open_sftp() as s:
            for i, target, old in reversed(activated):
                s.posix_rename(target, backup+f'/{i}.failed')
                if old is not None:
                    s.posix_rename(backup+f'/{i}.original', target)
        if stopped:
            _remote(c, 'sudo -n systemctl start final-work-pi-bridge.service')
            _remote(c, 'sudo -n systemctl start final-work-pi-tunnel.service')
        raise
    finally:
        c.close()


if __name__ == '__main__':
    main()
