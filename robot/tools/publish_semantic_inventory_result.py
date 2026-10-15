"""Copy quality-filtered candidates to Pi, without changing default map or motors."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.deploy_task_localization_and_start_inventory import (
    ROOT, PI_ROOT, GPU_ROOT, connect, install, remote_hashes, pi_python, python,
)


def publish(report_path):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    folder = report_path.parent
    memory_path = folder / 'results/quality_filtered/semantic_object_memory.json'
    memory = json.loads(memory_path.read_text(encoding='utf-8'))
    quality_path = folder / 'results/quality_filtered/quality_report.json'
    quality = json.loads(quality_path.read_text(encoding='utf-8'))
    if quality.get('raw_input_complete') is not True:
        raise ValueError('Raw inference incomplete: not publishing as completed run')
    pi = connect('dachuang.local', 22, 'dachuang', os.environ['PI_SSH_PASSWORD'])
    gpu = None
    stamp = time.strftime('%Y%m%d_%H%M%S')
    try:
        protected = list(report['protected_pi_hashes'])
        current = remote_hashes(pi, PI_ROOT, protected)
        if current != report['protected_pi_hashes']:
            raise ValueError('Pi map/inspection source changed since deployment; refusing automatic publication')
        if current['maps/ble_scene_17.pgm'] != memory['map_sha256'] or current['maps/ble_scene_17.yaml'] != memory['map_yaml_sha256']:
            raise ValueError('Result map hash does not match Pi original map')
        interpreter = pi_python(pi)
        report['pi_task_runtime_files'] = install(pi, PI_ROOT,
            [(ROOT / 'raspberry_pi/task_photo_observer.py', 'raspberry_pi/task_photo_observer.py')],
            PI_ROOT + '/backups/task_photo_http_' + stamp)
        report['pi_runtime_check'] = python(pi, interpreter,
            'import sys; sys.path.insert(0,' + repr(PI_ROOT) + '); '
            + 'import PIL,numpy; from raspberry_pi.task_semantic_localization import TaskSemanticLocalizer; '
            + 'from raspberry_pi.task_photo_observer import QwenTaskPhotoObserver; print("TASK_RUNTIME_DEPENDENCIES_OK")')
        report['pi_python'] = interpreter
        report['pi_result_dir'] = PI_ROOT + '/semantic_maps/' + folder.name
        files = [(memory_path, 'semantic_object_memory.json'), (quality_path, 'quality_report.json'),
                 (folder / 'README.md', 'README.md')]
        report['pi_result_files'] = install(pi, report['pi_result_dir'], files,
                                          PI_ROOT + '/backups/semantic_result_' + stamp)
        # Check real memory loader against the actual Pi interpreter, not only syntax.
        code = ('import json,sys; sys.path.insert(0,' + repr(PI_ROOT) + '); '
                + 'from raspberry_pi.task_semantic_localization import TaskSemanticLocalizer; '
                + 'm=json.load(open(' + repr(report['pi_result_dir'] + '/semantic_object_memory.json') + ")); "
                + 'l=TaskSemanticLocalizer(m,' + repr(current['maps/ble_scene_17.pgm']) + '); '
                + "print(json.dumps({'schema':m['schema'],'candidate_count':len(m['objects']),'verified_anchors':len(l.anchors)}))")
        report['pi_result_load_check'] = json.loads(python(pi, interpreter, code))
        if remote_hashes(pi, PI_ROOT, protected) != current:
            raise RuntimeError('Protected Pi files changed during publication')
        gpu = connect('202.121.181.124', 2020, 'sjtu', os.environ['QWEN_SSH_PASSWORD'])
        report['final_inventory_cli'] = install(gpu, GPU_ROOT,
            [(ROOT / '3090/scripts/run_semantic_object_inventory.py', 'scripts/run_semantic_object_inventory.py')],
            GPU_ROOT + '/backups/inventory_final_' + stamp)
        report['publication_complete'] = True
        print(json.dumps({'pi_result_dir': report['pi_result_dir'], 'load_check': report['pi_result_load_check'],
                          'original_map_and_inspection_unchanged': True, 'robot_motion_started': False}, ensure_ascii=False), flush=True)
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        pi.close()
        if gpu is not None:
            gpu.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    publish(args.report)
