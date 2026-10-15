"""Inspect/fetch our photo-only inventory job. No robot commands or restarts."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.deploy_task_localization_and_start_inventory import connect, python


def check(report_path, fetch=False):
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    if not report.get('job_started'):
        raise ValueError('Deployment report does not record a started job')
    job, pid = report['job'], int(report['pid'])
    if not job.startswith('/home/sjtu/car_inspection_jobs/object_inventory_'):
        raise ValueError('Unexpected job path')
    client = connect('202.121.181.124', 2020, 'sjtu', os.environ['QWEN_SSH_PASSWORD'])
    try:
        code = """
import json
from pathlib import Path
job=Path(JOB)
proc=Path('/proc')/str(PID)
running=proc.exists() and str(job).encode() in (proc/'cmdline').read_bytes()
checkpoints=[]
for p in sorted((job/'output/checkpoints').glob('*.json')):
    try:
        d=json.loads(p.read_text())
        checkpoints.append({'name':p.name,'success':d.get('success') is True,
                            'observation_count':len(d.get('observations',[]))})
    except (OSError,ValueError): pass
log=job/'inventory.log'
text=log.read_bytes()[-3500:].decode('utf-8','replace') if log.exists() else ''
finished=(job/'output/inventory_report.json').exists()
result={'job':str(job),'pid':PID,'running':running,'finished_outputs_exist':finished,
        'checkpoint_count':len(checkpoints),'successful_checkpoints':sum(c['success'] for c in checkpoints),
        'checkpoints':checkpoints,'log_tail':text}
if finished: result['processing_report']=json.loads((job/'output/inventory_report.json').read_text())
quality=job/'quality_filtered/quality_report.json'
result['quality_output_ready']=quality.exists()
if quality.exists(): result['quality_report']=json.loads(quality.read_text())
quality_log=job/'quality.log'
if quality_log.exists(): result['quality_log_tail']=quality_log.read_bytes()[-2000:].decode('utf-8','replace')
print(json.dumps(result,ensure_ascii=False))
""".replace('JOB', repr(job)).replace('PID', str(pid))
        status = json.loads(python(client, 'python3', code))
        status['expected_checkpoints'] = report['capture_count'] * 2
        if fetch:
            output = report_path.parent / 'results'
            output.mkdir(exist_ok=True)
            with client.open_sftp() as sftp:
                for name in ['ble_scene_17.pgm', 'ble_scene_17.yaml']:
                    sftp.get(job + '/maps/' + name, str(output / name))
                sftp.get(job + '/photos/inspection_manifest.json', str(output / 'inspection_manifest.json'))
                checkpoints = output / 'checkpoints'
                checkpoints.mkdir(exist_ok=True)
                for row in status['checkpoints']:
                    name = row['name']
                    if Path(name).name != name:
                        raise ValueError('Invalid checkpoint basename')
                    sftp.get(job + '/output/checkpoints/' + name, str(checkpoints / name))
                if status['finished_outputs_exist']:
                    for name in ['semantic_object_memory.json', 'inventory_report.json', 'object_inventory.csv']:
                        sftp.get(job + '/output/' + name, str(output / name))
                if status['quality_output_ready']:
                    quality_output = output / 'quality_filtered'
                    quality_output.mkdir(exist_ok=True)
                    for name in ['semantic_object_memory.json', 'quality_report.json']:
                        sftp.get(job + '/quality_filtered/' + name, str(quality_output / name))
                elif status['successful_checkpoints']:
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '3090/src'))
                    from qwen_grounded_tracker.semantic_object_catalog import ReadOnlyMap, build_catalog
                    from qwen_grounded_tracker.semantic_inventory_quality import filter_observations, remove_ambiguous_world_positions
                    observations = []
                    for row in status['checkpoints']:
                        if row['success']:
                            observations.extend(json.loads((checkpoints / row['name']).read_text(encoding='utf-8'))['observations'])
                    clean, quality = filter_observations(observations)
                    catalog = remove_ambiguous_world_positions(build_catalog(clean,
                        ReadOnlyMap(output / 'ble_scene_17.pgm', output / 'ble_scene_17.yaml'),
                        output / 'inspection_manifest.json'))
                    catalog['quality_report'] = quality
                    catalog['source'] = {'mode': 'partial_new_qwen_grounding_not_final', 'job': job}
                    (output / 'partial_semantic_object_memory.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
            status['local_results'] = str(output.resolve())
        (report_path.parent / 'job_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        return status
    finally:
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    check(args.report, args.fetch)
