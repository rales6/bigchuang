"""Wait for one started inference job, audit its candidates into a separate sidecar."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qwen_grounded_tracker.semantic_object_catalog import ReadOnlyMap, build_catalog
from qwen_grounded_tracker.semantic_inventory_quality import filter_observations, remove_ambiguous_world_positions


def finalize(job, timeout_s=7200, inference_pid=None):
    source = job / 'output/inventory_report.json'
    deadline = time.monotonic() + timeout_s
    print('Waiting for raw Qwen inference outputs: ' + str(job), flush=True)
    while not source.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Inference did not finish within finalization deadline')
        if inference_pid is not None:
            proc = Path('/proc') / str(inference_pid)
            if not proc.exists() or str(job).encode() not in (proc / 'cmdline').read_bytes():
                raise RuntimeError('Inference exited without a processing report; raw checkpoints preserved')
        time.sleep(2)
    processing_report = json.loads(source.read_text(encoding='utf-8'))
    observations = []
    for path in sorted((job / 'output/checkpoints').glob('*.json')):
        checkpoint = json.loads(path.read_text(encoding='utf-8'))
        if checkpoint.get('success') is True:
            observations.extend(checkpoint.get('observations', []))
    clean, quality = filter_observations(observations)
    output = job / 'quality_filtered'
    output.mkdir(exist_ok=True)
    map_data = ReadOnlyMap(job / 'maps/ble_scene_17.pgm', job / 'maps/ble_scene_17.yaml')
    catalog = remove_ambiguous_world_positions(build_catalog(clean, map_data, job / 'photos/inspection_manifest.json'))
    catalog.update(source={'mode': 'fresh_qwen_grounding_quality_filtered_not_coordinate_verified', 'job': str(job)},
                   quality_report=quality, processing_report=processing_report)
    catalog['map_positions_still_need_calibration_and_verification'] = True
    quality['processing_failures'] = processing_report.get('failures', [])
    quality['raw_input_complete'] = processing_report.get('all_selected_inputs_processed') is True
    for name, data in [('semantic_object_memory.json', catalog), ('quality_report.json', quality)]:
        temporary = output / (name + '.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(output / name)
    print(json.dumps({'output': str(output), 'observations': len(clean), 'ambiguous_regions': quality['ambiguous_regions'],
                      'rejected': len(quality['rejected']), 'raw_input_complete': quality['raw_input_complete']}, ensure_ascii=False), flush=True)
    return quality


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--inference-pid', type=int)
    parser.add_argument('--timeout', type=float, default=7200)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('Timeout must be positive')
    finalize(args.job, args.timeout, args.inference_pid)
