"""Offline task-localization diagnostic. Never connects to Pi or controls motors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raspberry_pi.task_semantic_localization import (
    CameraCalibration, PoseCandidate, SemanticDetection, TaskPose,
    TaskSemanticLocalizer, detections_from_qwen,
)


def run(memory_path, pgm_path, frames_path):
    memory = json.loads(memory_path.read_text(encoding='utf-8'))
    map_hash = hashlib.sha256(pgm_path.read_bytes()).hexdigest()
    localizer = TaskSemanticLocalizer(memory, map_hash)
    payload = json.loads(frames_path.read_text(encoding='utf-8'))
    calibrations = {int(k): CameraCalibration(**v) for k, v in payload.get('cameras', {}).items()}
    results = []
    for frame in payload['frames']:
        detections = detections_from_qwen(frame.get('qwen_objects', []), calibrations)
        # Explicit bearing/identity observations must be externally calibrated
        # and reference-matched; the diagnostic does not certify either.
        detections += [SemanticDetection(**d) for d in frame.get('detections', [])]
        results.append(localizer.evaluate(frame_id=frame['frame_id'],
            captured_at_s=frame['captured_at_s'], now_s=frame['now_s'],
            capture_pose=TaskPose(**frame['capture_pose']), current_pose=TaskPose(**frame['current_pose']),
            detections=detections, candidates=[PoseCandidate(TaskPose(**c['pose']),
                c.get('radar_supported', False), c.get('name', 'radar')) for c in frame.get('candidates', [])]))
    return {'scope': 'offline_pose_proposals_no_motor_control', 'map_sha256': map_hash,
            'verified_anchor_count': len(localizer.anchors), 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for source in (args.memory, args.map, args.frames):
        if args.output.resolve() == source.resolve():
            parser.error('Output must not overwrite an input')
    report = run(args.memory, args.map, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'verified_anchors': report['verified_anchor_count'],
                      'accepted': sum(r['accepted'] for r in report['results'])}, ensure_ascii=False))
