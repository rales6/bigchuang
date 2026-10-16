"""Fail closed on private assets/credentials in the selected Git snapshot.

Reports paths and categories only, never matched values. --staged examines
Git blobs that will actually be committed (not just the working copy).
This is a bounded publishing guard, not a full Git-history secret scanner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PREFIXES = ('robot/grasp_range_integration/tests/fixtures/', 'robot/tests/fixtures/',
                    'services/grasp/gpu/arm_teach_artifacts/', '.local/', 'secrets/', 'credentials/')
PRIVATE_NAMES = {'docs/maintenance/source_manifest.json', 'docs/maintenance/deepseek_api_inventory.json',
    'docs/maintenance/workspace_archive.json', 'robot/grasp_range_integration/phase1/glass_boundaries.json',
    'robot/grasp_range_integration/DEPLOYMENT_RECORD.json', 'services/track/configs/stereo_calibration.yaml',
    'robot/grasp_range_integration/tools/robot_skills_deploy_manifest.json',
    'services/grasp/gpu/learned_response.json'}
PATTERNS = {
    'api_token_shape': r'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})',
    'private_key': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    'real_local_network': r'(?<![\d.])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}(?![\d.])',
    'original_external_network': r'(?<![\d.])202\.121\.\d{1,3}\.\d{1,3}(?![\d.])',
    'non_example_account_path': r'/home/(?!robot(?:/|\b)|gpu(?:/|\b))[A-Za-z0-9_-]+(?:/|\b)',
}


def violations(name, data):
    results = []
    path = Path(name)
    private = (name in PRIVATE_NAMES or name.startswith(PRIVATE_PREFIXES)
               or name.startswith('maps/current/') and name != 'maps/current/README.md'
               or name.startswith('docs/maintenance/2026')
               or path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.pgm', '.pt', '.pth', '.onnx',
                                          '.safetensors', '.mp4', '.avi', '.mkv', '.sqlite3', '.db', '.pem', '.key'}
               or path.name.startswith('.env') and path.name != '.env.example'
               or '/secrets/' in '/'+name or '/credentials/' in '/'+name)
    if private:
        results.append({'path': name, 'category': 'private_asset_or_configuration'})
    if b'\0' not in data:
        text = data.decode('utf-8-sig', errors='replace')
        for category, pattern in PATTERNS.items():
            if re.search(pattern, text):
                results.append({'path': name, 'category': category})
        macs = re.findall(r'(?i)\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b', text)
        if any(value.upper() not in {'AA:BB:CC:DD:EE:01', 'AA:BB:CC:DD:EE:FF',
                                     '00:00:00:00:00:00', 'FF:FF:FF:FF:FF:FF'} for value in macs):
            results.append({'path': name, 'category': 'non_example_hardware_address'})
    return results


def audit(staged=False):
    # ls-files reads the current index, including staged removals. Deleted
    # files are never pulled back out of HEAD for a cleanup audit.
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    results, checked = [], 0
    for name in filter(None, names):
        if staged:
            data = subprocess.check_output(['git', 'show', ':'+name], cwd=ROOT)
        else:
            path = ROOT/name
            if not path.is_file():
                continue
            data = path.read_bytes()
        checked += 1
        results.extend(violations(name, data))
    return {'files_checked': checked, 'staged': staged, 'violations': results,
            'history_scanned': False, 'matched_values_logged': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    report = audit(args.staged)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(report['violations'])


if __name__ == '__main__':
    raise SystemExit(main())
