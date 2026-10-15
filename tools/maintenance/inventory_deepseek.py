"""Read-only handoff inventory. Emits paths/field names, NEVER key values.

No provider request, service reload or robot action. Remote credentials come
from the existing SSH helper's environment variables. Browser storage and
provider-side keys require the operator's separate inspection/revocation.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shlex


def scan(roots):
    import json
    import os
    from pathlib import Path
    import re
    import sqlite3

    skip = {'.git', '.venv', '.venv-win', 'venv', 'venvs', 'node_modules', '__pycache__',
            '.gradle', '.kotlin', '.tooling', '.idea', '.local', 'site-packages',
            'models', 'recordings', 'videos', 'video', 'frames', 'raw_frames'}
    extensions = {'.py', '.sh', '.ps1', '.json', '.yaml', '.yml', '.toml', '.md', '.txt',
                  '.env', '.ini', '.conf', '.cfg', '.service', '.log', '.js', '.html', '.jsonl'}
    result = {'roots': [str(p) for p in roots], 'references': [], 'credential_locations': [],
              'databases': [], 'unreadable': [], 'files_examined': 0,
              'excluded_directory_names': sorted(skip), 'maximum_text_bytes': 3*1024*1024}
    seen = set()

    def configured(value):
        if not isinstance(value, str) or not value.strip():
            return False
        text = value.strip().lower()
        return not any(word in text for word in ('placeholder', 'replace', 'your_', 'your-',
            'redacted', 'set_', 'example', '填', 'test', 'dummy', 'fake'))

    def fields(value, prefix=''):
        found = []
        if isinstance(value, dict):
            for key, child in value.items():
                name = prefix + '.' + str(key) if prefix else str(key)
                if str(key).lower() in {'api_key', 'apikey', 'deepseek_api_key',
                                        'final_work_deepseek_api_key', 'deepseek_api', 'deepseek_flash_api'}:
                    if isinstance(child, str):
                        found.append({'field': name, 'configured': configured(child)})
                elif isinstance(child, (dict, list)):
                    found.extend(fields(child, name))
        elif isinstance(value, list):
            for i, child in enumerate(value):
                found.extend(fields(child, prefix+'['+str(i)+']'))
        return found

    def examine(path):
        absolute = str(path.absolute())
        if absolute in seen or path.is_symlink():
            return
        seen.add(absolute)
        try:
            if path.suffix in {'.sqlite3', '.sqlite', '.db'} and 'history' in path.name:
                connection = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
                try:
                    row = connection.execute("SELECT value_json FROM app_settings WHERE key='api_credentials'").fetchone()
                    value = json.loads(row[0]) if row else {}
                    result['databases'].append({'path': absolute, 'table': 'app_settings',
                        'setting': 'api_credentials', 'fields': [f for f in fields(value) if 'deepseek' in f['field']]})
                finally:
                    connection.close()
                return
            if not (set(path.suffixes) & extensions or path.name.startswith('.env')
                    or path.name in {'.bashrc', '.profile', '.bash_profile'}):
                return
            if path.stat().st_size > result['maximum_text_bytes']:
                return
            raw = path.read_bytes()
            if b'\x00' in raw:
                return
            text = raw.decode('utf-8-sig', errors='replace')
            result['files_examined'] += 1
            mentions = 'deepseek' in text.lower() or 'deepseek' in path.name.lower()
            literal = bool(re.search(r'\bsk-[A-Za-z0-9_-]{16,}', text))
            if not mentions and not literal:
                return
            result['references'].append(absolute)
            entries = []
            try:
                if mentions:
                    entries = fields(json.loads(text))
            except (ValueError, TypeError):
                pass
            env_names = sorted(set(re.findall(r'\b(?:[A-Z][A-Z_]*_)?DEEPSEEK[A-Z_]*(?:KEY|CONFIG)\b', text)))
            if entries or literal or (env_names and (path.name.startswith('.env') or path.suffix in {'.conf', '.service'})):
                result['credential_locations'].append({'path': absolute, 'fields': entries,
                    'contains_key_shaped_literal': literal, 'environment_variable_names': env_names})
        except (OSError, sqlite3.Error, ValueError) as exc:
            result['unreadable'].append({'path': absolute, 'error_type': type(exc).__name__})

    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            examine(root)
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d.lower() not in skip and not d.startswith('.venv')]
            for name in files:
                examine(Path(directory)/name)
    for key in ('references',):
        result[key] = sorted(result[key])
    return result


def main():
    from tools.maintenance.assemble_workspace import connect, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'.local/handoff/deepseek_inventory.json')
    args = parser.parse_args()
    report = {'local': scan([ROOT.parent, Path('D:/xingxianfeng_files/dachaung')])}
    if args.remote:
        remote_roots = {
            'pi': ['/home/dachuang/dachuang', '/home/dachuang/grab_visual', '/home/dachuang/backups',
                   '/home/dachuang/semantic_map_packages', '/home/dachuang/.config/systemd',
                   '/etc/systemd/system', '/home/dachuang/.bashrc', '/home/dachuang/.profile'],
            'gpu': ['/home/sjtu/track', '/home/sjtu/3090', '/home/sjtu/3090_new_method',
                    '/home/sjtu/grab_visual', '/home/sjtu/backups', '/home/sjtu/car_runtime',
                    '/home/sjtu/car_inspection_jobs', '/home/sjtu/car_inspections',
                    '/home/sjtu/semantic_map_packages', '/home/sjtu/.config/systemd',
                    '/etc/systemd/system', '/home/sjtu/.bashrc', '/home/sjtu/.profile']}
        for role, roots in remote_roots.items():
            client = connect(role)
            try:
                source = inspect.getsource(scan)+'\nimport json\nprint(json.dumps(scan('+repr(roots)+')))'
                _, output, errors = client.exec_command('python3 -c '+shlex.quote(source), timeout=90)
                text = output.read().decode('utf-8')
                if output.channel.recv_exit_status():
                    # Do not propagate arbitrary remote output into a credential report.
                    raise RuntimeError(role+' inventory failed; remote output suppressed')
                report[role] = json.loads(text)
            finally:
                client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for role, result in report.items():
        print(role, 'files:', result['files_examined'], 'references:', len(result['references']),
              'credential locations:', len(result['credential_locations']), 'databases:', len(result['databases']),
              'unreadable:', len(result['unreadable']))
    print('Path-only report:', args.output)


if __name__ == '__main__':
    main()
