#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
source .venv-pi/bin/activate

instruction="${1:?请提供自然语言指令或 @UTF-8文件，例如：./natural_search.sh @search_runs/command.txt real}"
mode="${2:-simulate}"
expected_class="${3:-}"
expected_mode="${4:-}"
decision="search_runs/latest_decision.json"

submit_args=("$instruction")
if [[ "$instruction" == @* ]]; then
  submit_args=("file" --instruction-file "${instruction#@}")
fi

python3 -u -m raspberry_pi.submit_search_command \
  "${submit_args[@]}" \
  --3090-url "${SEARCH_3090_URL:-http://192.168.55.33:8011}" \
  --output "$decision"

python3 - "$decision" "$expected_class" "$expected_mode" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
t = d.get("target") or {}
if not t.get("class_name"):
    raise SystemExit("任务目标类别为空，拒绝启动底盘")
if sys.argv[2] and t.get("class_name") != sys.argv[2]:
    raise SystemExit("目标类别校验失败：期望 {}，实际 {}；拒绝启动底盘".format(sys.argv[2], t.get("class_name")))
if sys.argv[3] and t.get("detection_mode") != sys.argv[3]:
    raise SystemExit("检测模式校验失败：期望 {}，实际 {}；拒绝启动底盘".format(sys.argv[3], t.get("detection_mode")))
print("任务执行确认：class={} mode={} anchor={}".format(
    t.get("class_name"), t.get("detection_mode"),
    (d.get("anchor") or {}).get("name") or (d.get("anchor") or {}).get("query"),
), flush=True)
PY

exec ./search_object.sh "$decision" "$mode"
