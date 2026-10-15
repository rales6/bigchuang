#!/bin/sh
set -eu

release_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(dirname "$release_dir")

export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$release_dir/semantic_release_complete_test.py" \
  --release "$release_dir" \
  --project-root "$project_root" \
  --output-dir "$release_dir/complete_test_output" \
  "$@"
