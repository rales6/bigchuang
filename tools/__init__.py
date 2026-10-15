"""Repository maintenance tools; allow legacy pure-tool imports in tests.

Operational code stays in robot/tools until its import/deployment contracts
are deliberately migrated, instead of silently breaking existing scripts.
"""
from pathlib import Path

_legacy = Path(__file__).resolve().parents[1] / "robot" / "tools"
if _legacy.is_dir():
    __path__.append(str(_legacy))
