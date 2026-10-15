"""Small SQLite repository for persistent conversations and task plans."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


class HistoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                    instruction TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    execution_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_conversation
                    ON tasks(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deleted_conversations (
                    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id),
                    deleted_at INTEGER NOT NULL
                );
                """
            )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for field in ("metadata_json", "plan_json", "execution_json"):
            if field in result:
                result[field.removesuffix("_json")] = json.loads(result.pop(field) or "{}")
        return result

    def create_conversation(self, title: str = "新任务") -> dict[str, Any]:
        conversation_id = uuid.uuid4().hex
        now = _now_ms()
        clean_title = (title.strip() or "新任务")[:80]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (conversation_id, clean_title, now, now),
            )
        return self.get_conversation(conversation_id) or {}

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """SELECT c.*, COUNT(m.id) AS message_count
                   FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.id
                   WHERE NOT EXISTS (SELECT 1 FROM deleted_conversations d WHERE d.conversation_id=c.id)
                   GROUP BY c.id ORDER BY c.updated_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM conversations WHERE id=? AND NOT EXISTS "
                "(SELECT 1 FROM deleted_conversations WHERE conversation_id=conversations.id)", (conversation_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_conversation(self, conversation_id: str) -> None:
        """Remove from the workspace, retaining diagnostic messages/task logs."""
        with self._lock, self._db:
            if self.get_conversation(conversation_id) is None:
                raise KeyError("会话不存在或已删除")
            tasks = self.list_tasks(conversation_id)
            if any(task.get("execution", {}).get("active_step")
                   or any(step.get("status") in {"running", "awaiting_verification"}
                          for step in task.get("execution", {}).get("steps", {}).values())
                   for task in tasks):
                raise ValueError("该聊天有正在执行或等待核验的动作，请先结束该动作再删除")
            self._db.execute(
                "INSERT INTO deleted_conversations VALUES(?,?)", (conversation_id, _now_ms())
            )

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        if self.get_conversation(conversation_id) is None:
            raise KeyError("会话不存在")
        clean_title = (title.strip() or "新任务")[:80]
        with self._lock, self._db:
            if self.get_conversation(conversation_id) is None:
                raise KeyError("会话不存在")
            cursor = self._db.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (clean_title, _now_ms(), conversation_id),
            )
        if not cursor.rowcount:
            raise KeyError("会话不存在")
        return self.get_conversation(conversation_id) or {}

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("消息角色无效")
        if self.get_conversation(conversation_id) is None:
            raise KeyError("会话不存在")
        message_id = uuid.uuid4().hex
        now = _now_ms()
        with self._lock, self._db:
            if self.get_conversation(conversation_id) is None:
                raise KeyError("会话不存在")
            self._db.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?)",
                (message_id, conversation_id, role, content, json.dumps(metadata or {}, ensure_ascii=False), now),
            )
            self._db.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id)
            )
        return self.get_message(message_id) or {}

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        return self._row(row)

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at,id",
                (conversation_id,),
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def create_task(
        self,
        conversation_id: str,
        source_message_id: str,
        instruction: str,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = uuid.uuid4().hex
        now = _now_ms()
        execution = {"steps": {}, "events": [], "active_step": None}
        with self._lock, self._db:
            if self.get_conversation(conversation_id) is None:
                raise KeyError("会话不存在")
            self._db.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    task_id, conversation_id, source_message_id, instruction,
                    json.dumps(plan, ensure_ascii=False), "draft",
                    json.dumps(execution, ensure_ascii=False), now, now,
                ),
            )
        return self.get_task(task_id) or {}

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM tasks WHERE id=? AND NOT EXISTS "
                "(SELECT 1 FROM deleted_conversations WHERE conversation_id=tasks.conversation_id)",
                (task_id,),
            ).fetchone()
        return self._row(row)

    def list_tasks(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM tasks WHERE conversation_id=? ORDER BY created_at DESC",
                (conversation_id,),
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        execution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._db:
            current = self.get_task(task_id)
            if current is None:
                raise KeyError("任务不存在")
            next_status = status or str(current["status"])
            next_execution = execution if execution is not None else current["execution"]
            self._db.execute(
                "UPDATE tasks SET status=?, execution_json=?, updated_at=? WHERE id=?",
                (next_status, json.dumps(next_execution, ensure_ascii=False), _now_ms(), task_id),
            )
            return self.get_task(task_id) or {}

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._db.execute(
                "SELECT value_json FROM app_settings WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(str(row["value_json"]))
        except json.JSONDecodeError:
            return default

    def set_setting(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO app_settings(key,value_json,updated_at) VALUES(?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     value_json=excluded.value_json,
                     updated_at=excluded.updated_at""",
                (key, payload, _now_ms()),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()
