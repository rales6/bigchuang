"""Workspace deletion only: no robot calls, and diagnostic data is retained."""
from fastapi import HTTPException


def install_conversation_delete(app):
    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str):
        try:
            app.state.history.delete_conversation(conversation_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, "deleted_id": conversation_id, "logs_retained": True}
