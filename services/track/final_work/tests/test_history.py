from pathlib import Path
import tempfile
import unittest

from final_work.gpu.history import HistoryStore


class HistoryStoreTest(unittest.TestCase):
    def test_conversation_message_and_task_survive_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            store = HistoryStore(path)
            conversation = store.create_conversation("测试")
            message = store.add_message(conversation["id"], "user", "寻找红色方块")
            task = store.create_task(
                conversation["id"], message["id"], "寻找红色方块",
                {"steps": [{"id": "step_01", "action": "locate_object"}]},
            )
            store.update_task(task["id"], status="confirmed")
            store.close()

            reopened = HistoryStore(path)
            self.assertEqual(reopened.list_messages(conversation["id"])[0]["content"], "寻找红色方块")
            self.assertEqual(reopened.get_task(task["id"])["status"], "confirmed")
            reopened.close()

    def test_settings_survive_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            store = HistoryStore(path)
            store.set_setting("agent_preferences", {
                "language": "en-US", "provider": "deepseek_api",
            })
            store.set_setting("api_credentials", {"deepseek_api": "secret"})
            store.close()

            reopened = HistoryStore(path)
            self.assertEqual(
                reopened.get_setting("agent_preferences")["provider"],
                "deepseek_api",
            )
            self.assertEqual(
                reopened.get_setting("api_credentials")["deepseek_api"],
                "secret",
            )
            reopened.close()


if __name__ == "__main__":
    unittest.main()
