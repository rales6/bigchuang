from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DeepSeekFrontendTests(unittest.TestCase):
    def test_flash_planning_disables_default_thinking_and_retries_empty_output(self) -> None:
        source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("thinking: { type: 'disabled' }", source)
        self.assertIn("reasoning_effort: 'none'", source)
        self.assertIn("DEEPSEEK_RESPONSES_URL", source)
        self.assertIn("reasoning: { effort: 'none' }", source)
        self.assertIn("max_output_tokens: 1536", source)
        self.assertIn("max_tokens: 1536", source)
        self.assertIn("deepSeekOutputText(value)", source)

    def test_index_forces_new_deepseek_bundle(self) -> None:
        source = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("pursuit-sync6", source)
        self.assertIn("/app-runtime.js", source)


if __name__ == "__main__":
    unittest.main()
