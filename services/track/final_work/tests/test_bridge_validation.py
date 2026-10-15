import unittest

from final_work.pi.robot_bridge import _safe_text


class BridgeValidationTest(unittest.TestCase):
    def test_safe_text(self):
        self.assertEqual(_safe_text("  红色方块  ", name="目标"), "红色方块")
        with self.assertRaises(ValueError):
            _safe_text("", name="目标")
        with self.assertRaises(ValueError):
            _safe_text("bad\x00value", name="目标")


if __name__ == "__main__":
    unittest.main()
