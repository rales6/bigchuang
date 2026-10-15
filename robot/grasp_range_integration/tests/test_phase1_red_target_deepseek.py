"""DeepSeek may verify grounded crops; a model failure never invents a target."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image, ImageDraw

from grasp_range_integration.phase1.observation_recovery import red_visual_cues
from grasp_range_integration.phase1.red_target_deepseek import verify_red_block_pair


class DeepSeekRedTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = {}
        for camera, x in (("left", 160), ("right", 284)):
            image = Image.new("RGB", (1280, 720), (100, 100, 100))
            ImageDraw.Draw(image).rectangle((x, 142, x+45, 187), fill=(175, 36, 42))
            path = Path(self.temporary.name) / f"{camera}.jpg"
            image.save(path)
            self.paths[camera] = path
        self.cues = {item["camera_id"]: item
                     for item in red_visual_cues(self.paths)}
        self.config = SimpleNamespace(configured=True, model="deepseek-flash",
                                      api_key="test-key", base_url="https://example.invalid",
                                      timeout_s=10., repeat_count=2)

    def test_two_independent_positive_votes_verify_only_the_given_crops(self):
        requests = []
        def transport(request):
            requests.append(json.loads(request.data.decode("utf-8")))
            return {"choices": [{"message": {"content": json.dumps({
                "left": {"is_red_block": True, "confidence": .92},
                "right": {"is_red_block": True, "confidence": .91},
                "same_object": True})}}]}
        result = verify_red_block_pair(self.paths, self.cues, config=self.config,
                                       transport=transport)
        self.assertTrue(result["accepted"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(len([row for row in requests[0]["messages"][0]["content"]
                              if row["type"] == "image_url"]), 4)
        self.assertNotIn("test-key", str(result))

    def test_one_uncertain_camera_abstains(self):
        def transport(_request):
            return {"choices": [{"message": {"content": json.dumps({
                "left": {"is_red_block": True, "confidence": .96},
                "right": {"is_red_block": False, "confidence": .96},
                "same_object": False})}}]}
        result = verify_red_block_pair(self.paths, self.cues, config=self.config,
                                       transport=transport)
        self.assertFalse(result["accepted"])

    def test_clipped_camera_never_calls_model(self):
        self.cues["left"]["complete_in_frame"] = False
        result = verify_red_block_pair(self.paths, self.cues, config=self.config,
                                       transport=lambda _request: self.fail("unexpected request"))
        self.assertEqual(result["reason"], "two_complete_crops_required")


if __name__ == '__main__':
    unittest.main()
