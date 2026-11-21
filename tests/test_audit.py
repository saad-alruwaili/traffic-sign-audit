"""Run with:  python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import traffic_sign_audit as tsa  # noqa: E402


class TestCore(unittest.TestCase):
    def test_missing_file_is_caught(self):
        with self.assertRaises(FileNotFoundError):
            tsa.read_image_bgr("nope.png")

    def test_corrupt_image_is_caught(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "bad.png"
            p.write_bytes(b"xxxx")
            with self.assertRaises(tsa.ImageReadError):
                tsa.extract_image_features(p)

    def test_feature_columns_and_values(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "a.png"
            cv2.imwrite(str(p), np.random.default_rng(1).integers(0, 255, (40, 50, 3), dtype=np.uint8))
            f = tsa.extract_image_features(p, roi=(5, 5, 30, 30))
            for c in tsa.FEATURE_COLUMNS + ["roi_brightness_gray", "width", "height", "brightness_hsv_v"]:
                self.assertIn(c, f)
            self.assertEqual((f["width"], f["height"]), (50, 40))
            self.assertTrue(f["roi_valid"])
            self.assertFalse(tsa.extract_image_features(p, roi=(30, 30, 5, 5))["roi_valid"])

    def test_flags_are_boolean(self):
        df = pd.DataFrame({"brightness_gray": np.linspace(10, 240, 100), "contrast_gray": np.linspace(5, 80, 100),
                           "sharpness_lapvar": np.linspace(1, 900, 100), "min_side_px": np.arange(15, 115)})
        out = tsa.assign_review_priority(df, tsa.compute_thresholds(df, 10, 90))
        self.assertEqual(out["review_flag"].dtype, bool)
        self.assertTrue(set(out["review_flag"].unique()) <= {True, False})
        self.assertTrue(out["priority_score"].between(0, 5).all())

    def test_roi_mask(self):
        m = tsa.roi_valid_mask([0, 5, np.nan], [0, 5, 0], [10, 2, 10], [10, 2, 10], 20, 20)
        self.assertEqual(m.tolist(), [True, False, False])

    def test_missing_dataset_folder(self):
        with self.assertRaises(FileNotFoundError):
            tsa.load_annotations("folder_that_does_not_exist")


if __name__ == "__main__":
    unittest.main()
