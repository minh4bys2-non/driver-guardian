import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from ai.OptimalAlgorithsm.DataSystem import engine_1
from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG
from ai.OptimalAlgorithsm.DataSystem.normalization import PhysicalNormalizer, FEATURE_ENCODING
from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import load_jsonl


class Engine1Tests(unittest.TestCase):
    def setUp(self):
        self.normalizer = PhysicalNormalizer(DEFAULT_CONFIG)
        self.raw_template = {
            "blink_frequency": 4.0,
            "blink_duration": 100.0,
            "perclos": 0.05,
            "yawn_frequency": 1.0,
            "nod_duration": 0.5,
            "nod_frequency": 2.0,
            "dominant_head_motion_frequency": 0.2,
            "neuralnetwork_score": 0.123,
        }
        self.norm_template = self.normalizer.transform(self.raw_template)
        self.sample_record = {
            **self.norm_template,
            "raw_features": dict(self.raw_template),
            "normalization_id": self.normalizer.id,
            "normalization_scales": dict(self.normalizer.scales),
            "source_video": "Fold1_part1/01/0.mov",
            "start_second": 60.0,
            "end_second": 120.0,
            "label": 0,
            "feature_encoding": FEATURE_ENCODING,
        }

    def test_verify_sample_match_success(self):
        # Tạo bản sao nhưng thay đổi điểm neural score
        updated = dict(self.sample_record)
        updated["raw_features"] = dict(self.sample_record["raw_features"])
        updated["raw_features"]["neuralnetwork_score"] = 0.85
        updated["neuralnetwork_score"] = 0.85

        # Không phát sinh lỗi
        engine_1.Engine._verify_sample_match(self.sample_record, updated)

    def test_verify_sample_match_rejects_altered_physical_features(self):
        updated = dict(self.sample_record)
        updated["raw_features"] = dict(self.sample_record["raw_features"])
        # Thay đổi một đặc trưng vật lý
        updated["perclos"] = 0.99
        with self.assertRaises(ValueError) as ctx:
            engine_1.Engine._verify_sample_match(self.sample_record, updated)
        self.assertIn("perclos", str(ctx.exception))

    def test_verify_sample_match_rejects_metadata_mismatch(self):
        for key, bad_val in (
            ("source_video", "Fold1_part1/01/other.mov"),
            ("start_second", 0.0),
            ("end_second", 180.0),
            ("label", 1),
            ("feature_encoding", "other_encoding"),
        ):
            with self.subTest(key=key):
                updated = dict(self.sample_record)
                updated[key] = bad_val
                with self.assertRaises(ValueError):
                    engine_1.Engine._verify_sample_match(self.sample_record, updated)

    @patch.object(engine_1, "NeuralNetwork")
    @patch.object(engine_1, "read_video")
    def test_process_video_runs_neural_and_preserves_physical(self, mock_read_video, mock_neural_cls):
        mock_neural_instance = MagicMock()
        mock_neural_instance.return_value = torch.tensor(0.777)
        mock_neural_cls.return_value.eval.return_value = mock_neural_instance

        # Giả lập 3600 frames (2 phút @ 30 FPS)
        dummy_frame = np.full((3, 10, 10), 128, dtype=np.uint8)
        frames = ((dummy_frame, i / 30.0) for i in range(3600))
        mock_read_video.return_value = frames

        with tempfile.TemporaryDirectory() as tmpdir:
            # Tạo file video giả để check _source
            vid_file = Path(tmpdir) / "Fold1_part1/01/0.mov"
            vid_file.parent.mkdir(parents=True, exist_ok=True)
            vid_file.touch()

            engine = engine_1.Engine(video_root=tmpdir)
            target_records = {
                (60.0, 120.0): self.sample_record
            }

            samples = list(engine.process_video(
                "Fold1_part1/01/0.mov",
                0,
                target_records=target_records,
            ))

            self.assertEqual(len(samples), 1)
            sample = samples[0]

            # Kiểm tra neuralnetwork_score mới
            self.assertAlmostEqual(sample["neuralnetwork_score"], 0.777, places=5)
            self.assertAlmostEqual(sample["raw_features"]["neuralnetwork_score"], 0.777, places=5)

            # Kiểm tra toàn bộ 7 đặc trưng vật lý được bảo tồn
            for k in engine_1.PHYSICAL_FEATURES:
                self.assertAlmostEqual(sample[k], self.sample_record[k], places=7)
                self.assertAlmostEqual(sample["raw_features"][k], self.sample_record["raw_features"][k], places=7)

    @patch.object(engine_1, "NeuralNetwork")
    @patch.object(engine_1, "read_video")
    def test_run_e2e_with_resume_and_load_jsonl_compatibility(self, mock_read_video, mock_neural_cls):
        mock_neural_instance = MagicMock()
        mock_neural_instance.return_value = torch.tensor(0.654)
        mock_neural_cls.return_value.eval.return_value = mock_neural_instance

        # 3600 frames
        dummy_frame = np.full((3, 10, 10), 128, dtype=np.uint8)
        mock_read_video.side_effect = lambda *args, **kwargs: (
            (dummy_frame, i / 30.0) for i in range(3600)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vid_file = tmp_path / "Fold1_part1/01/0.mov"
            vid_file.parent.mkdir(parents=True, exist_ok=True)
            vid_file.touch()

            # Tạo input jsonl với 2 cửa sổ
            rec1 = dict(self.sample_record)
            rec2 = dict(self.sample_record)
            rec2["start_second"] = 0.0
            rec2["end_second"] = 60.0

            input_jsonl = tmp_path / "input.jsonl"
            output_jsonl = tmp_path / "output.jsonl"

            with input_jsonl.open("w", encoding="utf-8") as f:
                f.write(json.dumps(rec2) + "\n")
                f.write(json.dumps(rec1) + "\n")

            engine = engine_1.Engine(video_root=tmpdir)
            count = engine.run(input_jsonl, output_jsonl, strict=True, resume=False)
            self.assertEqual(count, 2)

            # Đọc lại bằng fitnessFunction.load_jsonl xem có tương thích 100% không
            X, y, meta = load_jsonl(output_jsonl, include_metadata=True)
            self.assertEqual(len(X), 2)
            self.assertEqual(len(y), 2)
            self.assertEqual(X.shape, (2, 8))
            self.assertEqual(meta["input_schema"]["feature_encoding"], FEATURE_ENCODING)

            # Test resume: chạy lại với resume=True, không có mẫu mới nào cần ghi
            count_resumed = engine.run(input_jsonl, output_jsonl, strict=True, resume=True)
            self.assertEqual(count_resumed, 0)


if __name__ == "__main__":
    unittest.main()
