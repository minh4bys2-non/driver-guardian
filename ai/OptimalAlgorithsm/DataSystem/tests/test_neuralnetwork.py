from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from ai.LSTM.src.model import DrowsyClassifier
from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG
from ai.OptimalAlgorithsm.DataSystem.neuralnetwork import NeuralNetwork


class RecordingModel(nn.Module):
    def forward(self, frames, seq_lens):
        self.frames, self.lengths = frames, seq_lens
        self.inference = torch.is_inference_mode_enabled()
        return frames.new_tensor([[0., 2.]], dtype=torch.float32)


class NeuralNetworkTests(unittest.TestCase):
    def test_preprocessing_and_real_score(self):
        for dtype in (np.uint8, np.float64):
            with self.subTest(dtype=dtype):
                core = RecordingModel()
                model = NeuralNetwork(model=core, device='cpu', chunk_size=0)
                frames = np.ones((50, 3, 20, 40), dtype=dtype)
                score = model(frames)
                self.assertAlmostEqual(score, torch.tensor([0., 2.]).softmax(0)[1].item())
                self.assertEqual(core.frames.shape, (1, 50, 3, 224, 224))
                self.assertEqual(core.frames.dtype, torch.uint8 if dtype == np.uint8 else torch.float32)
                self.assertEqual(core.lengths.tolist(), [50])
                self.assertEqual(core.chunk_size, 0)
                self.assertFalse(core.training)
                self.assertTrue(core.inference)
                self.assertAlmostEqual(core.frames[0, 0, 0, 0, 0].item(), 114 if dtype == np.uint8 else 114 / 255)

    def test_invalid_frames_and_logits(self):
        core = RecordingModel()
        model = NeuralNetwork(model=core, device='cpu')
        for frames in (torch.zeros(0, 3, 8, 8), torch.zeros(2, 1, 8, 8),
                       torch.zeros(3, 8, 8), torch.zeros(2, 3, 0, 8),
                       torch.full((50, 3, 8, 8), float('nan')),
                       torch.full((50, 3, 8, 8), 2.), torch.zeros(50, 3, 8, 8, dtype=torch.int32)):
            with self.subTest(shape=frames.shape), self.assertRaises(ValueError):
                model(frames)
        with self.assertRaises(TypeError):
            model([])
        for logits in (torch.zeros(1, 1), torch.zeros(1, 3), torch.tensor([[0., float('inf')]])):
            with patch.object(core, 'forward', return_value=logits), self.assertRaises(ValueError):
                model(torch.zeros(50, 3, 224, 224))
        for chunk in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                NeuralNetwork(model=core, chunk_size=chunk)

    def test_rejects_unsampled_or_incomplete_windows(self):
        core = RecordingModel()
        model = NeuralNetwork(model=core, device='cpu')
        for count in (49, 51, 300):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'Expected 50 frames'):
                model(np.zeros((count, 3, 8, 8), dtype=np.uint8))
        for field, value in (('neural_fps', 0), ('neural_sec', float('nan')),
                             ('neural_sec', .25), ('neural_fps', float('inf'))):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                NeuralNetwork(model=core, config=replace(DEFAULT_CONFIG, **{field: value}))

    def test_versioned_checkpoint_roundtrip_and_config_validation(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        core = DrowsyClassifier(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1),
                                input_dim=4, hidden_dim=4, chunk_size=1).eval()
        state = dict(format_version=1, model=core.state_dict(), model_config=core.model_config,
                     config=dict(image_size=224, target_fps=5, window_sec=10), epoch=0)
        frames = torch.randint(0, 256, (50, 3, 224, 224), dtype=torch.uint8)
        with torch.inference_mode():
            expected = core(frames.unsqueeze(0)).softmax(-1)[0, 1].item()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.pt'
            torch.save(state, path)
            model = NeuralNetwork.from_checkpoint(path, device='cpu', chunk_size=0)
            self.assertAlmostEqual(model(frames), expected, places=6)
            self.assertAlmostEqual(model(frames.float() / 255), expected, places=6)
            for field, value in (('neural_size', 112), ('neural_fps', 10), ('neural_sec', 5)):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Checkpoint'):
                    NeuralNetwork(path, device='cpu', config=replace(DEFAULT_CONFIG, **{field: value}))
            broken = dict(state['model'])
            broken.pop(next(iter(broken)))
            torch.save(state | {'model': broken}, path)
            with self.assertRaises(RuntimeError):
                NeuralNetwork(path, device='cpu')
            torch.save({'state_dict': state['model']}, path)
            with self.assertRaisesRegex(ValueError, 'Unsupported checkpoint'):
                NeuralNetwork(path, device='cpu')
