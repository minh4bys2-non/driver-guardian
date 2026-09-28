"""Generate full Python ORT reference outputs for the Android instrumentation test."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parents[1]
model = root / 'app/src/main/assets/models/backbone_neck.onnx'
output = root / 'app/src/androidTest/assets/backbone_neck'
output.mkdir(parents=True, exist_ok=True)
session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
images = (np.arange(3 * 640 * 640) % 251).astype(np.float32).reshape(1, 3, 640, 640) / np.float32(250)
features = session.run(['p3', 'p4', 'p5'], {'images': images})
for name, values in zip(['p3', 'p4', 'p5'], features):
    (output / f'{name}.f32').write_bytes(values.astype('<f4').tobytes())
    print(name, values.shape)
(output / 'reference.json').write_text(json.dumps({
    'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
    'onnxruntime': ort.__version__,
    'provider': 'CPUExecutionProvider',
    'input': 'float32(arange(3 * 640 * 640) % 251) / float32(250), NCHW',
    'atol': 0.0001,
    'rtol': 0.0001,
}, indent=2) + '\n')
