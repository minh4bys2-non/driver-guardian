"""Tiện ích dữ liệu cũ 60 giây, không chuyển số đo thành cửa sổ 30 giây.

Giữ cửa sổ [0,60), [60,120), ... có sẵn trong JSONL bước 10 giây."""

import json
import math
from pathlib import Path


def convert(input_path, output_path):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("Input and output must be different files")

    kept, seen = [], set()
    total = 0
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
                video, start, end = row["source_video"], row["start_second"], row["end_second"]
                if not isinstance(video, str) or not video.strip():
                    raise ValueError("Missing source_video")
                if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
                        or start < 0 or not math.isclose(end - start, 60, rel_tol=0, abs_tol=1e-6)):
                    raise ValueError("Expected a finite, nonnegative start and a 60-second window")
                if not math.isclose(start, round(start / 10) * 10, rel_tol=0, abs_tol=1e-6):
                    raise ValueError("Window start must align with the 10-second grid")
                total += 1
                slot = round(start / 60)
                if not math.isclose(start, slot * 60, rel_tol=0, abs_tol=1e-6):
                    continue
                key = (video, slot)
                if key in seen:
                    raise ValueError("Duplicate 60-second window")
                seen.add(key)
                kept.append(line.rstrip("\r\n") + "\n")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{input_path}: line {line_number}: {exc}") from exc

    if not kept:
        raise ValueError("No windows align with the 60-second grid")
    # Validate all rows before creating output; never overwrite an existing dataset.
    with output_path.open("x", encoding="utf-8") as destination:
        destination.writelines(kept)
    return total, len(kept)


if __name__ == "__main__":
    INPUT_JSONL = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/drowsiness_risk_10.jsonl"
    OUTPUT_JSONL = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/drowsiness_risk_60.jsonl"

    total, kept = convert(INPUT_JSONL, OUTPUT_JSONL)
    print(f"Kept {kept}/{total} samples; removed {total - kept} intermediate windows.")
    print(f"Output: {OUTPUT_JSONL}")
