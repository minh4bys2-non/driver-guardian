import hashlib
import json
from pathlib import Path


MODEL_FIELDS = ("nc", "reg_max", "backbone_w", "backbone_n", "neck_n", "strides")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def signature(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def architecture(config):
    return json.loads(canonical_json({key: getattr(config, key) for key in MODEL_FIELDS}))

def normalize_categories(records):
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Categories rỗng hoặc trùng id")
    result = []
    for index, record in enumerate(sorted(records, key=lambda r: r["id"])):
        if not isinstance(record["id"], int) or not isinstance(record["name"], str) or not record["name"].strip():
            raise ValueError("Category phải có id số nguyên và name không rỗng")
        if "index" in record and record["index"] != index:
            raise ValueError("Category index không khớp thứ tự output của model")
        result.append({"index": index, "id": record["id"], "name": record["name"]})
    return result


def build_metadata(cfg, categories):
    model = architecture(cfg)
    categories = normalize_categories(categories)
    if model["nc"] != len(categories):
        raise ValueError(f"nc={model['nc']} không khớp {len(categories)} categories")
    return {"format_version": 1, "task": "object_detection", "architecture": model,
            "architecture_sha256": signature(model), "categories": categories,
            "categories_sha256": signature(categories)}


def checkpoint_dir(cfg):
    metadata = cfg.checkpoint_metadata
    if not cfg.run_name or Path(cfg.run_name).name != cfg.run_name or cfg.run_name in (".", ".."):
        raise ValueError("run_name phải là một tên thư mục")
    return Path(cfg.ckpt_dir) / metadata["architecture_sha256"] / metadata["categories_sha256"] / cfg.run_name


def write_metadata(directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "architecture.json": {"model": metadata["architecture"], "sha256": metadata["architecture_sha256"]},
        "categories.json": metadata["categories"],
    }
    for name, value in files.items():
        path = directory / name
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError(f"Metadata không khớp, không ghi đè: {path}")
        else:
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_metadata(path, checkpoint, model=None, expected=None):
    metadata = checkpoint.get("metadata")
    if not metadata:
        raise ValueError("Checkpoint thiếu chữ ký kiến trúc/categories; không thể xác minh để nạp")
    if metadata.get("format_version") != 1 or metadata.get("task") != "object_detection":
        raise ValueError("Checkpoint không đúng định dạng object detection")
    if signature(metadata["architecture"]) != metadata["architecture_sha256"]:
        raise ValueError("Chữ ký SHA-256 kiến trúc không hợp lệ")
    categories = normalize_categories(metadata["categories"])
    if signature(categories) != metadata["categories_sha256"] or len(categories) != metadata["architecture"]["nc"]:
        raise ValueError("Chữ ký/mapping categories không hợp lệ")
    if model is not None and architecture(model) != metadata["architecture"]:
        raise ValueError("Kiến trúc model không khớp checkpoint; không tự thay head")
    if expected is not None and metadata != expected:
        raise ValueError("Kiến trúc hoặc categories của dataset không khớp checkpoint")
    directory = Path(path).parent
    expected_files = {
        "architecture.json": {"model": metadata["architecture"], "sha256": metadata["architecture_sha256"]},
        "categories.json": categories,
    }
    for name, value in expected_files.items():
        file = directory / name
        if not file.is_file() or json.loads(file.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Thiếu hoặc sai metadata đi kèm checkpoint: {file}")
    return metadata
