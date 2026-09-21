"""Interactive multi-object SAM3.1 and SeC-4B video segmentation."""

from __future__ import annotations

import asyncio
import json
import math
import base64
import io as py_io
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import logging
import importlib
import warnings
from fractions import Fraction
import uuid
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image
from typing_extensions import override

import comfy.model_management
import comfy.model_detection
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io
from tqdm import tqdm


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Importing from timm\.models\.layers is deprecated.*",
)


NODE_ID = "CS_Video_Segment_SAM3"
PROMPT_VERSION = 2
MULTI_ANCHOR_PROMPT_VERSION = 3
_PROPAGATION_OPTIONS = ["both", "forward", "backward"]
_PREVIEW_ROUTE_REGISTERED = False
_LAST_MODEL: Any = None
# SAM3.1 preview models are keyed by source path. ComfyUI's ModelPatcher
# remains responsible for GPU residency; this cache only avoids loading a
# second Python model object when Preview is clicked repeatedly.
_PREVIEW_MODEL_CACHE: dict[str, Any] = {}
_SAM3_CLIP_CACHE: dict[str, Any] = {}
_LAST_MODEL_SOURCE: dict[str, Any] | None = None
_SEC_MODEL_REGISTRY: dict[str, Any] = {}
_SEC_MODEL_NODE_TOKENS: dict[str, str] = {}
_SEC_MODEL_LOCK = threading.RLock()
_SEC_MODEL_LOAD_LOCK = threading.Lock()
_SEC_MODEL_DOWNLOAD_LOCK = threading.Lock()
_SEC_PREVIEW_MODEL_TOKEN: str | None = None
_SEC_PACKAGE_PATH = Path(__file__).resolve().parent / "sec_inference"
_SEC_CONFIG_PATH = Path(__file__).resolve().parent / "sec_configs"
_SEC_MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "sec_model_config"
_SEC_MODEL_FOLDER = "sec_models"
_SEC_WEIGHT_SPECS = {
    "SeC-4B-bf16.safetensors": (torch.bfloat16, "https://huggingface.co/VeryAladeen/Sec-4B/resolve/main/SeC-4B-bf16.safetensors"),
    "SeC-4B-fp16.safetensors": (torch.float16, "https://huggingface.co/VeryAladeen/Sec-4B/resolve/main/SeC-4B-fp16.safetensors"),
}
_SEC_DEFAULT_WEIGHT_FILENAME = "SeC-4B-bf16.safetensors"
_SELECTOR_CACHE_LIMIT = 8
_SELECTOR_CACHE_MAX_BYTES = 4 * 1024**3
_SEGMENT_LOGGER = logging.getLogger("CineStyleVideoSegment")
_NESTED_TQDM_LOCK = threading.Lock()
_PREVIEW_CACHE_STORE = None


def _preview_cache_store():
    global _PREVIEW_CACHE_STORE
    if _PREVIEW_CACHE_STORE is None:
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        if module is None:
            raise RuntimeError("CineStyle preview cache module is unavailable.")
        _PREVIEW_CACHE_STORE = module.PreviewCacheStore("video_segment", max_entries=_SELECTOR_CACHE_LIMIT, max_bytes=_SELECTOR_CACHE_MAX_BYTES)
    return _PREVIEW_CACHE_STORE


def _loader_preview_cache():
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_loader_preview_cache")
    return module.get_loader_preview_cache() if module is not None else None


def _cache_wait_input(
    node_id: Any,
    prompt: Any,
    images: torch.Tensor,
    input_names: tuple[str, ...],
    video_input: Any = None,
) -> dict[str, Any] | None:
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_preview_cache")
    if module is None:
        return None
    chain = module.build_input_chain(prompt, node_id, input_names)
    if chain is None:
        return None
    try:
        return module.get_wait_input_cache_store().put_chain(
            chain,
            images,
            _video_input_fps(video_input, prompt, node_id),
            info={
                "producer_node_id": str(node_id or ""),
                "producer_node_type": "CS_Video_Segment",
            },
            force=True,
        )
    except Exception as exc:
        _segment_info("CS Video Segment", f"wait input cache failed: {exc}")
        return None


def _no_nested_tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
    return iterable


class _NestedTqdmSilencer:
    """Temporarily hide progress bars emitted inside the bundled model kernels."""

    def __init__(self, module_names: tuple[str, ...]):
        self.module_names = module_names
        self._patched: list[tuple[Any, Any]] = []
        self._locked = False

    def start(self) -> None:
        _NESTED_TQDM_LOCK.acquire()
        self._locked = True
        for module_name in self.module_names:
            module = sys.modules.get(module_name)
            if module is None:
                try:
                    module = importlib.import_module(module_name)
                except Exception:
                    continue
            if module is None or not hasattr(module, "tqdm"):
                continue
            self._patched.append((module, module.tqdm))
            module.tqdm = _no_nested_tqdm

    def stop(self) -> None:
        for module, original in reversed(self._patched):
            module.tqdm = original
        self._patched.clear()
        if self._locked:
            self._locked = False
            _NESTED_TQDM_LOCK.release()


def _segment_info(node_name: str, message: str) -> None:
    _SEGMENT_LOGGER.info("[%s] %s", node_name, message)


class _SegmentProgress:
    """Forward ComfyUI progress updates while emitting throttled tqdm-style logs."""

    def __init__(self, node_name: str, total: int, backend: Any = None):
        self.node_name = node_name
        self.total = max(1, int(total))
        self.backend = backend
        self.bar = tqdm(
            total=self.total,
            desc=f"[INFO] [{node_name}] frame processing",
            unit="frame",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            mininterval=0.1,
            dynamic_ncols=True,
            leave=True,
        )

    def update(self, amount: int = 1) -> None:
        step = max(0, int(amount))
        if self.backend is not None:
            self.backend.update(step)
        self.bar.update(step)

    def close(self) -> None:
        self.bar.close()


def _segment_expected_frames(frame_count: int, anchor: int, direction: str, limit: int | None = None) -> int:
    direction = "both" if direction == "bidirectional" else direction
    cap = None if limit is None or int(limit) < 0 else max(1, int(limit))
    total = 0
    if direction in {"both", "forward"} and anchor + 1 < frame_count:
        total += min(cap, frame_count - anchor) if cap is not None else frame_count - anchor
    if direction in {"both", "backward"} and anchor > 0:
        total += min(cap, anchor + 1) if cap is not None else anchor + 1
    return max(1, total)

try:
    folder_paths.add_model_folder_path("sams", os.path.join(folder_paths.models_dir, "sams"))
except Exception:
    pass


def _video_files() -> list[str]:
    import os

    input_dir = folder_paths.get_input_directory()
    names = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    return sorted(folder_paths.filter_files_content_types(names, ["video"]))


def _resolve_video_path(video: str) -> str:
    value = str(video or "").strip()
    if not value:
        raise ValueError("Choose a video file before requesting a preview.")
    if folder_paths.exists_annotated_filepath(value):
        return folder_paths.get_annotated_filepath(value)
    candidate = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if candidate.is_file():
        return str(candidate)
    raise ValueError(f"Video file not found: {value}")


def _decode_video(video: str) -> torch.Tensor:
    from comfy_api.latest import InputImpl

    components = InputImpl.VideoFromFile(
        _resolve_video_path(video)
    ).get_components()
    images = components.images
    if not isinstance(images, torch.Tensor):
        raise ValueError("Video decoder did not return a tensor of frames.")
    return images


def _decode_video_frame(video: str, frame_index: int) -> torch.Tensor:
    path = _resolve_video_path(video)
    target = max(0, int(frame_index))
    with av.open(path, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The selected file contains no video stream.")
        stream = container.streams.video[0]
        for index, decoded in enumerate(container.decode(stream)):
            if index == target:
                array = decoded.to_ndarray(format="rgb24")
                return torch.from_numpy(array).unsqueeze(0).float().div_(255.0)
    raise ValueError(f"Frame {target} is outside the selected video.")


def _looks_like_image_file(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if " [" in text:
        text = text.split(" [", 1)[0]
    return text.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"))


def _resolve_image_path(image: str) -> str:
    value = str(image or "").strip()
    if not value:
        raise ValueError("Choose an image file before requesting a preview.")
    if folder_paths.exists_annotated_filepath(value):
        return folder_paths.get_annotated_filepath(value)
    candidate = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if candidate.is_file():
        return str(candidate)
    raise ValueError(f"Image file not found: {value}")


def _decode_image_frame(image: str, frame_index: int) -> torch.Tensor:
    if int(frame_index) != 0:
        raise ValueError("A Load Image source contains one frame.")
    with Image.open(_resolve_image_path(image)) as decoded:
        array = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).unsqueeze(0).float().div_(255.0)


def _prompt_node(prompt: Any, node_id: Any) -> dict[str, Any] | None:
    if not isinstance(prompt, dict):
        return None
    return prompt.get(str(node_id)) or prompt.get(node_id)


def _looks_like_video_file(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if " [" in text:
        text = text.split(" [", 1)[0]
    return text.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"))


def _prompt_has_file_video_source(prompt: Any, node_id: Any) -> bool:
    node = _prompt_node(prompt, node_id)
    if not node:
        return False
    inputs = node.get("inputs") if isinstance(node, dict) else None
    pending: list[Any] = []
    if isinstance(inputs, dict):
        pending.extend(inputs.get(name) for name in ("images", "video_input"))
    visited: set[str] = set()
    while pending:
        link = pending.pop(0)
        if not isinstance(link, (list, tuple)) or len(link) < 2:
            continue
        upstream_id = str(link[0])
        if upstream_id in visited:
            continue
        visited.add(upstream_id)
        upstream = _prompt_node(prompt, upstream_id)
        if not isinstance(upstream, dict):
            continue
        class_type = str(upstream.get("class_type") or "")
        upstream_inputs = upstream.get("inputs")
        if not isinstance(upstream_inputs, dict):
            continue
        if re.search(r"load.*video|video.*load", class_type, re.IGNORECASE):
            return True
        for name, value in upstream_inputs.items():
            if any(token in str(name).lower() for token in ("video", "file", "filename", "path")):
                if _looks_like_video_file(value):
                    return True
        pending.extend(
            value
            for value in upstream_inputs.values()
            if isinstance(value, (list, tuple)) and len(value) >= 2
        )
    return False


def _prompt_loader_id(prompt: Any, node_id: Any) -> str:
    """Find a CS Load Video upstream when only its IMAGE output is connected."""
    node = _prompt_node(prompt, node_id)
    if not isinstance(node, dict):
        return ""
    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
    pending: list[Any] = [value for name, value in inputs.items() if "video" in str(name).lower() or "image" in str(name).lower()]
    visited: set[str] = set()
    while pending:
        link = pending.pop(0)
        if not isinstance(link, (list, tuple)) or len(link) < 2:
            continue
        upstream_id = str(link[0])
        if upstream_id in visited:
            continue
        visited.add(upstream_id)
        upstream = _prompt_node(prompt, upstream_id)
        if not isinstance(upstream, dict):
            continue
        class_type = str(upstream.get("class_type") or "")
        if class_type == "CS_Load_Video" or class_type.endswith(".CS_Load_Video") or class_type.endswith("::CS_Load_Video"):
            return upstream_id
        upstream_inputs = upstream.get("inputs") if isinstance(upstream.get("inputs"), dict) else {}
        pending.extend(value for value in upstream_inputs.values() if isinstance(value, (list, tuple)) and len(value) >= 2)
    return ""


def _prompt_selector_fps(prompt: Any, node_id: Any) -> float | None:
    node = _prompt_node(prompt, node_id)
    if not node:
        return None
    inputs = node.get("inputs") if isinstance(node, dict) else None
    pending: list[Any] = []
    if isinstance(inputs, dict):
        pending.extend(inputs.get(name) for name in ("images", "video_input"))
    visited: set[str] = set()
    while pending:
        link = pending.pop(0)
        if not isinstance(link, (list, tuple)) or len(link) < 2:
            continue
        upstream_id = str(link[0])
        if upstream_id in visited:
            continue
        visited.add(upstream_id)
        upstream = _prompt_node(prompt, upstream_id)
        if not isinstance(upstream, dict):
            continue
        upstream_inputs = upstream.get("inputs")
        if not isinstance(upstream_inputs, dict):
            continue
        for name in ("fps", "frame_rate", "target_fps"):
            value = upstream_inputs.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
                return float(value)
        pending.extend(
            value
            for value in upstream_inputs.values()
            if isinstance(value, (list, tuple)) and len(value) >= 2
        )
    return None


def _video_input_fps(video_input: Any, prompt: Any = None, node_id: Any = None) -> float:
    if video_input is not None:
        try:
            # get_components() decodes the whole clip to float32 just to read
            # the rate; get_frame_rate() reads it from the container metadata.
            fps = float(video_input.get_frame_rate())
            if math.isfinite(fps) and fps > 0:
                return fps
        except Exception:
            pass
    return _prompt_selector_fps(prompt, node_id) or 24.0


def _cache_selector_input(node_id: Any, images: torch.Tensor, fps: float) -> str | None:
    key = str(node_id or "").strip()
    if not key or not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        return None
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 24.0
    try:
        entry = _preview_cache_store().put_preview(key, images, safe_fps, encode_video=True)
    except Exception as exc:
        print(f"[CineStyle] Selector input cache failed for node {key}: {exc}")
        return None
    print(
        f"[CineStyle] Cached selector input for node {key}: "
        f"{entry['info']['frames']} frames at {safe_fps:.3f} fps."
    )
    return str(entry["token"])


def _selector_cache_for_node(node_id: Any) -> dict[str, Any] | None:
    return _preview_cache_store().get_preview_variant(node_id, proxy=False)


def _selector_cache_for_token(token: Any) -> dict[str, Any] | None:
    return _preview_cache_store().get_token(token)


def _decode_selector_frame(payload: dict[str, Any], frame_index: int) -> torch.Tensor:
    token = str(payload.get("source_token") or "").strip()
    if not token:
        source = str(payload.get("video") or "")
        if str(payload.get("source_kind") or "").lower() == "image" or _looks_like_image_file(source):
            return _decode_image_frame(source, frame_index)
        return _decode_video_frame(source, frame_index)
    if token.startswith("loader_preview:"):
        cache = _loader_preview_cache()
        if cache is None:
            raise ValueError("The shared loader preview cache is unavailable.")
        return cache.decode_frame(token, frame_index)
    if token.startswith("wait_input:"):
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        if module is None:
            raise ValueError("The shared wait input preview cache is unavailable.")
        return module.get_wait_input_cache_store().decode_frame(payload, frame_index)
    entry = _selector_cache_for_token(token)
    if entry is None:
        raise ValueError("The cached Selector input is no longer available. Run the workflow once again.")
    try:
        frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("The cached Selector frames are unavailable. Run the workflow once again.") from exc
    target = int(frame_index)
    if target < 0 or target >= int(frames.shape[0]):
        raise ValueError(f"Frame {target} is outside the cached input.")
    frame = np.array(frames[target : target + 1], copy=True)
    return torch.from_numpy(frame).to(torch.float32).div_(255.0)


async def _selector_cache_info_route(request: web.Request) -> web.Response:
    requested_node_id = str(request.query.get("node_id", ""))
    entry = _selector_cache_for_node(requested_node_id)
    if entry is None:
        return web.json_response({"error": "No cached Selector input."}, status=404)
    token = str(entry["token"])
    return web.json_response(
        {
            "token": token,
            "label": "Cached input from the last workflow run",
            "video_url": f"/cinestyle/video-selector-cache-video?token={token}",
            "info": entry["info"],
        }
    )


async def _selector_cache_video_route(request: web.Request) -> web.StreamResponse:
    entry = _selector_cache_for_token(request.query.get("token", ""))
    if entry is None or not Path(str(entry.get("path") or "")).is_file():
        return web.json_response({"error": "Cached Selector video not found."}, status=404)
    return web.FileResponse(
        path=str(entry["path"]),
        headers={"Cache-Control": "no-store"},
    )


def _parse_json(value: str | None, name: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc.msg}.") from exc


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _pixel_coordinate(value: Any, size: int, name: str) -> float:
    result = _number(value, name)
    # The selector writes normalized coordinates. Pixel coordinates are also
    # accepted so values copied from the official SAM3 Detect node work too.
    if 0.0 <= result <= 1.0:
        result *= size
    return max(0.0, min(float(size), result))


def _parse_points(value: str | None, width: int, height: int) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    raw = _parse_json(value, "points")
    if raw is None:
        return [], []
    if isinstance(raw, dict):
        raw = raw.get("points", [raw])
    if not isinstance(raw, list):
        raise ValueError("points must be a JSON list of {x, y, label} objects.")

    positive: list[dict[str, float]] = []
    negative: list[dict[str, float]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            if "x" not in item or "y" not in item:
                raise ValueError(f"points[{index}] needs x and y.")
            label = item.get("label", 1)
            if isinstance(label, str):
                label = 0 if label.lower() in {"negative", "neg", "background", "0"} else 1
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label = item[2] if len(item) >= 3 else 1
            item = {"x": item[0], "y": item[1]}
        else:
            raise ValueError(f"points[{index}] must be an object or [x, y, label].")

        point = {
            "x": _pixel_coordinate(item["x"], width, f"points[{index}].x"),
            "y": _pixel_coordinate(item["y"], height, f"points[{index}].y"),
        }
        (negative if int(label) == 0 else positive).append(point)

    if not positive and not negative:
        raise ValueError("points must contain at least one point.")
    return positive, negative


def _parse_bbox(value: str | None, width: int, height: int) -> dict[str, float]:
    raw = _parse_json(value, "bbox")
    if raw is None:
        raise ValueError("bbox mode requires a JSON box.")
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], (dict, list)):
            raw = raw[0]
        elif len(raw) >= 4:
            raw = {"x": raw[0], "y": raw[1], "w": raw[2], "h": raw[3]}
    if not isinstance(raw, dict):
        raise ValueError("bbox must be {x, y, w, h} or [x, y, w, h].")

    x = _pixel_coordinate(raw.get("x", 0), width, "bbox.x")
    y = _pixel_coordinate(raw.get("y", 0), height, "bbox.y")
    width_value = raw.get("w", raw.get("width"))
    height_value = raw.get("h", raw.get("height"))
    if width_value is None or height_value is None:
        raise ValueError("bbox needs w/h (or width/height).")
    box_width = _number(width_value, "bbox.w")
    box_height = _number(height_value, "bbox.h")
    if 0.0 <= box_width <= 1.0:
        box_width *= width
    if 0.0 <= box_height <= 1.0:
        box_height *= height
    box_width = min(float(width) - x, box_width)
    box_height = min(float(height) - y, box_height)
    if box_width <= 0 or box_height <= 0:
        raise ValueError("bbox must have positive width and height inside the frame.")
    return {"x": x, "y": y, "width": box_width, "height": box_height}


def _decode_prompt_mask(value: Any, width: int, height: int) -> torch.Tensor | None:
    """Decode a Selector PNG mask into a soft ``[H, W]`` tensor."""
    if not value:
        return None
    if isinstance(value, dict):
        encoded = value.get("data") or value.get("png") or value.get("base64")
    else:
        encoded = value
    text = str(encoded or "").strip()
    if not text:
        return None
    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(text, validate=True)
        with Image.open(py_io.BytesIO(image_bytes)) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError("prompt_data contains an invalid mask PNG.") from exc
    alpha = rgba[..., 3]
    if alpha.size == 0 or float(alpha.max()) <= 0.0:
        alpha = rgba[..., :3].mean(axis=-1)
    mask = torch.from_numpy(alpha.copy()).float()
    if tuple(mask.shape) != (height, width):
        mask = F.interpolate(mask[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    if not bool((mask > 0.01).any()):
        return None
    return mask.clamp_(0.0, 1.0)


def _parse_prompt_data(value: str | None, width: int, height: int) -> list[dict[str, Any]]:
    """Parse the Selector's versioned per-object mask/box/point protocol."""
    raw = _parse_json(value, "prompt_data")
    if raw is None:
        raise ValueError("prompt_data is empty. Open the Selector and define at least one object.")
    objects = raw.get("objects") if isinstance(raw, dict) else raw
    if not isinstance(objects, list):
        raise ValueError("prompt_data must contain an objects list.")
    parsed_objects: list[dict[str, Any]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"prompt_data.objects[{index}] must be an object.")
        raw_points = item.get("points") or []
        points: list[dict[str, float | int]] = []
        if raw_points:
            positive, negative = _parse_points(json.dumps(raw_points), width, height)
            points = [{**point, "label": 1} for point in positive] + [{**point, "label": 0} for point in negative]
        box = None
        raw_box = item.get("bbox") or item.get("box")
        if raw_box:
            box = _parse_bbox(json.dumps(raw_box), width, height)
        mask = _decode_prompt_mask(item.get("mask"), width, height)
        text = str(item.get("text") or item.get("semantic") or "").strip()
        if not text and not points and box is None and mask is None:
            raise ValueError(f"prompt_data.objects[{index}] has no semantic, mask, bbox, or point prompt.")
        parsed_objects.append({"text": text, "points": points, "bbox": box, "mask": mask})
    if not parsed_objects:
        raise ValueError("prompt_data must contain at least one prompted object.")
    return parsed_objects


def _sam3_mask_logits(mask: torch.Tensor, device: Any, dtype: torch.dtype) -> torch.Tensor:
    """Convert a ComfyUI 0..1 brush mask into a coarse SAM3 prompt logit."""
    # SAM3 receives decoder-style logits rather than a binary output mask.
    return torch.logit(mask.to(device=device, dtype=dtype).clamp(0.05, 0.95))[None, None]


def _preview_model(source: Any) -> Any:
    global _LAST_MODEL_SOURCE
    if isinstance(source, dict) and source.get("name"):
        _LAST_MODEL_SOURCE = dict(source)
    if not isinstance(source, dict):
        return _LAST_MODEL
    kind = str(source.get("kind") or "")
    name = str(source.get("name") or "")
    if not name:
        return _LAST_MODEL
    if kind == "checkpoint":
        path = folder_paths.get_full_path_or_raise("checkpoints", name)
        model = _PREVIEW_MODEL_CACHE.get(path)
        if model is None:
            model, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
                path,
                output_vae=False,
                output_clip=True,
                output_clipvision=False,
                output_model=True,
            )
            if model is None:
                raise ValueError(f"Unable to load SAM3 checkpoint: {name}")
            _PREVIEW_MODEL_CACHE[path] = model
            if clip is not None:
                _SAM3_CLIP_CACHE[path] = clip
        return model
    if kind == "diffusion_model":
        path = folder_paths.get_full_path_or_raise("diffusion_models", name)
        model = _PREVIEW_MODEL_CACHE.get(path)
        if model is None:
            model = comfy.sd.load_diffusion_model(path)
            _PREVIEW_MODEL_CACHE[path] = model
        return model
    return _LAST_MODEL


def _sam3_clip_from_path(path: str) -> Any:
    """Load the SAM3 text encoder paired with a checkpoint, once per path."""
    clip = _SAM3_CLIP_CACHE.get(path)
    if clip is not None:
        return clip
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    config = comfy.model_detection.model_config_from_unet(state_dict, prefix, metadata=metadata)
    if config is None:
        raise ValueError(f"Unable to identify SAM3 checkpoint: {Path(path).name}")
    # The official ComfyUI SAM3 config stashes its embedded CLIP weights while
    # normalizing the model state dict. This avoids constructing a duplicate
    # full SAM3 model just to obtain the text encoder.
    config.process_unet_state_dict(state_dict)
    clip_sd = config.process_clip_state_dict({})
    if not clip_sd:
        raise ValueError(f"SAM3 checkpoint has no text encoder: {Path(path).name}")
    clip_target = config.clip_target(state_dict=clip_sd)
    if clip_target is None:
        raise ValueError(f"SAM3 checkpoint has no compatible text encoder: {Path(path).name}")
    clip = comfy.sd.CLIP(
        clip_target,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        tokenizer_data=clip_sd,
        parameters=comfy.utils.calculate_parameters(clip_sd),
        state_dict=clip_sd,
    )
    _SAM3_CLIP_CACHE[path] = clip
    return clip


def _sam3_clip_from_model(model: Any) -> Any:
    """Build the paired Comfy CLIP from the SAM3 model's stashed text weights."""
    cache_key = f"model:{id(model)}"
    if cache_key in _SAM3_CLIP_CACHE:
        return _SAM3_CLIP_CACHE[cache_key]
    base = getattr(model, "model", None)
    config = getattr(base, "model_config", None)
    stash = getattr(config, "_clip_stash", None)
    if not stash:
        return None
    clip_sd = config.process_clip_state_dict({})
    if not clip_sd:
        return None
    clip_target = config.clip_target(state_dict=clip_sd)
    if clip_target is None:
        return None
    clip = comfy.sd.CLIP(
        clip_target,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        tokenizer_data=clip_sd,
        parameters=comfy.utils.calculate_parameters(clip_sd),
        state_dict=clip_sd,
    )
    _SAM3_CLIP_CACHE[cache_key] = clip
    return clip


def _sam3_text_embeddings(model: Any, text: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode a Selector semantic prompt with ComfyUI's official SAM3 CLIP."""
    attached = None
    getter = getattr(model, "get_attachment", None)
    if callable(getter):
        attached = getter("sam3_clip")
    clip = attached or _sam3_clip_from_model(model)
    if clip is None and _LAST_MODEL_SOURCE and _LAST_MODEL_SOURCE.get("kind") == "checkpoint":
        path = folder_paths.get_full_path_or_raise("checkpoints", str(_LAST_MODEL_SOURCE.get("name")))
        clip = _sam3_clip_from_path(path)
    if clip is None:
        candidates = [
            name for name in folder_paths.get_filename_list("checkpoints")
            if "sam3" in str(name).lower()
        ]
        if not candidates:
            raise ValueError("SAM3 Semantic needs a SAM3 checkpoint with its official CLIP text encoder.")
        clip = _sam3_clip_from_path(folder_paths.get_full_path_or_raise("checkpoints", candidates[0]))
    setter = getattr(model, "set_attachments", None)
    if callable(setter):
        setter("sam3_clip", clip)
    encoded = clip.encode_from_tokens(clip.tokenize(text), return_dict=True)
    cond = encoded.get("cond")
    if cond is None:
        raise ValueError("SAM3 CLIP did not return text conditioning.")
    attention = encoded.get("attention_mask")
    return cond, attention


def _sam3_anchor_masks(model: Any, image: torch.Tensor, prompt_data: str | None) -> torch.Tensor:
    """Run SAM3.1's official model kernel with one mixed prompt per object."""
    _, height, width, _ = image.shape
    prompts = _parse_prompt_data(prompt_data, width, height)
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3_model = model.model.diffusion_model
    image_in = comfy.utils.common_upscale(
        image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled"
    ).to(device=device, dtype=dtype)
    masks: list[torch.Tensor] = []
    for prompt in prompts:
        points = prompt["points"]
        point_inputs = None
        if points:
            point_inputs = {
                "point_coords": torch.tensor(
                    [[[point["x"] / width * 1008, point["y"] / height * 1008] for point in points]],
                    dtype=dtype,
                    device=device,
                ),
                "point_labels": torch.tensor(
                    [[int(point["label"]) for point in points]],
                    dtype=torch.int32,
                    device=device,
                ),
            }
        box_inputs = None
        if prompt["bbox"] is not None:
            box = prompt["bbox"]
            box_inputs = torch.tensor(
                [[
                    [box["x"] / width * 1008, box["y"] / height * 1008],
                    [(box["x"] + box["width"]) / width * 1008, (box["y"] + box["height"]) / height * 1008],
                ]],
                dtype=dtype,
                device=device,
            )
        mask_inputs = None
        if prompt["mask"] is not None:
            mask_inputs = _sam3_mask_logits(prompt["mask"], device, dtype)

        with torch.no_grad():
            if prompt["text"]:
                text_embeddings, text_mask = _sam3_text_embeddings(model, prompt["text"])
                # The official detector consumes normalized cxcywh boxes.  Keep
                # point prompts on the interactive decoder path below, then use
                # the highest-scoring text detection as this object's anchor.
                detector_boxes = None
                if prompt["bbox"] is not None:
                    box = prompt["bbox"]
                    detector_boxes = torch.tensor([[
                        [(box["x"] + box["width"] / 2) / width,
                         (box["y"] + box["height"] / 2) / height,
                         box["width"] / width,
                         box["height"] / height]
                    ]], dtype=dtype, device=device)
                detected = sam3_model(
                    image_in,
                    text_embeddings=text_embeddings.to(device=device, dtype=dtype),
                    text_mask=text_mask.to(device=device) if text_mask is not None else None,
                    boxes=detector_boxes,
                    threshold=0.0,
                    orig_size=(height, width),
                )
                scores = detected.get("scores")
                detected_masks = detected.get("masks")
                if detected_masks is None or detected_masks.numel() == 0:
                    raise ValueError(f"SAM3 returned no detection for semantic prompt: {prompt['text']}")
                score_row = scores[0] if scores is not None and scores.numel() else None
                best = int(score_row.argmax().item()) if score_row is not None else 0
                coarse = detected_masks[0, best:best + 1].unsqueeze(1)
                if point_inputs is not None or box_inputs is not None or mask_inputs is not None:
                    coarse_1008 = F.interpolate(coarse, size=(1008, 1008), mode="bilinear", align_corners=False)
                    if mask_inputs is not None:
                        coarse_1008 = (coarse_1008 + F.interpolate(mask_inputs, size=(1008, 1008), mode="bilinear", align_corners=False)) * 0.5
                    mask_logits = sam3_model.forward_segment(
                        image_in,
                        point_inputs=point_inputs,
                        box_inputs=box_inputs,
                        mask_inputs=coarse_1008,
                    )
                    mask_logits = sam3_model.forward_segment(
                        image_in,
                        point_inputs=point_inputs,
                        box_inputs=box_inputs,
                        mask_inputs=mask_logits,
                    )
                else:
                    mask_logits = coarse
            else:
                mask_logits = sam3_model.forward_segment(
                    image_in,
                    point_inputs=point_inputs,
                    box_inputs=box_inputs,
                    mask_inputs=mask_inputs,
                )
                # Match the official Detect node's default interactive refinement pass.
                mask_logits = sam3_model.forward_segment(
                    image_in,
                    point_inputs=point_inputs,
                    box_inputs=box_inputs,
                    mask_inputs=mask_logits,
                )
        resized = F.interpolate(mask_logits, size=(height, width), mode="bilinear", align_corners=False)
        masks.append((resized[0, 0] > 0).float().to("cpu"))
    return torch.stack(masks, dim=0)


def _preview_data_url(frame: torch.Tensor, mask: torch.Tensor, masks: torch.Tensor | None = None) -> str:
    source = frame[..., :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    alpha = mask.to("cpu", dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(-1) * 0.52
    if masks is not None and masks.ndim == 3 and masks.shape[0] > 0:
        # One colour per object, matching the node's color_mask output, so two
        # people can be told apart in the preview.
        palette = _parse_object_colors(None, int(masks.shape[0]))
        strongest, owner = masks.to("cpu", dtype=torch.float32).max(dim=0)
        color = palette[torch.where(strongest > 0, owner + 1, torch.zeros_like(owner))]
    else:
        color = source.new_tensor([0.20, 0.77, 0.71])
    composite = source * (1.0 - alpha) + color * alpha
    array = (composite * 255.0).round().to(torch.uint8).numpy()
    image = Image.fromarray(array, mode="RGB")
    buffer = py_io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


async def _video_segment_preview_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        frame_index = max(0, int(payload.get("frame", 0)))
        frame = _decode_selector_frame(payload, frame_index)

        # ``SAM3_Detect`` updates a ComfyUI ProgressBar even when it is
        # invoked outside the normal prompt queue.  Some portable builds do
        # not create ``last_prompt_id`` until the first queued prompt runs;
        # initialize the optional state so the global progress hook remains
        # usable for an interactive preview request.
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None and not hasattr(server_instance, "last_prompt_id"):
            server_instance.last_prompt_id = "cinestyle-preview"

        model = _preview_model(payload.get("model_source"))
        if model is None:
            raise ValueError(
                "Connect a CheckpointLoaderSimple or Load Diffusion Model node to MODEL, "
                "or run this SAM3 node once before using Preview."
            )
        masks = _sam3_anchor_masks(model, frame, payload.get("prompt_data"))
        mask = masks.amax(dim=0).to("cpu").float().clamp_(0.0, 1.0)
        return web.json_response(
            {
                "frame": frame_index,
                "image": _preview_data_url(frame[0], mask, masks),
                "mask_area": float((mask > 0.5).float().mean().item()),
            }
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


def _sec_model_roots() -> list[str]:
    try:
        roots = folder_paths.get_folder_paths("sams")
    except KeyError:
        roots = []
    return [str(root) for root in (list(roots) or [os.path.join(folder_paths.models_dir, "sams")])]


def _sec_sync_model_folder() -> None:
    roots = [os.path.join(root, "SeC-4B") for root in _sec_model_roots()]
    extensions = set(getattr(folder_paths, "supported_pt_extensions", {".safetensors", ".bin", ".pth"}))
    folder_paths.folder_names_and_paths[_SEC_MODEL_FOLDER] = (roots, extensions)


def _sec_model_file_options() -> list[str]:
    """Return supported single-file names using ComfyUI's standard model listing."""
    _sec_sync_model_folder()
    try:
        files = folder_paths.get_filename_list(_SEC_MODEL_FOLDER)
    except (KeyError, OSError):
        files = []
    supported_names = set(_SEC_WEIGHT_SPECS)
    return [name for name in files if Path(name).name in supported_names]


def _sec_download_weights(target: Path, url: str) -> None:
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.download")
    print(f"[CineStyle] SeC-4B weights not found; downloading {target.name} to {target}.")
    print(f"[CineStyle] Download source: {url}")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "CineStyle-ComfyUI/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            expected = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            next_report = 256 * 1024 * 1024
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if expected:
                        print(f"[CineStyle] SeC-4B download: {downloaded / expected:.0%}")
                    else:
                        print(f"[CineStyle] SeC-4B downloaded: {downloaded / (1024 ** 3):.2f} GiB")
                    next_report += 256 * 1024 * 1024
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise RuntimeError("the downloaded file is empty")
        os.replace(str(partial), str(target))
    except Exception as exc:
        try:
            partial.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(
            f"Unable to download {target.name} automatically. "
            f"Download it manually from {url} and place it at {target}. "
            f"Original error: {exc}"
        ) from exc


def _sec_weight_dtype(weight_path: str) -> torch.dtype:
    filename = Path(weight_path).name.lower()
    if filename == "sec-4b-fp16.safetensors":
        return torch.float16
    if filename == "sec-4b-bf16.safetensors":
        return torch.bfloat16
    raise ValueError(
        "Unsupported SeC weight file. Choose SeC-4B-bf16.safetensors or SeC-4B-fp16.safetensors."
    )


def _sec_default_weight_filename() -> str:
    options = _sec_model_file_options()
    for filename in _SEC_WEIGHT_SPECS:
        if filename in options:
            return filename
    return _SEC_DEFAULT_WEIGHT_FILENAME


def _sec_weight_path(filename: str | None = None) -> str:
    filename = str(filename or _sec_default_weight_filename())
    basename = Path(filename).name
    spec = _SEC_WEIGHT_SPECS.get(basename)
    if spec is None:
        allowed = ", ".join(_SEC_WEIGHT_SPECS)
        raise ValueError(f"Unsupported SeC weight file {filename!r}. Supported files: {allowed}.")
    _, url = spec
    _sec_sync_model_folder()
    existing = folder_paths.get_full_path(_SEC_MODEL_FOLDER, filename)
    if existing:
        return existing
    roots = _sec_model_roots()
    target = Path(roots[0]) / "SeC-4B" / basename
    with _SEC_MODEL_DOWNLOAD_LOCK:
        if not target.is_file() or target.stat().st_size <= 0:
            _sec_download_weights(target, url)
    return str(target)


def _sec_model_config_path() -> str:
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    missing = [name for name in required if not (_SEC_MODEL_CONFIG_PATH / name).is_file()]
    if missing:
        raise RuntimeError(
            "CineStyle SeC model configuration is incomplete: "
            + ", ".join(missing)
        )
    return str(_SEC_MODEL_CONFIG_PATH)


def _sec_imports() -> tuple[Any, Any, Any]:
    package_root = str(_SEC_PACKAGE_PATH.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"Importing from timm\.models\.layers is deprecated.*",
    )
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
    except Exception:
        pass
    from sec_inference.configuration_sec import SeCConfig
    from sec_inference.modeling_sec import SeCModel
    from transformers import AutoTokenizer

    return SeCConfig, SeCModel, AutoTokenizer


def _sec_install_dtype_hooks(model: Any) -> None:
    def dtype_conversion_hook(module, args, kwargs):
        try:
            module_dtype = next(module.parameters(recurse=False)).dtype
        except StopIteration:
            return args, kwargs
        except Exception:
            return args, kwargs
        if isinstance(module, torch.nn.Embedding):
            return args, kwargs

        def convert(value):
            if not isinstance(value, torch.Tensor) or value.dtype in {
                torch.long, torch.int, torch.int32, torch.int64,
            } or value.dtype == module_dtype:
                return value
            return value.to(dtype=module_dtype)

        return tuple(convert(value) for value in args), {
            key: convert(value) for key, value in kwargs.items()
        }

    if getattr(model, "_cinestyle_dtype_hooks", False):
        return
    for module in model.modules():
        if any(True for _ in module.parameters(recurse=False)):
            module.register_forward_pre_hook(dtype_conversion_hook, with_kwargs=True)
    model._cinestyle_dtype_hooks = True


def _sec_create_model(
    weight_path: str,
    torch_dtype: torch.dtype,
    device: str,
    use_flash_attn: bool,
    allow_mask_overlap: bool,
) -> Any:
    SeCConfig, SeCModel, AutoTokenizer = _sec_imports()
    config_path = _sec_model_config_path()
    config = SeCConfig.from_pretrained(config_path)
    config.hydra_overrides_extra = [
        f"++model.non_overlap_masks={'false' if allow_mask_overlap else 'true'}"
    ]

    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
    except ImportError:
        init_empty_weights = None
        set_module_tensor_to_device = None

    from safetensors.torch import load_file

    if init_empty_weights is not None and set_module_tensor_to_device is not None:
        with init_empty_weights():
            model = SeCModel(config, use_flash_attn=use_flash_attn)
        state_dict = load_file(weight_path, device="cpu")
        try:
            for name, value in state_dict.items():
                set_module_tensor_to_device(model, name, device="cpu", value=value)
        finally:
            del state_dict
    else:
        model = SeCModel(config, use_flash_attn=use_flash_attn)
        state_dict = load_file(weight_path, device="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
        finally:
            del state_dict

    model = model.eval().to(device=device, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)
    if use_flash_attn and device.startswith("cuda:"):
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("[CineStyle] SeC flash attention is unavailable; using standard attention.")
    if device.startswith("cuda:") and torch_dtype != torch.float32:
        _sec_install_dtype_hooks(model)
    model._sec_loading_metadata = {
        "weight_path": weight_path,
        "torch_dtype": torch_dtype,
        "device": device,
        "use_flash_attn": use_flash_attn,
        "allow_mask_overlap": allow_mask_overlap,
    }
    model._sec_unloaded = False
    return model


def _sec_reload_model(model: Any) -> Any:
    metadata = getattr(model, "_sec_loading_metadata", None)
    if not metadata:
        raise RuntimeError("SeC model has been unloaded and has no reload metadata.")
    fresh = _sec_create_model(**metadata)
    model.__dict__.update(fresh.__dict__)
    model._sec_loading_metadata = metadata
    model._sec_unloaded = False
    del fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _sec_ensure_loaded(model: Any) -> Any:
    if getattr(model, "_sec_unloaded", False):
        return _sec_reload_model(model)
    return model


def _sec_unload_model(model: Any) -> None:
    if getattr(model, "_sec_unloaded", False):
        return
    for component in ("vision_model", "language_model", "grounding_encoder", "tokenizer"):
        value = getattr(model, component, None)
        if value is not None:
            try:
                if hasattr(value, "cpu"):
                    value.cpu()
                delattr(model, component)
            except Exception:
                pass
    model._sec_unloaded = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sec_executing_node_id() -> str | None:
    """Return the Comfy node currently executing, when supported by this build."""
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        node_id = getattr(context, "node_id", None)
        return str(node_id) if node_id is not None else None
    except Exception:
        # Older ComfyUI builds do not expose an execution context. The global
        # latest-token fallback still keeps Preview functional there.
        return None


def _sec_register_model(model: Any, node_id: str | None = None) -> str:
    token = getattr(model, "_cinestyle_sec_token", None)
    if not token:
        token = f"sec-{uuid.uuid4().hex}"
        model._cinestyle_sec_token = token
    node_id = node_id or _sec_executing_node_id()
    with _SEC_MODEL_LOCK:
        _SEC_MODEL_REGISTRY[token] = model
        if node_id:
            _SEC_MODEL_NODE_TOKENS[node_id] = token
        global _SEC_PREVIEW_MODEL_TOKEN
        _SEC_PREVIEW_MODEL_TOKEN = token
    return token


def _sec_default_model_settings() -> tuple[torch.dtype, str, bool, bool]:
    """Return the same defaults used by the SeC Loader for cold Preview."""
    if torch.cuda.is_available():
        return torch.bfloat16, "cuda:0", True, True
    return torch.float32, "cpu", False, True


def _sec_cold_load_model() -> Any:
    """Load the default SeC model once when Preview has no Loader token."""
    with _SEC_MODEL_LOAD_LOCK:
        # Another request may have populated the registry while this request
        # was waiting for the load lock.
        with _SEC_MODEL_LOCK:
            latest = _SEC_PREVIEW_MODEL_TOKEN
            existing = _SEC_MODEL_REGISTRY.get(latest) if latest else None
        if existing is not None:
            return _sec_ensure_loaded(existing)

        _, device, use_flash_attn, allow_mask_overlap = _sec_default_model_settings()
        weight_path = _sec_weight_path()
        torch_dtype = _sec_weight_dtype(weight_path) if device != "cpu" else torch.float32
        print(
            f"[CineStyle] SeC Preview has no registered Loader model; "
            f"cold-loading default SeC-4B {Path(weight_path).stem.removeprefix('SeC-4B-').upper()} weights from {weight_path} on {device}."
        )
        model = _sec_create_model(
            weight_path,
            torch_dtype,
            device,
            use_flash_attn,
            allow_mask_overlap,
        )
        model._cinestyle_sec_cache_key = (
            weight_path,
            str(torch_dtype),
            device,
            bool(use_flash_attn),
            bool(allow_mask_overlap),
        )
        token = _sec_register_model(model)
        print(f"[CineStyle] SeC-4B cold Preview model ready; token={token}")
        return model


def _sec_model_for_token(token: str | None) -> Any:
    with _SEC_MODEL_LOCK:
        resolved = str(token or "") or _SEC_PREVIEW_MODEL_TOKEN
        model = _SEC_MODEL_REGISTRY.get(resolved) if resolved else None
    if model is None:
        if token and str(token).strip():
            raise ValueError("The requested SeC model token is no longer available.")
        return _sec_cold_load_model()
    return _sec_ensure_loaded(model)


def _sec_model_registry_response(loader_node_id: str | None = None) -> web.Response:
    with _SEC_MODEL_LOCK:
        models = [
            {
                "token": token,
                "loaded": not bool(getattr(model, "_sec_unloaded", False)),
                "device": str(getattr(model, "_sec_loading_metadata", {}).get("device", "")),
                "node_id": next((node for node, mapped in _SEC_MODEL_NODE_TOKENS.items() if mapped == token), None),
            }
            for token, model in _SEC_MODEL_REGISTRY.items()
        ]
        if loader_node_id:
            # Newer ComfyUI builds expose the executing node context, so a
            # connected Selector must use that Loader's token. On older builds
            # without context support, a single registered model is unambiguous.
            latest = _SEC_MODEL_NODE_TOKENS.get(str(loader_node_id))
            if latest is None and len(_SEC_MODEL_REGISTRY) == 1:
                latest = _SEC_PREVIEW_MODEL_TOKEN
        else:
            latest = _SEC_PREVIEW_MODEL_TOKEN
    return web.json_response({"models": models, "latest": latest, "loader_node_id": loader_node_id})


def _sec_frame_dir(images: torch.Tensor) -> str:
    temp_dir = tempfile.mkdtemp(prefix="cinestyle_sec_")
    for index, image in enumerate(images):
        array = (image[..., :3].to("cpu", dtype=torch.float32).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(os.path.join(temp_dir, f"{index:05d}.jpg"), "JPEG", quality=95)
    return temp_dir


def _sec_mask_2d(mask: Any) -> np.ndarray:
    """Reduce a SeC mask/logit result to one object's HxW mask."""
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    while array.ndim > 2:
        # SeC returns an object dimension (and some model versions add a
        # singleton mask dimension). This node tracks one object, so select it.
        array = array[0]
    if array.ndim != 2:
        raise ValueError("SeC returned a mask with an unsupported shape.")
    return array


def _sec_mask_for_object(mask: Any, object_ids: Any, object_id: int) -> np.ndarray:
    """Select one object's mask from SeC's consolidated multi-object output."""
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    ids = list(object_ids or [])
    try:
        object_index = ids.index(object_id)
    except ValueError:
        object_index = 0
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 3:
        array = array[object_index if array.shape[0] > object_index else 0]
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("SeC returned a mask with an unsupported shape.")
    return array


def _sec_add_prompt(model: Any, state: dict[str, Any], frame_index: int, object_id: int, points, labels, box, init_mask):
    if init_mask is not None:
        # The bundled SeC/SAM2 configs enable mask-as-output for video
        # conditioning. Temporarily disable that shortcut on the anchor so a
        # rough brush mask is actually passed through the SAM decoder.
        encoder = model.grounding_encoder
        previous_mask_mode = getattr(encoder, "use_mask_input_as_output_without_sam", None)
        if previous_mask_mode is not None:
            encoder.use_mask_input_as_output_without_sam = False
        try:
            _, object_ids, mask_logits = encoder.add_new_mask(
                inference_state=state,
                frame_idx=frame_index,
                obj_id=object_id,
                mask=_sec_mask_2d(init_mask),
            )
        finally:
            if previous_mask_mode is not None:
                encoder.use_mask_input_as_output_without_sam = previous_mask_mode
        init_mask = _sec_mask_for_object(mask_logits > 0.0, object_ids, object_id)
    if points is not None or box is not None:
        _, object_ids, logits = model.grounding_encoder.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_index,
            obj_id=object_id,
            points=points,
            labels=labels,
            box=box,
        )
        init_mask = _sec_mask_for_object(logits > 0.0, object_ids, object_id)
    return init_mask


def _sec_anchor_preview(model: Any, frame: torch.Tensor, prompt_data: str | None) -> torch.Tensor:
    height, width = map(int, frame.shape[1:3])
    prompts = _parse_prompt_data(prompt_data, width, height)
    if any(prompt["text"] for prompt in prompts):
        raise ValueError("SeC-4B does not support Semantic prompts. Use Point, BBox, or Draw Mask.")
    temp_dir = _sec_frame_dir(frame)
    try:
        states = getattr(model.grounding_encoder, "_states", None)
        if hasattr(states, "clear"):
            states.clear()
        state = model.grounding_encoder.init_state(video_path=temp_dir, offload_video_to_cpu=False, offload_state_to_cpu=False)
        model.grounding_encoder.reset_state(state)
        masks = []
        for object_index, prompt in enumerate(prompts, start=1):
            point_values = prompt["points"]
            point_array = np.asarray([[item["x"], item["y"]] for item in point_values], dtype=np.float32) if point_values else None
            labels = np.asarray([int(item["label"]) for item in point_values], dtype=np.int32) if point_values else None
            box = prompt["bbox"]
            box_array = None if box is None else np.asarray(
                [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]],
                dtype=np.float32,
            )
            init_mask = _sec_add_prompt(
                model,
                state,
                0,
                object_index,
                point_array,
                labels,
                box_array,
                prompt["mask"].numpy() if isinstance(prompt["mask"], torch.Tensor) else prompt["mask"],
            )
            if init_mask is None:
                raise ValueError(f"SeC did not produce a mask for object {object_index}.")
            masks.append(torch.from_numpy((_sec_mask_2d(init_mask) > 0).astype(np.float32)))
        return torch.stack(masks, dim=0).amax(dim=0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _sec_video_segment_models_route(request: web.Request) -> web.Response:
    loader_node_id = request.query.get("loader_node_id")
    with _SEC_MODEL_LOCK:
        if loader_node_id:
            has_latest = bool(
                _SEC_MODEL_NODE_TOKENS.get(str(loader_node_id))
                or (len(_SEC_MODEL_REGISTRY) == 1 and _SEC_PREVIEW_MODEL_TOKEN)
            )
        else:
            has_latest = bool(_SEC_PREVIEW_MODEL_TOKEN)
    if not has_latest:
        print(
            "[CineStyle] SeC Preview requested without a registered model; "
            "default cold loading will be used."
        )
    return _sec_model_registry_response(loader_node_id)


async def _sec_video_segment_preview_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        frame_index = max(0, int(payload.get("frame", 0)))
        frame = _decode_selector_frame(payload, frame_index)
        model = _sec_model_for_token(payload.get("model_token"))
        mask = _sec_anchor_preview(model, frame, payload.get("prompt_data"))
        return web.json_response({
            "frame": frame_index,
            "image": _preview_data_url(frame[0], mask),
            "mask_area": float((mask > 0.5).float().mean().item()),
        })
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


SEC_MODEL = io.Custom("SEC_MODEL")


class CSSeCModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            devices.extend(f"gpu{index}" for index in range(torch.cuda.device_count()))
        return io.Schema(
            node_id="CS_SeC_ModelLoader",
            display_name="CS SeC-4B Model Loader",
            category="😺dzNodes/CineStyle/Video",
            search_aliases=["sec", "segment concept", "seC-4B", "video segmentation"],
            inputs=[
                io.Combo.Input(
                    "model_file",
                    options=_sec_model_file_options() or list(_SEC_WEIGHT_SPECS),
                    default=_sec_default_weight_filename(),
                    tooltip="SeC single-file weights found in ComfyUI/models/sams/SeC-4B.",
                ),
                io.Combo.Input("device", options=devices, default="auto"),
                io.Boolean.Input("use_flash_attn", default=True, advanced=True),
                io.Boolean.Input("allow_mask_overlap", default=True, advanced=True),
            ],
            outputs=[SEC_MODEL.Output("model", display_name="SEC_MODEL")],
        )

    @classmethod
    def execute(cls, model_file=_SEC_DEFAULT_WEIGHT_FILENAME, device="auto", use_flash_attn=True, allow_mask_overlap=True) -> io.NodeOutput:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("gpu"):
            device = f"cuda:{int(device[3:])}"
        weight_path = _sec_weight_path(model_file)
        dtype = _sec_weight_dtype(weight_path) if device != "cpu" else torch.float32
        if device == "cpu":
            use_flash_attn = False
        key = (weight_path, str(dtype), device, bool(use_flash_attn), bool(allow_mask_overlap))
        with _SEC_MODEL_LOCK:
            model = next((candidate for candidate in _SEC_MODEL_REGISTRY.values() if getattr(candidate, "_cinestyle_sec_cache_key", None) == key), None)
        if model is None:
            precision = Path(weight_path).stem.removeprefix("SeC-4B-").upper()
            print(f"[CineStyle] Loading SeC-4B {precision} weights from {weight_path} on {device}.")
            model = _sec_create_model(weight_path, dtype, device, bool(use_flash_attn), bool(allow_mask_overlap))
            model._cinestyle_sec_cache_key = key
        else:
            model = _sec_ensure_loaded(model)
        token = _sec_register_model(model)
        print(f"[CineStyle] SeC-4B ready; preview token={token}")
        return io.NodeOutput(model)


class CSVideoSegmentSeC(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Video_Segment_SeC",
            display_name="CS Video Segment (SeC-4B)",
            category="😺dzNodes/CineStyle/Video",
            search_aliases=["sec video", "segment concept", "longsam2", "video mask"],
            inputs=[
                SEC_MODEL.Input("model", tooltip="Loaded SeC-4B model."),
                io.Image.Input("images", optional=True, tooltip="Video frames as an IMAGE batch."),
                io.Video.Input("video_input", optional=True, tooltip="Optional VIDEO input."),
                io.Int.Input("anchor_frame", default=0, min=0, max=10000000, step=1),
                io.String.Input("prompt_data", default='{"version":2,"objects":[]}', multiline=True, optional=True, tooltip="Selector multi-object mask, bbox, and point prompts."),
                io.Combo.Input("tracking_direction", options=["forward", "backward", "bidirectional"], default="bidirectional", advanced=True),
                io.Int.Input("max_frames_to_track", default=-1, min=-1, max=10000000, step=1, advanced=True),
                io.Int.Input("mllm_memory_size", default=12, min=1, max=20, step=1, advanced=True),
                io.Boolean.Input("offload_video_to_cpu", default=False, advanced=True),
                io.Boolean.Input("auto_unload_model", default=True, advanced=True),
                io.Boolean.Input(
                    "wait_for_input_cache",
                    display_name="wait for input cache",
                    default=False,
                    advanced=True,
                    tooltip="Interrupt execution when this node is reached after caching its input.",
                ),
            ],
            outputs=[
                io.Mask.Output("mask", display_name="MASK"),
                io.Mask.Output("anchor_mask", display_name="anchor_mask"),
                io.Dict.Output("video_info", display_name="video_info"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        images: torch.Tensor | None = None,
        video_input: Any = None,
        anchor_frame: int = 0,
        prompt_data: str = '{"version":2,"objects":[]}',
        tracking_direction: str = "bidirectional",
        max_frames_to_track: int = -1,
        mllm_memory_size: int = 12,
        offload_video_to_cpu: bool = False,
        auto_unload_model: bool = True,
        wait_for_input_cache: bool = False,
    ) -> io.NodeOutput:
        node_name = "CS Video Segment (SeC-4B)"
        _segment_info(node_name, "start")
        model = _sec_ensure_loaded(model)
        _segment_info(node_name, "model ready")
        if images is None and video_input is not None:
            images = video_input.get_components().images
        if images is None:
            raise ValueError("Connect CS Load Video to images or video_input.")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, 3 or 4].")
        images = images[..., :3].to("cpu", dtype=torch.float32).clamp_(0, 1)
        _segment_info(node_name, f"input ready: frames={images.shape[0]}, size={images.shape[2]}x{images.shape[1]}")
        loader_origin = _prompt_loader_id(cls.hidden.prompt, cls.hidden.unique_id)
        if loader_origin:
            _segment_info(node_name, f"using shared CS Load Video preview cache: loader={loader_origin}")
        else:
            _cache_selector_input(
                cls.hidden.unique_id,
                images,
                _video_input_fps(video_input, cls.hidden.prompt, cls.hidden.unique_id),
            )
        if bool(wait_for_input_cache):
            _cache_wait_input(
                cls.hidden.unique_id,
                cls.hidden.prompt,
                images,
                ("images", "video_input"),
                video_input,
            )
            from comfy.model_management import InterruptProcessingException

            raise InterruptProcessingException()
        frame_count, height, width = map(int, images.shape[:3])
        anchor = int(anchor_frame)
        if not 0 <= anchor < frame_count:
            raise ValueError(f"anchor_frame must be between 0 and {frame_count - 1}.")
        if tracking_direction not in {"forward", "backward", "bidirectional"}:
            raise ValueError("tracking_direction must be forward, backward, or bidirectional.")
        prompts = _parse_prompt_data(prompt_data, width, height)
        _segment_info(node_name, f"prompts parsed: objects={len(prompts)}, anchor={anchor}")
        if any(prompt["text"] for prompt in prompts):
            raise ValueError("SeC-4B does not support Semantic prompts. Use Point, BBox, or Draw Mask.")

        temp_dir = _sec_frame_dir(images)
        _segment_info(node_name, "temporary frame sequence prepared")
        state = None
        nested_tqdm = _NestedTqdmSilencer(
            (
                "sec_inference.modeling_sec",
                "sec_inference.sam2_video_predictor",
                "sec_inference.sam2.sam2_video_predictor",
                "sec_inference.sam2.utils.misc",
            )
        )
        nested_tqdm.start()
        try:
            state = model.grounding_encoder.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=bool(offload_video_to_cpu),
                offload_state_to_cpu=str(model._sec_loading_metadata.get("device")) == "cpu",
            )
            model.grounding_encoder.reset_state(state)
            _segment_info(node_name, "video tracking state initialized")
            object_masks: list[np.ndarray] = []

            def add_all_prompts() -> list[np.ndarray]:
                added: list[np.ndarray] = []
                for object_index, prompt in enumerate(prompts, start=1):
                    point_values = prompt["points"]
                    point_array = np.asarray([[item["x"], item["y"]] for item in point_values], dtype=np.float32) if point_values else None
                    labels = np.asarray([int(item["label"]) for item in point_values], dtype=np.int32) if point_values else None
                    box = prompt["bbox"]
                    box_array = None if box is None else np.asarray(
                        [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]],
                        dtype=np.float32,
                    )
                    init_mask = _sec_add_prompt(
                        model,
                        state,
                        anchor,
                        object_index,
                        point_array,
                        labels,
                        box_array,
                        prompt["mask"].numpy() if isinstance(prompt["mask"], torch.Tensor) else prompt["mask"],
                    )
                    if init_mask is None:
                        raise RuntimeError(f"SeC did not produce an initial mask for object {object_index}.")
                    added.append(_sec_mask_2d(init_mask))
                return added

            object_masks = add_all_prompts()
            _segment_info(node_name, f"anchor prompts applied: objects={len(object_masks)}")
            initial_union = torch.from_numpy(np.asarray(object_masks).astype(np.float32).max(axis=0))
            limit = frame_count if int(max_frames_to_track) < 0 else max(1, int(max_frames_to_track))
            segments: dict[int, torch.Tensor] = {}
            progress_total = _segment_expected_frames(frame_count, anchor, tracking_direction, limit)
            if tracking_direction == "bidirectional":
                _segment_info(node_name, "propagating forward and backward")
            else:
                _segment_info(node_name, f"propagating {tracking_direction}")
            progress = _SegmentProgress(node_name, progress_total)

            def collect(reverse: bool):
                for frame_index, object_ids, mask_logits in model.propagate_in_video(
                    state,
                    start_frame_idx=anchor,
                    max_frame_num_to_track=limit,
                    reverse=reverse,
                    init_mask=initial_union.numpy(),
                    tokenizer=None,
                    mllm_memory_size=max(1, int(mllm_memory_size)),
                ):
                    union = (mask_logits > 0.0).any(dim=0).to("cpu", dtype=torch.float32)
                    segments[int(frame_index)] = union
                    progress.update()

            if tracking_direction in {"forward", "bidirectional"}:
                collect(False)
            if tracking_direction == "bidirectional":
                model.grounding_encoder.reset_state(state)
                object_masks = add_all_prompts()
                collect(True)
            elif tracking_direction == "backward":
                collect(True)
            progress.close()

            output = torch.zeros(frame_count, height, width, dtype=torch.float32)
            for frame_index, mask in segments.items():
                if 0 <= frame_index < frame_count:
                    output[frame_index] = mask
            output[anchor] = initial_union
            info = {
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "anchor_frame": anchor,
                "tracking_direction": tracking_direction,
                "object_count": len(prompts),
            }
            _segment_info(node_name, f"complete: frames={frame_count}, object_count={len(prompts)}")
            return io.NodeOutput(output, initial_union, info)
        finally:
            nested_tqdm.stop()
            try:
                if state is not None:
                    model.grounding_encoder.reset_state(state)
            except Exception:
                pass
            shutil.rmtree(temp_dir, ignore_errors=True)
            if auto_unload_model:
                _sec_unload_model(model)
                _segment_info(node_name, "model unloaded")
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()


_OBJECT_PALETTE = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0),
    (0.5, 0.0, 1.0),
    (0.0, 1.0, 0.5),
    (1.0, 0.0, 0.5),
    (0.5, 1.0, 0.0),
    (0.0, 0.5, 1.0),
    (1.0, 1.0, 1.0),
    (0.5, 0.25, 0.0),
    (0.5, 0.5, 0.5),
    (1.0, 0.75, 0.8),
)


def _object_is_prompted(item: Any) -> bool:
    """Return whether one serialized Selector object carries any prompt."""
    if not isinstance(item, dict):
        return False
    if str(item.get("text") or item.get("semantic") or "").strip():
        return True
    if item.get("points") or item.get("bbox") or item.get("box"):
        return True
    mask = item.get("mask")
    if isinstance(mask, dict):
        mask = mask.get("data") or mask.get("png")
    return bool(str(mask or "").strip())


def _parse_anchor_prompts(
    prompt_data: str | None,
    anchor_frame: int,
    frame_count: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Normalize v2 single-anchor and v3 multi-anchor Selector data.

    Returns the global object count and one entry per anchor with the anchor
    frame, a v2 ``prompt_data`` string containing only the prompted objects,
    and the global object index of each of those prompts.  Object indices are
    shared by every anchor so an object keeps its colour across shots.
    """
    raw = _parse_json(prompt_data, "prompt_data")
    if raw is None:
        raise ValueError("prompt_data is empty. Open the Selector and define at least one object.")
    if isinstance(raw, dict) and isinstance(raw.get("anchors"), list):
        anchor_items = raw["anchors"]
        object_count = int(raw.get("object_count") or 0)
    else:
        objects = raw.get("objects") if isinstance(raw, dict) else raw
        anchor_items = [{"frame": anchor_frame, "objects": objects}]
        object_count = 0

    anchors: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(anchor_items):
        if not isinstance(item, dict) or not isinstance(item.get("objects"), list):
            raise ValueError(f"prompt_data.anchors[{index}] must contain an objects list.")
        try:
            frame = int(item.get("frame"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prompt_data.anchors[{index}] has an invalid frame.") from exc
        if not 0 <= frame < frame_count:
            raise ValueError(f"Anchor frame {frame} is outside the input range 0..{frame_count - 1}.")
        object_count = max(object_count, len(item["objects"]))
        prompted = [(slot, obj) for slot, obj in enumerate(item["objects"]) if _object_is_prompted(obj)]
        if not prompted:
            continue
        anchors[frame] = {
            "frame": frame,
            "prompt_data": json.dumps({"version": PROMPT_VERSION, "objects": [obj for _, obj in prompted]}),
            "object_indices": [slot for slot, _ in prompted],
        }
    if not anchors:
        raise ValueError("prompt_data must contain at least one prompted object.")
    return max(1, object_count), [anchors[frame] for frame in sorted(anchors)]


def _detect_shot_cuts(images: torch.Tensor) -> list[int]:
    """Find hard cuts as isolated spikes in the colour-histogram distance.

    A cut changes the colour distribution in one frame while the frames on
    either side stay stable.  Motion blur, whip pans and muzzle flashes change
    it over several consecutive frames instead, so a spike must dominate both
    neighbours to count.  Returns the first frame index of every new shot.
    """
    frame_count = int(images.shape[0])
    if frame_count < 2:
        return []
    histograms: list[torch.Tensor] = []
    thumbs: list[torch.Tensor] = []
    for start in range(0, frame_count, 64):
        chunk = images[start : start + 64, ..., :3].movedim(-1, 1).float()
        if not images.is_floating_point():
            chunk = chunk / 255.0
        small = F.interpolate(chunk, size=(36, 64), mode="area").clamp_(0.0, 1.0)
        thumbs.append(small)
        bins = (small * 7.999).long()
        index = (bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]).flatten(1)
        counts = torch.zeros(index.shape[0], 512, dtype=torch.float32)
        counts.scatter_add_(1, index, torch.ones_like(index, dtype=torch.float32))
        histograms.append(counts / index.shape[1])
    hist = torch.cat(histograms)
    thumb = torch.cat(thumbs)
    hist_delta = 0.5 * (hist[1:] - hist[:-1]).abs().sum(dim=1)
    pixel_delta = (thumb[1:] - thumb[:-1]).abs().mean(dim=(1, 2, 3))

    candidates: list[tuple[int, float]] = []
    count = int(hist_delta.shape[0])
    for k in range(count):
        value = float(hist_delta[k])
        neighbour = max(
            float(hist_delta[k - 1]) if k > 0 else 0.0,
            float(hist_delta[k + 1]) if k + 1 < count else 0.0,
        )
        if value >= 0.06 and value >= 2.5 * neighbour and float(pixel_delta[k]) >= 0.03:
            candidates.append((k + 1, value))
    # Keep the strongest cut when several spikes land within a few frames.
    cuts: list[tuple[int, float]] = []
    for frame, value in candidates:
        if cuts and frame - cuts[-1][0] < 8:
            if value > cuts[-1][1]:
                cuts[-1] = (frame, value)
            continue
        cuts.append((frame, value))
    return [frame for frame, _ in cuts]


def _planner_chunk_window(prompt: Any, cuts: list[int], frame_count: int) -> tuple[tuple[int, int], str] | None:
    """Frames of the chunk a downstream CS Shot Planner is about to render.

    The planner runs after this node, so its settings are read from the prompt
    and its plan is recomputed here.  That way one ``chunk_index`` drives both
    nodes and only the shot being tested has to be segmented.
    """
    if not isinstance(prompt, dict):
        return None
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_shot_planner")
    planner = getattr(module, "plan_chunks", None)
    if planner is None:
        return None
    node = next((item for item in prompt.values()
                 if isinstance(item, dict) and item.get("class_type") == "CS_Shot_Planner"), None)
    if node is None:
        return None
    values = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}

    def number(name: str, fallback: float) -> float:
        value = values.get(name, fallback)
        if isinstance(value, (list, tuple)):  # linked input; not readable here
            raise ValueError(
                f"CS Shot Planner's {name} is connected to another node, so this node cannot follow it. "
                "Set it as a widget value or use work_frame instead."
            )
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    plan = planner(
        frame_count, cuts, number("fps", 24.0), number("target_seconds", 8.0),
        number("min_seconds", 4.0), number("max_seconds", 10.0),
        int(number("frame_step", 17)), int(number("frame_offset", 5)),
        int(number("max_shots_per_chunk", 0)),
    )
    chunks = plan["chunks"]
    if not chunks:
        return None
    located = int(number("locate_frame", -1))
    if located >= 0:
        chunk = next((item for item in chunks if item["start"] <= located < item["end"]), None)
    else:
        index = int(number("chunk_index", 0))
        chunk = chunks[index] if 0 <= index < len(chunks) else None
    if chunk is None:
        raise ValueError(f"CS Shot Planner has {len(chunks)} chunks; its current selection is out of range.")
    return (chunk["start"], chunk["end"]), f"chunk #{chunk['index']}"


def _parse_frame_range(value: str | None, frame_count: int) -> tuple[int, int] | None:
    """Parse ``"448-463"`` (inclusive) into a half-open range; ``None`` = all."""
    text = str(value or "").strip()
    if not text:
        return None
    parts = [part for part in re.split(r"[\s,;:\-]+", text) if part]
    try:
        numbers = [int(float(part)) for part in parts]
    except ValueError as exc:
        raise ValueError("frame_range must look like 448-463.") from exc
    if not numbers:
        return None
    first = max(0, min(numbers[0], frame_count - 1))
    last = max(first, min(numbers[-1] if len(numbers) > 1 else numbers[0], frame_count - 1))
    return first, last + 1


def _parse_cut_frames(value: str | None, frame_count: int) -> list[int] | None:
    """Parse a manual ``12, 80, 144`` cut list; ``None`` means auto-detect."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        frames = [int(float(part)) for part in re.split(r"[\s,;\[\]]+", text) if part]
    except ValueError as exc:
        raise ValueError("shot_cut_frames must be a comma-separated list of frame numbers.") from exc
    return sorted({frame for frame in frames if 0 < frame < frame_count})


def _anchor_segments(
    anchor_frames: list[int],
    cuts: list[int],
    frame_count: int,
    direction: str,
) -> list[tuple[int, int, int]]:
    """Assign every frame to at most one anchor as ``(anchor, start, end)``.

    Segments never cross a shot cut, and shots without an anchor stay empty
    rather than inheriting a track that has drifted onto someone else.

    With ``both``, the first anchor of a shot also fills the frames before it,
    and every later anchor owns the frames from itself until the next one: a
    correction made on a frame therefore applies from that frame onwards
    instead of being split at the midpoint between two anchors.
    """
    bounds = [0, *[cut for cut in cuts if 0 < cut < frame_count], frame_count]
    segments: list[tuple[int, int, int]] = []
    for shot_start, shot_end in zip(bounds[:-1], bounds[1:]):
        inside = [frame for frame in anchor_frames if shot_start <= frame < shot_end]
        for index, anchor in enumerate(inside):
            previous = inside[index - 1] if index > 0 else None
            following = inside[index + 1] if index + 1 < len(inside) else None
            if direction == "forward":
                start, end = anchor, following if following is not None else shot_end
            elif direction == "backward":
                start, end = (previous + 1) if previous is not None else shot_start, anchor + 1
            else:
                start = shot_start if previous is None else anchor
                end = shot_end if following is None else following
            segments.append((anchor, start, end))
    return segments


def _parse_object_colors(value: str | None, object_count: int) -> torch.Tensor:
    """Return ``[object_count + 1, 3]`` colours; row 0 is the black background."""
    colors = [list(color) for color in _OBJECT_PALETTE]
    for index, part in enumerate(re.split(r"[\s,;]+", str(value or "").strip())):
        if not part:
            continue
        match = re.fullmatch(r"#?([0-9a-fA-F]{6})", part)
        if match is None:
            raise ValueError(f"object_colors entry {part!r} must be a hex colour such as #FF0000.")
        rgb = [int(match.group(1)[offset : offset + 2], 16) / 255.0 for offset in (0, 2, 4)]
        if index < len(colors):
            colors[index] = rgb
        else:
            colors.append(rgb)
    while len(colors) < object_count:
        colors.append(colors[len(colors) % len(_OBJECT_PALETTE)])
    return torch.tensor([[0.0, 0.0, 0.0], *colors[:object_count]], dtype=torch.float32)


def _write_track(
    result: dict[str, Any],
    frame_indices: list[int],
    object_indices: list[int],
    union: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Resize a packed SAM3 track and store its union mask and object labels."""
    packed = result.get("packed_masks")
    if packed is None:
        return
    from comfy.ldm.sam3.tracker import unpack_masks

    unpacked = unpack_masks(packed)  # [T, N_obj, Hm, Wm]
    if unpacked.ndim != 4:
        return
    height, width = int(union.shape[1]), int(union.shape[2])
    lookup = torch.tensor([index + 1 for index in object_indices], dtype=torch.uint8)
    frames = min(int(unpacked.shape[0]), len(frame_indices))
    objects = min(int(unpacked.shape[1]), len(object_indices))
    for start in range(0, frames, 16):
        chunk = unpacked[start : min(frames, start + 16), :objects].to("cpu").float()
        steps = int(chunk.shape[0])
        soft = F.interpolate(
            chunk.reshape(steps * objects, 1, *chunk.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(steps, objects, height, width)
        strongest, owner = soft.max(dim=1)
        for step in range(steps):
            target = frame_indices[start + step]
            union[target] = strongest[step]
            labels[target] = torch.where(strongest[step] > 0, lookup[owner[step]], 0)


def _propagate_segments(
    model: Any,
    images: torch.Tensor,
    anchors: list[dict[str, Any]],
    segments: list[tuple[int, int, int]],
    pbar: Any,
    max_objects: int,
    union: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Track each anchor's objects only inside that anchor's segment."""
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3_model = model.model.diffusion_model
    frames_chw = images[..., :3].movedim(-1, 1)
    by_frame = {anchor["frame"]: anchor for anchor in anchors}

    def run(sequence: torch.Tensor, masks: torch.Tensor, frame_indices: list[int], object_indices: list[int]) -> None:
        with torch.no_grad():
            result = sam3_model.forward_video(
                images=sequence,
                initial_masks=masks,
                pbar=pbar,
                text_prompts=None,
                max_objects=max_objects,
                target_device=device,
                target_dtype=dtype,
            )
        _write_track(result, frame_indices, object_indices, union, labels)

    for anchor_frame, start, end in segments:
        anchor = by_frame[anchor_frame]
        masks = anchor["masks"]
        object_indices = anchor["object_indices"]
        if end - anchor_frame > 1:
            run(frames_chw[anchor_frame:end], masks, list(range(anchor_frame, end)), object_indices)
        if anchor_frame - start > 0:
            run(
                frames_chw[start : anchor_frame + 1].flip(0),
                masks,
                list(range(anchor_frame, start - 1, -1)),
                object_indices,
            )
        # The prompted anchor result is authoritative on its own frame.
        strongest, owner = masks.max(dim=0)
        lookup = torch.tensor([index + 1 for index in object_indices], dtype=torch.uint8)
        union[anchor_frame] = strongest
        labels[anchor_frame] = torch.where(strongest > 0, lookup[owner], 0)


class CSVideoSegmentSAM3(io.ComfyNode):
    """Select an object on any video frame and propagate its SAM3.1 mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_ID,
            display_name="CS Video Segment (SAM3.1)",
            category="😺dzNodes/CineStyle/Video",
            essentials_category="Video Tools",
            search_aliases=["video segment", "sam3.1", "sam3 video", "propagate mask"],
            description=(
                "Define semantic, mask, bounding-box, and positive/negative point prompts "
                "on any video frame, then propagate them in both directions."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Official ComfyUI SAM3/SAM3.1 model."),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Video frames as an IMAGE batch. Connect CS Load Video for Selector input.",
                ),
                io.Video.Input(
                    "video_input",
                    optional=True,
                    tooltip="Optional VIDEO input from CS Load Video.",
                ),
                io.Int.Input(
                    "anchor_frame",
                    default=0,
                    min=0,
                    max=10000000,
                    step=1,
                    tooltip="Frame where the selector prompt is defined.",
                ),
                io.String.Input(
                    "prompt_data",
                    default='{"version":2,"objects":[]}',
                    multiline=True,
                    optional=True,
                    tooltip="Selector multi-object semantic, mask, bbox, and point prompts.",
                ),
                io.Combo.Input(
                    "propagation_direction",
                    options=_PROPAGATION_OPTIONS,
                    default="both",
                    advanced=True,
                    tooltip="Propagate toward both sides of the anchor or only one side.",
                ),
                io.Int.Input(
                    "max_objects",
                    default=16,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip="Maximum SAM3.1 multiplex object slots.",
                ),
                io.Boolean.Input(
                    "stop_at_shot_cuts",
                    default=True,
                    tooltip=(
                        "Stop tracking at detected shot cuts. Each shot is tracked only from the anchors "
                        "placed inside it; shots without an anchor produce an empty mask."
                    ),
                ),
                io.Boolean.Input(
                    "follow_planner_chunk",
                    display_name="only segment the planner's chunk",
                    default=False,
                    tooltip=(
                        "Testing switch: segment only the frames of the chunk that a downstream "
                        "CS Shot Planner will render, and leave the rest of the mask empty. Turn it off "
                        "for the final run."
                    ),
                ),
                io.Int.Input(
                    "work_frame",
                    default=-1,
                    min=-1,
                    max=100000000,
                    step=1,
                    tooltip=(
                        "Testing shortcut: process only the shot that contains this frame and leave the rest "
                        "of the mask empty. -1 processes the whole video. Link it to the Shot Planner's "
                        "locate_frame so one number drives both."
                    ),
                ),
                io.String.Input(
                    "frame_range",
                    default="",
                    advanced=True,
                    tooltip=(
                        "Testing shortcut: segment only these frames, e.g. 448-463, and leave the rest of "
                        "the mask empty. Empty processes the whole video. Clear it before the final run."
                    ),
                ),
                io.String.Input(
                    "shot_cut_frames",
                    default="",
                    advanced=True,
                    tooltip=(
                        "Optional comma-separated first frames of each new shot, e.g. 266, 422. "
                        "Leave empty to detect cuts automatically."
                    ),
                ),
                io.String.Input(
                    "object_colors",
                    default="",
                    advanced=True,
                    tooltip=(
                        "Optional comma-separated hex colours for Object 1, 2, ... in color_mask. "
                        "Empty uses red, green, blue, yellow, magenta, cyan, ..."
                    ),
                ),
                io.Boolean.Input(
                    "wait_for_input_cache",
                    display_name="wait for input cache",
                    default=False,
                    advanced=True,
                    tooltip="Interrupt execution when this node is reached after caching its input.",
                ),
            ],
            outputs=[
                io.Mask.Output("mask", display_name="MASK"),
                io.Mask.Output("anchor_mask", display_name="anchor_mask"),
                io.Dict.Output("video_info", display_name="video_info"),
                io.Image.Output("color_mask", display_name="color_mask"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs: Any) -> Any:
        # The processed window comes from the planner's chunk_index, which is
        # not an input of this node, so a cached mask would belong to whichever
        # chunk ran last.
        return float("nan") if kwargs.get("follow_planner_chunk", False) else None

    @classmethod
    def execute(
        cls,
        model: Any,
        images: torch.Tensor | None = None,
        video_input: Any = None,
        anchor_frame: int = 0,
        prompt_data: str = '{"version":2,"objects":[]}',
        propagation_direction: str = "both",
        max_objects: int = 16,
        stop_at_shot_cuts: bool = True,
        follow_planner_chunk: bool = False,
        work_frame: int = -1,
        frame_range: str = "",
        shot_cut_frames: str = "",
        object_colors: str = "",
        wait_for_input_cache: bool = False,
    ) -> io.NodeOutput:
        global _LAST_MODEL
        node_name = "CS Video Segment (SAM3.1)"
        _segment_info(node_name, "start")
        _LAST_MODEL = model
        _segment_info(node_name, "model ready")
        if images is None and video_input is not None:
            images = video_input.get_components().images
        if images is None:
            raise ValueError("Connect CS Load Video to images or video_input.")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, 3 or 4].")
        if images.shape[0] == 0:
            raise ValueError("The video contains no frames.")

        images = images[..., :3].to(device="cpu", dtype=torch.float32).clamp_(0.0, 1.0)
        _segment_info(node_name, f"input ready: frames={images.shape[0]}, size={images.shape[2]}x{images.shape[1]}")
        loader_origin = _prompt_loader_id(cls.hidden.prompt, cls.hidden.unique_id)
        if loader_origin:
            _segment_info(node_name, f"using shared CS Load Video preview cache: loader={loader_origin}")
        else:
            _cache_selector_input(
                cls.hidden.unique_id,
                images,
                _video_input_fps(video_input, cls.hidden.prompt, cls.hidden.unique_id),
            )
        if bool(wait_for_input_cache):
            _cache_wait_input(
                cls.hidden.unique_id,
                cls.hidden.prompt,
                images,
                ("images", "video_input"),
                video_input,
            )
            from comfy.model_management import InterruptProcessingException

            raise InterruptProcessingException()
        frame_count, height, width = map(int, images.shape[:3])
        if propagation_direction not in _PROPAGATION_OPTIONS:
            raise ValueError(f"propagation_direction must be one of {_PROPAGATION_OPTIONS}.")

        # SAM3.1's multiplex tracker has sixteen object slots. Keep the
        # public control compatible with broader SAM3 workflows, but never
        # pass more than the architectural cap to the tracker.
        object_limit = min(16, max(1, int(max_objects)))

        object_count, anchors = _parse_anchor_prompts(prompt_data, int(anchor_frame), frame_count)
        for anchor in anchors:
            frame = anchor["frame"]
            if len(anchor["object_indices"]) > object_limit:
                dropped = [index + 1 for index in anchor["object_indices"][object_limit:]]
                _SEGMENT_LOGGER.warning(
                    "[%s] anchor %d has more objects than max_objects=%d; skipping Object %s",
                    node_name, frame, object_limit, dropped,
                )
            masks = _sam3_anchor_masks(model, images[frame : frame + 1], anchor["prompt_data"])
            anchor["masks"] = masks[:object_limit]
            anchor["object_indices"] = anchor["object_indices"][:object_limit]
            _segment_info(node_name, f"anchor {frame} prompts segmented: objects={int(anchor['masks'].shape[0])}")
        anchor_frames = [anchor["frame"] for anchor in anchors]

        cuts: list[int] = []
        if stop_at_shot_cuts:
            manual_cuts = _parse_cut_frames(shot_cut_frames, frame_count)
            cuts = manual_cuts if manual_cuts is not None else _detect_shot_cuts(images)
            _segment_info(node_name, f"{'manual' if manual_cuts is not None else 'detected'} shot cuts: {cuts}")
        segments = _anchor_segments(anchor_frames, cuts, frame_count, propagation_direction)
        window = _parse_frame_range(frame_range, frame_count)
        if window is None and bool(follow_planner_chunk):
            found = _planner_chunk_window(cls.hidden.prompt, cuts, frame_count)
            if found is None:
                raise ValueError("No CS Shot Planner was found downstream, so follow_planner_chunk has nothing to follow.")
            window, label = found
            _segment_info(node_name, f"follow_planner_chunk: {label} covers frames {window[0]}-{window[1] - 1}")
        if window is None and int(work_frame) >= 0:
            target = min(max(0, int(work_frame)), frame_count - 1)
            bounds = [0, *[cut for cut in cuts if 0 < cut < frame_count], frame_count]
            window = (
                max(value for value in bounds[:-1] if value <= target),
                min(value for value in bounds[1:] if value > target),
            )
            _segment_info(node_name, f"work_frame {target} selects shot {window[0]}-{window[1] - 1}")
        if window is not None:
            low, high = window
            segments = [
                (anchor, max(start, low), min(end, high))
                for anchor, start, end in segments
                if low <= anchor < high
            ]
            segments = [item for item in segments if item[2] > item[1]]
            _segment_info(
                node_name,
                f"frame_range limits processing to {low}-{high - 1}; segments={len(segments)} "
                "(the rest of the mask stays empty)",
            )
            if not segments:
                raise ValueError(f"No anchor lies inside frame_range {low}-{high - 1}.")
        empty_shots = [
            [start, end]
            for start, end in zip([0, *cuts], [*cuts, frame_count])
            if not any(start <= frame < end for frame in anchor_frames)
        ]
        if empty_shots:
            _segment_info(node_name, f"shots without an anchor (empty mask): {empty_shots}")

        # ``torch.zeros`` memsets, so every page is committed even when only one
        # chunk is tracked.  ``np.zeros`` hands back demand-zero pages, keeping
        # the untouched frames of a long video out of the working set.
        union = torch.from_numpy(np.zeros((frame_count, height, width), dtype=np.float32))
        labels = torch.from_numpy(np.zeros((frame_count, height, width), dtype=np.uint8))
        progress_total = max(1, sum((end - anchor if end - anchor > 1 else 0) + (anchor - start + 1 if anchor > start else 0) for anchor, start, end in segments))
        backend_pbar = comfy.utils.ProgressBar(progress_total)
        _segment_info(
            node_name,
            f"propagating masks: direction={propagation_direction}, anchors={anchor_frames}, segments={len(segments)}",
        )
        pbar = _SegmentProgress(node_name, progress_total, backend_pbar)
        nested_tqdm = _NestedTqdmSilencer(("comfy.ldm.sam3.tracker",))
        nested_tqdm.start()
        try:
            _propagate_segments(model, images, anchors, segments, pbar, object_limit, union, labels)
        finally:
            nested_tqdm.stop()
            pbar.close()
        low, high = window if window is not None else (0, frame_count)
        union[low:high].clamp_(0.0, 1.0)

        palette = _parse_object_colors(object_colors, object_count)
        color_mask = torch.from_numpy(np.zeros((frame_count, height, width, 3), dtype=np.float32))
        for start in range(low, high, 32):
            end = min(high, start + 32)
            color_mask[start:end] = palette[labels[start:end].long()] * union[start:end, ..., None]
        del labels

        anchor_mask = anchors[0]["masks"].amax(dim=0).to("cpu").float().clamp_(0.0, 1.0)
        info = {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "anchor_frame": anchor_frames[0],
            "anchor_frames": anchor_frames,
            "propagation_direction": propagation_direction,
            "prompt_version": PROMPT_VERSION if len(anchors) == 1 else MULTI_ANCHOR_PROMPT_VERSION,
            "object_count": object_count,
            "object_colors": ["#%02X%02X%02X" % tuple(round(float(v) * 255) for v in row) for row in palette[1:]],
            "stop_at_shot_cuts": bool(stop_at_shot_cuts),
            # Downstream nodes (the Shot Planner) follow this while testing a
            # single shot, so only one field has to be set.
            "work_frame": int(work_frame),
            "frame_window": list(window) if window is not None else None,
            "shot_cuts": cuts,
            "segments": [{"anchor": anchor, "start": start, "end": end} for anchor, start, end in segments],
        }
        _segment_info(node_name, f"complete: frames={frame_count}, anchors={len(anchors)}, object_count={object_count}")
        return io.NodeOutput(union, anchor_mask, info, color_mask)


_SHOT_PREVIEW_CUT_CACHE: dict[str, list[int]] = {}
_SHOT_PREVIEW_LOCK = threading.Lock()
# Per-object labels of the last shot preview, so a frame of that track can be
# adopted as a Selector prompt and corrected by hand.
_SHOT_PREVIEW_TRACK: dict[str, Any] = {}


def _selector_frame_batch(payload: dict[str, Any]) -> np.ndarray:
    """Load every cached Selector input frame as ``uint8 [F, H, W, 3]``.

    Shot previews need whole shots and cut detection, so only cached inputs
    are accepted: they are guaranteed to match the node's own frames.
    """
    token = str(payload.get("source_token") or "").strip()
    if not token:
        raise ValueError(
            "Preview Current Shot needs the node input cache. Set wait_for_input_cache to true, "
            "run the workflow once, set it back to false and reopen the Selector."
        )
    if token.startswith("loader_preview:"):
        cache = _loader_preview_cache()
        entry = cache.entry_for_token(token) if cache is not None else None
        if entry is None:
            raise ValueError("The shared loader preview cache is unavailable.")
        frames: list[np.ndarray] = []
        with av.open(str(entry["video_path"]), mode="r") as container:
            for decoded in container.decode(container.streams.video[0]):
                frames.append(decoded.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError("The loader preview cache contains no frames.")
        return np.stack(frames, axis=0)
    if token.startswith("wait_input:"):
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        entry = module.get_wait_input_cache_store().get_token(token) if module is not None else None
    else:
        entry = _selector_cache_for_token(token)
    if entry is None:
        raise ValueError("The cached Selector input is no longer available. Run the workflow once again.")
    try:
        return np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("The cached Selector frames are unavailable. Run the workflow once again.") from exc


def _encode_png(array: np.ndarray, mode: str) -> str:
    buffer = py_io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_jpeg(array: np.ndarray) -> str:
    buffer = py_io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _shot_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Track only the shot containing ``frame`` and summarize every frame."""
    frames_u8 = _selector_frame_batch(payload)
    frame_count = int(frames_u8.shape[0])
    frame = min(max(0, int(payload.get("frame", 0))), frame_count - 1)
    height, width = int(frames_u8.shape[1]), int(frames_u8.shape[2])

    model = _preview_model(payload.get("model_source"))
    if model is None:
        raise ValueError(
            "Connect a CheckpointLoaderSimple or Load Diffusion Model node to MODEL, "
            "or run this SAM3 node once before using Preview."
        )
    direction = str(payload.get("propagation_direction") or "both")
    if direction not in _PROPAGATION_OPTIONS:
        direction = "both"
    object_limit = min(16, max(1, int(payload.get("max_objects") or 16)))
    object_count, anchors = _parse_anchor_prompts(payload.get("prompt_data"), frame, frame_count)

    cuts: list[int] = []
    cut_source = "disabled"
    if bool(payload.get("stop_at_shot_cuts", True)):
        manual = _parse_cut_frames(payload.get("shot_cut_frames"), frame_count)
        if manual is not None:
            cuts, cut_source = manual, "manual"
        else:
            key = f"{payload.get('source_token')}:{frame_count}"
            with _SHOT_PREVIEW_LOCK:
                cached = _SHOT_PREVIEW_CUT_CACHE.get(key)
            if cached is None:
                cached = _detect_shot_cuts(torch.from_numpy(np.asarray(frames_u8)))
                with _SHOT_PREVIEW_LOCK:
                    _SHOT_PREVIEW_CUT_CACHE[key] = cached
            cuts, cut_source = cached, "detected"
    bounds = [0, *cuts, frame_count]
    start = max(value for value in bounds[:-1] if value <= frame)
    end = min(value for value in bounds[1:] if value > frame)

    shot_anchors = [dict(anchor) for anchor in anchors if start <= anchor["frame"] < end]
    if not shot_anchors:
        raise ValueError(f"Shot {start}-{end - 1} has no anchor. Prompt an object on a frame inside this shot first.")
    images = torch.from_numpy(np.array(frames_u8[start:end], copy=True)).to(torch.float32).div_(255.0)
    shot_frames = end - start
    for anchor in shot_anchors:
        local = anchor["frame"] - start
        anchor["frame"] = local
        anchor["masks"] = _sam3_anchor_masks(model, images[local : local + 1], anchor["prompt_data"])[:object_limit]
        anchor["object_indices"] = anchor["object_indices"][:object_limit]
    segments = _anchor_segments([anchor["frame"] for anchor in shot_anchors], [], shot_frames, direction)
    union = torch.zeros(shot_frames, height, width, dtype=torch.float32)
    labels = torch.zeros(shot_frames, height, width, dtype=torch.uint8)
    nested_tqdm = _NestedTqdmSilencer(("comfy.ldm.sam3.tracker",))
    nested_tqdm.start()
    try:
        _propagate_segments(model, images, shot_anchors, segments, None, object_limit, union, labels)
    finally:
        nested_tqdm.stop()

    palette = _parse_object_colors(payload.get("object_colors"), object_count)
    colors_u8 = (palette * 255.0).round().to(torch.uint8)
    pixels = float(height * width)
    areas = [
        [round(float((labels[step] == index + 1).sum()) / pixels, 5) for step in range(shot_frames)]
        for index in range(object_count)
    ]
    # Overlays only need to match the Selector canvas, not the source size.
    scale = min(1.0, 480.0 / max(1, width))
    overlay_size = (max(1, round(height * scale)), max(1, round(width * scale)))
    overlays: list[str] = []
    for step in range(shot_frames):
        small_labels = F.interpolate(labels[step][None, None].float(), size=overlay_size, mode="nearest")[0, 0].long()
        small_alpha = F.interpolate(union[step][None, None], size=overlay_size, mode="bilinear", align_corners=False)[0, 0]
        rgba = torch.zeros(*overlay_size, 4, dtype=torch.uint8)
        rgba[..., :3] = colors_u8[small_labels]
        rgba[..., 3] = torch.where(small_labels > 0, small_alpha * 150.0, torch.zeros_like(small_alpha)).round().to(torch.uint8)
        overlays.append(_encode_png(rgba.numpy(), "RGBA"))

    thumb_count = min(shot_frames, 10)
    thumb_steps = sorted({round(index * (shot_frames - 1) / max(1, thumb_count - 1)) for index in range(thumb_count)})
    thumb_size = (max(1, round(height * 160.0 / width)), 160)
    thumbs: list[dict[str, Any]] = []
    for step in thumb_steps:
        source = F.interpolate(images[step].movedim(-1, 0)[None], size=thumb_size, mode="area")[0].movedim(0, -1)
        small_labels = F.interpolate(labels[step][None, None].float(), size=thumb_size, mode="nearest")[0, 0].long()
        alpha = (small_labels > 0).float()[..., None] * 0.55
        mixed = source * (1.0 - alpha) + palette[small_labels] * alpha
        thumbs.append({"frame": start + step, "image": _encode_jpeg((mixed * 255.0).round().clamp(0, 255).to(torch.uint8).numpy())})

    with _SHOT_PREVIEW_LOCK:
        _SHOT_PREVIEW_TRACK.clear()
        _SHOT_PREVIEW_TRACK.update({
            "start": start,
            "end": end,
            "object_count": object_count,
            "labels": labels,
            "height": height,
            "width": width,
        })

    return {
        "start": start,
        "end": end,
        "cuts": cuts,
        "cut_source": cut_source,
        "anchors": [anchor["frame"] + start for anchor in shot_anchors],
        "object_colors": ["#%02X%02X%02X" % tuple(int(v) for v in row) for row in colors_u8[1:].tolist()],
        "areas": areas,
        "overlays": overlays,
        "thumbs": thumbs,
    }


def _tracked_frame_masks(frame: int) -> dict[str, Any]:
    """Return the last shot preview's per-object mask for one frame as PNGs."""
    with _SHOT_PREVIEW_LOCK:
        track = dict(_SHOT_PREVIEW_TRACK)
    labels = track.get("labels")
    if labels is None:
        raise ValueError("Run Preview Current Shot first, then adopt one of its frames.")
    start, end = int(track["start"]), int(track["end"])
    if not start <= frame < end:
        raise ValueError(f"Frame {frame} is outside the tracked shot {start}-{end - 1}.")
    step = labels[frame - start]
    objects = []
    for index in range(1, int(track["object_count"]) + 1):
        present = step == index
        if not bool(present.any()):
            continue
        rgba = torch.zeros(*present.shape, 4, dtype=torch.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = present.to(torch.uint8) * 255
        objects.append({
            "index": index,
            "area": round(float(present.float().mean()), 5),
            "mask": _encode_png(rgba.numpy(), "RGBA"),
        })
    if not objects:
        raise ValueError(f"No tracked object covers frame {frame}.")
    return {"frame": frame, "start": start, "end": end, "width": int(track["width"]), "height": int(track["height"]), "objects": objects}


async def _video_segment_shot_mask_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        result = await asyncio.to_thread(_tracked_frame_masks, int(payload.get("frame", 0)))
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _video_segment_shot_preview_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None and not hasattr(server_instance, "last_prompt_id"):
            server_instance.last_prompt_id = "cinestyle-preview"
        result = await asyncio.to_thread(_shot_preview, payload)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


class VideoSegmentExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        global _PREVIEW_ROUTE_REGISTERED
        if _PREVIEW_ROUTE_REGISTERED:
            return
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None:
            server_instance.routes.post("/cinestyle/video-segment-preview")(
                _video_segment_preview_route
            )
            server_instance.routes.post("/cinestyle/video-segment-shot-preview")(
                _video_segment_shot_preview_route
            )
            server_instance.routes.post("/cinestyle/video-segment-shot-mask")(
                _video_segment_shot_mask_route
            )
            server_instance.routes.get("/cinestyle/sec-models")(
                _sec_video_segment_models_route
            )
            server_instance.routes.post("/cinestyle/sec-video-segment-preview")(
                _sec_video_segment_preview_route
            )
            server_instance.routes.get("/cinestyle/video-selector-cache")(
                _selector_cache_info_route
            )
            server_instance.routes.get("/cinestyle/video-selector-cache-video")(
                _selector_cache_video_route
            )
            _PREVIEW_ROUTE_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVideoSegmentSAM3, CSSeCModelLoader, CSVideoSegmentSeC]


async def comfy_entrypoint() -> VideoSegmentExtension:
    return VideoSegmentExtension()
