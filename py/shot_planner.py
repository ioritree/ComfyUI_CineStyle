"""Plan cut-aligned generation chunks for long videos.

A long video is packed into consecutive chunks whose boundaries fall on shot
cuts, so every chunk can be generated independently without overlap.  Each
queued run selects one chunk with ``chunk_index``; ComfyUI's batch count with
an incrementing ``chunk_index`` processes the whole video in one click.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
from typing_extensions import override

import comfy.model_management
import folder_paths


_CATEGORY = "😺dzNodes/CineStyle/Video"
_LOGGER = logging.getLogger("CineStyleShotPlanner")
# Chunks shorter than min_seconds are allowed only when the shots around them
# leave no better packing; this penalty keeps them rare.
_SHORT_CHUNK_PENALTY = 50.0


def _parse_cut_list(value: Any, frame_count: int) -> list[int] | None:
    """Parse ``"213, 247"`` or a list of frames; ``None`` when nothing is given."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        parts = [part for part in re.split(r"[\s,;\[\]]+", text) if part]
    try:
        frames = [int(float(part)) for part in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError("shot_cut_frames must be a comma-separated list of frame numbers.") from exc
    return sorted({frame for frame in frames if 0 < frame < frame_count})


def _detect_cuts(images: torch.Tensor) -> list[int]:
    """Reuse the Video Segment cut detector so both nodes agree on shots."""
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_video_segment")
    detector = getattr(module, "_detect_shot_cuts", None)
    if detector is None:
        raise RuntimeError("Shot detection is unavailable. Enter shot_cut_frames manually.")
    return detector(images)


def _generation_length(frames: int, step: int, offset: int) -> int:
    """Smallest legal generation length ``offset + k * step`` covering ``frames``."""
    if step <= 1:
        return max(frames, offset, 1)
    if frames <= offset:
        return max(offset, 1)
    return offset + math.ceil((frames - offset) / step) * step


def _split_long_shots(bounds: list[int], max_frames: int) -> list[int]:
    """Insert extra boundaries inside shots longer than ``max_frames``."""
    result = [bounds[0]]
    for start, end in zip(bounds[:-1], bounds[1:]):
        length = end - start
        if length > max_frames:
            parts = math.ceil(length / max_frames)
            for index in range(1, parts):
                result.append(start + round(length * index / parts))
        result.append(end)
    return result


def plan_chunks(
    frame_count: int,
    cuts: list[int],
    fps: float,
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
    frame_step: int,
    frame_offset: int,
    max_shots: int = 0,
    overlap: int = 0,
) -> dict[str, Any]:
    """Pack shots into chunks close to ``target_seconds`` with dynamic programming.

    A chunk that starts inside a shot re-generates ``overlap`` frames of the
    previous chunk as its ``lead``, so the seam can be anchored on them.
    """
    max_frames = max(1, int(math.floor(max_seconds * fps)))
    shot_bounds = [0, *[cut for cut in cuts if 0 < cut < frame_count], frame_count]
    bounds = _split_long_shots(shot_bounds, max(1, max_frames - overlap))
    split_inside_shots = len(bounds) != len(shot_bounds)

    shot_edges = set(shot_bounds)
    best: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for end_index in range(1, len(bounds)):
        for start_index in range(end_index - 1, -1, -1):
            length = bounds[end_index] - bounds[start_index]
            if length > max_frames:
                break
            if bounds[start_index] not in shot_edges:
                length += overlap
                if length > max_frames:
                    continue
            if max_shots > 0:
                # A generated clip that contains a cut lets the video model lose
                # track of who is who, so the number of shots can be capped.
                inner = sum(
                    1 for edge in shot_edges
                    if bounds[start_index] < edge < bounds[end_index]
                )
                if inner + 1 > max_shots:
                    continue
            previous = best.get(start_index)
            if previous is None:
                continue
            seconds = length / fps
            cost = (seconds - target_seconds) ** 2
            if seconds < min_seconds:
                cost += _SHORT_CHUNK_PENALTY
            total = previous[0] + cost
            if end_index not in best or total < best[end_index][0]:
                best[end_index] = (total, [*previous[1], (bounds[start_index], bounds[end_index])])

    chunks = []
    for index, (start, end) in enumerate(best[len(bounds) - 1][1]):
        length = end - start
        continues = start not in shot_edges
        lead = overlap if continues else 0
        generation = _generation_length(length + lead, frame_step, frame_offset)
        shots = [
            [max(start - lead, shot_start), min(end, shot_end)]
            for shot_start, shot_end in zip(shot_bounds[:-1], shot_bounds[1:])
            if shot_start < end and shot_end > start
        ]
        chunks.append({
            "index": index,
            "start": start,
            "end": end,
            "frames": length,
            "continues": continues,
            "lead": lead,
            "generation_frames": generation,
            "extra_frames": generation - length - lead,
            "shots": shots,
        })
    return {
        "frame_count": frame_count,
        "fps": fps,
        "max_shots": max_shots,
        "overlap": overlap,
        "cuts": [cut for cut in cuts if 0 < cut < frame_count],
        "split_inside_shots": split_inside_shots,
        "target_seconds": target_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "frame_step": frame_step,
        "frame_offset": frame_offset,
        "chunks": chunks,
    }


def _slice_frames(batch: torch.Tensor, start: int, end: int, length: int, freeze: bool) -> torch.Tensor:
    """Return ``length`` frames for a chunk covering ``start`` to ``end``.

    Generation lengths are quantised, so a chunk is usually a few frames longer
    than its shots.  ``freeze`` holds the chunk's last real frame for the extra
    frames instead of borrowing the next shot, which would put a cut inside the
    generated clip.  Padding frames are trimmed when the chunks are assembled.
    """
    stop = min(int(batch.shape[0]), end if freeze else start + length)
    available = batch[start:stop]
    missing = length - int(available.shape[0])
    if missing > 0:
        tail = available[-1:] if available.shape[0] else batch[-1:]
        available = torch.cat([available, tail.expand(missing, *batch.shape[1:])], dim=0)
    return available[:length].clone()


def _slice_audio(audio: Any, start_seconds: float, seconds: float) -> dict[str, Any]:
    # VideoHelperSuite hands over a lazy Mapping rather than a plain dict.
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    rate = int(audio["sample_rate"])
    first = max(0, int(round(start_seconds * rate)))
    count = max(1, int(round(seconds * rate)))
    piece = waveform[..., first : first + count]
    if piece.shape[-1] < count:
        pad = torch.zeros(*piece.shape[:-1], count - piece.shape[-1], dtype=piece.dtype, device=piece.device)
        piece = torch.cat([piece, pad], dim=-1)
    return {"waveform": piece.clone(), "sample_rate": rate}


_COLOR_NAMES = {
    "#FF0000": "紅色", "#00FF00": "綠色", "#0000FF": "藍色", "#FFFF00": "黃色",
    "#FF00FF": "洋紅", "#00FFFF": "青色", "#FF8000": "橙色", "#8000FF": "紫色",
}


def _present_objects(color_chunk: torch.Tensor, colors: list[str]) -> list[dict[str, Any]]:
    """Measure how much of the chunk each coloured object actually covers.

    A person can be the subject of the shot, or a leg swinging through the
    frame for a few frames.  The prompt needs that difference: describing the
    full appearance of someone who is only partly visible makes the video model
    render the whole person.
    """
    if not isinstance(color_chunk, torch.Tensor) or color_chunk.ndim != 4 or not colors:
        return []
    # Sides are decided per shot, so a chunk of several shots needs enough
    # samples for the shortest one to still be measurable.
    stride = max(1, int(color_chunk.shape[0]) // 48)
    sample = color_chunk[::stride, ..., :3].float()
    frames = int(sample.shape[0])
    pixels = float(sample.shape[1] * sample.shape[2])
    present: list[dict[str, Any]] = []
    for index, value in enumerate(colors, start=1):
        text = str(value).lstrip("#")
        if len(text) != 6:
            continue
        target = torch.tensor([int(text[offset : offset + 2], 16) / 255.0 for offset in (0, 2, 4)])
        hit = ((sample - target).abs().amax(dim=-1) < 0.30) & (sample.amax(dim=-1) > 0.15)
        per_frame = hit.flatten(1).sum(dim=1) / pixels
        visible = per_frame > 0.002
        seen = int(visible.sum())
        if not seen:
            continue
        # A limb swinging into view is cut by the frame edge; a whole person
        # far from the camera is small too but stays inside the frame.
        band = max(1, round(min(hit.shape[1], hit.shape[2]) * 0.01))
        edge = (
            hit[:, :band].flatten(1).any(dim=1) | hit[:, -band:].flatten(1).any(dim=1)
            | hit[:, :, :band].flatten(1).any(dim=1) | hit[:, :, -band:].flatten(1).any(dim=1)
        )
        columns = hit.sum(dim=1).float()
        axis = torch.arange(columns.shape[1], dtype=torch.float32) + 0.5
        centre = (columns * axis).sum(dim=1) / columns.sum(dim=1).clamp(min=1.0) / columns.shape[1]
        present.append({
            "index": index,
            "mean": round(float(per_frame.mean()) * 100, 1),
            "active": round(float(per_frame[visible].mean()) * 100, 1),
            "cut_off": round(int((edge & visible).sum()) / seen, 2),
            "peak": round(float(per_frame.max()) * 100, 1),
            "frames_seen": seen,
            "frames": frames,
            "step": stride,
            "seen": visible.tolist(),
            "centres": [round(float(value), 3) for value in centre],
        })
    return present


def _sides(present: list[dict[str, Any]], start: int = 0, end: int = 1 << 30) -> list[str]:
    """Name the half of the frame each mask keeps to, or nothing for anybody.

    The colour mapping alone lets the video model drift between the two
    reference pictures, so the prompt also says which side each mask is on.
    With two people the claim is relative, so it is decided by their order in
    every frame where both appear: fighters at close range sit either side of
    the middle without ever crossing.  One person alone has no order to test
    and is placed against the middle instead.
    """
    blank = [""] * len(present)

    def inside(item: dict[str, Any], index: int) -> bool:
        return start <= index * int(item["step"]) < end

    if len(present) == 2:
        first, second = present
        pairs = [
            (left, right)
            for index, (left, right) in enumerate(zip(first["centres"], second["centres"]))
            if first["seen"][index] and second["seen"][index] and inside(first, index)
        ]
        if not pairs:
            return blank
        if all(left + 0.05 < right for left, right in pairs):
            return ["畫面左側", "畫面右側"]
        if all(right + 0.05 < left for left, right in pairs):
            return ["畫面右側", "畫面左側"]
        return blank
    if len(present) == 1:
        item = present[0]
        centres = [value for index, value in enumerate(item["centres"]) if item["seen"][index] and inside(item, index)]
        if not centres:
            return blank
        if max(centres) < 0.40:
            return ["畫面左側"]
        if min(centres) > 0.60:
            return ["畫面右側"]
    return blank


def _presence_text(present: list[dict[str, Any]], colors: list[str], located: bool = False) -> str:
    def describe(item: dict[str, Any]) -> str:
        index = int(item["index"])
        code = str(colors[index - 1]).upper() if index - 1 < len(colors) else ""
        ratio = item["frames_seen"] / max(1, item["frames"])
        # Size and duration are separate: a figure that fills a fifth of the
        # frame and then walks out is a real character, not a fragment.
        if item["active"] < 8.0 and item["cut_off"] >= 0.5:
            role = (
                f"僅局部入鏡（出現時平均佔畫面 {item['active']}%，出現於 {item['frames_seen']}/{item['frames']} 取樣幀）"
                "，只描述可見部位，不要描述看不到的臉、髮型或服裝整體，也不要生成完整人物"
            )
        elif ratio < 0.7:
            role = (
                f"只出現於本段的 {item['frames_seen']}/{item['frames']} 取樣幀（出現時平均佔畫面 {item['active']}%）"
                "，出現期間是完整人物，必須完整替換；離開畫面後不要再補出這個人"
            )
        elif item["active"] < 8.0:
            role = (
                f"本段主要角色，位於遠景（出現時平均佔畫面 {item['active']}%），畫面中是完整人物，"
                "必須完整替換臉部、髮型與服裝"
            )
        else:
            role = f"本段主要角色（平均佔畫面 {item['mean']}%）"
        return f"{_COLOR_NAMES.get(code, code)}（{code}）→ <Picture {index}>：{role}"

    if not present:
        return ""
    indices = [int(item["index"]) for item in present]
    missing = [index for index in range(1, len(colors) + 1) if index not in indices]
    line = "本段出現的人物：\n- " + "\n- ".join(describe(item) for item in present)
    if missing:
        line += "\n本段沒有出現 " + "、".join(
            f"<Picture {index}>" + (f"（{_COLOR_NAMES.get(str(colors[index - 1]).upper(), '')}）" if index - 1 < len(colors) else "")
            for index in missing
        ) + " 的人物，請完全不要使用這些參考圖，也不要憑空加入該人物。"
    elif len(present) < 2:
        if located:
            line += "\n每個鏡頭的左右方位已標在該鏡頭那一行。"
    elif located:
        line += "\n每個鏡頭的左右方位已標在該鏡頭那一行，請依遮罩顏色與該鏡頭的方位對應參考圖，兩人的對應關係不可互換。"
    else:
        line += "\n兩人的對應關係不可互換。"
    return line


def _format_seconds(frames: int, fps: float) -> str:
    return f"{frames / fps:.2f}s"


def _shot_text(chunk: dict[str, Any], fps: float, frame_count: int, freeze: bool = True,
               present: list[dict[str, Any]] | None = None, colors: list[str] | None = None) -> str:
    """Describe the chunk's shots with chunk-local frame numbers.

    Sides are decided per shot: across a cut the two people often swap halves,
    so one side for the whole chunk would be wrong for some of its shots and
    the prompt model would fill the gap with its own guess.
    """
    present = present or []
    colors = colors or []
    # Lead frames belong to the first shot, so they are numbered as part of it.
    start = chunk["start"] - chunk["lead"]
    located = False
    lines = []
    for number, (shot_start, shot_end) in enumerate(chunk["shots"], start=1):
        local_start, local_end = shot_start - start, shot_end - start - 1
        line = f"[Shot {number}] {local_start}–{local_end} 幀（{_format_seconds(local_start, fps)}–{_format_seconds(local_end + 1, fps)}）"
        sides = _sides(present, local_start, local_end + 1)
        named = [
            f"{_COLOR_NAMES.get(str(colors[item['index'] - 1]).upper(), '')}在{side}"
            for item, side in zip(present, sides)
            if side and item["index"] - 1 < len(colors)
        ]
        if named:
            line += "：" + "、".join(named)
            located = True
        lines.append(line)
    padding = ""
    if chunk["extra_frames"] > 0:
        local_start = chunk["lead"] + chunk["frames"]
        local_end = chunk["generation_frames"] - 1
        if not freeze and chunk["end"] < frame_count:
            lines.append(
                f"[Shot {len(chunk['shots']) + 1}] {local_start}–{local_end} 幀"
                f"（下一個鏡頭的開頭，僅為湊足生成幀數，生成後會裁掉）"
            )
        else:
            # Numbering this as a Shot makes the prompt model write it up as a
            # cut and invent a new framing, which loses the character binding.
            held = "畫面停在第" if freeze else "重複第"
            padding = (
                f"（{local_start}–{local_end} 幀為補齊幀，{held} {local_start - 1} 幀不動，"
                "沒有換鏡頭、沒有改變構圖或人物，生成後會裁掉）"
            )
    text = "；\n".join(lines) + f"\n（{fps:g}fps，共 {chunk['generation_frames']} 幀）"
    if padding:
        text += "\n" + padding
    presence = _presence_text(present, colors, located)
    if presence:
        text += "\n" + presence
    return text


class CSShotPlanner(io.ComfyNode):
    """Split a long video into cut-aligned chunks and output one chunk per run."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Shot_Planner",
            display_name="CS Shot Planner",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=["shot planner", "chunk video", "split long video", "cut aligned chunks"],
            description=(
                "Pack shots into generation chunks aligned to cuts. Set chunk_index to increment and the "
                "batch count to chunk_count to process a long video in one queue."
            ),
            inputs=[
                io.Image.Input("images", tooltip="All video frames, e.g. the IMAGE input of CS Video Segment (SAM3.1)."),
                io.Int.Input(
                    "chunk_index",
                    default=0,
                    min=0,
                    max=100000,
                    step=1,
                    control_after_generate=io.ControlAfterGenerate.increment,
                    tooltip="Chunk to output. Use increment with a batch count equal to chunk_count.",
                ),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.001, tooltip="Frame rate of images."),
                io.Float.Input("target_seconds", default=8.0, min=0.5, max=60.0, step=0.1, tooltip="Preferred chunk length."),
                io.Float.Input("min_seconds", default=4.0, min=0.0, max=60.0, step=0.1, tooltip="Chunks shorter than this are avoided when possible."),
                io.Float.Input("max_seconds", default=10.0, min=0.5, max=60.0, step=0.1, tooltip="Hard limit before rounding to legal generation frames. A shot longer than this is split inside the shot."),
                io.Int.Input("frame_step", default=17, min=1, max=512, step=1, advanced=True, tooltip="Legal generation length is frame_offset + k * frame_step. MiniMax: 17."),
                io.Int.Input("frame_offset", default=5, min=0, max=512, step=1, advanced=True, tooltip="Legal generation length is frame_offset + k * frame_step. MiniMax: 5."),
                io.Int.Input(
                    "locate_frame",
                    default=-1,
                    min=-1,
                    max=100000000,
                    step=1,
                    tooltip=(
                        "Render the chunk that contains this source frame and ignore chunk_index. "
                        "-1 follows video_info.work_frame from CS Video Segment (SAM3.1), and falls "
                        "back to chunk_index when that is -1 too."
                    ),
                ),
                io.Int.Input(
                    "max_shots_per_chunk",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip=(
                        "Maximum shots packed into one chunk; 0 means no limit. Set 1 to generate one shot "
                        "per run so a cut never lands inside a generated clip."
                    ),
                ),
                io.Combo.Input(
                    "padding",
                    options=["freeze last frame", "next shot frames"],
                    default="freeze last frame",
                    tooltip=(
                        "How the extra frames needed by the generation length rule are filled. Freezing the "
                        "last frame keeps the chunk inside one shot; borrowing the next shot puts a cut in it."
                    ),
                ),
                io.String.Input("shot_cut_frames", default="", optional=True, tooltip="Optional first frames of each new shot. Empty uses video_info.shot_cuts, then automatic detection. 0 means one continuous shot, split only by max_seconds."),
                io.Dict.Input("video_info", optional=True, tooltip="video_info from CS Video Segment (SAM3.1); its shot_cuts are reused."),
                io.Image.Input("color_mask", optional=True, tooltip="color_mask from CS Video Segment (SAM3.1), sliced like images."),
                io.Mask.Input("mask", optional=True, tooltip="Optional MASK batch, sliced like images."),
                io.Audio.Input("audio", optional=True, tooltip="Optional audio of the whole video, sliced to the chunk."),
                io.Boolean.Input(
                    "unload_models",
                    display_name="unload models before chunk",
                    default=True,
                    tooltip=(
                        "Unload models left in VRAM by the previous run before this chunk. Disable to keep "
                        "models loaded for faster runs when VRAM allows."
                    ),
                ),
                io.Int.Input(
                    "seam_overlap",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip=(
                        "Frames of the previous chunk re-generated at the start of a chunk that continues the "
                        "same shot, anchored by CS Seam Guide and trimmed when assembling. 5 carries texture and "
                        "motion across the seam; 0 anchors a single frame. Chunks that start on a cut get none."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("images", display_name="IMAGE"),
                io.Image.Output("color_mask", display_name="color_mask"),
                io.Mask.Output("mask", display_name="MASK"),
                io.Audio.Output("audio", display_name="AUDIO"),
                io.Int.Output("frame_count", display_name="frame_count"),
                io.Int.Output("chunk_count", display_name="chunk_count"),
                io.String.Output("shot_text", display_name="shot_text"),
                io.Dict.Output("plan", display_name="plan"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs: Any) -> Any:
        # Model unloading is a side effect, so the node must run for every chunk.
        return float("nan") if kwargs.get("unload_models", True) else None

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        chunk_index: int = 0,
        fps: float = 24.0,
        target_seconds: float = 8.0,
        min_seconds: float = 4.0,
        max_seconds: float = 10.0,
        frame_step: int = 17,
        frame_offset: int = 5,
        locate_frame: int = -1,
        max_shots_per_chunk: int = 0,
        seam_overlap: int = 0,
        padding: str = "freeze last frame",
        shot_cut_frames: str = "",
        video_info: dict[str, Any] | None = None,
        color_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        audio: dict[str, Any] | None = None,
        unload_models: bool = True,
    ) -> io.NodeOutput:
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("images must be a non-empty IMAGE batch.")
        if max_seconds < min_seconds:
            raise ValueError("max_seconds must be greater than or equal to min_seconds.")
        if unload_models:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            _LOGGER.info("[CS Shot Planner] unloaded models before chunk %d", int(chunk_index))

        frame_count = int(images.shape[0])
        cuts = _parse_cut_list(shot_cut_frames, frame_count)
        cut_source = "manual"
        if cuts is None and isinstance(video_info, dict) and video_info.get("shot_cuts") is not None:
            if int(video_info.get("frame_count", frame_count)) != frame_count:
                raise ValueError("video_info.frame_count does not match images. Connect the same frames used by CS Video Segment.")
            cuts = _parse_cut_list(video_info.get("shot_cuts") or [], frame_count) or []
            cut_source = "video_info"
        if cuts is None:
            cuts = _detect_cuts(images)
            cut_source = "detected"

        plan = plan_chunks(
            frame_count, cuts, float(fps), float(target_seconds), float(min_seconds), float(max_seconds),
            int(frame_step), int(frame_offset), max(0, int(max_shots_per_chunk)),
            overlap=max(0, int(seam_overlap)),
        )
        plan["cut_source"] = cut_source
        plan["padding"] = str(padding)
        chunk_count = len(plan["chunks"])
        index = int(chunk_index)
        located = int(locate_frame)
        if located < 0 and isinstance(video_info, dict):
            # CS Video Segment (SAM3.1) exposes the shot it is testing; follow it
            # so a single field drives both nodes.
            inherited = int(video_info.get("work_frame", -1) or -1)
            if inherited >= 0:
                located = inherited
                _LOGGER.info("[CS Shot Planner] following video_info.work_frame %d", located)
        if located >= 0:
            match = next((item for item in plan["chunks"] if item["start"] <= located < item["end"]), None)
            if match is None:
                raise ValueError(f"locate_frame {located} is outside 0..{frame_count - 1}.")
            index = int(match["index"])
            _LOGGER.info(
                "[CS Shot Planner] frame %d is in chunk #%d (%d-%d); chunk_index %s ignored",
                located, index, match["start"], match["end"] - 1, chunk_index,
            )
        if not 0 <= index < chunk_count:
            raise ValueError(
                f"chunk_index {index} is out of range: this video has {chunk_count} chunks (0..{chunk_count - 1}). "
                "Reset chunk_index to 0."
            )
        chunk = plan["chunks"][index]
        start, length = chunk["start"] - chunk["lead"], chunk["generation_frames"]
        summary = ", ".join(
            f"#{item['index']} {item['start']}-{item['end'] - 1} ({item['frames'] / fps:.2f}s→{item['generation_frames']}f)"
            for item in plan["chunks"]
        )
        _LOGGER.info("[CS Shot Planner] %s cuts=%s; chunks: %s", cut_source, plan["cuts"], summary)
        if plan["split_inside_shots"]:
            _LOGGER.warning("[CS Shot Planner] a shot is longer than max_seconds and was split inside the shot.")

        freeze = str(padding) != "next shot frames"
        end = chunk["end"]
        chunk_images = _slice_frames(images, start, end, length, freeze)
        chunk_color = (
            _slice_frames(color_mask, start, end, length, freeze)
            if isinstance(color_mask, torch.Tensor) and color_mask.ndim == 4
            else torch.zeros_like(chunk_images)
        )
        if isinstance(mask, torch.Tensor) and mask.ndim == 3:
            chunk_mask = _slice_frames(mask, start, end, length, freeze)
        else:
            chunk_mask = torch.zeros(length, chunk_images.shape[1], chunk_images.shape[2], dtype=torch.float32)
        seconds = length / float(fps)
        if isinstance(audio, Mapping) and "waveform" in audio:
            # Audio follows the same window as the frames.
            audio_seconds = (end - start) / float(fps) if freeze else seconds
            chunk_audio = _slice_audio(audio, start / float(fps), audio_seconds)
            if freeze and audio_seconds < seconds:
                waveform = chunk_audio["waveform"]
                rate = int(chunk_audio["sample_rate"])
                silence = torch.zeros(
                    *waveform.shape[:-1],
                    max(0, int(round(seconds * rate)) - int(waveform.shape[-1])),
                    dtype=waveform.dtype,
                )
                chunk_audio = {"waveform": torch.cat([waveform, silence], dim=-1), "sample_rate": rate}
        else:
            chunk_audio = {"waveform": torch.zeros(1, 2, max(1, int(round(seconds * 44100)))), "sample_rate": 44100}

        colors = [str(value) for value in (video_info or {}).get("object_colors") or []] if isinstance(video_info, dict) else []
        present = _present_objects(chunk_color, colors)
        chunk["objects_present"] = present
        plan["objects_present"] = present
        if present:
            _LOGGER.info(
                "[CS Shot Planner] chunk %d coverage: %s", index,
                ", ".join(f"Object {item['index']} mean {item['mean']}% seen {item['frames_seen']}/{item['frames']}" for item in present),
            )
        plan["chunk_index"] = index
        plan["locate_frame"] = located
        return io.NodeOutput(
            chunk_images,
            chunk_color,
            chunk_mask,
            chunk_audio,
            length,
            chunk_count,
            _shot_text(chunk, float(fps), frame_count, freeze, present, colors),
            plan,
        )


def _seam_dir() -> str:
    return os.path.join(folder_paths.get_output_directory(), "cs_seams")


def _seam_path(plan: dict[str, Any], index: int) -> str:
    """Seam file of one chunk; the name ties it to this video's chunk layout."""
    chunk = plan["chunks"][index]
    return os.path.join(_seam_dir(), f"seam_{plan['frame_count']}f_{chunk['start']}-{chunk['end']}.npy")


class CSSeamSave(io.ComfyNode):
    """Store the last real frames of a generated chunk for the next chunk's CS Seam Guide."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Seam_Save",
            display_name="CS Seam Save",
            category=_CATEGORY,
            description=(
                "Save the last frames of this chunk, before the padding frames, so the next chunk's "
                "CS Seam Guide can start from them."
            ),
            inputs=[
                io.Image.Input("images", tooltip="Decoded frames of the generated chunk, before upscaling."),
                io.Dict.Input("plan", tooltip="plan from CS Shot Planner."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, plan: dict[str, Any]) -> io.NodeOutput:
        index = int(plan["chunk_index"])
        chunk = plan["chunks"][index]
        end = int(chunk["lead"]) + int(chunk["frames"])
        count = min(max(1, int(plan["overlap"])), end)
        if images.shape[0] < end:
            raise ValueError(f"images has {images.shape[0]} frames; chunk {index} needs {end}.")
        path = _seam_path(plan, index)
        os.makedirs(_seam_dir(), exist_ok=True)
        np.save(path, (images[end - count : end, ..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy())
        _LOGGER.info("[CS Seam Save] chunk %d frames %d-%d -> %s", index, end - count, end - 1, path)
        return io.NodeOutput()


class CSSeamGuide(io.ComfyNode):
    """Anchor the previous chunk's last frames at this chunk's start so chunks join without a jump."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Seam_Guide",
            display_name="CS Seam Guide",
            category=_CATEGORY,
            description=(
                "Start this chunk from the frames CS Seam Save stored for the previous chunk. Chunk 0, a "
                "chunk that starts on a cut, or one whose previous seam was never saved passes through unchanged."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent", tooltip="LATENT from MiniMax H3 Reference to Video."),
                io.Vae.Input("vae", tooltip="MiniMax H3 video VAE."),
                io.Dict.Input("plan", tooltip="plan from CS Shot Planner."),
                io.Boolean.Input("enabled", default=True),
                io.Image.Input(
                    "seam_image",
                    optional=True,
                    tooltip=(
                        "Use these frames instead of the saved seam, e.g. the last frames of an earlier take you "
                        "prefer. The last frame lines up with the frame before this chunk."
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs: Any) -> Any:
        # The seam is read from disk, so a newly saved seam must invalidate the cache.
        folder = _seam_dir()
        if not os.path.isdir(folder):
            return ""
        return sorted((name, os.path.getmtime(os.path.join(folder, name))) for name in os.listdir(folder))

    @classmethod
    def execute(
        cls,
        positive: Any,
        latent: dict[str, Any],
        vae: Any,
        plan: dict[str, Any],
        enabled: bool = True,
        seam_image: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        index = int(plan["chunk_index"])
        chunk = plan["chunks"][index]
        if not enabled:
            return io.NodeOutput(positive)
        frames = seam_image
        if frames is None:
            if not chunk["continues"]:
                # Chunk 0, or a real cut: the new shot must not start from the old one.
                return io.NodeOutput(positive)
            path = _seam_path(plan, index - 1)
            if not os.path.isfile(path):
                _LOGGER.info("[CS Seam Guide] no seam saved for chunk %d yet; chunk %d starts unanchored", index - 1, index)
                return io.NodeOutput(positive)
            frames = torch.from_numpy(np.load(path)).float() / 255.0
            _LOGGER.info("[CS Seam Guide] chunk %d starts from %s", index, path)
        lead = int(chunk["lead"])
        # The guide takes one frame or a 5, 22, 39... frame clip; its last frame
        # sits on the last lead frame, which repeats the previous chunk's end.
        available = min(lead, int(frames.shape[0]))
        count = 1 if available < 5 else available - (available - 5) % 17
        return MiniMaxH3AddGuide.execute(
            positive=positive, latent=latent, frame_idx=max(0, lead - count), vae=vae, image=frames[-count:],
        )


class ShotPlannerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSShotPlanner, CSSeamSave, CSSeamGuide]


async def comfy_entrypoint() -> ShotPlannerExtension:
    return ShotPlannerExtension()
