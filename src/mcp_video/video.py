"""Core video processing: download, keyframe extraction, transcript."""

import base64
import json
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image


def download_video(url: str, output_dir: Path) -> dict:
    output_path = output_dir / "video.%(ext)s"
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(output_path),
        "--print-json",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    info = json.loads(result.stdout.strip().split("\n")[-1])
    video_file = output_dir / f"video.mp4"
    if not video_file.exists():
        candidates = list(output_dir.glob("video.*"))
        if candidates:
            video_file = candidates[0]
        else:
            raise FileNotFoundError("Downloaded video not found")

    return {
        "path": str(video_file),
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", "Unknown"),
        "description": info.get("description", ""),
    }


def extract_keyframes(video_path: str, max_frames: int = 10, threshold: float = 30.0) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    frames = []
    prev_hist = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > threshold:
                timestamp = frame_idx / fps if fps > 0 else 0
                b64 = _frame_to_base64(frame)
                frames.append({
                    "frame_index": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "timestamp_formatted": _format_time(timestamp),
                    "image_base64": b64,
                    "scene_change_score": round(diff, 2),
                })
        else:
            timestamp = frame_idx / fps if fps > 0 else 0
            b64 = _frame_to_base64(frame)
            frames.append({
                "frame_index": 0,
                "timestamp": 0.0,
                "timestamp_formatted": "00:00",
                "image_base64": b64,
                "scene_change_score": 0.0,
            })

        prev_hist = hist
        frame_idx += 1

    cap.release()

    if len(frames) > max_frames:
        frames.sort(key=lambda f: f["scene_change_score"], reverse=True)
        frames = frames[:max_frames]
        frames.sort(key=lambda f: f["timestamp"])

    return frames


def extract_frame_at_timestamp(video_path: str, timestamp: float) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame at timestamp {timestamp}s")

    return {
        "timestamp": timestamp,
        "timestamp_formatted": _format_time(timestamp),
        "image_base64": _frame_to_base64(frame),
    }


def extract_uniform_frames(video_path: str, num_frames: int = 10) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // (num_frames + 1))

    frames = []
    for i in range(1, num_frames + 1):
        frame_num = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = frame_num / fps if fps > 0 else 0
        frames.append({
            "frame_index": frame_num,
            "timestamp": round(timestamp, 2),
            "timestamp_formatted": _format_time(timestamp),
            "image_base64": _frame_to_base64(frame),
        })

    cap.release()
    return frames


def get_transcript(url: str) -> dict:
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-lang", "en",
        "--sub-format", "json3",
        "--print-json",
        "--no-warnings",
        url,
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd.extend(["--output", f"{tmpdir}/subs"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        sub_files = list(Path(tmpdir).glob("*.json3"))
        if sub_files:
            return _parse_json3_subs(sub_files[0])

        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if vtt_files:
            return _parse_vtt_subs(vtt_files[0])

    cmd_fallback = [
        "yt-dlp",
        "--skip-download",
        "--print", "%(subtitles)j",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=60)
    if result.returncode == 0 and result.stdout.strip():
        return {"segments": [], "full_text": "(No captions available for this video)"}

    return {"segments": [], "full_text": "(No captions available)"}


def _parse_json3_subs(path: Path) -> dict:
    data = json.loads(path.read_text())
    segments = []
    full_parts = []

    for event in data.get("events", []):
        if "segs" not in event:
            continue
        text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
        if not text or text == "\n":
            continue
        start_ms = event.get("tStartMs", 0)
        dur_ms = event.get("dDurationMs", 0)
        start = start_ms / 1000
        segments.append({
            "start": round(start, 2),
            "end": round((start_ms + dur_ms) / 1000, 2),
            "start_formatted": _format_time(start),
            "text": text,
        })
        full_parts.append(text)

    return {"segments": segments, "full_text": " ".join(full_parts)}


def _parse_vtt_subs(path: Path) -> dict:
    lines = path.read_text().splitlines()
    segments = []
    full_parts = []
    i = 0
    while i < len(lines):
        if "-->" in lines[i]:
            parts = lines[i].split("-->")
            start = _vtt_time_to_seconds(parts[0].strip())
            end = _vtt_time_to_seconds(parts[1].strip().split()[0])
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines)
            if text:
                segments.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "start_formatted": _format_time(start),
                    "text": text,
                })
                full_parts.append(text)
        i += 1

    return {"segments": segments, "full_text": " ".join(full_parts)}


def _vtt_time_to_seconds(time_str: str) -> float:
    parts = time_str.replace(",", ".").split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return 0.0


def _frame_to_base64(frame, max_width: int = 512) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
