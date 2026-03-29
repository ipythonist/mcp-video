"""Tests for core video processing functions."""

import base64
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mcp_video.video import (
    _format_time,
    _frame_to_base64,
    _vtt_time_to_seconds,
    extract_keyframes,
    extract_uniform_frames,
    extract_frame_at_timestamp,
)


def _make_test_video(path: str, num_frames: int = 30, fps: int = 10):
    """Create a minimal test video file."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (64, 64))

    for i in range(num_frames):
        color = (i * 8 % 256, (i * 4) % 256, (255 - i * 8) % 256)
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        if i % 10 == 0:
            frame[:32, :32] = (255, 255, 255)
        writer.write(frame)

    writer.release()


class TestFormatTime:
    def test_seconds_only(self):
        assert _format_time(45) == "00:45"

    def test_minutes_and_seconds(self):
        assert _format_time(125) == "02:05"

    def test_hours(self):
        assert _format_time(3661) == "1:01:01"

    def test_zero(self):
        assert _format_time(0) == "00:00"


class TestVttTimeToSeconds:
    def test_hms(self):
        assert _vtt_time_to_seconds("01:02:03.500") == 3723.5

    def test_ms(self):
        assert _vtt_time_to_seconds("05:30.000") == 330.0

    def test_comma_separator(self):
        assert _vtt_time_to_seconds("00:01,500") == 1.5


class TestFrameToBase64:
    def test_returns_valid_base64(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        b64 = _frame_to_base64(frame)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0
        assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_resizes_large_frames(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        b64 = _frame_to_base64(frame, max_width=256)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0


class TestExtractKeyframes:
    def test_extracts_frames_from_video(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            _make_test_video(f.name, num_frames=50)
            frames = extract_keyframes(f.name, max_frames=5, threshold=5.0)

        assert len(frames) > 0
        assert len(frames) <= 5
        assert "timestamp" in frames[0]
        assert "image_base64" in frames[0]
        assert "timestamp_formatted" in frames[0]

    def test_invalid_video_raises(self):
        with pytest.raises(RuntimeError):
            extract_keyframes("/nonexistent/video.mp4")


class TestExtractUniformFrames:
    def test_extracts_evenly_spaced(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            _make_test_video(f.name, num_frames=100)
            frames = extract_uniform_frames(f.name, num_frames=5)

        assert len(frames) == 5
        timestamps = [f["timestamp"] for f in frames]
        assert timestamps == sorted(timestamps)


class TestExtractFrameAtTimestamp:
    def test_extracts_single_frame(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            _make_test_video(f.name, num_frames=50, fps=10)
            frame = extract_frame_at_timestamp(f.name, 1.0)

        assert "image_base64" in frame
        assert frame["timestamp"] == 1.0
