"""MCP Video server - expose video intelligence tools via MCP."""

import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from mcp_video.video import (
    download_video,
    extract_frame_at_timestamp,
    extract_keyframes,
    extract_uniform_frames,
    get_transcript,
)

mcp = FastMCP(
    "mcp-video",
    description="Video intelligence MCP server - analyze any video with AI vision",
)


@mcp.tool()
def analyze_video(
    url: str,
    max_frames: int = 8,
    include_transcript: bool = True,
) -> list[TextContent | ImageContent]:
    """Analyze a video by extracting key frames and transcript.

    Supports YouTube, Vimeo, Twitter/X, and 1000+ sites via yt-dlp.
    Returns key scene-change frames as images plus the transcript text.

    Args:
        url: Video URL (YouTube, Vimeo, Twitter, or any yt-dlp supported site)
        max_frames: Maximum number of keyframes to extract (default: 8)
        include_transcript: Whether to include transcript/captions (default: True)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        info = download_video(url, Path(tmpdir))
        frames = extract_keyframes(info["path"], max_frames=max_frames)

        result = []

        meta = (
            f"**Video:** {info['title']}\n"
            f"**Duration:** {_fmt_duration(info['duration'])}\n"
            f"**Uploader:** {info['uploader']}\n"
            f"**Frames extracted:** {len(frames)}"
        )
        result.append(TextContent(type="text", text=meta))

        for frame in frames:
            result.append(TextContent(
                type="text",
                text=f"--- Frame at {frame['timestamp_formatted']} (scene change score: {frame['scene_change_score']}) ---",
            ))
            result.append(ImageContent(
                type="image",
                data=frame["image_base64"],
                mimeType="image/jpeg",
            ))

        if include_transcript:
            transcript = get_transcript(url)
            if transcript["full_text"]:
                result.append(TextContent(
                    type="text",
                    text=f"\n**Transcript:**\n{transcript['full_text'][:5000]}",
                ))

    return result


@mcp.tool()
def extract_video_frames(
    url: str,
    max_frames: int = 10,
    mode: str = "keyframe",
) -> list[TextContent | ImageContent]:
    """Extract frames from a video without transcript.

    Args:
        url: Video URL
        max_frames: Number of frames to extract (default: 10)
        mode: 'keyframe' for scene-change detection, 'uniform' for evenly spaced (default: keyframe)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        info = download_video(url, Path(tmpdir))

        if mode == "uniform":
            frames = extract_uniform_frames(info["path"], num_frames=max_frames)
        else:
            frames = extract_keyframes(info["path"], max_frames=max_frames)

        result = []
        result.append(TextContent(
            type="text",
            text=f"**{info['title']}** - {len(frames)} frames extracted ({mode} mode)",
        ))

        for frame in frames:
            result.append(TextContent(
                type="text",
                text=f"[{frame['timestamp_formatted']}]",
            ))
            result.append(ImageContent(
                type="image",
                data=frame["image_base64"],
                mimeType="image/jpeg",
            ))

    return result


@mcp.tool()
def get_video_transcript(url: str) -> str:
    """Get the transcript/captions of a video.

    Returns timestamped transcript segments. Works best with YouTube videos
    that have captions (auto-generated or manual).

    Args:
        url: Video URL (YouTube recommended for best caption support)
    """
    transcript = get_transcript(url)

    if not transcript["segments"]:
        return transcript["full_text"]

    lines = []
    for seg in transcript["segments"]:
        lines.append(f"[{seg['start_formatted']}] {seg['text']}")

    return "\n".join(lines)


@mcp.tool()
def get_frame_at_time(url: str, timestamp: float) -> list[TextContent | ImageContent]:
    """Extract a single frame at a specific timestamp from a video.

    Args:
        url: Video URL
        timestamp: Time in seconds to extract the frame at
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        info = download_video(url, Path(tmpdir))
        frame = extract_frame_at_timestamp(info["path"], timestamp)

        return [
            TextContent(
                type="text",
                text=f"Frame from **{info['title']}** at {frame['timestamp_formatted']}",
            ),
            ImageContent(
                type="image",
                data=frame["image_base64"],
                mimeType="image/jpeg",
            ),
        ]


@mcp.tool()
def video_to_strategy(
    url: str,
    framework: str = "backtrader",
    max_frames: int = 15,
) -> list[TextContent | ImageContent]:
    """Extract a trading strategy from a video and return structured data for code generation.

    Analyzes trading/investing tutorial videos (YouTube, Instagram, etc.) by extracting
    chart screenshots, indicator visuals, and spoken strategy rules. Returns everything
    the LLM needs to generate runnable trading strategy code.

    Supported frameworks: backtrader, zipline, pandas-ta, freqtrade

    Args:
        url: Video URL containing a trading strategy explanation
        framework: Target trading framework for code generation (default: backtrader)
        max_frames: Max frames to extract - higher catches more chart details (default: 15)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        info = download_video(url, Path(tmpdir))
        frames = extract_keyframes(info["path"], max_frames=max_frames, threshold=20.0)

        if len(frames) < 5:
            uniform = extract_uniform_frames(info["path"], num_frames=max_frames)
            seen_ts = {f["timestamp"] for f in frames}
            for uf in uniform:
                if uf["timestamp"] not in seen_ts:
                    frames.append(uf)
            frames.sort(key=lambda f: f["timestamp"])
            frames = frames[:max_frames]

        transcript = get_transcript(url)

        result = []

        strategy_prompt = (
            f"## Trading Strategy Extraction\n\n"
            f"**Video:** {info['title']}\n"
            f"**Duration:** {_fmt_duration(info['duration'])}\n"
            f"**Source:** {info['uploader']}\n"
            f"**Target framework:** {framework}\n\n"
            f"Below are the key frames from this trading strategy video, followed by the transcript. "
            f"Analyze the chart patterns, indicators, entry/exit rules, and timeframes shown. "
            f"Then generate a complete, runnable {framework} strategy.\n\n"
            f"**Look for in the frames:**\n"
            f"- Chart patterns (support/resistance, trendlines, candlestick patterns)\n"
            f"- Technical indicators (MA, RSI, MACD, Bollinger Bands, etc.)\n"
            f"- Entry signals and conditions\n"
            f"- Exit signals / stop-loss / take-profit levels\n"
            f"- Timeframe being used\n"
            f"- Position sizing rules\n"
        )
        result.append(TextContent(type="text", text=strategy_prompt))

        for frame in frames:
            ts = frame.get("timestamp_formatted", "")
            result.append(TextContent(type="text", text=f"--- Chart/Screen at {ts} ---"))
            result.append(ImageContent(
                type="image",
                data=frame["image_base64"],
                mimeType="image/jpeg",
            ))

        if transcript["segments"]:
            seg_text = "\n".join(
                f"[{s['start_formatted']}] {s['text']}" for s in transcript["segments"]
            )
            result.append(TextContent(
                type="text",
                text=f"\n## Transcript (timestamped)\n\n{seg_text[:8000]}",
            ))
        elif transcript["full_text"]:
            result.append(TextContent(
                type="text",
                text=f"\n## Transcript\n\n{transcript['full_text'][:8000]}",
            ))

        code_template = _get_framework_template(framework)
        result.append(TextContent(
            type="text",
            text=(
                f"\n## Code Generation Instructions\n\n"
                f"Generate a complete, runnable {framework} strategy based on the above analysis.\n"
                f"Include:\n"
                f"1. All indicator calculations seen in the video\n"
                f"2. Entry conditions (long/short)\n"
                f"3. Exit conditions (stop-loss, take-profit, trailing stop)\n"
                f"4. Position sizing if mentioned\n"
                f"5. Backtest setup with sample data download\n\n"
                f"**Template structure:**\n```python\n{code_template}\n```"
            ),
        ))

    return result


def _get_framework_template(framework: str) -> str:
    templates = {
        "backtrader": (
            "import backtrader as bt\n\n"
            "class VideoStrategy(bt.Strategy):\n"
            "    params = (('period', 14),)  # extracted from video\n\n"
            "    def __init__(self):\n"
            "        # indicators from video\n"
            "        pass\n\n"
            "    def next(self):\n"
            "        # entry/exit logic from video\n"
            "        pass\n\n"
            "if __name__ == '__main__':\n"
            "    cerebro = bt.Cerebro()\n"
            "    cerebro.addstrategy(VideoStrategy)\n"
            "    # add data feed\n"
            "    cerebro.run()\n"
            "    cerebro.plot()"
        ),
        "zipline": (
            "from zipline.api import order_target_percent, record, symbol\n"
            "from zipline import run_algorithm\n\n"
            "def initialize(context):\n"
            "    context.asset = symbol('SPY')\n\n"
            "def handle_data(context, data):\n"
            "    # strategy logic from video\n"
            "    pass"
        ),
        "pandas-ta": (
            "import pandas as pd\n"
            "import pandas_ta as ta\n"
            "import yfinance as yf\n\n"
            "df = yf.download('SPY', start='2023-01-01')\n\n"
            "# indicators from video\n"
            "# df.ta.rsi(length=14, append=True)\n\n"
            "# generate signals\n"
            "# backtest results"
        ),
        "freqtrade": (
            "from freqtrade.strategy import IStrategy, DecimalParameter\n"
            "import talib.abstract as ta\n\n"
            "class VideoStrategy(IStrategy):\n"
            "    minimal_roi = {'0': 0.1}\n"
            "    stoploss = -0.05\n"
            "    timeframe = '1h'\n\n"
            "    def populate_indicators(self, dataframe, metadata):\n"
            "        return dataframe\n\n"
            "    def populate_entry_trend(self, dataframe, metadata):\n"
            "        return dataframe\n\n"
            "    def populate_exit_trend(self, dataframe, metadata):\n"
            "        return dataframe"
        ),
    }
    return templates.get(framework, templates["backtrader"])


def _fmt_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
