"""Tests for MCP server tool registration."""

from mcp_video.server import mcp


class TestServerTools:
    def test_server_has_name(self):
        assert mcp.name == "mcp-video"

    def test_all_tools_registered(self):
        tools = mcp._tool_manager._tools
        tool_names = set(tools.keys())
        expected = {
            "analyze_video",
            "extract_video_frames",
            "get_video_transcript",
            "get_frame_at_time",
            "video_to_strategy",
        }
        assert expected.issubset(tool_names)
