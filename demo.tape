Output demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 900
Set Height 500
Set Theme "Catppuccin Mocha"
Set Padding 15
Set TypingSpeed 40ms

Type "# mcp-video: Video Intelligence MCP Server"
Enter
Sleep 500ms

Type "# Install"
Enter
Type "pip install mcp-video"
Enter
Sleep 1.5s

Type ""
Enter
Type "# Add to Claude Code config"
Enter
Type `cat ~/.claude/claude_code_config.json`
Enter
Sleep 500ms
Type `echo '{'`
Enter
Type `echo '  "mcpServers": {'`
Enter
Type `echo '    "video": { "command": "mcp-video" }'`
Enter
Type `echo '  }'`
Enter
Type `echo '}'`
Enter
Sleep 1.5s

Type ""
Enter
Type "# Now in Claude Code, just ask:"
Enter
Sleep 500ms

Type `echo ""`
Enter
Type `echo "User: Analyze this video https://youtube.com/watch?v=dQw4w9WgXcQ"`
Enter
Sleep 1s

Type `echo ""`
Enter
Type `echo "→ Downloading video..."`
Enter
Sleep 800ms
Type `echo "→ Extracting 8 keyframes (scene-change detection)..."`
Enter
Sleep 800ms
Type `echo "→ Extracting transcript with timestamps..."`
Enter
Sleep 800ms
Type `echo "→ Returning frames + transcript to Claude for analysis"`
Enter
Sleep 1.5s

Type `echo ""`
Enter
Type `echo "User: Convert this trading strategy to backtrader code"`
Enter
Type `echo "       https://youtube.com/watch?v=..."`
Enter
Sleep 1s

Type `echo ""`
Enter
Type `echo "→ Extracting 15 frames (charts, indicators, entry/exit rules)"`
Enter
Sleep 800ms
Type `echo "→ Parsing transcript for strategy rules..."`
Enter
Sleep 800ms
Type `echo "→ Generating runnable backtrader strategy..."`
Enter
Sleep 800ms
Type `echo ""`
Enter
Type `echo "✓ Strategy code generated with RSI + Moving Average crossover"`
Enter
Sleep 2s
