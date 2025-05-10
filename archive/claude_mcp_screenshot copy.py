#!/usr/bin/env python3
r"""
Claude Desktop & Claude Code MCP Screenshot Tool

This script implements a single-file Model Context Protocol (MCP) tool for Claude Desktop and Claude Code.
It takes a screenshot of the desktop (optionally a region), compresses it as JPEG, and returns the file path as JSON.
It is designed to be launched by Claude as a subprocess and communicate via stdin/stdout.

---

## Purpose

This tool allows Claude to capture screenshots OUTSIDE the browser, including:
1. Desktop applications
2. System dialogs
3. File explorers
4. External windows
5. Any content that isn't in a browser tab

The screenshots are saved in a 'screenshots' directory created in the current working directory.

### Important: When to Use This Tool vs. Puppeteer MCP

- **Use THIS TOOL** for capturing anything OUTSIDE the browser (desktop apps, system UI, etc.)
- **Use Puppeteer MCP** for capturing browser content (websites, web apps, browser UI)
- **Fallback case:** If Puppeteer MCP fails due to bot detection or website restrictions, this tool can be used as a fallback

This native screenshot tool captures exactly what appears on your screen, bypassing any browser-based restrictions.

---

## Installation Instructions

1. **Recommended File Location**

   Place this script in your Claude MCP tools directory:
   - **macOS:**   `~/claude-mcp-configs/claude_mcp_screenshot.py`
   - **Windows:** `%USERPROFILE%\\claude-mcp-configs\\claude_mcp_screenshot.py`
   - **Linux:**   `~/.claude/claude_mcp_screenshot.py` or `~/claude-mcp-configs/claude_mcp_screenshot.py`

2. **Install Dependencies**

   This script requires two Python packages. You have two options:

   **Option A: Use the shared `~/.venv` environment (simpler)**
   ```bash
   # Install in the default ~/.venv environment
   uv pip install pyautogui pillow
   ```

   **Option B: Create a dedicated environment (more isolated)**
   ```bash
   # Create and use a dedicated environment for MCP tools
   cd ~/claude-mcp-configs
   uv venv
   source .venv/bin/activate  # On macOS/Linux
   # OR .venv\Scripts\activate  # On Windows
   uv pip install pyautogui pillow
   ```

   The shared environment is simpler to manage, while a dedicated environment provides better isolation.
   

3. **Claude Desktop Integration**

   - Open Claude Desktop
   - Go to **Settings > Developer > Edit Config** (this opens `claude-desktop-config.json`)
   - Add this MCP server entry:

     **If using the shared environment (Option A):**
     ```json
     "mcpServers": {
       "screenshot": {
         "command": "uv",
         "args": [
           "run",
           "/ABSOLUTE/PATH/TO/claude_mcp_screenshot.py"
         ],
         "alwaysAllow": [
           "tools/screenshot"
         ]
       }
     }
     ```

     **If using a dedicated environment (Option B):**
     ```json
     "mcpServers": {
       "screenshot": {
         "command": "/ABSOLUTE/PATH/TO/claude-mcp-configs/.venv/bin/python",
         "args": [
           "/ABSOLUTE/PATH/TO/claude_mcp_screenshot.py"
         ],
         "alwaysAllow": [
           "tools/screenshot"
         ]
       }
     }
     ```
     Replace `/ABSOLUTE/PATH/TO/` with your actual file path. On Windows, use double backslashes.

   - Save and restart Claude Desktop
   - The tool will now appear in your MCP tools menu

4. **Claude Code Integration**

   Register the tool with Claude Code CLI:

   **If using the shared environment (Option A):**
   ```bash
   claude mcp add screenshot uv run /ABSOLUTE/PATH/TO/claude_mcp_screenshot.py
   ```

   **If using a dedicated environment (Option B):**
   ```bash
   # Point to the specific virtual environment
   claude mcp add screenshot "/ABSOLUTE/PATH/TO/claude-mcp-configs/.venv/bin/python /ABSOLUTE/PATH/TO/claude_mcp_screenshot.py"
   ```

---

## Usage for Developers

### Command Line Testing

1. Get the MCP schema:
   ```bash
   uv run claude_mcp_screenshot.py --mcp-schema
   ```

2. Test with default settings (full-screen, 70% quality):
   ```bash
   echo '{}' | uv run claude_mcp_screenshot.py
   ```

3. Test with custom settings:
   ```bash
   echo '{"quality": 80, "region": [100, 100, 500, 400]}' | uv run claude_mcp_screenshot.py
   ```

### Parameters

- **quality** (optional): Integer between 1-100 specifying JPEG quality. Higher values mean better
  quality but larger file sizes. Default: 70

- **region** (optional): Array of 4 integers [x, y, width, height] specifying screen coordinates to capture.
  If omitted, captures the entire screen.

### Output

The tool returns a JSON object with a single "file" key containing the path to the saved screenshot:
```json
{"file": "screenshots/screenshot_1746792863106.jpg"}
```

If an error occurs, the response will include an "error" key with the error message:
```json
{"error": "Screenshot failed: [error message]"}
```

---

## Usage for Claude Agents

### When to Use This Tool (IMPORTANT)

As a Claude agent, you should ONLY use this tool when:

1. The user needs to capture something OUTSIDE a browser window (desktop apps, system UI, etc.)
2. The Puppeteer MCP tool has failed due to bot detection or website restrictions
3. The user explicitly asks for a desktop screenshot

DO NOT use this tool for regular web content - use the Puppeteer MCP tool instead.

### How to Use This Tool

When appropriate, follow these steps:

1. Explain clearly to the user that you'll be taking a screenshot of their desktop/screen
2. Specify exactly what will be captured (full screen or region)
3. Ensure the user understands this captures EVERYTHING visible on their screen
4. Call the "screenshot" tool with appropriate parameters:
   ```
   # Full screen example:
   {"quality": 80}

   # Region example (coordinates in pixels):
   {"quality": 80, "region": [100, 100, 500, 400]}
   ```
5. The tool will return a file path that you can use to view the screenshot

### IMPORTANT: Tool Usage Requirements for Claude Agents

As a Claude agent, you MUST follow these requirements when using this tool:

1. Do NOT attempt to use pip commands to install dependencies - they are already installed
2. Do NOT try to run the script directly with Python - ALWAYS use the MCP tool interface
3. Do NOT forget that this tool is already configured and ready to use
4. Do NOT try to modify environment settings before using this tool
5. When viewing the screenshot captured by this tool, use the path returned in the response

### Example Prompts

For full screen capture:
"I'll take a screenshot of your entire desktop screen to help diagnose the issue with your application. This will capture everything currently visible on your screen and save it locally. Is that okay?"

For region capture:
"To capture just the dialog box, I'll need to take a screenshot of a specific region of your screen. Please position the dialog box and tell me the x, y coordinates and the width and height of the region in pixels, or describe where on screen it appears."

For fallback from Puppeteer:
"The website seems to be blocking automated screenshots. If you'd like, I can capture the webpage using your desktop screenshot tool instead, but this will capture everything visible on your screen. Would you prefer this approach?"

---

Author: Graham Anderson
Date: 2025-05-09

"""


import sys
import os
import time
import json
import pyautogui
from PIL import Image


def main():
    # Read a single JSON command from stdin (one line)
    try:
        line = sys.stdin.readline()
        if not line:
            print(json.dumps({"error": "No input received"}))
            sys.exit(1)
        command = json.loads(line)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse input: {e}"}))
        sys.exit(1)

    # Parse options
    quality = int(command.get("quality", 70))
    region = command.get("region", None)  # [x, y, width, height]

    # Take screenshot
    try:
        if region and isinstance(region, list) and len(region) == 4:
            x, y, w, h = region
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
        else:
            screenshot = pyautogui.screenshot()
    except Exception as e:
        print(json.dumps({"error": f"Screenshot failed: {e}"}))
        sys.exit(1)

    # Compress and save
    try:
        screenshot = screenshot.convert("RGB")
        output_dir = "screenshots"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        screenshot.save(filepath, format="JPEG", quality=quality)
    except Exception as e:
        print(json.dumps({"error": f"Failed to save screenshot: {e}"}))
        sys.exit(1)

    # Output result as JSON to stdout
    print(json.dumps({"file": filepath}))
    sys.stdout.flush()



if __name__ == "__main__":
    # Print MCP schema information to stdout when run with --mcp-schema flag
    if len(sys.argv) > 1 and sys.argv[1] == "--mcp-schema":
        schema = {
            "schema_version": "v1",
            "name": "screenshot",
            "description": "Take a screenshot of the desktop or specified region",
            "tools": [
                {
                    "name": "screenshot",
                    "description": "Capture a screenshot of the screen or a specified region",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "quality": {
                                "type": "integer",
                                "description": "JPEG quality (1-100). Higher is better quality but larger file size",
                                "default": 70
                            },
                            "region": {
                                "type": "array",
                                "description": "Region to capture [x, y, width, height]. If not provided, captures full screen",
                                "items": {
                                    "type": "integer"
                                },
                                "minItems": 4,
                                "maxItems": 4
                            }
                        }
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "Path to the saved screenshot file"
                            },
                            "error": {
                                "type": "string",
                                "description": "Error message if the screenshot failed"
                            }
                        }
                    }
                }
            ]
        }
        print(json.dumps(schema, indent=2))
        sys.exit(0)

    # Normal execution
    main()
