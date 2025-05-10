#!/usr/bin/env python3
import sys
import os
import time
import json
import pyautogui
from PIL import Image


# --- Screenshot logic ---
def take_screenshot(command):
    """Take a screenshot, optionally with quality and region, and save it as JPEG."""
    quality = int(command.get("quality", 70))
    region = command.get("region", None)
    try:
        # Capture full screen or region
        if region and isinstance(region, list) and len(region) == 4:
            x, y, w, h = region
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
        else:
            screenshot = pyautogui.screenshot()
    except Exception as e:
        return {"error": f"Screenshot failed: {e}"}

    try:
        screenshot = screenshot.convert("RGB")
        output_dir = "screenshots"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        screenshot.save(filepath, format="JPEG", quality=quality)
        return {"file": filepath}
    except Exception as e:
        return {"error": f"Failed to save screenshot: {e}"}


# --- MCP protocol server loop ---
def mcp_server():
    """
    Main MCP protocol loop.
    Handles 'initialize', 'tools/list', and 'tools/call' methods.
    """
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
            method = req.get("method")
            if method == "initialize":
                # Respond to MCP initialize handshake
                resp = {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "screenshot", "version": "1.0.0"},
                        "capabilities": {},
                    },
                }
            elif method == "tools/list":
                # List available tools (here: just 'screenshot')
                resp = {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "screenshot",
                                "description": "Take a screenshot of the desktop or a region",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "quality": {
                                            "type": "integer",
                                            "description": "JPEG quality (1-100)",
                                            "default": 70,
                                        },
                                        "region": {
                                            "type": "array",
                                            "description": "Region to capture [x, y, width, height]",
                                            "items": {"type": "integer"},
                                            "minItems": 4,
                                            "maxItems": 4,
                                        },
                                    },
                                },
                                "returns": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "description": "Path to the saved screenshot file",
                                        },
                                        "error": {
                                            "type": "string",
                                            "description": "Error message if the screenshot failed",
                                        },
                                    },
                                },
                            }
                        ]
                    },
                }
            elif method == "tools/call":
                # Execute the screenshot tool with given parameters
                params = req.get("params", {})
                result = take_screenshot(params)
                # MCP expects a 'content' array with type 'text' or 'resource'
                content_item = {"type": "text", "text": json.dumps(result)}
                resp = {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {"content": [content_item]},
                }
            else:
                # Unknown or unsupported method
                resp = {
                    "jsonrpc": "2.0",
                    "id": req.get("id"),
                    "error": {"code": -32601, "message": "Method not found"},
                }
            print(json.dumps(resp), flush=True)
        except Exception as e:
            # Log errors to stderr for debugging
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    # If called with --mcp-schema, print schema and exit (for dev tools)
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
                                "description": "JPEG quality (1-100)",
                                "default": 70,
                            },
                            "region": {
                                "type": "array",
                                "description": "Region to capture [x, y, width, height]",
                                "items": {"type": "integer"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                        },
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "Path to the saved screenshot file",
                            },
                            "error": {
                                "type": "string",
                                "description": "Error message if the screenshot failed",
                            },
                        },
                    },
                }
            ],
        }
        print(json.dumps(schema, indent=2))
        sys.exit(0)
    # Otherwise, run as MCP server
    mcp_server()
