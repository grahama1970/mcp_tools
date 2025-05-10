#!/usr/bin/env python3
"""
Claude MCP Screenshot Tool

A Model Context Protocol (MCP) tool for Claude Desktop and Claude Code that captures desktop screenshots
(full screen or specified region) and returns the file path as JSON. Designed for use outside the browser,
such as desktop apps, system dialogs, and file explorers.

--------------------------------------------------------
Quick Installation (with uv):

cd ~/claude-mcp-configs
uv venv
uv pip install pyautogui pillow loguru

--------------------------------------------------------
Third-Party Package Documentation:
- pyautogui: https://pyautogui.readthedocs.io/en/latest/
- Pillow (PIL): https://pillow.readthedocs.io/en/stable/
- loguru: https://loguru.readthedocs.io/en/stable/

--------------------------------------------------------
Sample Input (stdin, JSON-RPC MCP call):

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "quality": 80,
    "region": [100, 100, 500, 400]
  }
}

Sample Output (stdout, JSON-RPC MCP response):

{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"file\": \"screenshots/screenshot_123456789.jpg\"}"
      }
    ]
  }
}

On error, the response will include an "error" field instead of "result".

--------------------------------------------------------
Usage Notes:
- Run with: uv run claude_mcp_screenshot.py
- Screenshots are saved in the 'screenshots' directory relative to the script.
- Logs are written to 'mcp_server.log' in the script directory.
- MCP notifications (requests without "id") are ignored as per JSON-RPC 2.0.

Author: Graham Anderson
Date: 2025-05-09
"""
import sys
import os
import time
import json
import pyautogui
from PIL import Image
from loguru import logger

project_root = os.path.dirname(os.path.abspath(__file__))
# Log to file and stderr
logger.add(
    os.path.join(project_root, "mcp_server.log"), rotation="1 MB", retention="7 days"
)


def take_screenshot(command):
    quality = int(command.get("quality", 70))
    region = command.get("region", None)
    try:
        if region and isinstance(region, list) and len(region) == 4:
            x, y, w, h = region
            logger.info(f"Taking region screenshot: {region}")
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
        else:
            logger.info("Taking full screen screenshot")
            screenshot = pyautogui.screenshot()
    except Exception as e:
        logger.exception("Screenshot failed")
        raise RuntimeError(f"Screenshot failed: {e}")

    try:
        screenshot = screenshot.convert("RGB")
        output_dir = os.path.join(project_root, "screenshots")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        screenshot.save(filepath, format="JPEG", quality=quality)
        logger.info(f"Screenshot saved to {filepath}")
        return {"file": filepath}
    except Exception as e:
        logger.exception("Failed to save screenshot")
        raise RuntimeError(f"Failed to save screenshot: {e}")


def mcp_server():
    while True:
        line = sys.stdin.readline()
        if not line:
            logger.info("No more input; exiting MCP server loop.")
            break
        try:
            logger.debug(f"Received line: {line.strip()}")
            req = json.loads(line)
            logger.info(f"Parsed request: {req}")
            method = req.get("method")
            req_id = req.get("id", None)

            # --- Notification handling: do NOT respond to messages with no id ---
            if req_id is None:
                logger.info(
                    f"Received notification method: {method}, not sending response."
                )
                continue

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "screenshot", "version": "1.0.0"},
                        "capabilities": {},
                    },
                }
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
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
                params = req.get("params", {})
                try:
                    logger.info(f"Calling screenshot tool with params: {params}")
                    result = take_screenshot(params)
                    content_item = {"type": "text", "text": json.dumps(result)}
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"content": [content_item]},
                    }
                except Exception as e:
                    logger.error(f"Screenshot tool error: {e}")
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32000, "message": str(e)},
                    }
            else:
                logger.warning(f"Unknown method: {method}")
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            logger.debug(f"Sending response: {resp}")
            print(json.dumps(resp), flush=True)
        except Exception as e:
            # Try to extract id if possible, otherwise do not include it (per JSON-RPC spec)
            try:
                req2 = json.loads(line)
                req_id = req2.get("id")
            except Exception:
                req_id = None
            if req_id is not None:
                logger.exception("Internal error (with id)")
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": f"Internal error: {e}"},
                }
            else:
                logger.exception("Parse error (no id)")
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
            print(json.dumps(error_resp), flush=True)


if __name__ == "__main__":
    from sys import argv

    if len(argv) > 1 and argv[1] == "--mcp-schema":
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
        logger.info("Printing MCP schema and exiting.")
        print(json.dumps(schema, indent=2))
        sys.exit(0)
    logger.info("Starting MCP server loop.")
    mcp_server()
