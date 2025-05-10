#!/usr/bin/env python3
import sys
import json
import base64
import os

SCREENSHOT_PATH = "/absolute/path/to/screenshots/screenshot_123.jpg"
RESOURCE_URI = f"file://{SCREENSHOT_PATH}"


def read_resource(uri):
    # Only allow reading the screenshot resource
    if uri == RESOURCE_URI and os.path.exists(SCREENSHOT_PATH):
        with open(SCREENSHOT_PATH, "rb") as f:
            data = f.read()
        # Base64 encode for transport
        return base64.b64encode(data).decode("utf-8")
    else:
        return None


while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line)
        method = req.get("method")
        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "screenshot", "version": "0.1.0"},
                    "capabilities": {},
                },
            }
            print(json.dumps(resp), flush=True)
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "tools": [
                        {
                            "name": "screenshot",
                            "description": "Take a screenshot",
                            "parameters": {"type": "object", "properties": {}},
                            "returns": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "file": {"type": "string"},
                                },
                                "required": ["type", "file"],
                            },
                        }
                    ]
                },
            }
            print(json.dumps(resp), flush=True)
        elif method == "tools/call":
            # Respond with a resource reference
            resp = {
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "content": [
                        {
                            "type": "resource",
                            "uri": RESOURCE_URI,
                        }
                    ]
                },
            }
            print(json.dumps(resp), flush=True)
        elif method == "resources/list":
            # List available resources (just our screenshot)
            resources = []
            if os.path.exists(SCREENSHOT_PATH):
                resources.append(
                    {"uri": RESOURCE_URI, "description": "Screenshot file"}
                )
            resp = {
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {"resources": resources},
            }
            print(json.dumps(resp), flush=True)
        elif method == "resources/read":
            # Read resource content (base64-encoded)
            uri = req.get("params", {}).get("uri")
            content = read_resource(uri)
            if content is not None:
                resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"data": content}}
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "error": {"code": -32000, "message": "Resource not found"},
                }
            print(json.dumps(resp), flush=True)
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req.get("id"),
                "error": {"code": -32601, "message": "Method not found"},
            }
            print(json.dumps(resp), flush=True)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
