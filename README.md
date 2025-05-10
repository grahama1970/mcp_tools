# Claude MCP Tools

A collection of Model Context Protocol (MCP) tools for Claude.

## Tools

### Screenshot Tool with AI Description

The Screenshot tool captures screenshots using MSS, which is more reliable than PyAutoGUI. It now includes the ability to automatically describe screenshots using Vertex AI Gemini.

**Features**:
- Full screen or region capture
- Automatic resizing to fit Claude token limits
- Quality optimization to ensure compatibility
- Image saving with unique IDs
- **NEW**: AI-powered screenshot description using Vertex AI Gemini

## Installation

```bash
# With uv (recommended)
uv venv
uv pip install -e .

# Or with pip
pip install -e .
```

## Usage

### Standalone Usage

For the simplest deployment, use the standalone files in the `standalone/` directory:

```bash
python standalone/mss_screenshot.py
# Or directly if executable permission is set
./standalone/mss_screenshot.py
```

### Package Usage

```python
from claude_mcp_tools.screenshot import capture_screenshot

# Capture a screenshot
result = capture_screenshot(quality=70, region=[640, 0, 640, 480])
```

## Development

This project uses a hybrid approach:
1. Modular code in `src/` for development and maintenance
2. Standalone scripts in the `standalone/` directory for simple deployment

To create standalone scripts from the modular code:

```bash
python scripts/build_standalone.py
```

## Project Structure

```
claude_mcp_configs/
├── src/                  # Core package code
│   └── claude_mcp_tools/ # Main package
│       ├── screenshot/   # Screenshot module
│       └── utils/        # Internal utilities
├── standalone/           # Ready-to-use MCP tools
├── utils/                # Development utilities
├── dev/                  # Testing and validation scripts
├── scripts/              # Build and automation scripts
├── archive/              # Archived previous versions
├── logs/                 # Log output
├── screenshots/          # Screenshot outputs
├── pyproject.toml        # Package configuration
└── setup.py              # Package setup script
```