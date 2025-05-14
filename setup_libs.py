#!/usr/bin/env python3
"""
Setup script for the claude_mcp_configs libraries.

This script installs all the libraries in the 'libs' directory as development packages,
making them available for import in other scripts and tools.
"""

import os
import sys
import subprocess
from pathlib import Path

# Get the base directory
BASE_DIR = Path(__file__).parent.absolute()
LIBS_DIR = BASE_DIR / "libs"

def setup_libs():
    """Install all libraries in development mode."""
    print("Setting up claude_mcp_configs libraries...")
    
    # Check if the libs directory exists
    if not LIBS_DIR.exists():
        print(f"Error: Libs directory not found at {LIBS_DIR}")
        return 1
    
    # Find all libraries
    libraries = [d for d in LIBS_DIR.iterdir() if d.is_dir() and (
        (d / "__init__.py").exists() or (d / "pyproject.toml").exists()
    )]
    
    if not libraries:
        print("No libraries found to install.")
        return 0
    
    print(f"Found {len(libraries)} libraries:")
    for lib in libraries:
        print(f"  - {lib.name}")
    
    # Install each library in development mode
    for lib in libraries:
        print(f"\nInstalling {lib.name}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(lib)],
                check=True
            )
            print(f"Successfully installed {lib.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {lib.name}: {e}")
    
    print("\nAll libraries installed successfully.")
    return 0

def setup_task_manager():
    """Setup the claude_task_manager package specifically."""
    print("Setting up claude_task_manager...")
    
    # Check if the package exists
    task_manager_dir = LIBS_DIR / "claude_task_manager"
    if not task_manager_dir.exists():
        print(f"Error: claude_task_manager not found at {task_manager_dir}")
        return 1
    
    # Install dependencies
    print("Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "typer", "rich"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return 1
    
    # Install the package
    print("Installing claude_task_manager...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(task_manager_dir)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error installing claude_task_manager: {e}")
        return 1
    
    print("claude_task_manager installed successfully.")
    return 0

if __name__ == "__main__":
    # Check if a specific library was requested
    if len(sys.argv) > 1 and sys.argv[1] == "task-manager":
        sys.exit(setup_task_manager())
    else:
        sys.exit(setup_libs())
