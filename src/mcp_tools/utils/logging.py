"""
Logging utilities for Claude MCP Tools
"""

import os
from loguru import logger

def truncate_large_value(value, max_str_len=100):
    """
    Truncates large string values for logging purposes.
    
    Args:
        value: The string value to truncate
        max_str_len: Maximum string length to allow
        
    Returns:
        Truncated string
    """
    if isinstance(value, str):
        if len(value) > max_str_len:
            truncated = value[:max_str_len]
            return f"{truncated}... [truncated, {len(value)} chars total]"
    return value

def setup_logger(name, log_file, level="INFO"):
    """
    Setup a logger with proper configuration.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add file handler with appropriate formatting
    logger.add(
        log_file,
        rotation="10 MB",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    return logger