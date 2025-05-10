#!/usr/bin/env python3
"""
Standalone script to capture a screenshot of the right half of a MacBook Retina screen using MSS
"""

import mss
from PIL import Image
import os
import sys
import time


def setup_logging():
    """Configure basic logging to a file and console."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "mss_right_half.log")

    def log(message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp} | {message}\n")
        print(f"{timestamp} | {message}")

    return log


def capture_right_half_screenshot(monitor_index=1, scale_factor=2.0):
    """
    Capture a screenshot of the right half of the specified monitor.

    Args:
        monitor_index: Index of the monitor to capture (default: 1 for primary).
        scale_factor: Scaling factor for Retina displays (default: 2.0 for macOS).

    Returns:
        PIL.Image or None if capture fails.
    """
    log = setup_logging()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with mss.mss() as sct:
            # Log all monitors
            monitors = sct.monitors
            log(f"Available monitors: {monitors}")

            # Select monitor
            if monitor_index < 0 or monitor_index >= len(monitors):
                log(
                    f"Invalid monitor index {monitor_index}. Available: {list(range(len(monitors)))}"
                )
                return None
            monitor = monitors[monitor_index]
            log(f"Selected monitor {monitor_index}: {monitor}")

            # Calculate right half region in logical pixels
            logical_width = monitor["width"] // scale_factor
            logical_height = monitor["height"] // scale_factor
            logical_region = [
                logical_width // 2,  # x: Start at midpoint
                0,  # y: Top of screen
                logical_width // 2,  # width: Half the screen
                logical_height,  # height: Full height
            ]

            # Apply Retina scaling for physical pixels
            scaled_region = [int(coord * scale_factor) for coord in logical_region]
            log(f"Logical region (right half): {logical_region}")
            log(f"Scaled region (x{scale_factor}): {scaled_region}")

            # Validate region
            x, y, width, height = scaled_region
            if (
                x < 0
                or y < 0
                or x + width > monitor["width"]
                or y + height > monitor["height"]
            ):
                log(f"Region out of bounds: {scaled_region}, monitor: {monitor}")
                return None

            # Define capture region
            capture_region = {"top": y, "left": x, "width": width, "height": height}
            log(f"Capturing region: {capture_region}")

            # Capture screenshot
            sct_img = sct.grab(capture_region)

            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            # Save screenshot
            output_path = os.path.join(
                output_dir, f"right_half_{int(time.time() * 1000)}.png"
            )
            img.save(output_path)
            log(f"Saved screenshot: {output_path}")

            # Save full screen for reference
            full_img = sct.grab(monitor)
            full_img_pil = Image.frombytes(
                "RGB", full_img.size, full_img.bgra, "raw", "BGRX"
            )
            full_path = os.path.join(output_dir, "full_screen.png")
            full_img_pil.save(full_path)
            log(f"Saved full screen: {full_path}")

            return img

    except mss.exception.ScreenShotError as e:
        log(f"MSS ScreenShotError: {str(e)}")
        return None
    except Exception as e:
        log(f"Unexpected error: {str(e)}")
        return None


def main():
    """Run the script to capture the right half of the primary monitor."""
    log = setup_logging()
    log("Starting MSS right half screenshot script")

    # Capture right half of primary monitor (index 1) with Retina scaling
    img = capture_right_half_screenshot(monitor_index=1, scale_factor=2.0)

    if img:
        log(
            "Right half screenshot captured successfully. Check the 'screenshots' folder."
        )
    else:
        log("Screenshot capture failed. Check logs for details.")


if __name__ == "__main__":
    main()
