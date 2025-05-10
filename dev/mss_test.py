import sys, os, json, time
import mss


def main():
    line = sys.stdin.readline()
    if not line:
        print(json.dumps({"error": "No input received"}))
        sys.exit(1)
    try:
        command = json.loads(line)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse input: {e}"}))
        sys.exit(1)

    output_dir = "screenshots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        with mss.mss() as sct:
            sct.shot(output=filepath)
        print(json.dumps({"file": filepath}))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"error": f"Screenshot failed: {e}"}))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
