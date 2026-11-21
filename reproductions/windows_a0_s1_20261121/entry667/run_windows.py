"""Run the archived trainer with bounded retries for Windows JSON file locks."""
import runpy
import time
from pathlib import Path

from cnn_ps import io


original_save_json = io.save_json


def save_json_with_retry(path, value):
    for attempt in range(8):
        try:
            return original_save_json(path, value)
        except PermissionError as error:
            if error.winerror not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(0.1 * 2**attempt, 2))


if __name__ == "__main__":
    io.save_json = save_json_with_retry
    runpy.run_path(str(Path(__file__).with_name("trainer.py")), run_name="__main__")
