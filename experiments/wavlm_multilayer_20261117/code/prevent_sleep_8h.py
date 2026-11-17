"""Keep Windows awake for eight hours, without changing its power plan.

Run: python prevent_sleep_8h.py
Ctrl+C releases the request early. The display may turn off normally.
"""
import argparse
import ctypes
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import time


def keep_awake(seconds=8 * 3600, status_file=None):
    if os.name != 'nt':
        raise OSError('This script requires Windows')
    if seconds <= 0:
        raise ValueError('Duration must be positive')
    api = ctypes.windll.kernel32.SetThreadExecutionState
    api.argtypes = [ctypes.c_uint]
    api.restype = ctypes.c_uint
    continuous, system_required = 0x80000000, 0x00000001
    start = datetime.now(timezone.utc)
    deadline = time.monotonic() + seconds
    record = dict(pid=os.getpid(), started_utc=start.isoformat(),
                  expires_utc=(start + timedelta(seconds=seconds)).isoformat(),
                  duration_seconds=seconds, display_required=False)

    def status(state):
        record.update(state=state, updated_utc=datetime.now(timezone.utc).isoformat())
        if status_file:
            path = Path(status_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)

    if not api(continuous | system_required):
        raise ctypes.WinError()
    try:
        status('active')
        while time.monotonic() < deadline:
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        status('completed')
    except KeyboardInterrupt:
        status('cancelled')
    finally:
        api(continuous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--status-file', type=Path)
    args = parser.parse_args()
    keep_awake(args.hours * 3600, args.status_file)
