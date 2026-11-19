"""Run the A-group workflow with Windows automatic sleep disabled until exit.

Launch with pythonw.exe as a detached hidden process. Does not change the
permanent power plan or prevent the user from manually shutting down Windows.
Logs are written locally; no interactive progress monitor is started.
"""
import ctypes
import os
import sys
from datetime import datetime, timezone
from run_wavlm_ablation_a import ROOT, run
from cnn_ps.io import save_json


def main():
    api = ctypes.windll.kernel32.SetThreadExecutionState
    api.argtypes, api.restype = [ctypes.c_uint], ctypes.c_uint
    if not api(0x80000001):
        raise ctypes.WinError()
    state = dict(pid=os.getpid(), state='active', started_utc=datetime.now(timezone.utc).isoformat(),
                 release='Automatically at workflow exit', display_may_sleep=True)
    save_json(ROOT/'detached_awake.json', state)
    try:
        with (ROOT/'detached_training.log').open('a', encoding='utf-8', buffering=1) as log:
            sys.stdout = sys.stderr = log
            print('Detached restart', state, flush=True)
            try:
                run(2)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise
    finally:
        api(0x80000000)
        state.update(state='released', ended_utc=datetime.now(timezone.utc).isoformat())
        save_json(ROOT/'detached_awake.json', state)


if __name__ == '__main__':
    main()
