"""Continue the authorized evaluation/report after the active training finishes."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'result'
STATE=OUT/'pipeline_status.json'

def update(status,**extra):
    value=dict(status=status,updated=datetime.now().isoformat(),**extra)
    tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf8')
    tmp.replace(STATE)

ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    update('waiting_for_training')
    while True:
        status=json.loads((OUT/'logmel_sal_20261117/status.json').read_text(encoding='utf8'))
        if status['status']=='failed':
            raise RuntimeError(status.get('error','Training failed'))
        if status['status']=='trained':break
        time.sleep(15)
    for phase,args in [('evaluation',['run_logmel.py','evaluate']),('comparison',['compare_results.py'])]:
        update(phase)
        with (OUT/(phase+'_console.log')).open('w',encoding='utf8') as stream:
            subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    update('complete',report=str(OUT/'comparison.txt'))
except BaseException as exc:
    update('failed',error=repr(exc))
    raise
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
