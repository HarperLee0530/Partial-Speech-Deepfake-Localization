"""Run resumed training and its dependent evaluation in one tracked process."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'result'
def update(status,**extra):
    path=OUT/'pipeline_status.json'
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=status,updated=datetime.now().isoformat(),**extra),indent=2),encoding='utf8')
    tmp.replace(path)

ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    for phase,args in [('resumed_training',['run_logmel.py','resume']),('evaluation',['run_logmel.py','evaluate']),('comparison',['compare_results.py'])]:
        update(phase)
        with (OUT/(phase+'_console.log')).open('a',encoding='utf8') as stream:
            subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    update('complete',report=str(OUT/'comparison.txt'))
except BaseException as exc:
    update('failed',error=repr(exc))
    raise
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
