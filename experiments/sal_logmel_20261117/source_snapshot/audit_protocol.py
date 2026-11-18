"""Read-only check of historical code, settings and fixed role manifests."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
WORK=Path('C:/Users/35432/Desktop/学习打卡/NNDL')
REPO=WORK/'tmp/reduced_repo_20261111'
HISTORY=WORK/'tmp/publish_sal_tests_20261113/result/sal_baseline_20261112'
DATA=Path('D:/SAL_精简实验/data/reduced')

def funcs(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}

current=funcs(ROOT/'sal.py')
historical=funcs(HISTORY/'source_snapshot/code/sal.py')
shared=['cross_recording_mixup','make_data_module','training_config','get_dataset','calculate_metrics']
equivalent={name:current[name]==historical[name] for name in shared}
assert all(equivalent.values()),equivalent
old=json.loads((HISTORY/'training/resolved_config.json').read_text(encoding='utf8'))
new=json.loads((ROOT.parent/'result/logmel_sal_20261117/resolved_config.json').read_text(encoding='utf8'))
assert old['settings']==new['settings']
for section in ('model','data'):
    a=old[section].copy();b=new[section].copy()
    for key in (('net',) if section=='model' else ('root',)):
        a.pop(key);b.pop(key)
    assert a==b,(section,a,b)
checked={}
for file in (ROOT/'vendor/SAL').rglob('*.py'):
    rel=file.relative_to(ROOT).as_posix()
    if rel.endswith('ssl_encoder.py'):
        continue
    raw=subprocess.check_output(['git','show',f'77b2dc3:code/{rel}'],cwd=REPO)
    # CRLF is not a method difference.
    assert raw.decode('utf8').replace('\r\n','\n')==file.read_text(encoding='utf8'),rel
    checked[rel]=hashlib.sha256(raw).hexdigest()
roles={}
for role in ('train_detector','dev','eval','lps'):
    runtime=json.loads((DATA/'manifests'/f'{role}.json').read_text(encoding='utf8'))
    portable=json.loads((ROOT/'manifests/reduced'/f'{role}.json').read_text(encoding='utf8'))
    assert runtime==portable
    roles[role]={'count':runtime['count'],'manifest_sha256':runtime['manifest_sha256']}
ids={r:set(json.loads((DATA/'manifests'/f'{r}.json').read_text(encoding='utf8'))['utterance_ids']) for r in ('train_detector','dev','eval')}
assert not(ids['train_detector']&ids['dev'] or ids['train_detector']&ids['eval'] or ids['dev']&ids['eval'])
result=dict(status='passed',historical_commit='77b2dc3',shared_functions_ast_equal=equivalent,
            settings_equal=True,trainer_and_data_configs_equal_except_frontend_and_paths=True,
            unchanged_vendor_files=checked,roles=roles,uid_disjoint=True,
            note='UID independence does not independently prove source-recording or speaker independence.')
dest=ROOT.parent/'result/protocol_audit.json'
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'status':'passed','unchanged_vendor_files':len(checked),'roles':roles}))
