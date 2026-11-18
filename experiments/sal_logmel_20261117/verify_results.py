"""Verify packaged frozen predictions without audio, GPU, or checkpoint files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_comparison',ROOT/'source_snapshot/compare_results.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def main():
    manifest=json.loads((ROOT/'files_manifest.json').read_text(encoding='utf8'))
    for name,expected in manifest.items():
        raw=(ROOT/name).read_bytes()
        assert len(raw)==expected['bytes'],name
        assert hashlib.sha256(raw).hexdigest()==expected['sha256'],name
    results=json.loads((ROOT/'comparison.json').read_text(encoding='utf8'))
    checks={}
    with tempfile.TemporaryDirectory(prefix='sal_logmel_verify_') as folder:
        folder=Path(folder)
        with zipfile.ZipFile(ROOT/'prediction_evidence.zip') as archive:
            for info in archive.infolist():
                target=(folder/info.filename).resolve()
                assert target.is_relative_to(folder.resolve()),info.filename
            archive.extractall(folder)
        for split,dt in [('eval',.16),('lps',.02)]:
            rows={}
            for model in ('wavlm','logmel'):
                path=folder/model/split
                rows[model]=module.read(path/'predictions.sqlite')
                r=rows[model]
                metadata=json.loads((path/'metrics.json').read_text(encoding='utf8'))
                calculated=module.metrics(np.stack([module.counts(row,dt) for row in r]).sum(0))
                for key,value in calculated.items():
                    assert abs(float(value)-results[split][model][key])<1e-6,(split,model,key)
                logits=np.concatenate([np.frombuffer(row[1],np.float32).reshape(-1,2) for row in r])
                labels=np.concatenate([np.frombuffer(row[2],np.uint8) for row in r])
                score=logits[:,1]-logits[:,0]
                sorted_y=labels[np.argsort(-score)].astype(np.float64)
                tpr=np.cumsum(sorted_y)/sorted_y.sum()
                fpr=np.cumsum(1-sorted_y)/(len(sorted_y)-sorted_y.sum())
                eer=100*fpr[np.argmin(np.abs(fpr-(1-tpr)))]
                # NumPy/PyTorch may order tied scores differently; allow 2 frames.
                tolerance=max(.001,200/min(sorted_y.sum(),len(sorted_y)-sorted_y.sum()))
                assert abs(eer-metadata['eer_percent'])<=tolerance,(split,model,eer)
                checks[f'{model}_{split}']=dict(recordings=len(r),numpy_eer_percent=float(eer),
                    reported_eer_percent=metadata['eer_percent'],frame_and_event_metrics_verified=True)
            assert len(rows['wavlm'])==len(rows['logmel'])==results[split]['recordings']
            for a,b in zip(rows['wavlm'],rows['logmel']):
                assert a[0]==b[0] and a[2]==b[2] and abs(a[3]-b[3])<1e-8
    print(json.dumps(dict(status='passed',files_verified=len(manifest),checks=checks),indent=2))

if __name__=='__main__':main()
