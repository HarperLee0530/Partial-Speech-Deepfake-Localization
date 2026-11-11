"""Replay a deterministic real-audio sample on CPU, without changing experiments."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from cnn import baseline as p
from cnn.audit_generalization import TRAIN_RUN, TEST_RUN, read
from project_config import CODE_ROOT, RESULT_ROOT


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train = read(RESULT_ROOT/(TRAIN_RUN+'.json'))
    checkpoint = torch.load((CODE_ROOT/train['checkpoint']).resolve(), map_location='cpu')
    config = p.ModelConfig(**checkpoint['model_config'])
    model = p.CNNBiGRU(config).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    dataset = p.BaselineDataset('eval', config)
    index = {uid:i for i,uid in enumerate(dataset.utt_ids)}
    db = sqlite3.connect((RESULT_ROOT/TEST_RUN/'predictions.sqlite').as_uri()+'?mode=ro', uri=True)
    # Hash order avoids choosing examples based on observed correctness.
    candidates = [uid for uid, in db.execute('SELECT uid FROM predictions WHERE seconds <= 8')]
    selected = sorted(candidates, key=lambda uid:hashlib.sha256(uid.encode()).digest())[:32]
    rows=[]
    with torch.inference_mode():
        for uid in selected:
            old_blob, old_labels = db.execute('SELECT logits,labels FROM predictions WHERE uid=?',(uid,)).fetchone()
            old=np.frombuffer(old_blob,dtype=np.float32).reshape(-1,2)
            batch=p.collate([dataset[index[uid]]])
            output,_=p.infer_batch(model,batch,dataset.resolution,'cpu')
            new=p.sal_logits(output[0]).numpy()
            assert np.array_equal(np.frombuffer(old_labels,dtype=np.uint8),(1-batch['labels'][0]).numpy())
            difference=float(np.max(np.abs(old-new)))
            assert np.allclose(old,new,atol=1e-4,rtol=1e-4), (uid,difference)
            rows.append(dict(uid=uid,frames=len(new),max_abs_logit_difference=difference,
                             argmax_disagreements=int((old.argmax(1)!=new.argmax(1)).sum())))
    db.close()
    result=dict(sample_policy='32 SHA256-first eval IDs with audio <=8 seconds; independent of errors',
                cpu_only=True,all_labels_equal=True,all_logits_close=True,
                max_abs_logit_difference=max(r['max_abs_logit_difference'] for r in rows),
                argmax_disagreements=sum(r['argmax_disagreements'] for r in rows),rows=rows)
    output=RESULT_ROOT/('cnn_cpu_replay_'+p.datetime.now(p.ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'.json')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    print(output,flush=True)


if __name__=='__main__':
    main()
