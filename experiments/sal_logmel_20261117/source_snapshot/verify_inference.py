"""Exercise checkpoint/inference/export using dev only, never test data."""
from pathlib import Path
import torch
import run_logmel as run
import sal

torch.set_num_threads(2)
torch.manual_seed(888)
model=run.LogMelSAL().eval()
checkpoint=run.ROOT.parent/'result'/'inference_smoke.pt'
torch.save(dict(state_dict={'net.'+k:v for k,v in model.state_dict().items()},
                project_training=run.identity(),epoch=0,global_step=0),checkpoint)
loaded,_=run.load_model(.16,checkpoint,device='cpu')
x=torch.randn(1,64000)
with torch.inference_mode():
    for a,b in zip(model(x),loaded(x)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    fine=run.LogMelSAL(.02).eval()
    assert fine(x)[1].shape==(1,200,2)
sal.CONFIG['sal']['device']='cpu'
sal.load_model=run.load_model
sal.validate_training_metadata=run.validate
code=sal.run(['--split','dev','--limit','4','--checkpoint',str(checkpoint),
              '--run-name','logmel_inference_dev_smoke'])
assert code==0
print('Checkpoint roundtrip and dev-only full-utterance export passed.')
