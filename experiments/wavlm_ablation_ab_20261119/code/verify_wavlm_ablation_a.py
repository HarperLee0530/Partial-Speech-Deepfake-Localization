"""Train/dev-only checks before starting the preregistered A-group experiment."""
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from cnn_ps import train_wavlm_multilayer as legacy
from cnn_ps import train_wavlm_ablation_a as training
from cnn_ps.wavlm_ablation_a import ARMS, AblationCNN, AblationData, make_adapter
from cnn_ps.train_dual import infer
from cnn_ps.io import read_json, save_json, sha256


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'handoff_audit/ablation_a_20261118'
REFERENCE = ROOT/'handoff_audit/multilayer_20261117/full'


def main():
    tiny = ROOT/'handoff_audit/smoke_20261117/run1/mel'
    tiny_layers = ROOT/'handoff_audit/multilayer_20261117/smoke/layers'
    args = Namespace(output=OUTPUT/'smoke/training', seed=666, epochs=1, batch_size=8,
                     device='cuda', threads=4, train_cache=tiny/'train', dev_cache=tiny/'dev')
    for arm in ARMS:
        args.arm = arm
        train = AblationData(tiny/'train', tiny_layers/'train', arm=arm)
        dev = AblationData(tiny/'dev', tiny_layers/'dev', arm=arm)
        training.train_arm(args, train, dev)
    training.freeze_eval_plan(args.output, list(ARMS))
    results = {}
    for arm, original in [('A0_full', 'fusion6_9_12'), ('A1_layer12', 'layer12')]:
        legacy.setup(666, 4)
        folder = REFERENCE/'training'/original
        complete, protocol = read_json(folder/'complete.json'), read_json(folder/'protocol.json')
        for name, field in [('best.pt', 'checkpoint_sha256'), ('protocol.json', 'protocol_sha256'),
                            ('dev_scores20.npz', 'dev_scores20_sha256'), ('dev_metrics.json', 'dev_metrics_sha256')]:
            if sha256(folder/name) != complete[field]:
                raise ValueError(f'Reference changed: {original}/{name}')
        for name, digest in protocol['source_sha256'].items():
            if sha256(Path(__file__).parent/'cnn_ps'/name) != digest:
                raise ValueError(f'Reference source changed: {name}')
        payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        model = AblationCNN(legacy.model_config(), arm, 666).to('cuda')
        model.load_state_dict(payload['backbone_state_dict'], strict=True)
        adapter = make_adapter(ARMS[arm], 666, 'cuda')
        adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
        dataset = AblationData(REFERENCE/'reference/dev', REFERENCE/'layers/dev', arm=arm,
                              split='dev', seed=666, verify_hashes=True)
        _, scores, metadata = infer(model, adapter, dataset, 'cuda')
        with np.load(folder/'dev_scores20.npz') as previous:
            assert previous['uids'].tolist() == metadata['uids']
            np.testing.assert_array_equal(scores, previous['scores20'])
        results[arm] = dict(status='passed', source=str(folder), source_complete=complete,
            dev_recordings=len(dataset), max_absolute_score_difference=0.,
            checkpoint_sha256=sha256(folder/'best.pt'), protocol_sha256=sha256(folder/'protocol.json'))
        print(dict(arm=arm, full_dev_scores='bitwise identical'), flush=True)
    save_json(OUTPUT/'reference_reuse_verification.json', dict(status='passed', models=results,
        smoke=dict(arms=len(ARMS), train=4, dev=4, updates_per_arm=1, eval_read=False),
        synthetic_equivalence='CPU/CUDA output, loss, gradient and one AdamW update tested bitwise equal',
        reuse_scope='Existing seed666 full and layer12 arms only; no new test access in verification'))


if __name__ == '__main__':
    main()
