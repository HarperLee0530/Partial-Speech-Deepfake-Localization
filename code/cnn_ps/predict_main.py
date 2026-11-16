"""Default offline localization: frozen WavLM + projected CNN + boundary adapter.

Download the pinned public encoder with cnn_ps.download_wavlm first.
Use cnn_ps.predict_mel for the retained historical Mel control.
"""
import argparse
from pathlib import Path
from .predict_wavlm import predict as predict_backend

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / 'weights' / 'cnn_wavlm_frozen_seed666.pt'


def predict(audio, output, model_folder, checkpoint=DEFAULT_CHECKPOINT, device='cpu', threads=2):
    return predict_backend(audio, output, checkpoint, model_folder=model_folder,
                           device=device, threads=threads, expected_arm='wavlm_frozen')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-folder', type=Path, required=True,
                        help='Pinned microsoft/wavlm-base-plus directory from download_wavlm')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    result = predict(**vars(args))
    print(dict(arm=result['arm'], events=len(result['predicted_fake_intervals_seconds']),
               threshold=result['threshold'], output=str(args.output)))


if __name__ == '__main__':
    main()
