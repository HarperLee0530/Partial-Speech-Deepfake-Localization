"""Fetch the pinned public WavLM model; no audio is uploaded."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from .io import sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    snapshot_download('microsoft/wavlm-base-plus',
        revision='4c66d4806a428f2e922ccfa1a962776e232d487b',local_dir=str(args.output),
        allow_patterns=['config.json','preprocessor_config.json','pytorch_model.bin','README.md'])
    if sha256(args.output/'pytorch_model.bin')!='3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51':
        raise ValueError('Unexpected public checkpoint hash')
    print('Verified pinned WavLM:',args.output.resolve())

if __name__=='__main__':main()
