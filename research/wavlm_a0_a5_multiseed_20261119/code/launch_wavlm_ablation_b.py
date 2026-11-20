"""Detached B workflow entry; invoked only after explicit training approval."""
from pathlib import Path
import sys
import traceback


def main():
    workspace = Path(__file__).resolve().parents[2]
    with (workspace/'handoff_audit/ablation_b_20261119_launcher.log').open(
            'a', encoding='utf-8', buffering=1) as log:
        sys.stdout = sys.stderr = log
        try:
            from run_wavlm_ablation_b import run, DEFAULT_OUTPUT
            print('Starting authorized B1/B2/B3 workflow with generated-cache cleanup.', flush=True)
            run(DEFAULT_OUTPUT, discard_caches=True)
            print('B workflow complete.', flush=True)
        except BaseException:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
