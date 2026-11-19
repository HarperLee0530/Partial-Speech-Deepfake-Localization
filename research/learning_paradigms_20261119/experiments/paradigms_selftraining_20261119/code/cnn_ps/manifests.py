"""Build portable PS manifests and extract true 20 ms labels from the official archive."""
import argparse
import io
import os
from pathlib import Path
import tarfile

import numpy as np

from .io import read_json, save_json, sha256

LABEL_ARCHIVE_SHA256 = '714975e4d7b943ce665caec11d6f48726d39652b9441dfcbeb327240459602b2'


def relative(path, base):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()
    except ValueError:  # Windows data on another drive
        return str(Path(path).resolve())


def build(reduced_root, output, ids):
    """Use the frozen IDs; never redraw a subset or mix official partitions."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    fixed = read_json(ids)
    root = Path(reduced_root).resolve()
    result = {}
    for split in ('train', 'dev', 'eval'):
        records = []
        for entry in fixed[split]:
            uid, role = entry['uid'], entry['role']
            audio = root / role / 'audio' / (uid + '.wav')
            labels = root / role / 'segment_labels' / (split + '_seglab_0.16.npy')
            if not audio.is_file() or not labels.is_file():
                raise FileNotFoundError(f'Missing audio or labels for {uid} in {root / role}')
            records.append(dict(uid=uid, split=split, role=role,
                                audio=relative(audio, output), labels_file=relative(labels, output)))
        result[split] = dict(split=split, count=len(records), records=records)
    sets = [set(r['uid'] for r in result[s]['records']) for s in result]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Official splits overlap')
    for split, payload in result.items():
        save_json(output / f'{split}.json', payload)
    return {s: p['count'] for s, p in result.items()}


def full(audio_dir, labels_file, split, output):
    """Manifest all supplied official audio; the user must supply the intended full split."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    labels_file = Path(labels_file).resolve()
    if not labels_file.is_file():
        raise FileNotFoundError(labels_file)
    audios = sorted(Path(audio_dir).resolve().rglob('*.wav'))
    if not audios:
        raise ValueError('No .wav files found')
    records = [dict(uid=p.stem, split=split, audio=relative(p, output.parent),
                    labels_file=relative(labels_file, output.parent)) for p in audios]
    if len({r['uid'] for r in records}) != len(records):
        raise ValueError('Duplicate audio IDs')
    save_json(output, dict(split=split, count=len(records), records=records))
    return len(records)


def extract_labels(archive, ids, output):
    """Read archive members in memory; no archive paths are extracted to the filesystem."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if sha256(archive) != LABEL_ARCHIVE_SHA256:
        raise ValueError('Expected the official checked segment_labels_v1.2 archive')
    fixed = read_json(ids)
    output.mkdir(parents=True, exist_ok=True)
    done = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            for split in ('train', 'dev', 'eval'):
                if split in done or not member.name.endswith('/' + split + '_seglab_0.02.npy'):
                    continue
                with tar.extractfile(member) as stream:
                    labels = np.load(io.BytesIO(stream.read()), allow_pickle=True).item()
                selected = {}
                for row in fixed[split]:
                    y = np.asarray(labels[row['uid']])
                    if y.ndim != 1 or not np.isin(y, (0, 1)).all():
                        raise ValueError('Invalid official fine labels')
                    selected[row['uid']] = (1-y).astype('uint8')
                np.savez_compressed(output / f'{split}.npz', **selected)
                done.add(split)
    if done != {'train', 'dev', 'eval'}:
        raise ValueError(f'Archive is missing fine labels: {done}')
    save_json(output / 'provenance.json', dict(archive_sha256=LABEL_ARCHIVE_SHA256,
              encoding='1=fake; 0=real', grid_seconds=.02,
              files={s: sha256(output / f'{s}.npz') for s in sorted(done)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    default_ids = Path(__file__).resolve().parents[1] / 'manifests' / 'ps_fixed_ids.json'
    p = sub.add_parser('reduced')
    p.add_argument('--reduced-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ids', default=default_ids)
    p = sub.add_parser('full')
    p.add_argument('--audio-dir', required=True)
    p.add_argument('--labels-file', required=True)
    p.add_argument('--split', choices=('train', 'dev', 'eval'), required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('labels20')
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ids', default=default_ids)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'reduced':
        print(build(**args))
    elif action == 'full':
        print(full(**args))
    else:
        extract_labels(**args)


if __name__ == '__main__':
    main()
