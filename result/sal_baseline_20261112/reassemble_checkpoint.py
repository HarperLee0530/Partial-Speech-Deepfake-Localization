"""Reassemble the exact original checkpoint; standard library, no credentials/GPU."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parts-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().with_name('weights_manifest.json'))
    p.add_argument('--output',type=Path,default=Path('best-epoch=024.ckpt'))
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    target=args.output.resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise FileExistsError(f'Will not overwrite {target}')
    temp=target.with_suffix(target.suffix+'.partial')
    total=hashlib.sha256(); length=0
    with temp.open('xb') as out:
        for row in manifest['parts']:
            if Path(row['name']).name!=row['name']: raise ValueError('Invalid part name')
            source=args.parts_dir/row['name']
            h=hashlib.sha256(); n=0
            with source.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):
                    out.write(block); h.update(block); total.update(block); n+=len(block); length+=len(block)
            if n!=row['bytes'] or h.hexdigest()!=row['sha256']:
                raise ValueError(f'Part checksum mismatch: {source}; partial output is NOT usable')
            print(f'Verified {source.name}',flush=True)
    if length!=manifest['checkpoint_bytes'] or total.hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('Final checkpoint checksum mismatch; partial output is NOT usable')
    temp.replace(target)
    print(f'CHECKPOINT VERIFIED: {target}\nSHA256 {total.hexdigest()}')


if __name__=='__main__': main()
