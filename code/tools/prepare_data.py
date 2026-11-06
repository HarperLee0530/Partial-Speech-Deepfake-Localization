import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

"""Extract verified evaluation archives without changing source archives."""
from pathlib import Path
import concurrent.futures
import json
import tarfile
import time

from project_config import DATA_ROOT as STORE, RESULT_ROOT, LOG_ROOT
ROOT = Path(__file__).resolve().parents[1]

def extract(job):
    family, filename = job
    src = STORE / 'downloads' / family / filename
    dest = STORE / family
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / (filename + '.extracted.json')
    if marker.exists():
        return json.loads(marker.read_text())
    count = total = 0
    # LPS .tgz is actually an uncompressed tar; detect by content.
    with tarfile.open(src, 'r|*') as archive:
        for member in archive:
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError('Unexpected archive member: ' + member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != member.size:
                with archive.extractfile(member) as inp, target.open('wb') as out:
                    while block := inp.read(8 * 1024 * 1024):
                        out.write(block)
            count += 1
            total += member.size
            if count % 5000 == 0:
                print(filename, count, total, flush=True)
    result = dict(archive=str(src), destination=str(dest), files=count, bytes=total,
                  completed=time.strftime('%Y-%m-%dT%H:%M:%S'))
    marker.write_text(json.dumps(result, indent=2))
    print('COMPLETE', result, flush=True)
    return result

if __name__ == '__main__':
    jobs = [('PartialSpoof', n) for n in ['database_segment_labels_v1.2.tar.gz',
            'database_protocols.tar.gz', 'database_dev.tar.gz', 'database_eval.tar.gz']]
    jobs += [('LlamaPartialSpoof', 'R01TTS.0.a.tgz')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(extract, jobs))
    (ROOT / 'manifests' / 'extracted_evaluation_data.json').write_text(json.dumps(results, indent=2))
