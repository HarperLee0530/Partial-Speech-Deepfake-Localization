import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

"""Download public research assets, resume partial files, verify publisher hashes.
Does not run models, train, evaluate, or install dependencies.
"""
import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from project_config import DATA_ROOT as STORE, RESULT_ROOT, LOG_ROOT
LOCK = threading.Lock()

def log(message):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + message
    with LOCK:
        print(line, flush=True)
        with (LOG_ROOT/'download.log').open('a', encoding='utf8') as f:
            f.write(line+'\n')

def metadata(url, name):
    p = ROOT/'references'/name
    if p.exists():
        return json.loads(p.read_text(encoding='utf8'))
    with urllib.request.urlopen(url, timeout=60) as r:
        body = r.read()
    p.write_bytes(body)
    return json.loads(body)

def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def download(item):
    path = Path(item['path']); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+'.part')
    algo, expected = item['checksum'].split(':',1)
    if path.exists():
        if path.stat().st_size == item['size'] and digest(path,algo) == expected:
            log('ALREADY VERIFIED '+path.name)
            return dict(item, status='verified')
        raise RuntimeError('Existing final file invalid; preserved: '+str(path))
    if item['size'] > 100*1024*1024:
        return download_ranges(item, path, partial, algo, expected)
    for attempt in range(1,13):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < item['size']:
                headers={'User-Agent':'SAL-research-resource-downloader/1.0'}
                if offset: headers['Range']=f'bytes={offset}-'
                # Fresh query avoids expired signed CDN redirects cached by intermediaries.
                sep='&' if '?' in item['url'] else '?'
                req=urllib.request.Request(item['url']+sep+'download=1&resume_ts='+str(int(time.time())),headers=headers)
                with urllib.request.urlopen(req, timeout=60) as response:
                    if offset and response.status != 206:
                        raise RuntimeError('Server did not honor resume Range; partial file preserved')
                    if offset and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected Content-Range')
                    log(f'DOWNLOAD {path.name} {offset}/{item["size"]}')
                    last=time.monotonic()
                    with partial.open('ab' if offset else 'wb') as f:
                        while True:
                            block=response.read(1024*1024)
                            if not block: break
                            f.write(block); offset+=len(block)
                            if time.monotonic()-last>30:
                                log(f'PROGRESS {path.name} {offset}/{item["size"]} ({100*offset/item["size"]:.1f}%)')
                                last=time.monotonic()
            if partial.stat().st_size != item['size']:
                raise RuntimeError(f'Size mismatch {partial.stat().st_size}/{item["size"]}')
            actual=digest(partial,algo)
            if actual!=expected:
                raise RuntimeError('Hash mismatch; partial file preserved for inspection')
            partial.rename(path)
            result=dict(item,status='verified',verified_at=time.strftime('%Y-%m-%dT%H:%M:%S'),sha256=digest(path,'sha256'))
            (path.with_name(path.name+'.verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
            log('VERIFIED '+path.name)
            return result
        except Exception as e:
            log(f'RETRY {attempt} {path.name}: {e!r}')
            if attempt<12: time.sleep(min(5*attempt,45))
    return dict(item,status='download_failed_partial_preserved')

def download_ranges(item, path, partial, algo, expected):
    """Parallel HTTP ranges; existing sequential prefix is reused, never discarded."""
    chunks=path.with_name(path.name+'.chunks')
    chunks.mkdir(exist_ok=True)
    state=chunks/'state.json'
    if state.exists():
        start=json.loads(state.read_text())['prefix_bytes']
    else:
        start=partial.stat().st_size if partial.exists() else 0
        state.write_text(json.dumps({'prefix_bytes':start,'size':item['size']}))
    block_size=32*1024*1024
    ranges=[(a,min(a+block_size,item['size'])-1) for a in range(start,item['size'],block_size)]
    def one(bounds):
        a,b=bounds; dest=chunks/f'{a:015d}-{b:015d}.chunk'
        if dest.exists() and dest.stat().st_size==b-a+1:return dest
        tmp=dest.with_suffix('.partial')
        for attempt in range(12):
            try:
                n=tmp.stat().st_size if tmp.exists() else 0
                if n<b-a+1:
                    sep='&' if '?' in item['url'] else '?'
                    req=urllib.request.Request(item['url']+sep+f'range_start={a+n}&ts={int(time.time())}',headers={'Range':f'bytes={a+n}-{b}','User-Agent':'SAL-resource-downloader/1.0'})
                    with urllib.request.urlopen(req,timeout=45) as response:
                        cr=response.headers.get('Content-Range','')
                        if response.status!=206 or not cr.startswith(f'bytes {a+n}-{b}/'):
                            raise RuntimeError(f'Range not honored {response.status} {cr}')
                        with tmp.open('ab' if n else 'wb') as out:
                            while True:
                                data=response.read(1024*1024)
                                if not data:break
                                out.write(data)
                if tmp.stat().st_size!=b-a+1:raise RuntimeError('Incomplete range')
                tmp.rename(dest);return dest
            except Exception as e:
                log(f'RANGE RETRY {path.name} {a} attempt {attempt+1}: {e!r}')
                time.sleep(min(5*(attempt+1),30))
        raise RuntimeError('Exhausted range retries '+str(dest))
    log(f'PARALLEL {path.name}; preserved prefix {start}; {len(ranges)} chunks')
    try:
        last=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16 if item['size']>10_000_000_000 else 12) as executor:
            futures=[executor.submit(one,b) for b in ranges]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                if time.monotonic()-last>=30:
                    done=start+sum(p.stat().st_size for p in chunks.glob('*.chunk'))
                    log(f'PROGRESS {path.name} {done}/{item["size"]} ({100*done/item["size"]:.1f}%)')
                    last=time.monotonic()
        # Assembly can resume too; never append chunks twice after interruption.
        current=partial.stat().st_size if partial.exists() else 0
        with partial.open('ab') as out:
            for a,b in ranges:
                if current>b:continue
                if current<a:raise RuntimeError('Gap in assembly')
                with (chunks/f'{a:015d}-{b:015d}.chunk').open('rb') as src:
                    src.seek(current-a);shutil.copyfileobj(src,out,8*1024*1024)
                current=b+1
        if partial.stat().st_size!=item['size'] or digest(partial,algo)!=expected:
            raise RuntimeError('Final hash/size mismatch; all data preserved')
        partial.rename(path)
        result=dict(item,status='verified',verified_at=time.strftime('%Y-%m-%dT%H:%M:%S'),sha256=digest(path,'sha256'))
        path.with_name(path.name+'.verified.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        # Delete only our redundant verified chunks inside the exact download folder.
        for a,b in ranges:(chunks/f'{a:015d}-{b:015d}.chunk').unlink()
        log('VERIFIED '+path.name)
        return result
    except Exception as e:
        log(f'FAILED {path.name}: {e!r}')
        return dict(item,status='download_failed_partial_preserved',error=repr(e))

def main():
    STORE.mkdir(parents=True, exist_ok=True)
    items=[]
    hf=metadata('https://huggingface.co/api/models/MaoYC/SAL?blobs=true','hf.json')
    for f in hf['siblings']:
        if f['rfilename']=='SAL_WavLM.ckpt':
            items.append(dict(name=f['rfilename'],kind='trained_SAL',path=str(STORE/'models'/f['rfilename']),size=f['size'],checksum='sha256:'+f['lfs']['sha256'],url=f'https://huggingface.co/MaoYC/SAL/resolve/{hf["sha"]}/{f["rfilename"]}'))
    wavlm=metadata('https://huggingface.co/api/models/s3prl/converted_ckpts?blobs=true','wavlm_hf.json')
    for f in wavlm['siblings']:
        if f['rfilename']=='wavlm_large.pt':
            items.append(dict(name=f['rfilename'],kind='base_encoder',path=str(STORE/'models'/f['rfilename']),size=f['size'],checksum='sha256:'+f['lfs']['sha256'],url=f'https://huggingface.co/s3prl/converted_ckpts/resolve/{wavlm["sha"]}/{f["rfilename"]}'))
    for record,folder,include in [('5766198','PartialSpoof',{'database_dev.tar.gz','database_eval.tar.gz','database_protocols.tar.gz','database_segment_labels_v1.2.tar.gz','README_v1.2'}),('14214149','LlamaPartialSpoof',{'R01TTS.0.a.tgz','label_R01TTS.0.a.txt','metadata_crossfade.csv','LICENSE.txt','README.txt'})]:
        name='zenodo.json' if record=='5766198' else 'llama_zenodo.json'
        data=metadata('https://zenodo.org/api/records/'+record,name)
        for f in data['files']:
            if include is None or f['key'] in include:
                items.append(dict(name=f['key'],kind=folder,path=str(STORE/'downloads'/folder/f['key']),size=f['size'],checksum=f['checksum'],url=f['links']['self']))
    total=sum(x['size'] for x in items)
    if shutil.disk_usage(STORE).free < total+20*1024**3:
        raise RuntimeError('Insufficient disk headroom')
    (ROOT/'manifests/download_plan.json').write_text(json.dumps(items,indent=2),encoding='utf8')
    log(f'PLAN {len(items)} files; {total/1e9:.2f} GB; free {shutil.disk_usage(STORE).free/1e9:.2f} GB')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(download,x) for x in items]
        results=[]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            (ROOT/'manifests/download_results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    log('COMPLETE '+str({x['name']:x['status'] for x in results}))

if __name__=='__main__': main()
