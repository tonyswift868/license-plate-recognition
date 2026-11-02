"""Fixed detector-size study; inference never receives truth or scale labels."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .runtime_acceptance import _rows, compare


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(path):
    return str(path).replace("\\", "/").casefold()


def stratified_sample(rows, n=600, seed=20261028):
    if not 0 < n <= len(rows) or len({key(r['path']) for r in rows}) != len(rows):
        raise ValueError("sample requires unique paths and 0<n<=population")
    groups = defaultdict(list)
    for row in rows:
        groups[row['subset']].append(row)
    quotas = {s: n * len(group) // len(rows) for s, group in groups.items()}
    order = sorted(groups, key=lambda s: (-(n * len(groups[s]) % len(rows)), s))
    for s in order[:n-sum(quotas.values())]:
        quotas[s] += 1
    rng = random.Random(seed)
    chosen = {key(r['path']) for s in sorted(groups) for r in rng.sample(groups[s], quotas[s])}
    return [r for r in rows if key(r['path']) in chosen]


def metrics(rows):
    n = len(rows)
    correct = sum(r['plate'] == r['truth'] for r in rows)
    auto = [r for r in rows if r['status'] == 'OK']
    ac = sum(r['plate'] == r['truth'] for r in auto)
    return {'n': n, 'correct': correct, 'accuracy': correct/n if n else None,
            'auto_accepted': len(auto), 'auto_correct': ac,
            'auto_precision': ac/len(auto) if auto else None,
            'no_plate': sum(r['status'] == 'NO_PLATE' for r in rows)}


def summary(rows):
    subsets = {s: metrics([r for r in rows if r['subset'] == s]) for s in sorted({r['subset'] for r in rows})}
    overall = {p: metrics([r for r in rows if r['partition'] == p]) for p in ('val','devhard')}
    dark = metrics([r for r in rows if r['subset'] == 'ccpd_db' and r['brightness'] <= 19])
    def acc(s): return subsets[s]['accuracy']
    # Frozen composition from Batch 13: base120+green30, dark, tilt35+rotate15, weather.
    composition = {'clear': .8*acc('ccpd_base')+.2*acc('ccpd_green'),
                   'night': dark['accuracy'],
                   'tilt': .7*acc('ccpd_tilt')+.3*acc('ccpd_rotate'),
                   'dirty': acc('ccpd_weather')}
    return {'all':metrics(rows), 'partitions':overall, 'subsets':subsets,
            'dark':dark, 'composition_estimate':composition}


def paired_changes(rows, baseline):
    if [key(r['path']) for r in rows] != [key(r['path']) for r in baseline]:
        raise ValueError('paired result order differs')
    improved, regressed = [], []
    for row, old in zip(rows, baseline, strict=False):
        before, after = old['plate'] == row['truth'], row['plate'] == row['truth']
        detail = {'path':row['path'], 'truth':row['truth'], 'subset':row['subset'],
                  'before':{k:old.get(k) for k in ('status','plate','confidence','route')},
                  'after':{k:row.get(k) for k in ('status','plate','confidence','route')}}
        if after and not before:
            improved.append(detail)
        if before and not after:
            regressed.append(detail)
    return {'improved':len(improved), 'regressed':len(regressed),
            'improved_rows':improved, 'regressed_rows':regressed,
            'representatives':{'improved':improved[:5],'regressed':regressed[:5]}}


def prepare(args):
    from lpr_tools.data.manifest import image_path, read_manifest

    from .index_samples import path_key, read_samples
    ordered = [(partition, row, image_path(row, manifest)) for partition,manifest in
               (('val',args.val),('devhard',args.hard)) for row in read_manifest(manifest)]
    selected = {path_key(p) for _,_,p in ordered}
    indexed = {path_key(s.row.path):s for s in read_samples(args.index, only_paths=selected)}
    with Path(args.excluded_paths).open(encoding='utf-8-sig',newline='') as stream:
        excluded = [r['path'] for r in csv.DictReader(stream)]
    blocked = {key(p) for p in excluded}
    blocked_hashes = {sha(p) for p in excluded}
    rows=[]
    # Read only selected metadata, streaming the shared index once.
    selected_keys = {key(p) for _,_,p in ordered}
    with Path(args.index).open(encoding='utf-8-sig',newline='') as stream:
        brightness={key(r['path']):int(r['brightness']) for r in csv.DictReader(stream)
                    if key(r['path']) in selected_keys}
    for partition,row,path in ordered:
        sample=indexed[path_key(path)]
        if key(path) in blocked or sha(path) in blocked_hashes:
            raise ValueError('development input overlaps frozen test paths/content')
        if row.plate_no != sample.row.plate_no or not sample.bbox or not row.quad:
            raise ValueError('development manifest/index labels or geometry differ')
        rows.append({'path':str(path),'partition':partition,'subset':sample.subset,'truth':row.plate_no,
                     'bbox':sample.bbox,'quad':row.quad,'brightness':brightness[key(path)]})
    if len(rows)!=3100 or Counter(r['partition'] for r in rows)!={'val':1000,'devhard':2100}:
        raise ValueError('requires fixed val1000 + devhard2100')
    camera=stratified_sample(rows,args.n,args.seed)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'native-input.json',rows)
    dump(out/'camera-input.json',camera)
    with (out/'camera-manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','partition','subset'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in camera)
    video=[]
    for s in ('ccpd_base','ccpd_green'):
        video.extend([r for r in rows if r['subset']==s][:30])
    dump(out/'video-input.json',video)
    dump(out/'protocol.json',{'seed':args.seed,'sample_n':args.n,'strata':dict(Counter(r['subset'] for r in camera)),
         'sizes':[256,384,512],'camera_scales':[.6,.8,1.0],'camera_canvas':[1280,720],
         'frozen_path_overlap':0,'frozen_content_overlap':0,'composition':{'clear':'base*.8+green*.2','night':'ccpd_db brightness<=19','tilt':'tilt*.7+rotate*.3','dirty':'weather'},
         'recommendation_gate':'each native composition group >= baseline-0.01 AND camera1.0 whole-plate accuracy>=.95 AND G1 P95<=300ms',
         'threshold_policy':'current defaults fixed; no refit','sources':{str(p):sha(p) for p in (args.val,args.hard,args.index,args.excluded_paths)},
         'camera_input_sha256':sha(out/'camera-input.json')})


def evaluate(args):
    import cv2
    from loguru import logger
    from lpr.io_utils import imread
    from lpr.pipeline import PlateRecognizer

    from lpr_tools.eval_det import camera_canvas

    from .common import effective_config_info
    logger.remove()
    cv2.setNumThreads(1)
    rows=json.loads(Path(args.input).read_text(encoding='utf-8'))
    expected=_rows(args.expected) if args.expected else None
    if expected is not None and [key(r['path']) for r in rows]!=[key(r['path']) for r in expected]:
        raise ValueError('expected/input order differs')
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=False)
    rec=PlateRecognizer.from_config() if args.size==0 else PlateRecognizer.from_config(runtime__detector_input_size=args.size)
    if rec.corner_detector.image_size != (args.size or 256):
        raise ValueError('effective detector size differs')
    scales=[None] if args.mode=='native' else [.6,.8,1.0]
    results={str(s):[] for s in scales}
    with gzip.open(output/'actual.jsonl.gz','wt',encoding='utf-8') as stream:
        for i,row in enumerate(rows,1):
            image=imread(row['path'])
            for scale in scales:
                frame=image if scale is None else camera_canvas(image,row['bbox'],row['quad'],scale)[0]
                result=rec.recognize(frame)
                actual={**row,'scale':scale,'status':result.status.value,'plate':result.plate_no,
                        'confidence':float(result.confidence),'route':getattr(result,'route',None),
                        'candidate_index':getattr(result,'candidate_index',None),'rect_index':getattr(result,'rect_index',None)}
                results[str(scale)].append(actual)
                stream.write(json.dumps(actual,ensure_ascii=False,separators=(',',':'))+'\n')
                del result,frame
            del image
            if i%100==0:
                print(f'{args.mode} size={args.size or 256} {i}/{len(rows)}',flush=True)
    report={'size':rec.corner_detector.image_size,'config':effective_config_info(rec.config),
            'mode':args.mode,'model_version':rec.version,'input_sha256':sha(args.input)}
    if args.mode=='native':
        flat=results['None']
        report['metrics']=summary(flat)
        diffs=[{'path':r['path'],'differences':d} for r,e in zip(flat,expected, strict=False) if (d:=compare(r,e))]
        report.update(difference_count=len(diffs),differences=diffs,paired=paired_changes(flat,expected),expected_sha256=sha(args.expected))
    else:
        report['scales']={s:{'all':metrics(rs),'subsets':{sub:metrics([r for r in rs if r['subset']==sub]) for sub in sorted({r['subset'] for r in rs})}} for s,rs in results.items()}
    dump(output/'report.json',report)
    if args.mode=='native' and args.size==0 and report['difference_count']:
        raise RuntimeError('default behavior changed; stop study')


def video(args):
    import cv2
    from lpr.io_utils import imread

    from .common import letterbox
    rows=json.loads(Path(args.input).read_text(encoding='utf-8'))
    if len(rows)!=60 or any(r['partition']!='val' for r in rows):
        raise ValueError('video needs 60 val sources')
    path=Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25.,(1280,720))
    if not writer.isOpened():
        raise RuntimeError('mp4v writer unavailable')
    try:
        for row in rows:
            frame=letterbox(imread(row['path']),(1280,720))
            for _ in range(25):
                writer.write(frame)
            if path.stat().st_size>100_000_000:
                raise RuntimeError('video exceeds 100 MB')
    finally:
        writer.release()
    if path.stat().st_size>100_000_000:
        raise RuntimeError('video exceeds 100 MB')
    dump(path.with_suffix('.json'),{'seconds':60,'fps':25,'frames':1500,'size':[1280,720],
         'sources':len(rows),'bytes':path.stat().st_size,'sha256':sha(path),'input_sha256':sha(args.input),
         'protocol':'30 base + 30 green val images; each held 25 frames; synthetic speed/tool rehearsal only'})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','native','camera','video'])
    parser.add_argument('--out',required=True)
    parser.add_argument('--input')
    parser.add_argument('--expected')
    parser.add_argument('--size',type=int,choices=[0,256,384,512],default=0)
    parser.add_argument('--val')
    parser.add_argument('--hard')
    parser.add_argument('--index')
    parser.add_argument('--excluded-paths')
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--n',type=int,default=600)
    args=parser.parse_args(argv)
    if args.mode=='prepare':
        prepare(args)
    elif args.mode=='video':
        video(args)
    else:
        evaluate(args)
    return 0

if __name__=='__main__':
    raise SystemExit(main())