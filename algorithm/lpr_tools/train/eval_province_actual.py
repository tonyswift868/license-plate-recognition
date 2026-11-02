"""Single-reader fixed val/dev evaluation for batch12; no route arbitration."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from lpr.config import ALGORITHM_DIR


def select(attempts):
    best=None
    for a in attempts:
        if not isinstance(a,dict) or not a.get('valid') or a.get('status') in {'INVALID_FORMAT','BAD_INPUT'}:
            continue
        p=a.get('plate')
        c=a.get('confidence')
        if not isinstance(p,str) or not p or not isinstance(c,(int,float)) or not math.isfinite(c) or not 0<=c<=1:
            continue
        if best is None or c>best['confidence']:
            best=a
    return best

def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:
        province='anwei' if r['truth'].startswith('皖') else 'non_anwei'
        names=['all','split/'+r['split'],'subset/'+r['subset'],r['split']+'/'+r['subset'],province,province+'/'+r['split'],province+'/'+r['split']+'/'+r['subset']]
        if r.get('province_261'):
            names.append('nonwan_261')
        if r['subset']=='ccpd_green':
            names.append('green')
        for n in names:
            groups[n].append(r)
    out={}
    for n,rs in groups.items():
        count=len(rs)
        pc=sum(bool(r['prediction']) and r['prediction'][0]==r['truth'][0] for r in rs)
        fc=sum(r['prediction']==r['truth'] for r in rs)
        out[n]={'n':count,'province_correct':pc,'province_accuracy':pc/count if count else None,'plate_correct':fc,'plate_accuracy':fc/count if count else None,'wrong_as_anwei':sum(bool(r['prediction']) and r['prediction'][0]=='皖' and not r['truth'].startswith('皖') for r in rs)}
    return out

def nonwan_membership(val_samples, devhard_samples, path_key_fn):
    """Return index-authoritative non-Wan paths split by the fixed manifests."""
    val={path_key_fn(s.row.path) for s in val_samples if s.row.plate_no and s.row.plate_no[0]!='皖'}
    devhard={path_key_fn(s.row.path) for s in devhard_samples if s.row.plate_no and s.row.plate_no[0]!='皖'}
    if val & devhard:
        raise ValueError('fixed val and devhard non-Wan memberships overlap')
    union=val|devhard
    if len(union)!=len(val)+len(devhard):
        raise ValueError('fixed non-Wan union is incomplete')
    return union,{'val':len(val),'devhard':len(devhard)}

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--side',choices=['baseline','candidate'],required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f'refusing to overwrite fixed evaluation: {a.out}')
    import hashlib

    import cv2
    import torch
    from lpr.boxcrop import crop_box
    from lpr.config import load_config
    from lpr.pipeline import _decode_text, _layouts_to_try, _rank, _rectifications, decode
    from lpr.recognize import (
        load_box_reader,
        load_classifier,
        load_corner_detector,
        load_plate_reader,
    )
    from lpr.segment import segment
    from lpr.types import RecogStatus

    from lpr_tools.data.image_health import decode_image
    from lpr_tools.eval.box_direct import _actual_candidates, _read_brightness
    from lpr_tools.eval.common import effective_config_info
    from lpr_tools.eval.index_samples import (
        FIXED_HARD_MANIFEST,
        FIXED_HARD_N,
        FIXED_VAL_MANIFEST,
        FIXED_VAL_N,
        fixed_manifest_samples,
        path_key,
    )
    from lpr_tools.eval.three_route import _model_manifest_identity
    from lpr_tools.train.province_inference_timing import instrument_method, summarize
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cfg=load_config(a.config,runtime__torch_threads=2)
    models=('box_reader_v2.pt','crnn_v2.pt','char_cnn_v2.pt') if a.side=='baseline' else ('box_reader_v3.pt','crnn_v4.pt','char_cnn_v3.pt')
    root=Path('algorithm/models')
    box=load_box_reader(root/models[0])
    reader=load_plate_reader(root/models[1])
    cnn=load_classifier('cnn',root/models[2])
    det=load_corner_detector(root/'plate_det_v1.pt')
    expected_config={'model_path':models[2],'plate_reader_model_path':models[1],'corner_model_path':'plate_det_v1.pt'}
    for field,expected in expected_config.items():
      configured=cfg.resolve_path(getattr(cfg.runtime,field))
      expected_path=(root/expected).resolve()
      if configured is None or configured.resolve()!=expected_path:
          raise ValueError(f'{a.side} config {field} does not resolve to {expected_path}: {configured}')
    artifacts={name:_model_manifest_identity(root/name) for name in (*models,'plate_det_v1.pt')}
    config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest()
    effective=effective_config_info(cfg,a.config)
    rows=[]
    timings=defaultdict(list)
    sample_sets=[]
    for split,manifest,expected,purpose in [('val',FIXED_VAL_MANIFEST,FIXED_VAL_N,'val'),('devhard',FIXED_HARD_MANIFEST,FIXED_HARD_N,'dev-hard')]:
      samples,identity=fixed_manifest_samples(str(ALGORITHM_DIR / "data/index/ccpd_index.csv"),manifest,purpose=purpose)
      if len(samples)!=expected:
          raise ValueError(f'{split} fixed row count mismatch')
      sample_sets.append((split,samples,identity))
    if sum(len(x[1]) for x in sample_sets)!=3100:
        raise ValueError('fixed evaluation must contain exactly 3100 rows')
    nonwan261,nonwan_by_split=nonwan_membership(sample_sets[0][1],sample_sets[1][1],path_key)
    all_nonwan={path_key(s.row.path) for _,ss,_ in sample_sets for s in ss if s.row.plate_no and s.row.plate_no[0]!='皖'}
    if len(nonwan261)!=261 or nonwan261!=all_nonwan:
        raise ValueError(f'fixed nonwan union mismatch: union={len(nonwan261)}, all={len(all_nonwan)}')
    all_keys={path_key(s.row.path) for _,samples,_ in sample_sets for s in samples}
    brightness=_read_brightness(str(ALGORITHM_DIR / "data/index/ccpd_index.csv"),all_keys)
    for kind,obj,method,batch_fn in [('box',box,'read',lambda args,kw:1),('crnn',reader,'read',lambda args,kw:1),('char',cnn,'predict_proba',lambda args,kw:len(args[0]))]:
      instrument_method(obj,method,timings[kind],batch_size=batch_fn)
    progress_path=a.out.with_name(a.out.name+'.progress.jsonl')
    progress_path.parent.mkdir(parents=True,exist_ok=True)
    progress_stream=progress_path.open('w',encoding='utf-8')
    def report_progress(split):
      if len(rows)%100==0:
        progress_stream.write(json.dumps({'rows_done':len(rows),'total_rows':3100,'split':split},separators=(',',':'))+'\n')
        progress_stream.flush()
    for split,samples,_identity in sample_sets:
      for sample in samples:
        m=sample.row
        path=Path(m.path)
        try:
            image,error=decode_image(path)
        except Exception as exc:
            image,error=None,repr(exc)
        key=path_key(path)
        rec={'path':str(path),'truth':m.plate_no,'split':split,'subset':sample.subset,'source':m.source,'brightness':brightness[key],'province_261':key in nonwan261,'prediction':None,'attempts':{},'input_error':None}
        if error:
          rec['input_error']=str(error)
          rec['attempts']={k:[] for k in ('box','crnn','char')}
          rec.update({k:None for k in ('box','crnn','char')})
          rows.append(rec)
          report_progress(split)
          continue
        try:
            candidates=_actual_candidates(image,cfg,det)
        except Exception as e:
            rec['locator_error']=repr(e)
            candidates=[]
        rec['route_errors']=[]
        for kind,obj in [('box',box),('crnn',reader),('char',cnn)]:
          ats=[]
          for ci,c in enumerate(candidates):
            candidate_meta={'candidate_index':ci,'candidate_source':getattr(c,'source',None),'candidate_score':float(c.score),'bbox':[int(v) for v in c.bbox],'candidate_color':str(c.color)}
            if kind=='box':
              try:
                  crop=crop_box(image,c.bbox,*obj.box_pad)
              except Exception as exc:
                rec['route_errors'].append({'candidate_index':ci,'route':'box_crop','error':repr(exc)})
                ats.append({'plate':'','confidence':None,'status':'BAD_CROP','valid':False,**candidate_meta,'rect_index':None})
                continue
              if crop is None or crop.size==0:
                ats.append({'plate':'','confidence':None,'status':'BAD_CROP','valid':False,**candidate_meta,'rect_index':None})
                continue
              start=len(timings[kind])
              try:
                  text,conf=obj.read(crop)
              except Exception as exc:
                  timings[kind][start].update(path=str(path),candidate_index=ci,stage='read',exception=repr(exc))
                  rec['route_errors'].append({'candidate_index':ci,'route':'box_read','error':repr(exc)})
                  text,conf='',[]
              timings[kind][start].update(path=str(path),candidate_index=ci,stage='read')
              d=_decode_text(text,conf,c.color,cfg.postprocess.min_confidence)
              z={'plate':d.plate_no if d else '', 'confidence':float(d.confidence) if d else (min(conf) if conf else None),'status':d.status.value if d else 'INVALID_OUTPUT','valid':d is not None}
              ats.append({**z,**candidate_meta,'rect_index':None})
            else:
              try:
                  rects=_rectifications(image,c,cfg.rectify)
              except Exception as exc:
                rec['route_errors'].append({'candidate_index':ci,'route':'rectify','error':repr(exc)})
                continue
              for ri,rect in enumerate(rects):
                if kind=='crnn':
                  start=len(timings[kind])
                  try:
                      text,conf=obj.read(rect.image)
                  except Exception as exc:
                      timings[kind][start].update(path=str(path),candidate_index=ci,rect_index=ri,stage='read',exception=repr(exc))
                      rec['route_errors'].append({'candidate_index':ci,'rect_index':ri,'route':'crnn_read','error':repr(exc)})
                      text,conf='',[]
                  timings[kind][start].update(path=str(path),candidate_index=ci,rect_index=ri,stage='read')
                  d=_decode_text(text,conf,rect.color,cfg.postprocess.min_confidence)
                  z={'plate':d.plate_no if d else '', 'confidence':float(d.confidence) if d else (min(conf) if conf else None),'status':d.status.value if d else 'INVALID_OUTPUT','valid':d is not None}
                else:
                  z=None
                  best_rank=None
                  for layout in _layouts_to_try(rect):
                    try:
                        seg=segment(rect,layout,cfg.segment)
                    except Exception as exc:
                      rec['route_errors'].append({'candidate_index':ci,'rect_index':ri,'route':'segment','error':repr(exc)})
                      z=z or {'plate':'','confidence':None,'status':'READ_ERROR','valid':False}
                      break
                    if not seg.ok:
                        continue
                    start=len(timings[kind])
                    try:
                        probs=obj.predict_proba(seg.chars)
                    except Exception as exc:
                        timings[kind][start].update(path=str(path),candidate_index=ci,rect_index=ri,layout=layout,stage='predict_proba',exception=repr(exc))
                        rec['route_errors'].append({'candidate_index':ci,'rect_index':ri,'route':'char_predict_proba','error':repr(exc)})
                        z=z or {'plate':'','confidence':None,'status':'READ_ERROR','valid':False}
                        break
                    timings[kind][start].update(path=str(path),candidate_index=ci,rect_index=ri,layout=layout,stage='predict_proba')
                    try:
                        d=decode(probs,seg.layout,rect.color,cfg.postprocess)
                    except Exception as exc:
                      rec['route_errors'].append({'candidate_index':ci,'rect_index':ri,'route':'decode','error':repr(exc)})
                      z=z or {'plate':'','confidence':None,'status':'READ_ERROR','valid':False}
                      break
                    cand={'plate':d.plate_no,'confidence':float(d.confidence),'status':d.status.value,'valid':d.status in (RecogStatus.OK,RecogStatus.LOW_CONFIDENCE)}
                    if z is None or _rank(d)>best_rank:
                        z=cand
                        best_rank=_rank(d)
                    if d.status==RecogStatus.OK:
                        break
                  z=z or {'plate':'','confidence':None,'status':'NO_READ','valid':False}
                ats.append({**z,**candidate_meta,'rect_index':ri})
          chosen=select(ats)
          rec['attempts'][kind]=ats
          rec[kind]=chosen
        rows.append(rec)
        del image
        report_progress(split)
    progress_stream.close()
    if len(rows)!=3100 or sum(r['split']=='val' for r in rows)!=1000 or sum(r['split']=='devhard' for r in rows)!=2100:
        raise ValueError('actual evaluation denominator must be val=1000 + devhard=2100')
    if sum(r['province_261'] for r in rows)!=261:
        raise ValueError('province_261 denominator mismatch')
    timing_summary={k:{**summarize(v,buckets=(1,7,8)), 'reader_threads':2,'scope':'original read/predict_proba invocation only; excludes decode/preparation; includes cold calls'} for k,v in timings.items()}
    result={'schema':'batch12-single-reader-fixed-eval-v1','side':a.side,'rows':len(rows),'models':dict(zip(('box','crnn','char'),models, strict=False)),'artifacts':artifacts,'config':{'path':str(a.config.resolve()),'sha256':config_sha256,'effective_sha256':effective['effective_config_sha256']},'metrics':{k:aggregate([{**r,'prediction':r[k]['plate'] if r.get(k) else None} for r in rows]) for k in ('box','crnn','char')},'timings_ms':timing_summary,'manifest_identities':{split:identity for split,_,identity in sample_sets},'nonwan_261_n':len(nonwan261),'all_nonwan_n':len(nonwan261),'nonwan_by_split':nonwan_by_split,'selection':'per-route highest-confidence legal candidate/rect, stable first tie; CNN layout tries use pipeline status-first _rank; no route combination','input_errors':sum(bool(r['input_error']) for r in rows),'input_preparation':'one decode_image call per original path; decode, locating and rectification excluded from reader timing; no G1 claim','progress_file':str(progress_path.resolve()),'records':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.out,'wt',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False)
    for kind,records in timings.items():
      with a.out.with_name(f'actual-{a.side}-{kind}-timing.csv').open('w',encoding='utf-8',newline='') as stream:
        import csv as _csv
        fields=('call_index','path','stage','candidate_index','rect_index','layout','batch_size','elapsed_ms','exception')
        writer=_csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
