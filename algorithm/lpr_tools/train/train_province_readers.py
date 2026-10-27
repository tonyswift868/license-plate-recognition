"""Continue the three own readers with safe province balance and online swaps.

Each epoch visits every audited source once. Count-weighted replay keeps the
batch11 nine-group multiset, then adds bounded real non-Anhui exposure. Synthetic
copies are additional forwards, never a replacement for a real source visit.
"""
from __future__ import annotations
import os
from pathlib import Path
_RUNTIME = Path(__file__).resolve().parents[3] / 'algorithm/out/tmp/batch12-runtime'
_RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(_RUNTIME)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
from collections import Counter
import csv
import gc
import json
import math
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
from torch import nn
from lpr.charset import CHAR_TO_INDEX, NUM_CLASSES, PROVINCES, PROVINCE_IDX
from lpr.recognize import file_sha256
from lpr.recognize.box_reader import BOX_PAD, box_tensor, checkpoint, model_from_checkpoint
from lpr.recognize.box_reader_v2 import INPUT_SIZE_V2
from lpr.recognize.crnn import CrnnNet, plate_tensor
from lpr.recognize.cnn_model import CharCNN
from lpr_tools.train.box_data import bbox_from_quad, parse_quad
from lpr_tools.train.box_data_v2 import build_balanced_multiplicity
from lpr_tools.train.train_box_reader import BoxDataset, cache_contains, free_ram_bytes, process_ram_bytes, row_crop
from lpr_tools.train.train_box_reader_v2 import affine_target
from lpr_tools.train.province_data import province_repeat_additions
from lpr_tools.train.province_synth import replace_province, normalized_first_char

FORMAT='province-reader-training-v1'
render_lock=threading.Lock()


class _DeferredFinish(Exception):
    """A cached crop needs a GT full-image fallback on the main thread."""


def _replace_province_locked(*args, **kwargs):
    with render_lock:
        return replace_province(*args, **kwargs)


def local_quad(quad, bbox):
    x,y,w,h=bbox
    origin=np.float32([max(0,math.floor(x-BOX_PAD[0]*w)),max(0,math.floor(y-BOX_PAD[1]*h))])
    return np.float32(parse_quad(quad))-origin


def rectified(image, quad, length):
    width=276 if length==8 else 252
    dst=np.float32([[0,0],[width-1,0],[width-1,79],[0,79]])
    return cv2.warpPerspective(image,cv2.getPerspectiveTransform(np.float32(quad),dst),(width,80))


def frozen_batch_statistics(model):
    for layer in model.modules():
        if isinstance(layer,nn.modules.batchnorm._BatchNorm): layer.eval()


def real_plan(labels, subsets, seed, epoch):
    base=build_balanced_multiplicity(subsets,seed=seed,epoch=epoch)
    extra=province_repeat_additions([{'label':s} for s in labels],seed=seed,epoch=epoch)
    weights=np.asarray(base.multiplicity,dtype=np.float32)+np.asarray(extra.additions,dtype=np.float32)
    return weights,dict(base.group_exposures),dict(extra.added_by_province)


def synthetic_targets(labels, *, seed, epoch, probability):
    """Equal target counts with distinct source/target province whenever feasible."""
    if not 0 < probability < 1:raise ValueError('invalid synthesis probability')
    rng=np.random.default_rng([seed,epoch,83])
    selected=rng.permutation(len(labels))[:int(len(labels)*probability)]
    assigned=np.resize(np.arange(len(PROVINCES)),len(selected));rng.shuffle(assigned)
    cursor=0
    for position,index in enumerate(selected):
        if PROVINCES[int(assigned[position])]!=labels[int(index)][0]:continue
        for _ in range(len(selected)):
            other=cursor%len(selected);cursor+=1
            if (PROVINCES[int(assigned[other])]!=labels[int(index)][0] and
                    PROVINCES[int(assigned[position])]!=labels[int(selected[other])][0]):
                assigned[position],assigned[other]=assigned[other],assigned[position]
                break
        # Tiny smoke sets can contain only one source province. Re-rendering
        # its own glyph is recorded as such; the full-data audit forbids it.
    result=np.full(len(labels),-1,dtype=np.int16)
    result[selected]=assigned
    return result


class ProvinceDataset(BoxDataset):
    def __init__(self,*args,kind='box',synth_probability=.35,**kwargs):
        super().__init__(*args,**kwargs)
        self.kind=kind;self.synth_probability=synth_probability;self.synthetic_gt_fallbacks=0

    def specification(self,index):
        row,rng,bbox=super().specification(index)
        if self.kind!='box':bbox=bbox_from_quad(row['quad'])
        return row,rng,bbox

    def finish(self,image,row,index,rng,*,bbox,defer_gt_fallback=False):
        quad=local_quad(row['quad'],bbox)
        theta=affine_target(row['quad'],bbox,image.shape)
        variants=[(image,row['label'],False,quad,theta)]
        # shortcut: logical repeats reuse an epoch's augmentation, as in v2;
        # return to independent repeat augmentation if future quality needs it.
        if self.synthetic_targets[index]>=0:
            target=PROVINCES[int(self.synthetic_targets[index])]
            synth_quad=quad;synth_theta=theta
            try:
                generated,label,_=_replace_province_locked(image,quad,row['label'],target,
                    np.random.default_rng([self.seed,self.epoch,index]))
            except ValueError as error:
                if str(error)!='province glyph slot falls outside source image':raise
                if defer_gt_fallback:
                    # Do not let the CPU prefetch thread trigger torchvision
                    # CUDA decode or consume the Python photo RNG.
                    raise _DeferredFinish from error
                # Keep the measured real crop unchanged. A synthetic label must
                # correspond to a visible replacement, so only that additional
                # view falls back to its GT crop when jitter removed the slot.
                gt_box=bbox_from_quad(row['quad'])
                gt_image=row_crop(row,gt_box,gpu_fallback=self.gpu_fallback)
                synth_quad=local_quad(row['quad'],gt_box)
                synth_theta=affine_target(row['quad'],gt_box,gt_image.shape)
                generated,label,_=_replace_province_locked(gt_image,synth_quad,row['label'],target,
                    np.random.default_rng([self.seed,self.epoch,index]))
                self.synthetic_gt_fallbacks+=1
            variants.append((generated,label,True,synth_quad,synth_theta))
        result=[]
        for pixels,label,synthetic,view_quad,view_theta in variants:
            if self.kind=='char':
                pixels=normalized_first_char(pixels,view_quad,label)
                tensor=torch.from_numpy(pixels.copy()).unsqueeze(0).float().div_(255.)
            else:
                if rng.random()<.25:
                    gamma=rng.uniform(.8,1.25)
                    pixels=cv2.LUT(pixels,np.asarray([255*(v/255.)**gamma for v in range(256)],np.uint8))
                tensor=(box_tensor(pixels,INPUT_SIZE_V2) if self.kind=='box' else
                        plate_tensor(rectified(pixels,view_quad,len(label))))
            result.append((tensor,label,index,view_theta,row['subset'],synthetic))
        return result


class _PackedStream:
    """A TAR handle lazily opened and used only by the current prepare thread."""
    def __init__(self):
        self.stream=None
        self.path=None
        self.total_size=None

    def read_span(self,rows):
        paths={str(row['cache_pack']) for row in rows if row.get('cache_pack')}
        if not paths:
            return None,None,None
        if len(paths)!=1:
            raise ValueError('one packed epoch may use only one immutable JPEG TAR')
        path=Path(next(iter(paths)))
        if self.stream is None:
            self.path=path
            self.stream=path.open('rb')
            self.total_size=self.stream.seek(0,2)
        elif path!=self.path:
            raise ValueError('packed JPEG TAR changed within an epoch')
        offsets=[int(row['pack_offset']) for row in rows if row.get('cache_pack')]
        ends=[int(row['pack_offset'])+int(row['pack_length']) for row in rows if row.get('cache_pack')]
        start_pos=min(offsets);end_pos=max(ends)
        if (start_pos<0 or end_pos>self.total_size or end_pos-start_pos>40_000_000
                or any(int(row['pack_length'])<=0 for row in rows if row.get('cache_pack'))):
            raise ValueError('cache block span exceeds bounded IO or has invalid offsets')
        self.stream.seek(start_pos)
        return self.stream.read(end_pos-start_pos),start_pos,end_pos

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream=None


class PackedEpoch:
    """One bounded CPU-prepared block; optional single-thread prefetch overlaps GPU work."""
    def __init__(self,ds,labels,subsets,*,batch_size=128,block_size=128,prefetch=True):
        self.ds=ds;self.labels=labels;self.subsets=subsets
        self.batch_size=batch_size;self.block_size=block_size;self.raw_fallbacks=0
        self.prefetch=bool(prefetch);self.prepare_seconds=0.

    def __iter__(self):
        ds=self.ds
        ds.synthetic_targets=synthetic_targets(self.labels,seed=ds.seed,epoch=ds.epoch,probability=ds.synth_probability)
        self.weights,self.base_groups,self.extra_provinces=real_plan(self.labels,self.subsets,ds.seed,ds.epoch)
        rng=np.random.default_rng([ds.seed,ds.epoch])
        batch=[];weights=[]
        stream=_PackedStream()
        try:
            blocks=list(rng.permutation(math.ceil(len(ds)/self.block_size)))
            jobs=[(int(block)*self.block_size,
                   min(len(ds),(int(block)+1)*self.block_size)) for block in blocks]
            if self.prefetch:
                with ThreadPoolExecutor(max_workers=1,thread_name_prefix='batch12-cpu-prefetch') as executor:
                    yield from self._consume_jobs(jobs,executor,stream,rng,batch,weights)
            else:
                yield from self._consume_jobs(jobs,None,stream,rng,batch,weights)
            if batch:yield collate(batch,weights)
        finally:
            # Executor context exits and joins its sole worker before the TAR
            # and dataset mmap are closed by the training loop.
            stream.close()

    def _consume_jobs(self,jobs,executor,stream,rng,batch,weights):
        pending=iter(jobs)
        job=next(pending,None)
        future=executor.submit(self._prepare,stream,*job) if executor is not None and job else None
        while job is not None:
            begin,end=job
            prepared=(future.result() if future is not None else
                      self._prepare(stream,begin,end))
            next_job=next(pending,None)
            future=(executor.submit(self._prepare,stream,*next_job)
                    if executor is not None and next_job is not None else None)
            for relative in rng.permutation(end-begin):
                relative=int(relative)
                row,photo_rng,bbox,item=prepared[relative]
                index=begin+relative
                if item is None:
                    # Source JPEG decode and the resulting crop stay on the
                    # training thread when the audited cache does not cover it.
                    self.raw_fallbacks+=1
                    image=row_crop(row,bbox,gpu_fallback=self.ds.gpu_fallback)
                    items=self.ds.finish(image,row,index,photo_rng,bbox=bbox)
                elif item[0]=='deferred_finish':
                    _,image=item
                    # The failed replacement does not consume photo_rng.
                    items=self.ds.finish(image,row,index,photo_rng,bbox=bbox)
                else:
                    items=item[1]
                for value in items:
                    batch.append(value)
                    weights.append(float(self.weights[index])*
                                   (1/(1-self.ds.synth_probability) if value[-1] else 1.))
                    if len(batch)==self.batch_size:
                        yield collate(batch,weights)
                        batch.clear();weights.clear()
            job=next_job

    def _prepare(self,stream,begin,end):
        started=time.perf_counter()
        specifications=[self.ds.specification(index) for index in range(begin,end)]
        rows=[row for row,_,_ in specifications]
        data,start_pos,_=stream.read_span(rows)
        prepared=[]
        for relative,(row,photo_rng,bbox) in enumerate(specifications):
            index=begin+relative
            cached=None
            if row.get('cache_pack'):
                encoded=np.frombuffer(data,np.uint8,count=int(row['pack_length']),
                                      offset=int(row['pack_offset'])-start_pos)
                cached=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                if cached is None:raise ValueError('invalid immutable cache JPEG')
            if cached is not None and cache_contains(row,bbox,BOX_PAD,cached.shape):
                image=row_crop(row,bbox,cached_image=cached,gpu_fallback=False)
            elif self.ds.gpu_fallback:
                prepared.append((row,photo_rng,bbox,None))
                continue
            else:
                image=row_crop(row,bbox,gpu_fallback=False)
            try:
                items=self.ds.finish(image,row,index,photo_rng,bbox=bbox,
                                     defer_gt_fallback=True)
                prepared.append((row,photo_rng,bbox,('items',items)))
            except _DeferredFinish:
                prepared.append((row,photo_rng,bbox,('deferred_finish',image)))
        self.prepare_seconds+=time.perf_counter()-started
        return prepared


def collate(items,weights):
    images,labels,indices,thetas,groups,synthetic=zip(*items)
    lengths=torch.tensor([len(s) for s in labels],dtype=torch.long)
    targets=torch.tensor([CHAR_TO_INDEX[c] for label in labels for c in label],dtype=torch.long)
    return (torch.stack(images),targets,lengths,list(labels),list(indices),torch.stack(thetas),
            list(groups),list(synthetic),torch.tensor(weights,dtype=torch.float32))


def inference_state(model,kind):
    return checkpoint(model) if kind=='box' else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def atomic_save(payload,path):
    temporary=path.with_suffix('.writing')
    torch.save(payload,temporary);os.replace(temporary,path)


def restore_model(path,kind):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    state=saved['model_state'] if saved.get('format')==FORMAT else saved
    if kind=='box':return model_from_checkpoint(state)
    state=dict(state);state.pop('ctc_temperature',None)
    model=CrnnNet() if kind=='crnn' else CharCNN()
    model.load_state_dict(state,strict=True)
    return model


def run(args):
    if args.kind not in ('box','crnn','char'):raise ValueError('unknown model kind')
    minimum={'box':20,'crnn':15,'char':10}[args.kind]
    if args.epochs<minimum and not (args.smoke or args.profile_batches):raise ValueError('full epoch minimum not met')
    if not 0<args.synth_probability<1:raise ValueError('invalid synthesis share')
    if args.device=='cuda' and free_ram_bytes()<3_500_000_000:
        raise RuntimeError('GPU startup needs 3.5GB free RAM; no training started')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda':
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.48,5_500_000_000/total))
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('high')
    args.out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((args.data_dir/'province_audit.json').read_text(encoding='utf-8'))
    if not args.smoke:
        if json.loads((args.data_dir/'independent_audit.json').read_text(encoding='utf-8')).get('passed') is not True:
            raise ValueError('independent province isolation checks failed')
        if audit.get('schema')!='province-coverage-data-v1':raise ValueError('unsupported province audit')
        for name,identity in audit['outputs'].items():
            if file_sha256(args.data_dir/name)!=identity['sha256']:raise ValueError('audited metadata changed: '+name)
        if file_sha256(args.index)!=audit['outputs']['combined_train.csv']['sha256']:
            raise ValueError('training index differs from the audited full pool')
        if file_sha256(Path('algorithm/out/data/batch11/tune.csv'))!=audit['source_hashes']['batch11_tune_sha256']:
            raise ValueError('safe tune identity changed')
    index=args.index
    with index.open(encoding='utf-8-sig',newline='') as f:
        labels=[];subsets=[]
        for row in csv.DictReader(f):labels.append(row['label']);subsets.append(row['subset'])
    samples=json.loads(args.sampler.read_text(encoding='utf-8-sig'))
    samples=samples.get('samples_by_subset',samples.get('subsets',samples))
    ds=ProvinceDataset(index,samples,training=True,seed=args.seed,gpu_fallback=device.type=='cuda',
                       kind=args.kind,synth_probability=args.synth_probability)
    loader=PackedEpoch(ds,labels,subsets,batch_size=args.batch_size,block_size=args.block_size,
                       prefetch=not args.no_cpu_prefetch)
    model=restore_model(args.init,args.kind).to(device)
    # Frozen running statistics permit exact repeated-tensor weighting. CNN
    # dropout is disabled; none of the other model architectures has dropout.
    for layer in model.modules():
        if isinstance(layer,nn.Dropout):layer.p=0.
    if args.kind=='char':
        # Keep the feature extractor and every alphanumeric output logit exactly
        # unchanged. Only province rows of the final own-v2 classifier adapt.
        for parameter in model.parameters():parameter.requires_grad_(False)
        head=model.classifier[-1]
        head.weight.requires_grad_(True);head.bias.requires_grad_(True)
        head.weight.register_hook(lambda gradient: torch.cat((gradient[:31],torch.zeros_like(gradient[31:]))))
        head.bias.register_hook(lambda gradient: torch.cat((gradient[:31],torch.zeros_like(gradient[31:]))))
    parameters=[p for p in model.parameters() if p.requires_grad]
    if args.kind=='box':
        # Retain v2's smaller localization LR and geometry supervision while
        # adapting the recognition features to the new province distribution.
        local_ids={id(p) for p in model.stn.parameters()}
        parameters=[{'params':model.stn.parameters(),'lr':args.lr*.2},
                    {'params':[p for p in parameters if id(p) not in local_ids],'lr':args.lr}]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,
        weight_decay=0. if args.kind=='char' else 1e-4,fused=device.type=='cuda')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.08)
    ctc=nn.CTCLoss(blank=NUM_CLASSES,reduction='none',zero_infinity=False)
    provenance={'init':str(args.init.resolve()),'init_sha256':file_sha256(args.init),
                'index_sha256':file_sha256(index),'audit_sha256':file_sha256(args.data_dir/'province_audit.json'),
                'sampler_sha256':file_sha256(args.sampler),
                'augmentation_source_sha256':file_sha256(Path(__file__).with_name('province_synth.py')),
                'font_sha256':file_sha256(Path(r'C:\Windows\Fonts\simhei.ttf'))}
    history=[];best=-1.;best_epoch=0;elapsed=0.;start_epoch=0;skip=0
    if args.resume:
        saved=torch.load(args.out/'last.pt',map_location='cpu',weights_only=True)
        if saved['provenance']!=provenance or saved['kind']!=args.kind:raise ValueError('resume provenance differs')
        model.load_state_dict(saved['model_state']['state_dict'] if args.kind=='box' else saved['model_state'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best_score'];best_epoch=saved['best_epoch']
        start_epoch=saved['epoch'];skip=saved['cursor'];elapsed=saved['elapsed_seconds']
    run_started=time.perf_counter()
    for epoch in range(start_epoch,args.epochs):
        ds.epoch=epoch;model.train();frozen_batch_statistics(model)
        epoch_started=time.perf_counter();real_seen=np.zeros(len(ds),dtype=np.uint8)
        synthetic_provinces=Counter();synthetic_province_weights=Counter()
        real_draws=synthetic_draws=real_physical=synthetic_physical=0;loss_total=0.;steps=0;epoch_prior_seconds=0.
        if args.resume and epoch==start_epoch and skip:
            state=saved['epoch_counts'];real_seen=np.asarray(state['real_seen'],dtype=np.uint8)
            real_draws=state['real_draws'];synthetic_draws=state['synthetic_draws']
            real_physical=state['real_physical'];synthetic_physical=state['synthetic_physical'];loss_total=state['loss_total']
            steps=state.get('steps',skip);epoch_prior_seconds=state.get('active_seconds',0.)
            synthetic_provinces.update(state.get('synthetic_provinces',{}));synthetic_province_weights.update(state.get('synthetic_province_weights',{}))
        def payload(cursor,counts):
            return {'format':FORMAT,'kind':args.kind,'model_state':inference_state(model,args.kind),
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':epoch,'cursor':cursor,'epoch_counts':counts,'history':history,
                    'best_score':best,'best_epoch':best_epoch,'provenance':provenance,
                    'elapsed_seconds':elapsed+time.perf_counter()-run_started,
                    'recipe':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
        for step,batch in enumerate(loader):
            if step<skip:continue
            images,targets,lengths,texts,indices,thetas,groups,synthetic,weights=batch
            optimizer.zero_grad(set_to_none=True)
            outputs=model(images.to(device),return_theta=True) if args.kind=='box' else model(images.to(device))
            logits,pred_theta=outputs if args.kind=='box' else (outputs,None)
            if args.kind=='char':
                y=torch.tensor([CHAR_TO_INDEX[t[0]] for t in texts],device=device)
                per_example=nn.functional.cross_entropy(logits[:,:31],y,reduction='none')
            else:
                per_example=ctc(logits.float().log_softmax(-1),targets.to(device),
                    torch.full((len(texts),),logits.shape[0],dtype=torch.long),lengths)/lengths.to(device)
                if pred_theta is not None:per_example+=(pred_theta-thetas.to(device)).square().mean((1,2))
            gpu_weights=weights.to(device)
            loss=(per_example*gpu_weights).sum()/gpu_weights.sum()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            for i,is_synth,weight,label in zip(indices,synthetic,weights.tolist(),texts):
                if is_synth:
                    synthetic_physical+=1;synthetic_draws+=weight
                    synthetic_provinces[label[0]]+=1;synthetic_province_weights[label[0]]+=weight
                else:real_physical+=1;real_draws+=weight;real_seen[i]+=1
            loss_total+=float(loss.detach());steps=step+1
            if step%100==0 or (args.out/'stop-request.json').exists():
                rss=process_ram_bytes();free=free_ram_bytes()
                print(json.dumps({'epoch':epoch+1,'step':steps,'loss':float(loss.detach()),'rss':rss,
                                 'free_ram':free,'real_physical':real_physical,'synthetic_physical':synthetic_physical}),flush=True)
                if (rss>2_500_000_000 or (args.out/'stop-request.json').exists() or (device.type=='cuda' and free<2_000_000_000)) and not args.smoke:
                    counts={'real_seen':real_seen.tolist(),'real_draws':real_draws,'synthetic_draws':synthetic_draws,
                            'real_physical':real_physical,'synthetic_physical':synthetic_physical,'loss_total':loss_total,
                            'steps':steps,'active_seconds':epoch_prior_seconds+time.perf_counter()-epoch_started,
                            'synthetic_provinces':dict(synthetic_provinces),'synthetic_province_weights':dict(synthetic_province_weights)}
                    atomic_save(payload(steps,counts),args.out/'last.pt')
                    request=args.out/'stop-request.json'
                    reason=(json.loads(request.read_text(encoding='utf-8-sig')).get('reason') if request.exists() else 'insufficient memory; safely checkpointed')
                    (args.out/'paused.json').write_text(json.dumps({'reason':reason,
                        'epoch':epoch+1,'cursor':steps,'rss':rss,'free_ram':free,'required_free_bytes':3_500_000_000}))
                    raise SystemExit(75)
            if args.profile_batches and steps>=args.profile_batches:
                result={'kind':args.kind,'steps':steps,'seconds':time.perf_counter()-epoch_started,
                        'real_physical':real_physical,'synthetic_physical':synthetic_physical,
                        'rss_peak':process_ram_bytes(peak=True),'probe_updates_discarded':True}
                (args.out/'profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
                return result
        if real_physical!=len(ds) or not np.all(real_seen==1):raise RuntimeError('full epoch unique coverage failed')
        if abs(real_draws-float(loader.weights.sum()))>1:raise RuntimeError('real balanced exposure mismatch')
        optimizer.zero_grad(set_to_none=True)
        if 'images' in locals():del images,outputs,logits,loss,per_example
        if ds._map is not None:ds._map.close();ds._map=None
        if ds._file is not None:ds._file.close();ds._file=None
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        counts={'real_seen':real_seen.tolist(),'real_draws':real_draws,'synthetic_draws':synthetic_draws,
                'real_physical':real_physical,'synthetic_physical':synthetic_physical,'loss_total':loss_total,
                'steps':steps,'active_seconds':epoch_prior_seconds+time.perf_counter()-epoch_started,
                            'synthetic_provinces':dict(synthetic_provinces),'synthetic_province_weights':dict(synthetic_province_weights)}
        atomic_save(payload(steps,counts),args.out/'last.pt')
        if args.smoke:score={'selection_score':0.,'province_accuracy':0.,'plate_accuracy':0.}
        else:
            scored=subprocess.run([str(args.cpu_python),'-B','-m','lpr_tools.train.score_province_readers',
                '--kind',args.kind,'--model',str(args.out/'last.pt'),'--data-dir',str(args.data_dir),
                '--out',str(args.out/'selection-latest.json')],capture_output=True,text=True,encoding='utf-8')
            if scored.returncode:raise RuntimeError('CPU selection failed: '+scored.stderr[-4000:])
            score=json.loads(scored.stdout)
        record={'epoch':epoch+1,'real_unique':len(ds),'real_physical':real_physical,
                'synthetic_physical':synthetic_physical,'synthetic_provinces':dict(synthetic_provinces),
                'synthetic_province_weights':dict(synthetic_province_weights),'real_logical':real_draws,'synthetic_logical':synthetic_draws,
                'synthetic_logical_fraction':synthetic_draws/(real_draws+synthetic_draws),
                'base_group_logical':loader.base_groups,'extra_province_logical':loader.extra_provinces,
                'loss':loss_total/max(1,steps),'seconds':epoch_prior_seconds+time.perf_counter()-epoch_started,
                'selection':{k:v for k,v in score.items() if k not in ('records',)},'rss_peak':process_ram_bytes(peak=True)}
        history.append(record);scheduler.step();skip=0
        improved=score['selection_score']>best
        if improved:best=score['selection_score'];best_epoch=epoch+1
        saved_out=payload(0,{});saved_out['epoch']=epoch+1
        atomic_save(saved_out,args.out/'last.pt')
        if improved:atomic_save(saved_out,args.out/'best.pt')
        report={'kind':args.kind,'epochs_completed':len(history),'best_epoch':best_epoch,'best_score':best,
                'history':history,'provenance':provenance,'raw_gpu_fallbacks':loader.raw_fallbacks,'synthetic_gt_fallbacks':ds.synthetic_gt_fallbacks,
                'effective_seconds':elapsed+time.perf_counter()-run_started}
        (args.out/'history.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        projection={'kind':args.kind,'epoch_seconds':record['seconds'],'projected_total_seconds':record['seconds']*args.epochs,
                    'epochs':args.epochs,'includes_cpu_selection':True}
        if epoch==0:
            (args.out/'first-epoch-projection.json').write_text(json.dumps(projection,indent=2))
            with Path('algorithm/out/receipts/第十二批.md').open('a',encoding='utf-8') as f:
                f.write('\n首轮实测：'+json.dumps(projection,ensure_ascii=False)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('box','crnn','char'),required=True)
    p.add_argument('--index',type=Path,default=Path('algorithm/out/data/batch12/combined_train.csv'))
    p.add_argument('--data-dir',type=Path,default=Path('algorithm/out/data/batch12'))
    p.add_argument('--sampler',type=Path,default=Path('algorithm/out/data/batch11/sampler.json'))
    p.add_argument('--init',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--block-size',type=int,default=128)
    p.add_argument('--synth-probability',type=float,default=.35);p.add_argument('--lr',type=float,default=.00008)
    p.add_argument('--seed',type=int,default=12042);p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--profile-batches',type=int,default=0)
    p.add_argument('--no-cpu-prefetch',action='store_true',
                   help='disable the single bounded CPU prefetch thread for serial comparisons')
    p.add_argument('--cpu-python',type=Path,default=Path('.venv/Scripts/python.exe').resolve())
    print(json.dumps(run(p.parse_args()),default=str),flush=True)

if __name__=='__main__':main()
