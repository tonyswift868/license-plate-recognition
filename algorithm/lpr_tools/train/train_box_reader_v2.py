"""Full-pool, nine-group-balanced v2 training with a supervised internal STN.

Reuse the immutable v1 JPEG TAR. Each source is decoded/prepared once per epoch;
actual repeated tensor draws balance all groups without repeated JPEG decoding.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from lpr.charset import CHAR_TO_INDEX, NUM_CLASSES
from lpr.recognize import file_sha256
from lpr.recognize.box_reader import BOX_PAD, box_tensor, checkpoint
from lpr.recognize.box_reader_v2 import INPUT_SIZE_V2, BoxNetV2
from lpr_tools.train.box_data import parse_quad
from lpr_tools.train.box_data_v2 import GROUPS, build_balanced_multiplicity
from lpr_tools.train.train_box_reader import (
    BoxDataset,
    cache_contains,
    free_ram_bytes,
    process_ram_bytes,
    row_crop,
)
from torch import nn


def affine_target(quad, bbox, crop_shape, pad=BOX_PAD) -> torch.Tensor:
    """Fit a pixel-center affine output->input grid from training-only LT/RT/RB/LB.

    The 4-point annotation is used only for auxiliary training supervision. An
    affine fit is not an exact projective warp; inference receives no annotation.
    """
    points = np.asarray(parse_quad(quad), dtype=np.float32)
    x, y, width, height = bbox
    left, top = max(0, math.floor(x-pad[0]*width)), max(0, math.floor(y-pad[1]*height))
    h, w = crop_shape[:2]
    if min(w, h) <= 0:
        raise ValueError("affine target needs a nonempty crop")
    normalized = (points-np.float32([left-.5, top-.5]))*np.float32([2/w, 2/h])-1
    horizontal = (normalized[1]+normalized[2]-normalized[0]-normalized[3])*.25
    vertical = (normalized[2]+normalized[3]-normalized[0]-normalized[1])*.25
    # Keep a narrow learned margin around the labelled plate after internal STN.
    theta = np.stack((horizontal*1.08, vertical*1.12, normalized.mean(axis=0)), axis=1)
    if not np.isfinite(theta).all():
        raise ValueError("nonfinite affine training target")
    return torch.from_numpy(theta)


class BoxDatasetV2(BoxDataset):
    def finish(self, image, row, index, rng, *, bbox=None):
        if not image.size:
            raise ValueError(f"empty v2 training crop: {row['path']}")
        if bbox is None:
            _, _, bbox = self.specification(index)
        theta = affine_target(row["quad"], bbox, image.shape)
        if self.training:
            if rng.random() < .4:
                gamma = rng.uniform(.65, 1.5)
                lut = np.asarray([255*(v/255.)**gamma for v in range(256)], np.uint8)
                image = cv2.LUT(image, lut)
            if rng.random() < .15:
                image = cv2.GaussianBlur(image, (3, 3), rng.uniform(.2, 1.))
        return box_tensor(image, INPUT_SIZE_V2), row["label"], index, theta, row["subset"]


def collate_v2(items):
    images, labels, indices, thetas, groups = zip(*items, strict=False)
    targets = torch.tensor([CHAR_TO_INDEX[c] for label in labels for c in label], dtype=torch.long)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return (torch.stack(images), targets, lengths, list(labels), list(indices),
            torch.stack(thetas), list(groups))


class BalancedPackedBatches:
    def __init__(self, dataset, subsets, batch_size=128, *, block_size=256):
        if batch_size <= 0 or block_size <= 0:
            raise ValueError("batch and block size must be positive")
        self.dataset, self.subsets = dataset, subsets
        self.batch_size, self.block_size = batch_size, block_size
        self.original_fallbacks = 0
        self.reduced_for_memory = False
        self.plan = None
        self.prepare_seconds = 0.
        self.fallback_seconds = 0.

    def __iter__(self):
        ds = self.dataset
        self.plan = build_balanced_multiplicity(self.subsets, seed=ds.seed, epoch=ds.epoch)
        rng = np.random.default_rng(ds.seed+ds.epoch)
        blocks = rng.permutation(math.ceil(len(ds)/self.block_size))
        batch, counts = [], []
        def jobs():
            for block in blocks:
                if free_ram_bytes() < 2_000_000_000:
                    self.reduced_for_memory = True
                begin = int(block)*self.block_size
                end = min(len(ds), begin+self.block_size)
                quantum = min(64, self.block_size) if self.reduced_for_memory else self.block_size
                for part in range(begin,end,quantum):
                    yield part,min(end,part+quantum)
        with Path(ds.row(0)["cache_pack"]).open("rb") as stream, ThreadPoolExecutor(max_workers=1) as executor:
            total_size = stream.seek(0, 2)
            pending=iter(jobs())
            job=next(pending,None)
            future=None
            while job is not None:
                prepared=future.result() if future is not None else self._prepare(stream,total_size,*job)
                next_job=next(pending,None)
                # One bounded prefetch thread overlaps cached-JPEG decoding with
                # GPU work. No image pool/extra files, and no DataLoader workers.
                future=(executor.submit(self._prepare,stream,total_size,*next_job)
                        if next_job is not None and not self.reduced_for_memory else None)
                begin=job[0]
                for relative in rng.permutation(len(prepared)):
                    relative = int(relative)
                    # shortcut: repeats share this epoch's augmentation. Frozen
                    # backbone BN and per-image GN make count-weighted CTC/MSE
                    # exactly execute the logical repeated sampling multiset.
                    item=prepared[relative]
                    if isinstance(item[0],str) and item[0]=='raw_fallback':
                        _,row,index,photo_rng,bbox=item
                        fallback_started=time.perf_counter()
                        image=row_crop(row,bbox,gpu_fallback=ds.gpu_fallback)
                        item=ds.finish(image,row,index,photo_rng,bbox=bbox)
                        self.fallback_seconds+=time.perf_counter()-fallback_started
                    batch.append(item)
                    counts.append(self.plan.multiplicity[begin+relative])
                    if len(batch)==self.batch_size:
                        yield (*collate_v2(batch),torch.tensor(counts,dtype=torch.float32))
                        batch,counts=[],[]
                job=next_job
        if batch:
            yield (*collate_v2(batch),torch.tensor(counts,dtype=torch.float32))

    def _prepare(self, stream, total_size, begin, end):
        prepare_started=time.perf_counter()
        ds = self.dataset
        rows = [ds.row(index) for index in range(begin, end)]
        offsets = [int(row["pack_offset"]) for row in rows]
        sizes = [int(row["pack_length"]) for row in rows]
        if (any(n <= 0 for n in sizes) or offsets != sorted(offsets)
                or offsets[0] < 0 or offsets[-1]+sizes[-1] > total_size):
            raise ValueError("invalid packed JPEG offsets")
        stream.seek(offsets[0])
        data = stream.read(offsets[-1]+sizes[-1]-offsets[0])
        prepared = []
        for relative, index in enumerate(range(begin, end)):
            row, rng, bbox = ds.specification(index)
            encoded = np.frombuffer(data, np.uint8, count=sizes[relative],
                                    offset=offsets[relative]-offsets[0])
            cached = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if cached is None:
                raise ValueError(f"unreadable packed JPEG: {row['path']}")
            if not cache_contains(row, bbox, BOX_PAD, cached.shape):
                self.original_fallbacks += 1
                # GPU JPEG decode stays on the training thread/default stream.
                # The prefetch worker only touches cached JPEGs and CPU tensors.
                prepared.append(('raw_fallback',row,index,rng,bbox))
            else:
                image=row_crop(row,bbox,cached_image=cached,gpu_fallback=False)
                prepared.append(ds.finish(image,row,index,rng,bbox=bbox))
        self.prepare_seconds+=time.perf_counter()-prepare_started
        return prepared


def validate_inputs(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8-sig"))
    if audit.get("passed") is not True:
        raise ValueError("v2 requires a passed isolation audit")
    hashes = audit["source_hashes"]
    if file_sha256(args.train) != hashes["packed_cache_index_sha256"]:
        raise ValueError("immutable training cache index changed")
    if file_sha256(args.tune) != audit["outputs"]["tune.csv"]["sha256"]:
        raise ValueError("safe tune index changed")
    if file_sha256(args.sampler) != audit["outputs"]["sampler.json"]["sha256"]:
        raise ValueError("empirical box jitter distribution changed")
    for section, keys in (("training", ("train_tune_sha_overlap", "train_protected_sha_overlap",
                                        "train_batch8_holdout_sha_overlap")),
                           ("tune", ("train_tune_sha_overlap", "tune_protected_sha_overlap",
                                     "green_batch8_holdout_sha_overlap"))):
        if any(audit[section][key] != 0 for key in keys):
            raise ValueError("training/tune isolation failed")
    return audit


def run(args):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    profiling = args.profile_batches > 0
    if args.epochs < 60 and not (args.smoke or profiling):
        raise ValueError("v2 release training requires at least 60 complete epochs")
    if args.workers != 0:
        raise ValueError("v2 uses zero DataLoader workers")
    audit = None if args.smoke else validate_inputs(args)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if free_ram_bytes() < 4_000_000_000:
            raise RuntimeError("less than 4GB available; do not start GPU training")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.42, 5_000_000_000/total))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    samples = json.loads(args.sampler.read_text(encoding="utf-8-sig"))
    samples = samples.get("samples_by_subset", samples.get("subsets", samples))
    ds = BoxDatasetV2(args.train, samples, training=True, seed=args.seed,
                      gpu_fallback=device.type=="cuda")
    with args.train.open(encoding="utf-8-sig",newline="") as stream:
        subsets = [row["subset"] for row in csv.DictReader(stream)]
    loader = BalancedPackedBatches(ds, subsets, args.batch_size, block_size=args.block_size)
    plan = build_balanced_multiplicity(subsets, seed=args.seed, epoch=0)
    if audit and plan.unique_samples != audit["training"]["unique_rows"]:
        raise ValueError("full unique training count differs from audit")
    model = BoxNetV2(imagenet=args.imagenet).to(device)
    # Lower STN LR keeps the initially identity grid stable while affine labels
    # teach localization. Recognition loss and annotation loss jointly update it.
    local_ids = {id(p) for p in model.stn.parameters()}
    init_source = None
    if args.init_backbone:
        saved=torch.load(args.init_backbone,map_location='cpu',weights_only=True)
        from lpr.recognize.box_reader import ARCH
        if saved.get('architecture') != ARCH:
            raise ValueError('v2 backbone initialization requires own v1 checkpoint')
        model.features.load_state_dict({k[len('features.'):]:v for k,v in saved['state_dict'].items()
                                        if k.startswith('features.')},strict=True)
        init_source={'path':str(args.init_backbone.resolve()),'sha256':file_sha256(args.init_backbone),
                     'scope':'matching first three residual blocks only; new head/STN random'}
    optimizer = torch.optim.AdamW([
        {"params":model.stn.parameters(), "lr":args.lr*.2},
        {"params":[p for p in model.parameters() if id(p) not in local_ids], "lr":args.lr}],
        weight_decay=1e-4, fused=device.type=="cuda")
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-5)
    ctc = nn.CTCLoss(blank=NUM_CLASSES,zero_infinity=False,reduction="none")
    history=[]
    best=-1.
    best_epoch=0
    ram_peak=0
    started=time.perf_counter()
    input_hashes={"train_sha256":file_sha256(args.train),"tune_sha256":file_sha256(args.tune),
                  "sampler_sha256":file_sha256(args.sampler),"audit_sha256":file_sha256(args.audit) if args.audit else None}
    for epoch in range(args.epochs):
        ds.epoch=epoch
        model.train()
        visits=np.zeros(len(ds),dtype=np.uint16)
        group_counts={g:0 for g in GROUPS}
        seen=0
        physical_seen=0
        loss_sum=0.
        geom_sum=0.
        epoch_started=time.perf_counter()
        for step,(images,targets,lengths,labels,indices,theta,groups,repeats) in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True)
            weights=repeats.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type=="cuda"):
                logits,pred_theta=model(images.to(device),return_theta=True)
            per_example=ctc(logits.float().log_softmax(-1),targets.to(device),
                            torch.full((len(images),),logits.shape[0],dtype=torch.long),lengths)
            recognition=(per_example/lengths.to(device)*weights).sum()/weights.sum()
            geometry=((pred_theta-theta.to(device)).square().mean((1,2))*weights).sum()/weights.sum()
            loss=recognition+args.geometry_weight*geometry
            if not torch.isfinite(loss):
                detail={"epoch":epoch+1,"step":step,"logical_seen":seen,"physical_seen":physical_seen,
                        "input_finite":bool(torch.isfinite(images).all()),"input_max":float(images.abs().max()),
                        "logits_finite":bool(torch.isfinite(logits).all()),"theta_finite":bool(torch.isfinite(pred_theta).all()),
                        "recognition":float(recognition.detach()),"geometry":float(geometry.detach()),
                        "labels":labels,"indices":indices,
                        "nonfinite_parameters":[name for name,p in model.named_parameters() if not torch.isfinite(p).all()]}
                (args.out/'nonfinite-loss.json').write_text(json.dumps(detail,indent=2))
                raise RuntimeError("nonfinite v2 training loss; numeric diagnostics saved")
            loss.backward()
            try:
                nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            except RuntimeError:
                detail={"epoch":epoch+1,"step":step,"logical_seen":seen,"physical_seen":physical_seen,
                        "recognition":float(recognition.detach()),"geometry":float(geometry.detach()),
                        "logits_min":float(logits.detach().min()),"logits_max":float(logits.detach().max()),
                        "input_finite":bool(torch.isfinite(images).all()),"input_max":float(images.abs().max()),
                        "original_fallbacks":loader.original_fallbacks,
                        "bad_gradients":[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()],
                        "indices":indices,"labels":labels}
                (args.out/'nonfinite-gradient.json').write_text(json.dumps(detail,indent=2))
                raise
            optimizer.step()
            draw_counts=repeats.to(dtype=torch.int64).tolist()
            np.add.at(visits,indices,draw_counts)
            for group,count in zip(groups,draw_counts, strict=False):
                group_counts[group]+=count
            draws=sum(draw_counts)
            seen+=draws
            physical_seen+=len(images)
            loss_sum+=float(recognition.detach())*draws
            geom_sum+=float(geometry.detach())*draws
            if step==1 or step%100==0:
                ram=process_ram_bytes()
                ram_peak=max(ram_peak,ram)
                limit=900_000_000 if args.smoke and device.type=="cpu" else 2_550_000_000
                if ram>limit:
                    raise RuntimeError(f"training process RAM over internal reserve: {ram}")
                print(json.dumps({"epoch":epoch+1,"step":step,"seen":seen,
                    "unique_seen":int(np.count_nonzero(visits)),"loss":float(recognition.detach()),
                    "geometry_loss":float(geometry.detach()),"rss":ram,"seconds":time.perf_counter()-started}),flush=True)
            if profiling and step>=args.profile_batches:
                report={"profile_batches":step,"exposures":seen,"unique_sources":int(np.count_nonzero(visits)),"physical_forward_samples":physical_seen,
                    "wall_seconds":time.perf_counter()-started,"seconds_per_exposure":(time.perf_counter()-started)/seen,
                    "training_peak_rss_bytes":process_ram_bytes(peak=True),"discarded_probe_updates":True,
                    "prepare_seconds":loader.prepare_seconds,"fallback_seconds":loader.fallback_seconds,
                    "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved() if device.type=="cuda" else 0,
                    "group_exposures":group_counts,"epoch_unique_pool":len(ds),"full_exposures_per_epoch":plan.total_exposures}
                (args.out/'profile.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(report),flush=True)
                return report
        expected=np.asarray(loader.plan.multiplicity,dtype=np.uint16)
        if not np.array_equal(visits,expected) or group_counts!=dict(loader.plan.group_exposures):
            raise RuntimeError("balanced epoch coverage/counts violated")
        if not args.smoke:
            optimizer.zero_grad(set_to_none=True)
            del images,targets,lengths,theta,repeats,logits,pred_theta,per_example,loss,recognition,geometry
            # Release the 86MB mmap's resident pages before the own-venv CPU
            # scorer starts; next epoch reopens it from the immutable index.
            if ds._map is not None:
                ds._map.close()
                ds._map=None
            if ds._file is not None:
                ds._file.close()
                ds._file=None
            gc.collect()
            if device.type=='cuda':
                torch.cuda.empty_cache()
            payload=checkpoint(model)
            torch.save(payload,args.out/'last.pt')
            scored=subprocess.run([str(args.cpu_python),'-m','lpr_tools.train.score_box_reader_v2',
                '--model',str(args.out/'last.pt'),'--tune',str(args.tune)],check=True,capture_output=True,text=True,encoding='utf-8')
            score=json.loads(scored.stdout)
        else:
            score={"accuracy":0.,"correct":0,"n":0,"smoke_only":True}
        record={"epoch":epoch+1,"samples":seen,"unique_samples":int(np.count_nonzero(visits)),
                "group_exposures":group_counts,"physical_forward_samples":physical_seen,"loss":loss_sum/seen,"geometry_loss":geom_sum/seen,
                "tune":score,"lr":optimizer.param_groups[1]['lr'],"seconds":time.perf_counter()-epoch_started}
        history.append(record)
        if score['accuracy']>best:
            best=score['accuracy']
            best_epoch=epoch+1
            if not args.smoke:
                torch.save(payload,args.out/'best.pt')
        schedule.step()
        report={"recipe":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "history":history,"epochs_completed":len(history),"best_epoch":best_epoch,"best_accuracy":best,
            "best_sha256":file_sha256(args.out/'best.pt') if not args.smoke else None,
            "training_unique_samples":len(ds),"exposures_per_epoch":plan.total_exposures,
            "group_source_counts":dict(plan.group_unique),"total_exposures":sum(r['samples'] for r in history),
            "ram_peak_bytes":max(ram_peak,process_ram_bytes(peak=True)),
            "gpu_reserved_peak":torch.cuda.max_memory_reserved() if device.type=='cuda' else 0,
            "original_gpu_fallbacks":loader.original_fallbacks,"io_reduced_for_low_memory":loader.reduced_for_memory,"prepare_seconds":loader.prepare_seconds,"fallback_seconds":loader.fallback_seconds,
            "wall_seconds":time.perf_counter()-started,"init_source":init_source,
            "resampling_execution":"identical repeated draws collapsed with frozen backbone BN, per-image GroupNorm and unreduced CTC",**input_hashes}
        (args.out/'history.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        mean=sum(r['seconds'] for r in history)/len(history)
        projection={"completed_epochs":len(history),"epoch_seconds":record['seconds'],
                    "mean_epoch_seconds":mean,"projected_total_seconds":mean*args.epochs,
                    "exceeds_six_hours":mean*args.epochs>21600}
        (args.out/'projection.json').write_text(json.dumps(projection,indent=2))
        if epoch==0 and not args.smoke:
            with Path('algorithm/out/receipts/第十一批.md').open('a',encoding='utf-8') as f:
                f.write('\n首个完整epoch：'+json.dumps(projection,ensure_ascii=False)+'\n')
            if projection['exceeds_six_hours']:
                raise RuntimeError('first full epoch projects beyond 6 hours; revise runtime before continuing')
        print(json.dumps(record),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--tune',type=Path,required=True)
    parser.add_argument('--sampler',type=Path,required=True)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--block-size',type=int,default=256)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--geometry-weight',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=11042)
    parser.add_argument('--imagenet',action='store_true')
    parser.add_argument('--init-backbone',type=Path)
    parser.add_argument('--amp',action='store_true',help='CUDA BF16 autocast; CTC loss remains float32')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--profile-batches',type=int,default=0)
    parser.add_argument('--cpu-python',type=Path,default=Path('.venv/Scripts/python.exe').resolve())
    args=parser.parse_args()
    print(json.dumps(run(args),default=str),flush=True)


if __name__=='__main__':
    main()
