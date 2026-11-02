"""Continue own CRNN v2 with measured crop errors; inference architecture is unchanged."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from lpr.charset import CHAR_TO_INDEX
from lpr.recognize import file_sha256, write_manifest_entry
from lpr.recognize.crnn import BLANK, INPUT_SIZE, CrnnNet, plate_tensor
from lpr_tools.data.ccpd import parse_ccpd_name
from lpr_tools.train import train_crnn_v2 as v2
from lpr_tools.train.train_crnn import _collate, _worker_init
from torch import nn
from torch.utils.data import DataLoader, Dataset

SUBSETS = v2.SUBSETS
MIXTURE = {**{s: .62 / 7 for s in SUBSETS}, 'base7': .12, 'real8': .18, 'font': .08}
UNIT = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
EMPIRICAL_PROBABILITY = .5


def sampling_weights(rows):
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[row.group, row.gt].append(i)
    if {k[0] for k in strata} != set(MIXTURE):
        raise ValueError('all sampling groups must exist')
    weights = torch.zeros(len(rows), dtype=torch.double)
    for (group, gt), positions in strata.items():
        counts = Counter(rows[i].text[0] for i in positions)
        raw = torch.tensor([min(4., (max(counts.values()) / counts[rows[i].text[0]]) ** .5)
                            for i in positions], dtype=torch.double)
        gt_share = .30 if group in SUBSETS else .65
        source_share = (gt_share if gt else 1 - gt_share) if (group, not gt) in strata else 1.
        weights[positions] = raw / raw.sum() * MIXTURE[group] * source_share
    return weights


def validate_audit(index_dir):
    audit_path = index_dir / 'audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    for key in ('train', 'tune', 'val'):
        if file_sha256(index_dir / (key + '.csv')) != audit['output_sha256'][key]:
            raise ValueError('audited index changed: ' + key)
    checks = audit['content_checks']
    pairs = checks.get('partition_pairwise_sha256_overlap', {})
    if any(pairs.get(k) != 0 for k in ('train_tune','train_val','tune_val')) or any(
        checks.get(k) != 0 for k in ('train_selected_sha_overlap','fit_report_sha_overlap',
        'selected_sha_seen_by_v1','protected_hash_overlap_train','protected_hash_overlap_tune')):
        raise ValueError('data leakage audit failed')
    return audit


def load_empirical(path, index_dir):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if (data.get('schema') != 'crop-error-v1' or data.get('frame') != 'GT canonical unit square'
            or data.get('corner_order') != ['LT','RT','RB','LB'] or data.get('fit_partition') != 'train only'):
        raise ValueError('unknown or unsafe empirical geometry frame/partition')
    hashes = data.get('indices_sha256', {})
    if hashes.get('safe_train_csv') != file_sha256(index_dir / 'train.csv') or hashes.get(
        'batch8_data_audit_json') != file_sha256(index_dir / 'audit.json'):
        raise ValueError('empirical geometry uses different safe data')
    groups = {}
    for subset in SUBSETS:
        values = np.asarray(data['subsets'][subset], dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (4,2) or not len(values) or not np.isfinite(values).all():
            raise ValueError('invalid empirical corner vectors: ' + subset)
        if not all(cv2.isContourConvex(np.float32(UNIT + delta)) for delta in values):
            raise ValueError('nonconvex empirical crop')
        groups[subset] = values
    return groups


def original_lookup(index_path):
    result = {}
    with index_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            is_gt = row['source'] == 'ground_truth' or row['source'].endswith(':ground_truth')
            if not is_gt or row['subset'] not in SUBSETS:
                continue
            if row['partition'] != 'train' or not row.get('orig_path'):
                raise ValueError('empirical augmentation requires a safe training original')
            record = parse_ccpd_name(row['orig_path'])
            if record.plate_no != row['plate_no']:
                raise ValueError('original name and crop label differ')
            quad = np.float32(record.quad)
            if cv2.isContourConvex(quad) and abs(cv2.contourArea(quad)) > 1:
                result[row['crop_path']] = (row['orig_path'], quad)
    return result


def empirical_crop(image, gt_quad, delta, width):
    """Sample all four correlated corners on the original, retaining real background."""
    if (np.shape(gt_quad) != (4,2) or np.shape(delta) != (4,2)
            or not np.isfinite(gt_quad).all() or not np.isfinite(delta).all()):
        raise ValueError('invalid crop geometry')
    canonical = np.float32(UNIT + delta)
    if not cv2.isContourConvex(canonical):
        raise ValueError('sampled geometry must be convex')
    to_original = cv2.getPerspectiveTransform(UNIT, np.float32(gt_quad))
    quad = cv2.perspectiveTransform(canonical[None], to_original)[0]
    if not np.isfinite(quad).all() or abs(cv2.contourArea(quad)) <= 1:
        raise ValueError('singular empirical crop')
    target = UNIT * np.float32([width - 1, 79])
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(image, matrix, (width,80), borderMode=cv2.BORDER_CONSTANT)


class Crops(Dataset):
    def __init__(self, rows, *, draws=None, epoch=0, seed=42, empirical=None, originals=None):
        self.rows, self.draws, self.epoch, self.seed = rows, draws, epoch, seed
        self.empirical, self.originals = empirical, originals

    def __len__(self):
        return len(self.rows) if self.draws is None else len(self.draws)

    def __getitem__(self, i):
        row = self.rows[i if self.draws is None else self.draws[i]]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i]))
        geometry = False
        if (self.draws is not None and row.gt and row.group in SUBSETS
                and row.path in self.originals and rng.random() < EMPIRICAL_PROBABILITY):
            original, gt_quad = self.originals[row.path]
            image = cv2.imdecode(np.fromfile(original,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('unreadable safe original: ' + original)
            delta = self.empirical[row.group][int(rng.integers(len(self.empirical[row.group])))]
            image = empirical_crop(image,gt_quad,delta,276 if len(row.text)==8 else 252)
            geometry = True
        else:
            image = cv2.imdecode(np.fromfile(row.path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape != (80,276 if len(row.text)==8 else 252,3):
            raise ValueError('invalid native plate: ' + row.path)
        if self.draws is not None:
            image = v2.augment(image,rng,geometry=False)
        target = torch.tensor([CHAR_TO_INDEX[c] for c in row.text],dtype=torch.long)
        source = f"{row.group}/{'GT' if row.gt else 'pipeline'}/layout{len(row.text)}"
        return plate_tensor(image),target,len(target),source+('/empirical' if geometry else '')


def eligible(tune, val, baseline_tune, baseline_val):
    # Acceptance reporting only; it must not alter the dispatch's v2 checkpoint ranking.
    return (tune['GT']['correct'] >= baseline_tune['GT']['correct']
            and val['all']['accuracy'] >= baseline_val['all']['accuracy']-.005
            and val['layout8']['accuracy'] >= baseline_val['layout8']['accuracy']-.005)


def select_best_trained(history):
    if not history:
        raise ValueError('no trained checkpoints')
    return max(history,key=lambda r:v2.selection_score(r['tune'],r['val']))


def stop_reason(history, baseline_pipeline, maximum):
    if len(history) == 4 and history[-1]['tune']['pipeline']['accuracy'] - baseline_pipeline < .03 - 1e-12:
        return 'four-round stop loss: pipeline gain below 3 percentage points'
    if len(history) >= 4 and history[-1]['eligible'] and history[-1]['tune']['pipeline']['accuracy'] >= .82:
        return 'candidate target reached with GT and safe-validation non-regression'
    return 'predeclared round budget reached' if len(history) >= maximum else None


def train(index_dir, fonts, init, empirical_path, out, *, epochs=8, samples_per_epoch=65536,
          batch_size=128, lr=.0001, seed=142, num_workers=1, resume=False, stop_after=None):
    if (epochs < 1 or samples_per_epoch < 1 or batch_size < 1 or not 0 < lr <= .01
            or not 0 <= num_workers <= 2 or (stop_after is not None and stop_after < 1)):
        raise ValueError('invalid training limits')
    if out.resolve() == init.resolve() or init.resolve() == (out/'best.pt').resolve():
        raise ValueError('initializer cannot be overwritten')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    audit = validate_audit(index_dir)
    empirical = load_empirical(empirical_path,index_dir)
    originals = original_lookup(index_dir/'train.csv')
    rows = v2.read_index(index_dir/'train.csv','train') + v2.read_fonts(fonts)
    tune_rows, val_rows = v2.read_index(index_dir/'tune.csv','tune'),v2.read_index(index_dir/'val.csv','val')
    weights = sampling_weights(rows)
    metadata = dict(version='crnn_v3_raw',epochs=epochs,samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,lr=lr,seed=seed,num_workers=num_workers,torch_threads=4,opencv_threads=1,
        initialization=dict(path=str(init.resolve()),sha256=file_sha256(init),kind='own_crnn_v2_continuation'),
        indices_sha256=audit['output_sha256'],data_audit_sha256=file_sha256(index_dir/'audit.json'),
        empirical_path=str(empirical_path.resolve()),empirical_sha256=file_sha256(empirical_path),
        empirical_counts={k:len(v) for k,v in empirical.items()},eligible_original_crops=len(originals),empirical_probability=EMPIRICAL_PROBABILITY,
        synthetic_labels_sha256=file_sha256(fonts/'labels.csv'),mixture=MIXTURE,
        hard_gt_share=.30,other_real_gt_share=.65,
        train_rows=len(rows),tune_rows=len(tune_rows),val_rows=len(val_rows),
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[3])).replace('\\','/'):file_sha256(p)
            for p in (Path(__file__).resolve(),Path(v2.__file__).resolve(),Path(__file__).with_name('train_crnn.py').resolve(),Path(__file__).resolve().parents[2]/'lpr/recognize/crnn.py')},
        selection='unchanged v2 lexicographic minimum GT, macro GT, safe val exact; trained rounds only; GT/green acceptance reported separately',
        stop_loss=f'after round4, current pipeline gain <.03; otherwise maximum{epochs}, or pipeline>=.82 with eligibility',
        learning_rate_schedule='initial LR through round4, half from round5')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint,best_path,metadata_path = out/'checkpoint.pt',out/'best.pt',out/'metadata.json'
    if not resume and any(p.exists() for p in (checkpoint,best_path,metadata_path)):
        raise FileExistsError('training outputs already exist')
    model = CrnnNet()
    state = dict(torch.load(init,map_location='cpu',weights_only=True))
    state.pop('ctc_temperature',None)
    model.load_state_dict(state,strict=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=BLANK,zero_infinity=False)
    if resume:
        old = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if old['metadata'] != metadata:
            raise ValueError('resume provenance differs')
        model.load_state_dict(old['model'],strict=True)
        optimizer.load_state_dict(old['optimizer'])
        history,start,best_score,best_epoch = old['history'],old['epoch'],tuple(old['best_score']),old['best_epoch']
        baseline_tune,baseline_val=old['baseline_tune'],old['baseline_val']
        torch.set_rng_state(old['torch_rng'])
    else:
        baseline_tune,baseline_val=v2.evaluate(model,tune_rows),v2.evaluate(model,val_rows)
        history,start,best_score,best_epoch=[],0,(-1.,-1.,-1.),0
        (out/'baseline.json').write_text(json.dumps(dict(tune=baseline_tune,val=baseline_val),indent=2),encoding='utf-8')
    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    end = epochs if stop_after is None else min(epochs,start+stop_after)
    reason = stop_reason(history,baseline_tune['pipeline']['accuracy'],epochs)
    for epoch in range(start,end):
        if reason:
            break
        began=time.perf_counter()
        current_lr=lr*(1. if epoch<4 else .5)
        for group in optimizer.param_groups:
            group['lr']=current_lr
        draws=torch.multinomial(weights,samples_per_epoch,replacement=True,
            generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        loader=DataLoader(Crops(rows,draws=draws,epoch=epoch,seed=seed,empirical=empirical,originals=originals),
            batch_size=batch_size,num_workers=num_workers,collate_fn=_collate,
            worker_init_fn=_worker_init if num_workers else None,
            generator=torch.Generator().manual_seed(seed+epoch+100000))
        counts,losses=Counter(),[]
        model.train()
        for step,(images,targets,lengths,sources) in enumerate(loader,1):
            counts.update(sources)
            optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=criterion(logits.log_softmax(2),targets,torch.full((len(images),),logits.shape[0],dtype=torch.long),lengths)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite CTC loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step%50==0:
                print(json.dumps(dict(epoch=epoch+1,step=step,loss=sum(losses)/len(losses),seconds=time.perf_counter()-began)),flush=True)
        if sum(counts.values()) != samples_per_epoch:
            raise RuntimeError('draw count mismatch')
        tune,val=v2.evaluate(model,tune_rows),v2.evaluate(model,val_rows)
        qualifies=eligible(tune,val,baseline_tune,baseline_val)
        score=v2.selection_score(tune,val)
        if score>best_score:
            best_score,best_epoch=score,epoch+1
            v2._atomic_save(model.state_dict(),best_path)
        record=dict(epoch=epoch+1,loss=sum(losses)/len(losses),lr=current_lr,tune=tune,val=val,
            eligible=qualifies,actual_draw_counts=dict(counts),epoch_seconds=time.perf_counter()-began,best_epoch=best_epoch)
        history.append(record)
        reason=stop_reason(history,baseline_tune['pipeline']['accuracy'],epochs)
        v2._atomic_save(dict(metadata=metadata,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
            epoch=epoch+1,best_score=best_score,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),
            baseline_tune=baseline_tune,baseline_val=baseline_val),checkpoint)
        (out/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(record,ensure_ascii=False),flush=True)
    result=dict(best_epoch=best_epoch,best_score=best_score,epochs_completed=len(history),
        stop_reason=reason or 'requested checkpoint pause',best_sha256=file_sha256(best_path))
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    selected=history[best_epoch-1]['val'] if best_epoch else baseline_val
    write_manifest_entry(best_path,version='crnn_v3_raw',train_data=json.dumps(metadata,ensure_ascii=False),
        val_char_acc=selected['characters']['accuracy'],input_size=INPUT_SIZE)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index-dir','fonts','init','empirical','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--samples-per-epoch',type=int,default=65536)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=.0001)
    parser.add_argument('--num-workers',type=int,default=1)
    parser.add_argument('--seed',type=int,default=142)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    result=train(args.index_dir,args.fonts,args.init,args.empirical,args.out,epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,batch_size=args.batch_size,lr=args.lr,seed=args.seed,
        num_workers=args.num_workers,resume=args.resume,stop_after=args.stop_after)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
