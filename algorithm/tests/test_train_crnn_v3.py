from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from lpr.recognize import file_sha256
from lpr_tools.train import train_crnn_v2 as v2
from lpr_tools.train import train_crnn_v3 as v3
from .test_train_crnn_v2 import _make_audited_data, _assert_tree_equal


def test_sampling_retains_green_share_and_increases_hard_candidates():
    rows=[]
    for group in v3.MIXTURE:
        rows.append(v2.Sample('a','皖A12345',group,False))
        if group!='font': rows.append(v2.Sample('b','皖A12345',group,True))
    weights=v3.sampling_weights(rows)
    total=defaultdict(float); gt=defaultdict(float)
    for row,w in zip(rows,weights.tolist()):
        total[row.group]+=w
        if row.gt: gt[row.group]+=w
    assert sum(total.values())==pytest.approx(1)
    for group in v3.MIXTURE:
        assert total[group]==pytest.approx(v3.MIXTURE[group])
        if group!='font': assert gt[group]/total[group]==pytest.approx(.30 if group in v3.SUBSETS else .65)
    assert sum(total[s]-gt[s] for s in v3.SUBSETS)==pytest.approx(.434)
    assert total['real8']==pytest.approx(.18)


def test_correlated_reprojection_retains_real_background_and_rejects_folded_quad():
    image=np.full((100,200,3),[10,20,30],dtype=np.uint8)
    image[30:71,50:151]=[210,220,230]
    quad=np.float32([[50,30],[150,30],[150,70],[50,70]])
    clean=v3.empirical_crop(image,quad,np.zeros((4,2),np.float32),252)
    delta=np.float32([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]])
    expanded=v3.empirical_crop(image,quad,delta,252)
    assert clean.shape==expanded.shape==(80,252,3)
    assert np.array_equal(clean[0,0],[210,220,230])
    assert np.array_equal(expanded[0,0],[10,20,30])
    assert np.array_equal(expanded[40,126],[210,220,230])
    folded=np.float32([[0,0],[-1,1],[1,-1],[0,0]])
    with pytest.raises(ValueError): v3.empirical_crop(image,quad,folded,252)


def test_photometric_reuse_can_disable_old_mild_geometry(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('old geometry must not run')
    monkeypatch.setattr(v2.cv2,'warpPerspective',forbidden)
    image=np.full((80,252,3),120,np.uint8)
    a=v2.augment(image,np.random.default_rng(1),geometry=False)
    b=v2.augment(image,np.random.default_rng(1),geometry=False)
    assert np.array_equal(a,b) and a.dtype==np.uint8


def test_four_round_stop_uses_exact_percentage_point_gain():
    base=.7059773828756059
    def records(gain):
        return [dict(tune={'pipeline':{'accuracy':base+gain}},eligible=True) for _ in range(4)]
    assert 'below 3' in v3.stop_reason(records(.0299),base,8)
    assert v3.stop_reason(records(.03),base,8) is None
    assert v3.stop_reason(records(.01)[:3],base,8) is None


def test_eligibility_preserves_clean_gt_and_safe_green():
    base_t={'GT':{'correct':1805}}
    base_v={'all':{'accuracy':.928},'layout8':{'accuracy':.82}}
    assert v3.eligible(base_t,base_v,base_t,base_v)
    assert not v3.eligible({'GT':{'correct':1804}},base_v,base_t,base_v)
    bad={'all':{'accuracy':.93},'layout8':{'accuracy':.814}}
    assert not v3.eligible(base_t,bad,base_t,base_v)


def prepare(root):
    index,fonts,init=_make_audited_data(root)
    # Test originals are separate from native crops, with valid CCPD metadata and real context.
    name='001-0_0-50&30_150&70-150&70_50&70_50&30_150&30-0_0_25_26_27_28_29-80-0.jpg'
    original=root/name
    image=np.full((100,200,3),[20,30,40],np.uint8)
    cv2.imencode('.jpg',image)[1].tofile(original)
    path=index/'train.csv'
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f); fields=list(reader.fieldnames); rows=list(reader)
    for row in rows: row['orig_path']=str(original) if row['subset'] else ''
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields+['orig_path']);w.writeheader();w.writerows(rows)
    # Exercise both real candidate scoring and the safe eight-character guard.
    for part in ('tune','val'):
        target=index/(part+'.csv')
        with target.open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f); columns=list(reader.fieldnames); values=list(reader)
        extra=dict(values[0])
        if part=='tune':
            extra['source']='pipeline_candidate:0:0'
        else:
            crop=root/'green_val.png'
            cv2.imencode('.png',np.full((80,276,3),100,np.uint8))[1].tofile(crop)
            extra.update(plate_no='皖AD12345',layout='8',crop_path=str(crop))
        values.append(extra)
        with target.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(values)
    audit=json.loads((index/'audit.json').read_text(encoding='utf-8'))
    audit['output_sha256']={key:file_sha256(index/(key+'.csv')) for key in ('train','tune','val')}
    (index/'audit.json').write_text(json.dumps(audit),encoding='utf-8')
    data=dict(schema='crop-error-v1',frame='GT canonical unit square',corner_order=['LT','RT','RB','LB'],fit_partition='train only',
        indices_sha256={'safe_train_csv':file_sha256(path),'batch8_data_audit_json':file_sha256(index/'audit.json')},
        subsets={s:[np.zeros((4,2)).tolist()] for s in v3.SUBSETS})
    empirical=root/'empirical.json'; empirical.write_text(json.dumps(data),encoding='utf-8')
    return index,fonts,init,empirical


def test_sampler_rejects_wrong_partition_or_data_hash(tmp_path):
    index,fonts,init,p=prepare(tmp_path)
    v3.load_empirical(p,index)
    data=json.loads(p.read_text());data['fit_partition']='devhard'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError): v3.load_empirical(p,index)
    data['fit_partition']='train only';data['indices_sha256']['safe_train_csv']='0'*64
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError): v3.load_empirical(p,index)


def test_resume_replays_model_optimizer_rng_and_baseline_selection(tmp_path):
    index,fonts,init,p=prepare(tmp_path)
    kwargs=dict(epochs=2,samples_per_epoch=16,batch_size=8,num_workers=0)
    whole,split=tmp_path/'whole',tmp_path/'split'
    v3.train(index,fonts,init,p,whole,**kwargs)
    v3.train(index,fonts,init,p,split,stop_after=1,**kwargs)
    v3.train(index,fonts,init,p,split,resume=True,**kwargs)
    a=torch.load(whole/'checkpoint.pt',weights_only=True)
    b=torch.load(split/'checkpoint.pt',weights_only=True)
    for key in ('model','optimizer','torch_rng','baseline_tune','baseline_val','best_score','best_epoch'):
        _assert_tree_equal(a[key],b[key])
    assert a['epoch']==b['epoch']==2



def test_audit_rejects_partition_overlap(tmp_path):
    index,fonts,init,p=prepare(tmp_path)
    v3.validate_audit(index)
    path=index/'audit.json';data=json.loads(path.read_text())
    data['content_checks']['partition_pairwise_sha256_overlap']['train_tune']=1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='leakage'): v3.validate_audit(index)
