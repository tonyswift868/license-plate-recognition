"""Sampling and character preservation contracts for province continuation."""
import numpy as np
import torch
from lpr.charset import PROVINCES
from lpr.recognize.cnn_model import CharCNN
from lpr_tools.train.train_province_readers import synthetic_targets, local_quad, frozen_batch_statistics


def test_targets_are_uniform_and_different_on_imbalanced_sources():
    labels=['皖A12345']*9000+[p+'A12345' for p in PROVINCES[1:] for _ in range(40)]
    first=synthetic_targets(labels,seed=12042,epoch=0,probability=.35)
    assert np.array_equal(first,synthetic_targets(labels,seed=12042,epoch=0,probability=.35))
    selected=np.flatnonzero(first>=0)
    counts=np.bincount(first[selected],minlength=31)
    assert counts.max()-counts.min()<=1
    assert all(PROVINCES[first[i]]!=labels[i][0] for i in selected)
    assert not np.array_equal(first,synthetic_targets(labels,seed=12042,epoch=1,probability=.35))


def test_quad_translation_uses_clipped_padding_origin():
    quad='10 20 30 20 30 40 10 40'
    assert np.allclose(local_quad(quad,(10,20,20,20)),[[2,3],[22,3],[22,23],[2,23]])
    assert np.allclose(local_quad(quad,(0,0,20,20)),[[10,20],[30,20],[30,40],[10,40]])


def test_province_only_adam_update_preserves_alphanumeric_logits():
    torch.set_num_threads(2);torch.manual_seed(12)
    model=CharCNN()
    for p in model.parameters():p.requires_grad_(False)
    head=model.classifier[-1]
    head.weight.requires_grad_(True);head.bias.requires_grad_(True)
    head.weight.register_hook(lambda g:torch.cat((g[:31],torch.zeros_like(g[31:]))))
    head.bias.register_hook(lambda g:torch.cat((g[:31],torch.zeros_like(g[31:]))))
    model.eval();images=torch.rand(4,1,32,32)
    before=model(images).detach().clone()
    opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=0.)
    torch.nn.functional.cross_entropy(model(images)[:,:31],torch.tensor([0,1,2,3])).backward();opt.step()
    after=model(images).detach()
    assert torch.equal(before[:,31:],after[:,31:])
    assert not torch.equal(before[:,:31],after[:,:31])

def test_missing_glyph_in_jitter_keeps_real_view_and_recovers_only_synthetic(monkeypatch):
    import random
    import lpr_tools.train.train_province_readers as train
    ds=train.ProvinceDataset.__new__(train.ProvinceDataset)
    ds.kind='box';ds.seed=12;ds.epoch=0;ds.gpu_fallback=False;ds.synthetic_gt_fallbacks=0
    ds.synthetic_targets=np.array([1],dtype=np.int16)
    row={'label':'皖A12345','quad':'0 0 100 0 100 40 0 40','subset':'base'}
    jitter=np.full((20,20,3),90,dtype=np.uint8)
    recovered=np.full((50,110,3),110,dtype=np.uint8)
    calls=[]
    def replace(image,quad,label,target,rng):
        calls.append(image.shape)
        if image.shape==jitter.shape:raise ValueError('province glyph slot falls outside source image')
        return image.copy(),target+label[1:],{}
    monkeypatch.setattr(train,'replace_province',replace)
    monkeypatch.setattr(train,'row_crop',lambda row,bbox,**kwargs:recovered)
    result=ds.finish(jitter,row,0,random.Random(7),bbox=(80,0,20,20))
    assert len(result)==2 and not result[0][-1] and result[1][-1]
    assert result[0][1]=='皖A12345' and result[1][1]=='沪A12345'
    assert ds.synthetic_gt_fallbacks==1 and calls==[jitter.shape,recovered.shape]
    expected=train.affine_target(row['quad'],(0,0,100,40),recovered.shape)
    assert torch.equal(result[1][3],expected)
