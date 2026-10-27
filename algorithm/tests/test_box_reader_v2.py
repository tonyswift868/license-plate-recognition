from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from lpr.charset import NUM_CLASSES
from lpr.recognize import load_box_reader, write_manifest_entry
from lpr.recognize.box_reader import BoxNet, BoxPlateReader, checkpoint, model_from_checkpoint
from lpr.recognize.box_reader_v2 import ARCH_V2, INPUT_SIZE_V2, BoxNetV2
from lpr_tools.train.box_data_v2 import GROUPS
from lpr_tools.train.train_box_reader_v2 import BalancedPackedBatches, affine_target


def test_v2_stn_identity_shape_grad_and_checkpoint_factory(tmp_path):
    torch.set_num_threads(2); torch.manual_seed(11)
    model=BoxNetV2()
    images=torch.randn(2,3,64,192)
    logits,theta=model(images,return_theta=True)
    assert logits.shape==(24,2,NUM_CLASSES+1)
    assert torch.equal(theta,torch.tensor([[[1.,0.,0.],[0.,1.,0.]]]).expand(2,-1,-1))
    logits.square().mean().backward()
    assert torch.isfinite(model.stn.regressor[-1].weight.grad).all()
    assert model.stn.regressor[-1].weight.grad.abs().sum()>0
    saved=checkpoint(model,temperature=1.1)
    assert saved['architecture']==ARCH_V2 and saved['input_size']==list(INPUT_SIZE_V2)
    path=tmp_path/'box_reader_v2.pt'; torch.save(saved,path)
    write_manifest_entry(path,version='box_reader_v2',train_data='unit',val_char_acc=None,input_size=INPUT_SIZE_V2)
    reader=load_box_reader(path)
    assert reader.input_size==INPUT_SIZE_V2 and reader.temperature==1.1
    assert reader.read(np.zeros((0,0,3),np.uint8))==('',[])
    assert path.stat().st_size<20_000_000


def test_v1_architecture_remains_loadable_and_dimension_mismatch_rejected():
    torch.set_num_threads(2)
    source=BoxNet().eval(); saved=checkpoint(source)
    loaded=model_from_checkpoint(saved).eval()
    x=torch.randn(1,3,48,160)
    with torch.inference_mode(): assert torch.equal(source(x),loaded(x))
    saved['input_size']=[192,64]
    with pytest.raises(ValueError,match='input_size'): model_from_checkpoint(saved)
    saved['architecture']='unregistered'
    with pytest.raises(ValueError,match='unsupported'): model_from_checkpoint(saved)


def test_affine_supervision_preserves_tilt_and_pixel_center_origin():
    target=affine_target('0 0 100 20 100 60 0 40',(0,0,100,60),(60,100,3),pad=(0,0))
    assert target.numpy()==pytest.approx(np.float32([[1.08,0,.01],[.36,.7466667,1/60]]),abs=1e-6)
    with pytest.raises(ValueError,match='nonempty'):
        affine_target('0 0 100 0 100 40 0 40',(0,0,100,40),(0,100,3))


def test_balanced_loader_draws_real_repeats_but_prepares_every_source_once(tmp_path,monkeypatch):
    subsets=['base','base',*GROUPS[1:]]
    pack=tmp_path/'fake.tar'; pack.write_bytes(b'unit')
    class Dataset:
        seed=11; epoch=0
        def __len__(self): return len(subsets)
        def row(self,index): return {'cache_pack':str(pack)}
    loader=BalancedPackedBatches(Dataset(),subsets,batch_size=4,block_size=4)
    prepared=[]
    def prepare(stream,size,begin,end):
        prepared.extend(range(begin,end))
        return [(torch.full((3,64,192),float(i)),'皖A12345',i,torch.eye(2,3),subsets[i]) for i in range(begin,end)]
    monkeypatch.setattr(loader,'_prepare',prepare)
    monkeypatch.setattr('lpr_tools.train.train_box_reader_v2.free_ram_bytes',lambda:0)
    counts=Counter(); groups=Counter()
    for _,_,_,_,indices,_,names,repeats in loader:
        for i,g,n in zip(indices,names,repeats.tolist()): counts[i]+=int(n); groups[g]+=int(n)
    assert sorted(prepared)==list(range(len(subsets)))
    assert [counts[i] for i in range(len(subsets))]==list(loader.plan.multiplicity)
    assert dict(groups)==dict(loader.plan.group_exposures)
    assert sum(counts.values())==18 and loader.reduced_for_memory


def test_repeated_source_compression_matches_full_training_network():
    torch.set_num_threads(2); torch.manual_seed(13)
    compressed=BoxNetV2().train(); expanded=BoxNetV2().train()
    expanded.load_state_dict(compressed.state_dict())
    assert all(not m.training for m in compressed.features.modules() if isinstance(m,torch.nn.BatchNorm2d))
    x=torch.randn(2,3,64,192); counts=torch.tensor([3.,1.]); order=torch.tensor([0,0,0,1])
    a=compressed(x); b=expanded(x[order])
    assert torch.allclose(a[:,order],b,atol=1e-5,rtol=1e-4)
    loss_a=(a.square().mean((0,2))*counts).sum()/counts.sum()
    loss_b=b.square().mean(); loss_a.backward(); loss_b.backward()
    for (name,p),(other,q) in zip(compressed.named_parameters(),expanded.named_parameters()):
        assert name==other
        assert torch.allclose(p.grad,q.grad,atol=2e-5,rtol=1e-3),name
    for name,value in compressed.named_buffers():
        assert torch.equal(value,dict(expanded.named_buffers())[name]),name
