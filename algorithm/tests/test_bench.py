import json
import cv2
import numpy as np
import pytest
from lpr.testing import FakeRecognizer
from lpr.io_utils import imwrite
from lpr_tools.data.manifest import ManifestRow, write_manifest
from lpr_tools.eval.common import letterbox, parse_size
from lpr_tools.eval.bench_latency import benchmark, main as latency_main
from lpr_tools.eval.bench_video import benchmark_video, main as video_main
import lpr_tools.eval.bench_latency as bl
import lpr_tools.eval.bench_video as bv


class CountingFake(FakeRecognizer):
    def __init__(self):
        super().__init__();self.shapes=[]
    def recognize(self,image,**kwargs):
        self.shapes.append(image.shape)
        return super().recognize(image,**kwargs)


def test_letterbox_and_warmup():
    image=np.full((1160,720,3),255,np.uint8)
    boxed=letterbox(image,(1280,720))
    assert boxed.shape==(720,1280,3) and not boxed[:,0].any()
    assert boxed[:,640].all()
    recognizer=CountingFake()
    report=benchmark(recognizer,[image],n=3,size=(1280,720))
    assert len(recognizer.shapes)==8 and set(recognizer.shapes)=={(720,1280,3)}
    assert report["n"]==3 and report["warmup"]==5
    assert report["wall_time"]["max_ms"]>=report["wall_time"]["p95_ms"]>=report["wall_time"]["p50_ms"]>0
    assert set(report["stage_ms"])==set(bl.STAGES)
    assert parse_size("1280x720")== (1280,720)
    for size in ("a","0x1","1x-2"):
        with pytest.raises(ValueError):parse_size(size)


def make_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),20.,(160,120))
    assert writer.isOpened()
    for i in range(20):writer.write(np.full((120,160,3),i*10,np.uint8))
    writer.release()


def test_all_frames_decoded_and_sampled(tmp_path):
    path=tmp_path/"synth.mp4";make_video(path)
    recognizer=CountingFake()
    report=benchmark_video(recognizer,path,sample_fps=4)
    assert report["decoded_frames"]==20 and report["sampled_frames"]==4
    assert len(recognizer.shapes)==4 and report["end_to_end_fps"]>0
    assert sum(report["vote_frames_used"])==3 and report["vote_outputs"]==1
    assert benchmark_video(FakeRecognizer(),path,sample_fps=30)["sampled_frames"]==20
    with pytest.raises(ValueError):benchmark_video(FakeRecognizer(),path,sample_fps=0)
    with pytest.raises(ValueError):benchmark_video(FakeRecognizer(),tmp_path/"missing.mp4")


def test_cli_sources(tmp_path,capsys,monkeypatch):
    monkeypatch.setattr(bl,"runtime_info",lambda:{"cpu":"test CPU","torch_threads":2})
    monkeypatch.setattr(bv,"runtime_info",lambda:{"cpu":"test CPU","torch_threads":2})
    p=tmp_path/"image.png";imwrite(p,np.zeros((120,200,3),np.uint8))
    manifest=tmp_path/"m.csv"
    write_manifest(manifest,[ManifestRow(p.name,"陕A12345","blue","clear","test")])
    for source in (["--images",str(p)],["--manifest",str(manifest)],[]):
        assert latency_main(["--fake","--n","2",*source])==0
        assert json.loads(capsys.readouterr().out)["acceptance_eligible"] is False
    path=tmp_path/"synth.mp4";make_video(path)
    assert video_main(["--fake","--video",str(path)])==0
    assert json.loads(capsys.readouterr().out)["decoded_frames"]==20
    for argv in (["--fake","--n","0"],["--fake","--images",str(tmp_path/"missing")]):
        with pytest.raises(SystemExit):latency_main(argv)
