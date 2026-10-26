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


@pytest.fixture(autouse=True)
def isolate_bench_config(tmp_path, monkeypatch):
    config = tmp_path / "isolated-bench.toml"
    config.write_text('[runtime]\nmodel_path = ""\n', encoding="utf-8")
    monkeypatch.setenv("LPR_CONFIG", str(config))


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


def test_bench_reports_crnn_accuracy_only_with_aligned_ground_truth():
    class SelectedRecognizer:
        version = "test"
        def recognize(self, image):
            self.last_selection = {"source": "color", "crnn_selected": True}
            return type("Result", (), {"status": type("Status", (), {"value": "OK"})(),
                                        "plate_no": "A", "stage_ms": {}})()
    images = [np.zeros((12, 24, 3), np.uint8), np.ones((12, 24, 3), np.uint8)]
    report = benchmark(SelectedRecognizer(), images, n=2, size=(24, 12), truths=["A", "B"])
    attribution = report["network_attribution"]
    assert attribution["crnn_selected_count"] == 2
    assert attribution["crnn_selected_correct_count"] == 1
    assert attribution["crnn_selected_accuracy"] == .5
    unknown = benchmark(SelectedRecognizer(), images, n=2, size=(24, 12))["network_attribution"]
    assert unknown["crnn_selected_accuracy"] is None
    assert unknown["crnn_selected_correct_count"] is None
    assert unknown["crnn_selected_accuracy_status"] == "unknown_no_ground_truth"


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
    monkeypatch.setattr(bl,"runtime_info",lambda cfg=None:{"cpu":"test CPU","torch_threads":2})
    monkeypatch.setattr(bv,"runtime_info",lambda cfg=None:{"cpu":"test CPU","torch_threads":2})
    p=tmp_path/"image.png";imwrite(p,np.zeros((120,200,3),np.uint8))
    manifest=tmp_path/"m.csv"
    write_manifest(manifest,[ManifestRow(p.name,"陕A12345","blue","clear","test")])
    for source in (["--images",str(p)],["--manifest",str(manifest)],[]):
        assert latency_main(["--fake","--n","2",*source])==0
        report=json.loads(capsys.readouterr().out)
        assert report["acceptance_eligible"] is False
        if source:
            assert (report["requested"],report["decoded"],report["skipped"])==(1,1,0)
        else:
            assert (report["requested"],report["decoded"],report["skipped"])==(0,0,0)
    path=tmp_path/"synth.mp4";make_video(path)
    custom = tmp_path / "video-config.toml"
    custom.write_text('[runtime]\ntorch_threads = 3\nmodel_path = ""\n', encoding="utf-8")
    monkeypatch.setattr(bv, "runtime_info", lambda cfg: {"cpu": "test CPU", "torch_threads": cfg.runtime.torch_threads})
    assert video_main(["--fake","--video",str(path),"--config",str(custom)])==0
    video_report = json.loads(capsys.readouterr().out)
    assert video_report["decoded_frames"] == 20 and video_report["torch_threads"] == 3
    with pytest.raises(SystemExit):latency_main(["--fake","--n","0"])
    assert latency_main(["--fake","--images",str(tmp_path/"missing")])==0
    report=json.loads(capsys.readouterr().out)
    assert (report["requested"],report["decoded"],report["skipped"])==(0,0,0)
    assert report["n"]==0

    missing=tmp_path/'missing-only.csv'
    write_manifest(missing,[ManifestRow('absent.jpg','陕A12345','blue','clear','test')])
    monkeypatch.setattr(bl,'make_recognizer',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('all bad must not load model')))
    assert latency_main(['--manifest',str(missing)])==0
    report=json.loads(capsys.readouterr().out)
    assert (report['requested'],report['decoded'],report['skipped'])==(1,0,1)
    assert report['n']==0 and report['warmup']==0 and report['acceptance_eligible'] is False


def test_cli_skips_undecodable_and_reports_empty_inputs(tmp_path,capsys,monkeypatch):
    monkeypatch.setattr(bl,"runtime_info",lambda cfg=None:{"cpu":"test CPU","torch_threads":2})
    good=tmp_path/"good.png";imwrite(good,np.zeros((32,48,3),np.uint8))
    bad=tmp_path/"bad.jpg";bad.write_bytes(b"not an image")
    manifest=tmp_path/"mixed.csv"
    write_manifest(manifest,[ManifestRow(good.name,"陕A12345","blue","clear","test"),
                             ManifestRow(bad.name,"陕A12345","blue","clear","test")])
    assert latency_main(["--fake","--manifest",str(manifest),"--n","2"])==0
    report=json.loads(capsys.readouterr().out)
    assert (report["requested"],report["decoded"],report["skipped"])==(2,1,1)
    assert report["n"]==2 and report["decode_errors"][0]=={"path":str(bad),"reason":"decode_failed"}

    all_bad=tmp_path/"bad-only.csv"
    write_manifest(all_bad,[ManifestRow(bad.name,"陕A12345","blue","clear","test")])
    assert latency_main(["--fake","--manifest",str(all_bad)])==0
    report=json.loads(capsys.readouterr().out)
    assert (report["requested"],report["decoded"],report["skipped"])==(1,0,1)
    assert report["n"]==0 and report["wall_time"] is None

    empty=tmp_path/"empty.csv";write_manifest(empty,[])
    assert latency_main(["--fake","--manifest",str(empty)])==0
    report=json.loads(capsys.readouterr().out)
    assert (report["requested"],report["decoded"],report["skipped"])==(0,0,0)
    assert report["n"]==0
