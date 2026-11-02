import json

import cv2
import numpy as np
import pytest
from lpr.testing import FakeRecognizer
from lpr_tools.eval import bench_video as bv


@pytest.mark.parametrize('threads', [0, -1, True, 1.5])
def test_video_rejects_invalid_decoder_threads(threads):
    with pytest.raises(ValueError, match='positive integer'):
        bv.benchmark_video(FakeRecognizer(), 'unused.mp4', decoder_threads=threads)


def test_video_explicit_threads_real_capture_and_cli(tmp_path, monkeypatch, capsys):
    path=tmp_path/'small.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25.,(160,120))
    assert writer.isOpened()
    try:
        for _ in range(5):
            writer.write(np.zeros((120,160,3),np.uint8))
    finally:
        writer.release()
    result=bv.benchmark_video(FakeRecognizer(),path,decoder_threads=1)
    assert (result['decoded_frames'],result['sampled_frames'])==(5,1)
    assert result['decoder_threads']==1 and result['decoder_backend']=='FFMPEG'
    monkeypatch.setattr(bv,'runtime_info',lambda cfg:{'cpu':'test','torch_threads':2})
    assert bv.main(['--fake','--video',str(path),'--decoder-threads','1'])==0
    assert json.loads(capsys.readouterr().out)['decoder_threads']==1


def test_video_refuses_unapplied_decoder_limit_and_releases(monkeypatch):
    class Capture:
        released=False
        def isOpened(self):return True
        def get(self,prop):return 2
        def release(self):self.released=True
        def read(self):raise AssertionError('must fail before decoding or recognition')
    cap=Capture()
    calls=[]
    def opened(*args):
        calls.append(args)
        return cap
    monkeypatch.setattr(bv.cv2,'VideoCapture',opened)
    with pytest.raises(ValueError,match='thread limit not applied'):
        bv.benchmark_video(FakeRecognizer(),'input.mp4',decoder_threads=1)
    assert calls==[('input.mp4',cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])]
    assert cap.released
