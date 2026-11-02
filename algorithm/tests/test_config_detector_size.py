import pytest
from lpr.config import load_config
from lpr.errors import ConfigError


@pytest.mark.parametrize('size',[0,128,256,384,512,1024])
def test_detector_size_valid(size):
    assert load_config(runtime__detector_input_size=size).runtime.detector_input_size==size

@pytest.mark.parametrize('size',[-32,1,127,129,255,1056,True,False,256.0,'256',None])
def test_detector_size_invalid(size):
    with pytest.raises(ConfigError):
        load_config(runtime__detector_input_size=size)

@pytest.mark.parametrize('size',[0,384,512])
def test_factory_preserves_verified_weights_and_passes_size(monkeypatch,size):
    import lpr.pipeline as pipeline
    import lpr.plate_det as module
    class Detector:
        def __init__(self,model,*,version,image_size,score_threshold):
            self.model,self.version,self.image_size,self.score_threshold=model,version,image_size,score_threshold
    monkeypatch.setattr(module,'PlateDetector',Detector)
    old=Detector(object(),version='verified',image_size=256,score_threshold=.37)
    monkeypatch.setattr(pipeline,'load_classifier',lambda *a:object())
    monkeypatch.setattr(pipeline,'load_corner_detector',lambda path:old)
    monkeypatch.setattr(pipeline,'load_plate_reader',lambda path:None)
    monkeypatch.setattr(pipeline,'load_box_reader',lambda path:None)
    rec=pipeline.PlateRecognizer.from_config(runtime__detector_input_size=size)
    assert rec.corner_detector.model is old.model
    assert rec.corner_detector.score_threshold==.37
    assert rec.corner_detector.version=='verified'
    assert rec.corner_detector.image_size==(size or 256)
    assert (rec.corner_detector is old)==(size==0)

@pytest.mark.parametrize('model_path',['','models/corner_net_v1.pt'])
def test_nonzero_size_rejects_non_plate_detector(monkeypatch,model_path):
    import lpr.pipeline as pipeline
    monkeypatch.setattr(pipeline,'load_classifier',lambda *a:object())
    monkeypatch.setattr(pipeline,'load_corner_detector',lambda path:object())
    with pytest.raises(ConfigError,match='requires a plate_det'):
        pipeline.PlateRecognizer.from_config(runtime__detector_input_size=384,runtime__corner_model_path=model_path)