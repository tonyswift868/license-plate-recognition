import numpy as np
import pytest
import torch

from lpr.boxcrop import crop_box
from lpr_tools.train.gpu_box_batches import GpuBoxBatches, rgb_region_for_box
from lpr_tools.train.train_box_reader import collate


@pytest.mark.parametrize("box", [(30, 20, 80, 25), (-10, 5, 60, 30), (175, 80, 30, 30)])
def test_roi_transfer_retains_exact_crop_box_pixels_at_image_boundaries(box):
    image = np.random.default_rng(11).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    rgb = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1)
    region, local_box = rgb_region_for_box(rgb, box, (30, 20, 80, 25))
    assert np.array_equal(crop_box(region, local_box, .08, .12),
                          crop_box(image, box, .08, .12))


def test_batched_decode_preserves_full_epoch_and_reduces_parallel_io_for_low_ram(monkeypatch):
    import lpr_tools.train.gpu_box_batches as module
    import lpr_tools.train.train_box_reader as train

    class Dataset:
        seed, epoch = 42, 0
        samples = {"db": [(0., 0., 0., 0.)]}
        def __len__(self):
            return 9
        def row(self, index):
            return {"path": str(index), "quad": "10 10 90 10 90 40 10 40",
                    "label": "皖A12345", "subset": "db"}

    devices = []
    monkeypatch.setattr(module, "read_file", lambda path: torch.tensor([int(path)]))
    def decode(encoded, *, mode, device):
        devices.append(device)
        return [torch.full((3, 60, 100), 100, dtype=torch.uint8) for _ in encoded]
    monkeypatch.setattr(module, "decode_jpeg", decode)
    monkeypatch.setattr(train, "free_ram_bytes", lambda: 1_500_000_000)
    loader = GpuBoxBatches(Dataset(), 4, collate)
    batches = list(loader)
    seen = [index for batch in batches for index in batch[4]]
    assert sorted(seen) == list(range(9))
    assert [len(batch[4]) for batch in batches] == [4, 4, 1]
    assert devices == ["cuda"]*3
    assert loader.reduced_for_memory
