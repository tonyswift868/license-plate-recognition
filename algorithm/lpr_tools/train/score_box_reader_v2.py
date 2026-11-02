"""CPU-only selection over safe GT boxes, including held-out eight-character plates."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# Keep helper-library temporary files in the explicitly authorized office.
_RUNTIME_TMP = Path(__file__).resolve().parents[3] / "algorithm/out/tmp/batch11-runtime"
_RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(_RUNTIME_TMP)

import cv2  # noqa: E402 - temporary directory must be configured before dependency imports
import torch  # noqa: E402 - temporary directory must be configured before dependency imports
from lpr.recognize.box_reader import BoxPlateReader  # noqa: E402 - temporary directory must be configured before dependency imports
from lpr.recognize.crnn import _greedy_decode  # noqa: E402 - temporary directory must be configured before dependency imports
from lpr_tools.train.train_box_reader import process_ram_bytes  # noqa: E402 - temporary directory must be configured before dependency imports
from lpr_tools.train.train_box_reader_v2 import BoxDatasetV2, collate_v2  # noqa: E402 - temporary directory must be configured before dependency imports
from torch.utils.data import DataLoader  # noqa: E402 - temporary directory must be configured before dependency imports


def score(model_path, tune_path):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    reader=BoxPlateReader.from_file(model_path,version='safe-v2-selection')
    ds=BoxDatasetV2(tune_path,{},training=False)
    loader=DataLoader(ds,batch_size=8,num_workers=0,collate_fn=collate_v2)
    counts=defaultdict(lambda:[0,0])
    with torch.inference_mode():
        for images,_,_,labels,_,_,groups in loader:
            logits=reader.model(images)
            for i,(label,group) in enumerate(zip(labels,groups, strict=False)):
                correct=int(_greedy_decode(logits[:,i])[0]==label)
                for key in ('all',group,'all_8char' if len(label)==8 else 'all_7char'):
                    counts[key][0]+=correct
                    counts[key][1]+=1
    metrics={g:{'correct':c,'n':n,'accuracy':c/n} for g,(c,n) in counts.items()}
    return {**metrics['all'],'groups':metrics,'cpu_peak_ram_bytes':process_ram_bytes(peak=True)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--tune',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(score(args.model,args.tune)),flush=True)


if __name__=='__main__':
    main()
