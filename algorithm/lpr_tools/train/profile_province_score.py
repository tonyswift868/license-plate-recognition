"""Run the frozen safe CRNN scorer once and record each production-reader call."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_TMP = _ROOT / "algorithm" / "out" / "tmp" / "batch12-runtime"
_RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_RUNTIME_TMP)
os.environ["TMP"] = str(_RUNTIME_TMP)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"


def profile(model_path: Path, data_dir: Path, output: Path):
    from lpr import recognize

    from lpr_tools.train.province_inference_timing import instrument_method, summarize
    from lpr_tools.train.score_province_readers import score

    model_path, data_dir, output = (Path(model_path).resolve(), Path(data_dir).resolve(),
                                    Path(output).resolve())
    records = []
    load_reader = recognize.load_plate_reader

    def load_timed_reader(path):
        reader = load_reader(path)
        instrument_method(reader, "read", records, batch_size=lambda _args, _kwargs: 1)
        return reader

    recognize.load_plate_reader = load_timed_reader
    try:
        result = score("crnn", model_path, data_dir, output=None)
    finally:
        recognize.load_plate_reader = load_reader
    if not records:
        raise RuntimeError("safe CRNN scoring produced no timed load_plate_reader.read calls")
    result["read_timing"] = {
        **summarize(records),
        "method": "CrnnReader.read on the scorer's existing GT quadrilateral-warped safe plate crop",
        "scope": "safe tune + province holdout + synthetic panel; one inference per scorer row; first cold call retained; no warmup or repeated inference",
        "fixed_val_dev_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("crnn",), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("algorithm/out/data/batch12"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = profile(args.model, args.data_dir, args.out)
    print(json.dumps({"kind": "crnn", "model": result["model"],
                      "out": str(args.out.resolve()), "calls": result["read_timing"]["calls"]},
                     ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
