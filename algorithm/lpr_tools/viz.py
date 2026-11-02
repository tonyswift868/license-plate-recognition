"""Export masks, candidates, rectifications, binary plates and segmentation boxes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from lpr.config import LprConfig, load_config
from lpr.errors import BadInputError
from lpr.io_utils import imread, imwrite, list_images
from lpr.locate import detection_masks, locate
from lpr.preprocess import preprocess
from lpr.rectify import rectify
from lpr.segment import binary_plate, segment


def stage_images(image: np.ndarray, cfg: LprConfig | None = None) -> dict[str, np.ndarray]:
    """Use the production stage functions; all text painted on images is ASCII."""
    cfg = cfg or load_config()
    pre = preprocess(image, cfg.preprocess)
    output = {'01_work': pre.work_bgr, '02_gray': pre.gray}
    for name, mask in detection_masks(pre, cfg.locate).items():
        output[f'03_mask_{name}'] = mask
    canvas = cv2.cvtColor(image.reshape(image.shape[:2]), cv2.COLOR_GRAY2BGR) if image.ndim == 2 or image.shape[2] == 1 else image.copy()
    candidates = locate(pre, cfg.locate, cfg.runtime.max_candidates)
    for i, candidate in enumerate(candidates):
        quad = np.round(candidate.corners).astype(np.int32)
        cv2.polylines(canvas, [quad], True, (0, 0, 255), 2)
        cv2.putText(canvas, f'{i}: {candidate.color} {candidate.score:.2f} {candidate.source}',
                    (max(0, int(quad[:, 0].min())), max(18, int(quad[:, 1].min())-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1, cv2.LINE_AA)
        rect = rectify(image, candidate, cfg.rectify)
        output[f'05_plate_{i}'] = rect.image
        output[f'06_binary_{i}'] = binary_plate(rect, cfg.segment)[1]
        layouts = [rect.layout] if rect.layout in (7, 8) else [7, 8]
        if rect.color == 'yellow':
            layouts = [rect.layout, 8 if rect.layout == 7 else 7]
        for layout in layouts:
            result = segment(rect, layout, cfg.segment)
            boxes = rect.image.copy()
            for j, (x, y, w, h) in enumerate(result.boxes):
                cv2.rectangle(boxes, (x, y), (x+w-1, y+h-1), (0, 0, 255), 1)
                cv2.putText(boxes, str(j), (x, max(10, y-2)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)
            cv2.putText(boxes, f'L{layout} ok={int(result.ok)}', (3, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)
            output[f'07_boxes_{i}_L{layout}'] = boxes
            if result.ok:
                output[f'08_chars_{i}_L{layout}'] = np.hstack(result.chars)
    output['04_candidates'] = canvas
    return output


def export_image(path: str | Path, out: str | Path, cfg: LprConfig | None = None) -> int:
    images = stage_images(imread(path), cfg)
    for name, image in images.items():
        imwrite(Path(out) / f'{name}.png', image)
    return len(images)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    paths = [path for inp in args.inputs for path in list_images(inp)]
    if not paths:
        print('No input images found', file=sys.stderr)
        return 1
    cfg = load_config()
    failures, count = 0, 0
    for i, path in enumerate(paths):
        # A numbered directory also keeps equal stems from different inputs separate.
        try:
            count += export_image(path, Path(args.out) / f'{i:04d}_{path.stem}', cfg)
        except (OSError, ValueError, BadInputError, cv2.error) as error:
            failures += 1
            print(f'Failed: {path}: {error}', file=sys.stderr)
    print(f'Visualized {len(paths)-failures}/{len(paths)} images; wrote {count} PNGs to {args.out}; failures={failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
