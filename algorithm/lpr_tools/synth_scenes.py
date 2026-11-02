"""Deterministic synthetic vehicle scenes; no dataset or classifier is required."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from lpr.io_utils import imwrite

COLORS = {"blue": (180, 65, 15), "yellow": (20, 205, 235), "green": (110, 210, 100)}
# Independent drawing geometry, in millimetres; GA 36-2018 figures 1 and 7.
CENTERS7 = np.array([38, 95, 174, 231, 288, 345, 402], np.float32)
CENTERS8 = np.array([38, 90, 182, 234, 286, 338, 390, 442], np.float32)


def draw_plate(text: str = "皖A12345", color: str = "blue") -> np.ndarray:
    """Render a physical-size plate. A hollow square stands for the province glyph."""
    if len(text) not in (7, 8) or color not in COLORS:
        raise ValueError("expected 7/8 characters and blue/yellow/green")
    width = 480 if len(text) == 8 else 440
    plate = np.full((140, width, 3), COLORS[color], np.uint8)
    fg = (245, 245, 245) if color == "blue" else (12, 12, 12)
    if color == "green":
        gradient = np.linspace(1.12, .85, 140)[:, None, None]
        plate = np.clip(plate * gradient, 0, 255).astype(np.uint8)
    cv2.rectangle(plate, (3, 3), (width - 4, 136), fg, 2)
    centers = CENTERS8 if len(text) == 8 else CENTERS7
    for i, (char, center) in enumerate(zip(text, centers, strict=False)):
        x = round(float(center))
        if i == 0:
            cv2.rectangle(plate, (x - 20, 26), (x + 20, 113), fg, 5)
        else:
            glyph = np.zeros((100, 80), np.uint8)
            cv2.putText(glyph, char, (2, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.8, 255, 5, cv2.LINE_AA)
            ys, xs = np.where(glyph > 0)
            glyph = cv2.resize(glyph[ys.min():ys.max()+1, xs.min():xs.max()+1], (39, 88))
            alpha = glyph[..., None] / 255.0
            patch = plate[26:114, x-19:x+20]
            patch[:] = (patch * (1-alpha) + np.array(fg) * alpha).astype(np.uint8)
    if len(text) == 7:
        cv2.circle(plate, (133, 70), 4, fg, -1)
    for x in (110, width - 110):
        cv2.circle(plate, (x, 12), 4, (180, 180, 180), -1)
    return plate


def make_scene(index: int = 0, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Return a 1280x720 BGR scene and a JSON/CSV-compatible ground truth row.

    Indices cycle colours and signed tilts; random position, scale, perspective,
    illumination and noise are reproducible and independent of call order.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    color = ("blue", "yellow", "green")[index % 3]
    layout = 8 if color == "green" else 7
    digits = "".join(rng.choice(list("0123456789"), 5 if layout == 7 else 4))
    text = "皖A" + digits if layout == 7 else "皖AD" + str(index % 10) + digits
    scene = np.full((720, 1280, 3), (100, 105, 110), np.uint8)
    cv2.rectangle(scene, (0, 440), (1279, 719), (65, 65, 65), -1)
    for _ in range(12):
        x, y = rng.integers([0, 0], [1200, 400])
        w, h = rng.integers([15, 10], [140, 90])
        cv2.rectangle(scene, (int(x), int(y)), (int(x+w), int(y+h)),
                      tuple(int(v) for v in rng.integers(30, 220, 3)), -1)
    body = COLORS["blue"] if index % 4 == 0 else (70, 85, 95)
    cv2.fillConvexPoly(scene, np.array([[310, 250], [930, 250], [1100, 620], [180, 620]]), body)
    cv2.fillConvexPoly(scene, np.array([[380, 280], [870, 280], [945, 400], [300, 400]]), (45, 50, 55))
    cv2.circle(scene, (285, 615), 65, (25, 25, 25), -1)
    cv2.circle(scene, (995, 615), 65, (25, 25, 25), -1)
    for x in (240, 920):
        cv2.rectangle(scene, (x, 430), (x+120, 470), (200, 205, 220), -1)
    width = float(rng.uniform(145, 410))
    height = width * 140 / (480 if layout == 8 else 440)
    angle = (-25, -15, 0, 15, 25)[index % 5]
    quad = np.array([[-width/2, -height/2], [width/2, -height/2],
                     [width/2, height/2], [-width/2, height/2]], np.float32)
    # Perspective deformation in addition to in-plane rotation.
    quad += rng.uniform(-.07, .07, (4, 2)).astype(np.float32) * [width, height]
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    quad = (quad @ rotation.T + [rng.uniform(540, 740), rng.uniform(495, 540)]).astype(np.float32)
    center = quad.mean(axis=0)
    cv2.fillConvexPoly(scene, np.round(center + (quad-center)*1.12).astype(np.int32), (22, 22, 22))
    plate = draw_plate(text, color)
    ph, pw = plate.shape[:2]
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw-1, 0], [pw-1, ph-1], [0, ph-1]]), quad)
    warped = cv2.warpPerspective(plate, matrix, (1280, 720))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), matrix, (1280, 720))
    scene[mask > 0] = warped[mask > 0]
    brightness = .43 if index % 6 == 0 else float(rng.uniform(.75, 1.1))
    sigma = 5 if index % 4 == 1 else 1.5
    scene = np.clip(scene.astype(np.float32)*brightness + rng.normal(0, sigma, scene.shape), 0, 255).astype(np.uint8)
    return scene, {"file": f"scene_{index:04d}.png", "plate_text": text,
                   "color": color, "layout": layout, "quad": quad.tolist()}


def generate(out: str | Path, n: int = 40, seed: int = 42) -> list[dict]:
    if n < 0 or seed < 0:
        raise ValueError("n and seed must be nonnegative")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        image, row = make_scene(i, seed)
        imwrite(out / row["file"], image)
        rows.append(row)
    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "plate_text", "color", "layout", "quad"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "quad": json.dumps(row["quad"])})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.n < 0 or args.seed < 0:
        parser.error("n and seed must be nonnegative")
    generate(args.out, args.n, args.seed)
    print(f"Generated {args.n} scenes; labels: {Path(args.out) / 'labels.csv'}; seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
