"""Render this repository's uncompressed mxGraph figure with Pillow, offline.

Supported: absolute rectangle/rounded/rhombus vertices, orthogonal connectors,
explicit ports and waypoints, plain multiline labels, colors and dashed lines.
This is a renderer for this flat figure, not a general draw.io exporter.
Compressed diagrams, HTML nodes and unsupported shapes/edge styles are rejected.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = Path(r"C:\Windows\Fonts\simhei.ttf")


def style_of(cell):
    return dict(item.split("=", 1) for item in cell.get("style", "").split(";") if "=" in item)


@dataclass
class Node:
    id: str
    label: str
    style: dict
    box: tuple


@dataclass
class Text:
    font: object
    lines: list
    positions: list
    boxes: list


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def expand(box, margin):
    return box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin


def segment_hits(a, b, box):
    """Closed segment/rectangle intersection, including diagonal fixture cases."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0] - box[0]), (dx, box[2] - a[0]),
                 (-dy, a[1] - box[1]), (dy, box[3] - a[1])):
        if p == 0:
            if q < 0:
                return False
        elif p < 0:
            low = max(low, q / p)
        else:
            high = min(high, q / p)
    return low <= high


def inside(node, box, margin=0):
    x0, y0, x1, y1 = node.box
    if node.style.get("shape") == "rhombus":
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2 - margin, (y1 - y0) / 2 - margin
        return rx > 0 and ry > 0 and all(
            abs(x - cx) / rx + abs(y - cy) / ry <= 1
            for x in (box[0], box[2]) for y in (box[1], box[3]))
    return (x0 + margin <= box[0] <= box[2] <= x1 - margin
            and y0 + margin <= box[1] <= box[3] <= y1 - margin)


def wrap(label, font, width):
    lines = []
    for paragraph in label.split("\n"):
        current = ""
        for char in paragraph:
            box = font.getbbox(current + char)
            if current and box[2] - box[0] > width:
                lines.append(current)
                current = char
            else:
                current += char
        lines.append(current)
    return lines


def fit_text(node, font_path, scale):
    pad = float(node.style.get("spacing", 12)) * scale
    x0, y0, x1, y1 = node.box
    maximum = round(float(node.style.get("fontSize", 16)) * scale)
    for size in range(maximum, 8, -1):
        font = ImageFont.truetype(str(font_path), size)
        lines = wrap(node.label, font, x1 - x0 - 2 * pad)
        metrics = [font.getbbox(line or " ") for line in lines]
        line_height = max(m[3] - m[1] for m in metrics) + max(3, round(size * .3))
        height = (len(lines) - 1) * line_height + max(m[3] - m[1] for m in metrics)
        top = (y0 + y1 - height) / 2
        positions, boxes = [], []
        for index, metric in enumerate(metrics):
            width, ink_height = metric[2] - metric[0], metric[3] - metric[1]
            left, y = (x0 + x1 - width) / 2, top + index * line_height
            boxes.append((left, y, left + width, y + ink_height))
            positions.append((left - metric[0], y - metric[1]))
        if all(inside(node, box, pad) for box in boxes):
            return Text(font, lines, positions, boxes)
    raise ValueError(f"Text does not fit node {node.id}; shorten its label")


def port(node, style, prefix):
    x0, y0, x1, y1 = node.box
    return (x0 + float(style[prefix + "X"]) * (x1 - x0),
            y0 + float(style[prefix + "Y"]) * (y1 - y0))


def route(cell, nodes, scale):
    style = style_of(cell)
    start, end = port(nodes[cell.get("source")], style, "exit"), port(nodes[cell.get("target")], style, "entry")
    points = [(float(p.get("x")) * scale, float(p.get("y")) * scale)
              for p in cell.findall("mxGeometry/Array[@as='points']/mxPoint")]
    anchors = [start, *points, end]
    path = [start]
    for index, dest in enumerate(anchors[1:]):
        prev = path[-1]
        if prev[0] != dest[0] and prev[1] != dest[1]:
            # Complete mxGraph orthogonal endpoint stubs; keep every XML waypoint.
            if index == len(anchors) - 2:
                vertical = style["entryY"] in ("0", "1")
                bend = (dest[0], prev[1]) if vertical else (prev[0], dest[1])
            else:
                horizontal = style["exitX"] in ("0", "1")
                bend = (dest[0], prev[1]) if horizontal else (prev[0], dest[1])
            if bend != prev:
                path.append(bend)
        if dest != path[-1]:
            path.append(dest)
    return path


def arrow(path, scale):
    a, b = path[-2:]
    length = math.dist(a, b)
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    size = 9 * scale
    return [b, (b[0] - ux * size - uy * size / 2, b[1] - uy * size + ux * size / 2),
            (b[0] - ux * size + uy * size / 2, b[1] - uy * size - ux * size / 2)]


def edge_label(cell, path, font_path, scale, obstacles, segments, page):
    label = cell.get("value", "")
    if not label:
        return None
    style = style_of(cell)
    for size in range(round(float(style.get("fontSize", 14)) * scale), 11, -1):
        font = ImageFont.truetype(str(font_path), size)
        metric = font.getbbox(label)
        width, height = metric[2] - metric[0], metric[3] - metric[1]
        padding, gap = 3 * scale, 7 * scale
        ordered = sorted(zip(path, path[1:]), key=lambda seg: -math.dist(*seg))
        for a, b in ordered:
            for fraction in (.5, .25, .75, .1, .9):
                x, y = a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction
                candidates = ([(x - width / 2, y - height - gap), (x - width / 2, y + gap)]
                              if a[1] == b[1] else [(x + gap, y - height / 2), (x - width - gap, y - height / 2)])
                for left, top in candidates:
                    ink = (left, top, left + width, top + height)
                    box = expand(ink, padding)
                    if not (0 <= box[0] < box[2] <= page[0] and 0 <= box[1] < box[3] <= page[1]):
                        continue
                    if any(overlaps(box, other) for other in obstacles):
                        continue
                    if any(segment_hits(c, d, expand(box, scale * 2)) for c, d in segments):
                        continue
                    text = Text(font, [label], [(left - metric[0], top - metric[1])], [ink])
                    return box, text
    raise ValueError(f"No unobstructed label position for {cell.get('id')}")


def draw_text(draw, text, fill):
    for line, position in zip(text.lines, text.positions):
        draw.text(position, line, font=text.font, fill=fill)


def draw_path(draw, path, style, scale):
    color, width = style.get("strokeColor", "#527A9B"), max(2, round(2 * scale))
    if style.get("dashed") == "1":
        for a, b in zip(path, path[1:]):
            length = math.dist(a, b)
            offset = 0
            while offset < length:
                stop = min(length, offset + 6 * scale)
                draw.line([(a[0] + (b[0] - a[0]) * t / length, a[1] + (b[1] - a[1]) * t / length)
                           for t in (offset, stop)], fill=color, width=width)
                offset += 10 * scale
    else:
        draw.line(path, fill=color, width=width, joint="curve")
    draw.polygon(arrow(path, scale), fill=color)


def render(source, output, width=2400, font_path=FONT):
    if not 2000 <= width <= 2600:
        raise ValueError("Width must be between 2000 and 2600 pixels")
    model = ET.parse(source).find("diagram/mxGraphModel")
    if model is None:
        raise ValueError("Expected an uncompressed mxGraphModel")
    scale = width / float(model.get("pageWidth"))
    page = (width, round(float(model.get("pageHeight")) * scale))
    if page[1] <= 0 or width * page[1] > 12_000_000:
        raise ValueError("Page exceeds the lightweight render budget")
    cells = model.findall("root/mxCell")
    ids = [c.get("id") for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate cell id")
    nodes, edges = {}, []
    for cell in cells:
        style = style_of(cell)
        if cell.get("vertex") == "1":
            if style.get("shape", "rectangle") not in ("rectangle", "rhombus") or style.get("html") != "0":
                raise ValueError("Only plain rectangle/rhombus nodes are supported")
            geo = cell.find("mxGeometry")
            x, y, w, h = [float(geo.get(key)) * scale for key in ("x", "y", "width", "height")]
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > page[0] or y + h > page[1]:
                raise ValueError(f"Invalid node geometry: {cell.get('id')}")
            nodes[cell.get("id")] = Node(cell.get("id"), cell.get("value", ""), style, (x, y, x + w, y + h))
        elif cell.get("edge") == "1":
            if style.get("edgeStyle") != "orthogonalEdgeStyle" or style.get("endArrow") != "block":
                raise ValueError("Only orthogonal block-arrow edges are supported")
            edges.append(cell)
    texts = {key: fit_text(node, font_path, scale) for key, node in nodes.items()}
    paths = {cell.get("id"): route(cell, nodes, scale) for cell in edges}
    segments = [seg for path in paths.values() for seg in zip(path, path[1:])]
    if any(segment_hits(a, b, expand(box, scale * 2))
           for a, b in segments for text in texts.values() for box in text.boxes):
        raise ValueError("A connector crosses node text")
    obstacles = [expand(node.box, scale * 2) for node in nodes.values()]
    for path in paths.values():
        triangle = arrow(path, scale)
        obstacles.append((min(x for x, y in triangle), min(y for x, y in triangle),
                          max(x for x, y in triangle), max(y for x, y in triangle)))
    labels = {}
    for cell in edges:
        label = edge_label(cell, paths[cell.get("id")], font_path, scale, obstacles, segments, page)
        if label:
            labels[cell.get("id")] = label
            obstacles.append(expand(label[0], scale * 2))
    image = Image.new("RGB", page, "white")
    draw = ImageDraw.Draw(image)
    for cell in edges:
        draw_path(draw, paths[cell.get("id")], style_of(cell), scale)
    for key, node in nodes.items():
        x0, y0, x1, y1 = node.box
        fill, stroke = node.style.get("fillColor", "white"), node.style.get("strokeColor", "#527A9B")
        border = max(2, round(1.5 * scale))
        if node.style.get("shape") == "rhombus":
            draw.polygon([((x0 + x1) / 2, y0), (x1, (y0 + y1) / 2),
                          ((x0 + x1) / 2, y1), (x0, (y0 + y1) / 2)], fill=fill, outline=stroke, width=border)
        elif node.style.get("rounded") == "1":
            radius = min(x1 - x0, y1 - y0) * float(node.style.get("arcSize", 10)) / 100
            draw.rounded_rectangle(node.box, radius=radius, fill=fill, outline=stroke, width=border)
        else:
            draw.rectangle(node.box, fill=fill, outline=stroke, width=border)
        draw_text(draw, texts[key], node.style.get("fontColor", "#17324D"))
    for cell in edges:
        if cell.get("id") in labels:
            box, text = labels[cell.get("id")]
            draw.rectangle(box, fill=style_of(cell).get("labelBackgroundColor", "white"))
            draw_text(draw, text, style_of(cell).get("fontColor", "#17324D"))
    output = Path(output)
    image.save(output, format="PNG", dpi=(150, 150))
    image.close()
    return {"nodes": len(nodes), "edges": len(edges), "edge_labels": len(labels),
            "png_size": list(page), "png_bytes": output.stat().st_size,
            "node_font_px": {key: text.font.size for key, text in texts.items()},
            "node_text_overflow": 0, "connector_text_collisions": 0,
            "edge_label_overlaps": 0, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def self_check():
    """Small geometry fixtures; no models, external data or output files."""
    assert segment_hits((0, 0), (10, 0), (4, -1, 6, 1))
    assert segment_hits((0, 0), (10, 10), (4, 4, 6, 6))
    assert not segment_hits((0, 0), (10, 0), (4, 1, 6, 2))
    diamond = Node("d", "", {"shape": "rhombus"}, (0, 0, 100, 100))
    assert inside(diamond, (40, 40, 60, 60))
    assert not inside(diamond, (0, 0, 20, 20))
    rect = Node("r", "", {}, (0, 0, 100, 100))
    assert inside(rect, (10, 10, 90, 90), 10)
    nodes = {"a": Node("a", "", {}, (0, 0, 100, 100)),
             "b": Node("b", "", {}, (200, 200, 300, 300))}
    cell = ET.fromstring('<mxCell source="a" target="b" style="exitX=0.5;exitY=1;entryX=0.5;entryY=0;"><mxGeometry><Array as="points"><mxPoint x="50" y="150"/><mxPoint x="220" y="150"/></Array></mxGeometry></mxCell>')
    path = route(cell, nodes, 1)
    assert path == [(50, 100), (50, 150), (220, 150), (250, 150), (250, 200)]
    assert all(a[0] == b[0] or a[1] == b[1] for a, b in zip(path, path[1:]))
    font = ImageFont.truetype(str(FONT), 20)
    lines = wrap("中文长标签abcdef\n第二行", font, 60)
    assert "".join(lines) == "中文长标签abcdef第二行"
    text = fit_text(Node("fit", "中文第一行\n第二行", {"fontSize": "30", "shape": "rhombus"}, (0, 0, 180, 100)), FONT, 1)
    assert text.font.size < 30
    print(json.dumps({"geometry_self_check": "PASS"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "pipeline.drawio")
    parser.add_argument("--output", type=Path, default=HERE.parent / "pipeline.png")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--font", type=Path, default=FONT)
    parser.add_argument("--self-check", action="store_true", help="Run small in-memory geometry fixtures only")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        print(json.dumps(render(args.source, args.output, args.width, args.font), ensure_ascii=True))


if __name__ == "__main__":
    main()
