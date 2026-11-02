"""Bitset acceleration for exhaustive agree-threshold search.

Each bit represents one trace row. Rectification order is retained, so the
first OK returned by the production pipeline remains the selected outcome.
"""
from __future__ import annotations

import numpy as np


def _pack(values) -> int:
    """Pack a bool vector into a Python int with row zero in the low bit."""
    return int.from_bytes(np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little").tobytes(),
                          "little")


class DecisionBits:
    """Precompile trace rows into per-rect, cross-row integer masks.

    Parameters use the same keys as ``agree_threshold.exact_branch_bound``:
    ``agree_min``, ``solo_seg``, ``solo_reader``, and ``prefer``.
    """

    def __init__(self, records):
        self.records = list(records)
        self.n = len(self.records)
        self.all = (1 << self.n) - 1
        self.groups = {
            "val": _pack([r["split"] == "val" for r in self.records]),
            "devhard": _pack([r["split"] != "val" for r in self.records]),
        }
        self.truth = [r.get("truth") for r in self.records]
        max_rects = max((len(r.get("rects", ())) for r in self.records), default=0)
        self.rects = []
        self.conf_s = []
        self.conf_r = []
        for j in range(max_rects):
            segs, readers = [], []
            for row in self.records:
                rects = row.get("rects", ())
                rect = rects[j] if j < len(rects) else {}
                s = rect.get("segment")
                r = rect.get("reader")
                if s and not s.get("valid", True):
                    s = None
                if r and not r.get("valid", True):
                    r = None
                segs.append(s)
                readers.append(r)
            sv = [bool(s) for s in segs]
            rv = [bool(r) for r in readers]
            equal = [bool(s and r and s["plate"] == r["plate"]) for s, r in zip(segs, readers, strict=False)]
            st = [bool(s and s["plate"] == truth) for s, truth in zip(segs, self.truth, strict=False)]
            rt = [bool(r and r["plate"] == truth) for r, truth in zip(readers, self.truth, strict=False)]
            self.rects.append({"sv": _pack(sv), "rv": _pack(rv), "equal": _pack(equal),
                               "st": _pack(st), "rt": _pack(rt)})
            self.conf_s.append(np.asarray([float(s["confidence"]) if s else float("-inf")
                                           for s in segs], dtype=np.float64))
            self.conf_r.append(np.asarray([float(r["confidence"]) if r else float("-inf")
                                           for r in readers], dtype=np.float64))
        self.pair_conf = [np.minimum(s, r) for s, r in zip(self.conf_s, self.conf_r, strict=False)]
        self._mask_cache = {"agree": {}, "solo_seg": {}, "solo_reader": {}}

    def _threshold(self, axis: str, threshold: float) -> tuple[int, ...]:
        """Return one >= threshold mask per rect, cached on first use."""
        threshold = float(threshold)
        cached = self._mask_cache[axis].get(threshold)
        if cached is not None:
            return cached
        confidence = {"agree": self.pair_conf, "solo_seg": self.conf_s,
                      "solo_reader": self.conf_r}[axis]
        masks = tuple(_pack(per_rect >= threshold) for per_rect in confidence)
        self._mask_cache[axis][threshold] = masks
        return masks

    def _route_masks(self, params):
        agree = self._threshold("agree", params["agree_min"])
        solo_s = self._threshold("solo_seg", params["solo_seg"])
        solo_r = self._threshold("solo_reader", params["solo_reader"])
        return agree, solo_s, solo_r

    def counts(self, params):
        """Exact accepted/correct counts per fixed split for one threshold point."""
        agree, solo_s, solo_r = self._route_masks(params)
        prefer_seg = params["prefer"] == "segment"
        prefix_ok = 0
        accepted = correct = 0
        for j, rect in enumerate(self.rects):
            a = agree[j] & rect["equal"]
            s = solo_s[j] & rect["sv"]
            r = solo_r[j] & rect["rv"]
            ok = a | s | r
            first = ok & ~prefix_ok & self.all
            # Agree has precedence and always selects segment. Otherwise solo
            # selection uses prefer when both qualify, or the sole qualifying side.
            if prefer_seg:
                seg_selected = a | (~a & s)
                reader_selected = ~a & ~s & r
            else:
                reader_selected = ~a & r
                seg_selected = a | (~a & ~r & s)
            correct |= first & ((seg_selected & rect["st"]) | (reader_selected & rect["rt"]))
            accepted |= first
            prefix_ok |= ok
        return {name: {"accepted": (accepted & mask).bit_count(),
                       "correct": (correct & mask).bit_count()}
                for name, mask in self.groups.items()}

    def bounds(self, lower_params, upper_params):
        """Safe per-split accept/correct bounds over a threshold box.

        ``lower_params`` and ``upper_params`` contain each numerical axis at
        the low/high ends of its interval and share the same ``prefer``.
        possible_accept is the low-corner accepted population; since raising
        any threshold can only remove per-rect OK events, it bounds every point
        from above. guaranteed_accept is the high-corner population. A row may
        contribute to possible_correct if a correct route could be selected no
        later than its first guaranteed-OK rect; earlier possible wrong OKs are
        deliberately ignored, making this optimistic and safe.
        """
        if lower_params["prefer"] != upper_params["prefer"]:
            raise ValueError("threshold-box bounds require one fixed disagree_prefer")
        low_a, low_s, low_r = self._route_masks(lower_params)
        high_a, high_s, high_r = self._route_masks(upper_params)
        possible_accept = guaranteed_accept = possible_correct = 0
        not_before_guaranteed = self.all
        prefer_seg = lower_params["prefer"] == "segment"
        for j, rect in enumerate(self.rects):
            ap = low_a[j] & rect["equal"]
            sp = low_s[j] & rect["sv"]
            rp = low_r[j] & rect["rv"]
            ag = high_a[j] & rect["equal"]
            sg = high_s[j] & rect["sv"]
            rg = high_r[j] & rect["rv"]
            possible_accept |= ap | sp | rp
            guaranteed_here = ag | sg | rg

            # On the same rect, agree always returns segment. Solo segment can
            # be selected only when agree is not guaranteed; if reader is
            # preferred it must also be possible to disable its solo gate.
            maybe_seg = ((ap & rect["st"]) |
                         (~ag & sp & (self.all if prefer_seg else ~rg) & rect["st"]))
            maybe_reader = (~ag & rp & (self.all if not prefer_seg else ~sg) & rect["rt"])
            possible_correct |= not_before_guaranteed & (maybe_seg | maybe_reader)
            not_before_guaranteed &= ~guaranteed_here
            guaranteed_accept |= guaranteed_here

        return {name: {"possible_accept": (possible_accept & mask).bit_count(),
                       "guaranteed_accept": (guaranteed_accept & mask).bit_count(),
                       "possible_correct": (possible_correct & mask).bit_count()}
                for name, mask in self.groups.items()}
