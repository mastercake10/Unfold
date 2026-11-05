#!/usr/bin/env python3
"""Scan-free distortion metric: printed documents are dominated by horizontal and
vertical lines (rules, boxes, text baselines). Detect line segments, measure how far
each deviates from the nearest axis after removing the global rotation, and how
straight long lines stay over their length.

Scores (length-weighted, segments >= --min-mm):
  axis_0.25  share of near-axis line length within 0.25 deg of 0/90 deg
  rms_deg    RMS angle deviation of near-axis segments (within 5 deg)
  bend_mm    median deviation from a straight line of long merged lines
"""
import argparse
import json

import cv2
import numpy as np


def segments(gray, mm_per_px, min_mm):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.empty((0, 4))
    L = lines.reshape(-1, 4).astype(float)
    length = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]) * mm_per_px
    return L[length >= min_mm]


def axis_dev(L):
    ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180
    dev = (ang + 45) % 90 - 45          # deviation from nearest axis, in [-45, 45)
    vert = np.abs(((ang - 90 + 90) % 180) - 90) < 45
    return dev, vert


def long_line_bend(gray, mm_per_px, vert, min_len_mm=60):
    """Follow long rules: Hough on dark thin structures, then for each long line measure
    the max perpendicular deviation of its edge pixels from the fitted straight line."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    k = int(min_len_mm / mm_per_px / 4)
    kern = np.ones((k, 1), np.uint8) if vert else np.ones((1, k), np.uint8)
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines)
    bends = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        span = (h if vert else w) * mm_per_px
        if span < min_len_mm:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        a, b = (ys, xs) if vert else (xs, ys)   # a along the line, b across
        order = np.argsort(a)
        # centre line per position along the line
        ua, inv = np.unique(a, return_inverse=True)
        c = np.bincount(inv, b) / np.bincount(inv)
        p = np.polyfit(ua, c, 1)
        r = c - np.polyval(p, ua)
        bends.append(np.percentile(np.abs(r), 95) * mm_per_px)
    return bends


def score(img, page_w_mm=210.0, min_mm=8.0):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mm_per_px = page_w_mm / gray.shape[1]
    L = segments(gray, mm_per_px, min_mm)
    if len(L) == 0:
        return dict(segments=0)
    length = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]) * mm_per_px
    dev, vert = axis_dev(L)
    near = np.abs(dev) < 5
    rot = np.average(dev[near], weights=length[near])   # global page rotation
    d = dev - rot
    near = np.abs(d) < 5
    w = length[near]
    res = dict(segments=int(near.sum()),
               rotation_deg=float(rot),
               axis_025=float(w[np.abs(d[near]) < 0.25].sum() / w.sum()),
               axis_050=float(w[np.abs(d[near]) < 0.5].sum() / w.sum()),
               rms_deg=float(np.sqrt(np.average(d[near] ** 2, weights=w))),
               rms_vert_deg=float(np.sqrt(np.average(d[near & vert] ** 2, weights=length[near & vert])))
               if (near & vert).any() else None,
               rms_horiz_deg=float(np.sqrt(np.average(d[near & ~vert] ** 2, weights=length[near & ~vert])))
               if (near & ~vert).any() else None)
    bv = long_line_bend(gray, mm_per_px, True)
    bh = long_line_bend(gray, mm_per_px, False)
    res["bend_vert_mm"] = float(np.median(bv)) if bv else None
    res["bend_horiz_mm"] = float(np.median(bh)) if bh else None
    res["_L"], res["_d"] = L, d
    return res


def histogram_png(path, results, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    bins = np.linspace(-3, 3, 121)
    for r, lab in zip(results, labels):
        L, d = r["_L"], r["_d"]
        dev, vert = axis_dev(L)
        length = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
        for k, (m, name) in enumerate(((~vert, "horizontal"), (vert, "vertical"))):
            ax[k].hist(d[m], bins=bins, weights=length[m] / length.sum(), histtype="step", lw=1.5, label=lab)
            ax[k].set_title(f"{name} lines: deviation from axis [deg]")
    for a in ax:
        a.legend(); a.set_xlabel("deg")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def vis(img, r, path):
    out = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 2 + 127
    for (x0, y0, x1, y1), d in zip(r["_L"], r["_d"]):
        if abs(d) >= 5:
            continue
        t = min(abs(d) / 1.0, 1.0)
        col = (0, int(200 * (1 - t)), int(255 * t))
        cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--page-w", type=float, default=210.0)
    ap.add_argument("--hist", help="write an angle histogram PNG comparing the inputs")
    ap.add_argument("--vis", help="draw segments of the first image coloured by deviation")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = []
    for p in a.images:
        img = cv2.imread(p)
        r = score(img, a.page_w)
        res.append(r)
        pub = {k: v for k, v in r.items() if not k.startswith("_")}
        if a.json:
            print(json.dumps(dict(file=p, **pub)))
        else:
            f = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"{p}: within 0.25 deg {pub['axis_025']*100:.1f} %, within 0.5 deg {pub['axis_050']*100:.1f} %, "
                  f"rms {pub['rms_deg']:.3f} deg (vert {f(pub['rms_vert_deg'])}, horiz {f(pub['rms_horiz_deg'])}), "
                  f"long-line bend vert {f(pub['bend_vert_mm'])} mm / horiz {f(pub['bend_horiz_mm'])} mm, "
                  f"{pub['segments']} segments")
    if a.hist:
        histogram_png(a.hist, res, [p.split('/')[-2] if '/' in p else p for p in a.images])
    if a.vis:
        vis(cv2.imread(a.images[0]), res[0], a.vis)


if __name__ == "__main__":
    main()
