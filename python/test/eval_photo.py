#!/usr/bin/env python3
"""Photometric comparison of a flattened page with a flatbed scan:
paper uniformity, ink contrast, and similarity (NCC) to the registered scan."""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_real import compare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flat")
    ap.add_argument("scan")
    ap.add_argument("--scan-dpi", type=float, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    flat = cv2.imread(args.flat)
    _, M, scan = compare(flat, cv2.imread(args.scan), args.scan_dpi, 210.0, return_transform=True)
    h, w = flat.shape[:2]
    sw = cv2.warpAffine(cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY), cv2.invertAffineTransform(M), (w, h),
                        flags=cv2.INTER_AREA, borderValue=0)
    valid = cv2.warpAffine(np.full(scan.shape[:2], 255, np.uint8), cv2.invertAffineTransform(M), (w, h)) > 0
    valid &= cv2.erode(valid.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = sw.astype(np.float32)
    # paper: bright in the scan and far from any ink
    ink = cv2.dilate((s < 160).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    paper = valid & (s > 200) & ~ink
    paper_std = float(np.std(g[paper]) / np.mean(g[paper]))
    dark = valid & (s < 90)
    contrast = float((np.mean(g[paper]) - np.mean(g[dark])) / np.mean(g[paper]))
    # similarity in local windows (robust to the scanner's own tone curve)
    def ncc(a, b):
        a = a - cv2.blur(a, (31, 31)); b = b - cv2.blur(b, (31, 31))
        num = cv2.blur(a * b, (31, 31))
        den = np.sqrt(cv2.blur(a * a, (31, 31)) * cv2.blur(b * b, (31, 31))) + 1e-3
        return num / den
    textish = valid & ink
    sim = float(np.mean(ncc(g, s)[textish]))
    # sharpness: gradient energy at ink edges relative to the scan
    def gm(x):
        return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))
    sharp = float(np.mean(gm(g)[textish]) / np.mean(gm(s)[textish]) * np.mean(s[paper]) / np.mean(g[paper]))
    res = dict(paper_std=paper_std, contrast=contrast, ncc=sim, sharp=sharp)
    print(json.dumps(res) if args.json else
          f"paper non-uniformity {paper_std*100:.2f} %, ink contrast {contrast:.3f}, "
          f"local NCC to scan {sim:.4f}, edge sharpness vs scan {sharp:.3f}")


if __name__ == "__main__":
    main()
