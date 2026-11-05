#!/usr/bin/env python3
"""Compare a flattened page with a flatbed scan of the same page.

The scan may have padding and a slight rotation; a similarity transform is
fitted first (global placement), then the remaining local warp is reported.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def homography_flat(photo, W, H, dpi):
    from docscan3d import find_page_quad, refine_quad
    img = cv2.imread(photo)
    q, _ = find_page_quad(img)
    q = refine_quad(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), q)
    ppm = dpi / 25.4
    ow, oh = int(round(W * ppm)), int(round(H * ppm))
    M = cv2.getPerspectiveTransform(q.astype(np.float32),
                                    np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]]))
    return cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def compare(flat, scan, scan_dpi, page_w_mm, vis_path=None, max_err_mm=6.0):
    mm_per_px = page_w_mm / flat.shape[1]
    s = (25.4 / scan_dpi) / mm_per_px
    scan = cv2.resize(scan, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ga = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY) if flat.ndim == 3 else flat
    gb = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY) if scan.ndim == 3 else scan
    clahe = cv2.createCLAHE(2.0, (8, 8))
    ga, gb = clahe.apply(ga), clahe.apply(gb)
    sift = cv2.SIFT_create(40000)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64)).knnMatch(da, db, k=2)
    m = [x for x, y in m if x.distance < 0.75 * y.distance]
    a = np.float64([ka[x.queryIdx].pt for x in m])
    b = np.float64([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=max_err_mm / mm_per_px,
                                         maxIters=20000, confidence=0.999)
    a, b = a[inl.ravel() > 0], b[inl.ravel() > 0]
    # refit on inliers by least squares, then robust trimming of gross mismatches
    for _ in range(3):
        M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.LMEDS)
        r = np.linalg.norm(a @ M[:, :2].T + M[:, 2] - b, axis=1) * mm_per_px
        keep = r < max(3 * np.median(r) * 1.4826 + 0.3, 1.0)
        a, b = a[keep], b[keep]
    rvec = (a @ M[:, :2].T + M[:, 2] - b) * mm_per_px
    r = np.linalg.norm(rvec, axis=1)
    scale = np.hypot(M[0, 0], M[1, 0])
    ys = a[:, 1] * mm_per_px
    bands = {f"{lo}-{hi}mm": float(np.median(r[(ys >= lo) & (ys < hi)])) if np.any((ys >= lo) & (ys < hi)) else None
             for lo, hi in ((0, 99), (99, 198), (198, 297))}
    # coverage: share of page tiles with content in the scan that also got verified matches
    ny, nx = 8, 6
    kbp = np.float64([k.pt for k in kb])
    content = np.histogram2d(kbp[:, 1], kbp[:, 0], [ny, nx], [[0, gb.shape[0]], [0, gb.shape[1]]])[0] >= 15
    hit = np.histogram2d(b[:, 1], b[:, 0], [ny, nx], [[0, gb.shape[0]], [0, gb.shape[1]]])[0] >= 3
    coverage = float((content & hit).sum() / max(content.sum(), 1))
    res = dict(matches=int(len(r)), coverage=coverage, scale=float(scale),
               rotation_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
               local_median_mm=float(np.median(r)), local_p95_mm=float(np.percentile(r, 95)),
               local_max_mm=float(r.max()), bands=bands)
    if vis_path:
        vis = cv2.cvtColor(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 2 + 127
        lim = 0.5
        for p, d, e in zip(a, rvec, r):
            c = int(np.clip(e / lim * 255, 0, 255))
            col = (0, 255 - c, c) if e < lim else (0, 0, 255)
            cv2.arrowedLine(vis, tuple(int(x) for x in p), tuple(int(x) for x in p + d / mm_per_px * 40),
                            col, 1, tipLength=0.2)
        cv2.putText(vis, f"local median {res['local_median_mm']:.3f} mm, p95 {res['local_p95_mm']:.3f} mm (arrows x40)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(vis_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flat", help="flattened image (or a photo with --homography)")
    ap.add_argument("scan")
    ap.add_argument("--scan-dpi", type=float, default=300)
    ap.add_argument("--page-w", type=float, default=210)
    ap.add_argument("--page-h", type=float, default=297)
    ap.add_argument("--homography", action="store_true", help="first flatten the photo with a 4-corner homography")
    ap.add_argument("--vis")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.homography:
        flat = homography_flat(args.flat, args.page_w, args.page_h, 200)
    else:
        flat = cv2.imread(args.flat)
    res = compare(flat, cv2.imread(args.scan), args.scan_dpi, args.page_w, args.vis)
    if args.json:
        print(json.dumps(res))
    else:
        print(f"matches {res['matches']}, coverage {res['coverage']:.2f}, scale {res['scale']:.4f}, rot {res['rotation_deg']:.2f} deg, "
              f"local median {res['local_median_mm']:.3f} mm, p95 {res['local_p95_mm']:.3f} mm, "
              f"bands {{{', '.join(f'{k}: {v:.3f}' for k, v in res['bands'].items() if v is not None)}}}")


if __name__ == "__main__":
    main()
