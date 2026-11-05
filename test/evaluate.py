#!/usr/bin/env python3
"""Compare docscan3d output with the synthetic ground truth."""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_synthetic import surface  # noqa: E402


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum() if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("synth")
    ap.add_argument("out")
    args = ap.parse_args()
    T = np.load(os.path.join(args.synth, "truth.npz"))
    pts = np.loadtxt(os.path.join(args.out, "points.csv"), delimiter=",", skiprows=1)
    uvz, x1 = pts[:, :3], pts[:, 5:7]
    p = {k: float(T[k]) for k in ("fold1", "fold2", "bend", "curl", "corner")}

    uvmap = T["uvmap"]
    tuv = np.stack([cv2.remap(uvmap[..., c], x1[:, 0:1].astype(np.float32), x1[:, 1:2].astype(np.float32),
                              cv2.INTER_LINEAR)[:, 0] for c in range(2)], 1)
    ok = np.isfinite(tuv).all(1)
    uvz, tuv = uvz[ok], tuv[ok]
    P = surface(tuv[:, 0], tuv[:, 1], p)

    s, R, t = umeyama(uvz, P, True)
    res = P - (s * uvz @ R.T + t)
    print(f"points evaluated: {len(P)}")
    print(f"3D error after similarity fit: rms {np.sqrt((res**2).sum(1).mean()):.3f} mm, "
          f"p95 {np.percentile(np.linalg.norm(res, axis=1), 95):.3f} mm  (scale factor {s:.4f})")
    _, R0, t0 = umeyama(uvz, P, False)
    res0 = P - (uvz @ R0.T + t0)
    print(f"3D error after rigid fit (own scale): rms {np.sqrt((res0**2).sum(1).mean()):.3f} mm")
    zt = P @ R0[:, 2]
    zt -= np.median(zt - uvz[:, 2])
    dz = uvz[:, 2] - zt
    print(f"height error only: rms {np.sqrt(np.mean(dz**2)):.3f} mm, true z span {np.ptp(zt):.2f} mm")

    flat = cv2.imread(os.path.join(args.out, "flat.png"), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(os.path.join(args.synth, "letter_truth.png"), cv2.IMREAD_GRAYSCALE)
    ref = cv2.resize(ref, (flat.shape[1], flat.shape[0]), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(20000)
    ka, da = sift.detectAndCompute(flat, None)
    kb, db = sift.detectAndCompute(ref, None)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    m = [a for a, b in m if a.distance < 0.7 * b.distance]
    a = np.float64([ka[x.queryIdx].pt for x in m])
    b = np.float64([kb[x.trainIdx].pt for x in m])
    d = np.linalg.norm(a - b, axis=1)
    d = d[d < 40]
    mm_per_px = 210.0 / flat.shape[1]
    print(f"flat output vs. original letter: {len(d)} matches, displacement median "
          f"{np.median(d)*mm_per_px:.3f} mm, p95 {np.percentile(d, 95)*mm_per_px:.3f} mm, "
          f"max {d.max()*mm_per_px:.3f} mm")
    keep = np.linalg.norm(a - b, axis=1) < 40
    a, b = a[keep], b[keep]
    M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.LMEDS)
    r = np.linalg.norm(a @ M[:, :2].T + M[:, 2] - b, axis=1) * mm_per_px
    sc = np.hypot(M[0, 0], M[1, 0])
    print(f"  global placement: scale {sc:.4f}, rotation {np.degrees(np.arctan2(M[1,0], M[0,0])):.3f} deg, "
          f"shift {np.linalg.norm(M[:, 2])*mm_per_px:.2f} mm")
    print(f"  remaining local warp after similarity fit: median {np.median(r):.3f} mm, "
          f"p95 {np.percentile(r, 95):.3f} mm, max {r.max():.3f} mm")
    by_v = np.digitize(b[:, 1] * mm_per_px, [99, 198])
    for k, name in enumerate(["panel 1", "panel 2", "panel 3"]):
        if np.any(by_v == k):
            print(f"    {name}: median {np.median(r[by_v == k]):.3f} mm")


if __name__ == "__main__":
    main()
