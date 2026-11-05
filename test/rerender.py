#!/usr/bin/env python3
"""Re-run only the photometric stage (render, shading, fusion) from a saved
heightfield.npz, for fast iteration. Usage: rerender.py OUT_DIR img0 img1 ... [-o file]"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import docscan3d as d  # noqa: E402


def load(out_dir, images, W=210.0, H=297.0, dpi=200):
    z = np.load(os.path.join(out_dir, "heightfield.npz"))
    fulls = [cv2.imread(p) for p in images]
    h, w = fulls[0].shape[:2]
    cam = dict(f=float(z["f"]), cx=(w - 1) / 2, cy=(h - 1) / 2, k1=float(z["k1"]), k2=float(z["k2"]))
    F = dict(o=z["origin"], eu=z["eu"], ev=z["ev"], en=z["en"], scale=float(z["scale"]))
    gu, gv, Z, st = z["u"], z["v"], z["z"], z["st"]
    inv = d.flat_to_uv(st, gu, gv, W, H)
    Xr, shape, nu, nv, page = d.flat_geometry(cam, F, gu, gv, Z, W, H, dpi, inv)
    rend = [d.render_view(fulls[j], Xr, shape, cam, z["R"][j], z["t"][j]) for j in range(len(images))]
    return rend, nu, nv, page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--output", default="flat_rerender.png")
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    a = ap.parse_args()
    rend, nu, nv, page = load(a.out_dir, a.images if a.fuse else a.images[:1])
    albs = [d.remove_shading(r[0], r[1] & page, nu, nv) for r in rend]
    flat = d.fuse_views(albs, [r[1] & page for r in rend], [r[2] for r in rend]) if len(rend) > 1 else albs[0]
    out8 = d.to_uint8(flat)
    cv2.imwrite(a.output, out8 if a.no_clean else d.clean_paper(out8))


if __name__ == "__main__":
    main()
