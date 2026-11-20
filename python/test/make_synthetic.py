#!/usr/bin/env python3
"""Render two photos of a synthetic folded A4 letter with known geometry.

Writes view1.jpg, view2.jpg and truth.npz (K, camera poses, surface model and a
per-pixel page-coordinate map for view 1, used by evaluate.py).
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 210.0, 297.0
PPM = 6.0  # texture pixels per mm
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

WORDS = ("Sehr geehrter Herr Müller hiermit erhalten Sie die Abrechnung für den Zeitraum "
         "Bitte überweisen Sie den Betrag bis zum genannten Datum auf das unten angegebene Konto "
         "Rechnung Kundennummer Vertrag Leistung Menge Preis Summe Steuer gesamt Hannover "
         "Mit freundlichen Grüßen Ihre Stadtwerke Versicherung Beitrag Zahlung Mahnung").split()


def make_letter(seed):
    rng = np.random.default_rng(seed)
    tw, th = int(W * PPM), int(H * PPM)
    im = Image.new("RGB", (tw, th), (250, 249, 245))
    d = ImageDraw.Draw(im)
    mm = lambda x: int(x * PPM)
    f = lambda s, b=False: ImageFont.truetype(FONT_B if b else FONT, int(s * PPM))
    d.rectangle([mm(140), mm(12), mm(190), mm(32)], fill=(30, 80, 160))
    d.text((mm(144), mm(17)), "ACME AG", font=f(6, True), fill=(255, 255, 255))
    for i, line in enumerate(["Herrn", "Linus Müller", "Musterstraße 12", "30000 Musterstadt"]):
        d.text((mm(25), mm(50 + 5 * i)), line, font=f(3.5), fill=(20, 20, 20))
    d.text((mm(25), mm(95)), "Rechnung Nr. 2026-11-0042", font=f(4.5, True), fill=(10, 10, 10))
    y = 108.0
    for _ in range(12):
        x = 25.0
        while x < 180:
            w = WORDS[rng.integers(len(WORDS))]
            d.text((mm(x), mm(y)), w, font=f(3.2), fill=(25, 25, 25))
            x += len(w) * 1.9 + 2.0
        y += 5.2
    y += 4
    for r in range(8):
        d.line([mm(25), mm(y + r * 6), mm(185), mm(y + r * 6)], fill=(60, 60, 60), width=2)
        if r < 7:
            for c, x in enumerate([27, 95, 125, 155]):
                d.text((mm(x), mm(y + r * 6 + 1)), f"{rng.integers(1, 999)},{rng.integers(10, 99)}" if c else
                       WORDS[rng.integers(len(WORDS))], font=f(3.0), fill=(25, 25, 25))
    for x in (25, 93, 123, 153, 185):
        d.line([mm(x), mm(y), mm(x), mm(y + 42)], fill=(60, 60, 60), width=2)
    d.line([mm(3), mm(105), mm(8), mm(105)], fill=(80, 80, 80), width=2)  # fold mark
    d.line([mm(3), mm(148.5), mm(10), mm(148.5)], fill=(80, 80, 80), width=2)  # punch mark
    d.text((mm(25), mm(275)), "Seite 1 von 1   IBAN DE00 0000 0000 0000 0000 00", font=f(2.6), fill=(60, 60, 60))
    tex = np.array(im)[:, :, ::-1].copy()
    noise = rng.normal(0, 2.0, tex.shape[:2])[..., None]
    return np.clip(tex + noise, 0, 255).astype(np.uint8)


def surface(u, v, p):
    """3D point (mm) of page coordinate (u, v) in the page's own frame.
    Folds parallel to u at v=99 and v=198; each panel slightly curved;
    plus a gentle non-developable warp along u."""
    vs = np.linspace(0, H, 2001)
    th = np.zeros_like(vs)
    th += np.where(vs > 99, p["fold1"], 0) + np.where(vs > 198, p["fold2"], 0)
    th += p["bend"] * np.sin(np.pi * (vs % 99) / 99)  # bulge within each panel
    th = np.radians(th)
    dv = vs[1] - vs[0]
    Y = np.concatenate([[0], np.cumsum(np.cos(th[:-1]) * dv)])
    Z = np.concatenate([[0], np.cumsum(np.sin(th[:-1]) * dv)])
    y = np.interp(v, vs, Y)
    z = np.interp(v, vs, Z)
    z = z + p["curl"] * ((u - W / 2) / (W / 2)) ** 2 + p["corner"] * np.clip((u - 150) / 60, 0, 1) ** 2 * np.clip((v - 240) / 57, 0, 1) ** 2
    return np.stack([u, y, z], -1)


def look_at(cam, target, up=(0, -1, 0)):
    f = target - cam
    f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])  # world -> camera rows: x right, y down, z forward
    return R, -R @ cam


def render(tex, K, R, t, img_wh, p, page_pose, bg, extra_tex=None, nu=70, nv=100):
    w, h = img_wh
    out = bg.copy()
    extra = None if extra_tex is None else np.zeros((h, w, extra_tex.shape[2]), np.float32)
    us, vs = np.linspace(0, W, nu + 1), np.linspace(0, H, nv + 1)
    U, V = np.meshgrid(us, vs)
    P = surface(U.ravel(), V.ravel(), p)
    Rp, tp = page_pose
    Pw = P @ Rp.T + tp
    Pc = Pw @ R.T + t
    x = Pc @ K.T
    px = (x[:, :2] / x[:, 2:3]).reshape(nv + 1, nu + 1, 2)
    tx = np.stack([U * PPM - 0.5, V * PPM - 0.5], -1)
    zbuf = np.full((h, w), np.inf, np.float32)
    depth = Pc[:, 2].reshape(nv + 1, nu + 1)
    for j in range(nv):
        for i in range(nu):
            for tri in (((j, i), (j, i + 1), (j + 1, i)), ((j, i + 1), (j + 1, i + 1), (j + 1, i))):
                src = np.float32([tx[a] for a in tri])
                dst = np.float32([px[a] for a in tri])
                x0, y0 = np.floor(dst.min(0)).astype(int) - 1
                x1, y1 = np.ceil(dst.max(0)).astype(int) + 2
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
                if x1 <= x0 or y1 <= y0:
                    continue
                M = cv2.getAffineTransform(src, dst - np.float32([x0, y0]))
                patch = cv2.warpAffine(tex, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillConvexPoly(m, np.round((dst - [x0, y0]) * 16).astype(np.int32), 1, cv2.LINE_8, 4)
                dz = np.mean([depth[a] for a in tri])
                zb = zbuf[y0:y1, x0:x1]
                m = (m > 0) & (dz < zb)
                zb[m] = dz
                out[y0:y1, x0:x1][m] = patch[m]
                if extra is not None:
                    pe = cv2.warpAffine(extra_tex, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    extra[y0:y1, x0:x1][m] = pe[m]
    return out, extra, np.isfinite(zbuf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="synth")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--baseline", type=float, default=150.0, help="mm between cameras")
    ap.add_argument("--fold1", type=float, default=9.0)
    ap.add_argument("--fold2", type=float, default=-7.0)
    ap.add_argument("--bend", type=float, default=2.0)
    ap.add_argument("--curl", type=float, default=2.5)
    ap.add_argument("--corner", type=float, default=6.0)
    ap.add_argument("--noise", type=float, default=2.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    w, h = 3000, 4000  # portrait phone photo
    f = 26.0 * np.hypot(w, h) / 43.27
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])

    p = dict(fold1=args.fold1, fold2=args.fold2, bend=args.bend, curl=args.curl, corner=args.corner)
    a = np.radians(4)
    Rp = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    tp = np.array([-W / 2, -H / 2, 0.0])
    tex = make_letter(args.seed)
    uvtex = np.dstack(np.meshgrid((np.arange(tex.shape[1]) + 0.5) / PPM,
                                  (np.arange(tex.shape[0]) + 0.5) / PPM)).astype(np.float32)

    bgl = rng.normal(0, 1, (h // 8, w // 8)).astype(np.float32)
    bgl = cv2.GaussianBlur(bgl, (0, 0), 3)
    bgl = cv2.resize(bgl, (w, h))
    stripes = 8 * np.sin(np.arange(w)[None, :] / 37.0 + 3 * bgl)
    bg = np.clip(np.dstack([60 + stripes + 10 * bgl, 80 + stripes + 10 * bgl, 105 + stripes + 10 * bgl]), 0, 255).astype(np.uint8)

    cams = [np.array([-args.baseline / 2, 15.0, -360.0]), np.array([args.baseline / 2, -10.0, -350.0])]
    truth = dict(K=K, Rp=Rp, tp=tp, **{k: v for k, v in p.items()})
    for k, cam in enumerate(cams):
        R, t = look_at(cam, np.array([0.0, 0.0, 0.0]) + rng.normal(0, 5, 3))
        img, uvmap, mask = render(tex, K, R, t, (w, h), p, (Rp, tp), bg, uvtex if k == 0 else None)
        shade = 1.0 - 0.15 * ((np.arange(w)[None, :] - w / 2) ** 2 + (np.arange(h)[:, None] - h / 2) ** 2) / (w * w / 4 + h * h / 4)
        img = img.astype(np.float32) * shade[..., None]
        img = cv2.GaussianBlur(img, (0, 0), 0.7)
        img += rng.normal(0, args.noise, img.shape)
        cv2.imwrite(os.path.join(args.out, f"view{k+1}.jpg"), np.clip(img, 0, 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        truth[f"R{k+1}"], truth[f"t{k+1}"] = R, t
        if k == 0:
            truth["uvmap"] = np.where(mask[..., None], uvmap, np.nan).astype(np.float32)
    cv2.imwrite(os.path.join(args.out, "letter_truth.png"), cv2.resize(tex, None, fx=200 / 25.4 / PPM, fy=200 / 25.4 / PPM, interpolation=cv2.INTER_AREA))
    np.savez_compressed(os.path.join(args.out, "truth.npz"), **truth)
    print(f"focal_px={f:.3f}")


if __name__ == "__main__":
    main()
