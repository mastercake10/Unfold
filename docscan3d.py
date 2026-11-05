#!/usr/bin/env python3
"""Two-view 3D reconstruction of a curved/folded document page.

Outputs measured surface points as (u, v, z) in mm:
  u, v  coordinates in the best-fit page plane, aligned to the page edges
        (u along the short edge 0..W, v along the long edge 0..H)
  z     height above that plane (towards the camera is positive)
  s, t  where that point lands on the flattened page (isometric unrolling)

Also writes a smooth height field z(u, v), the unrolled grid and flat.png.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- camera

def exif_focal_px(path, width, height):
    try:
        exif = Image.open(path).getexif()
        sub = exif.get_ifd(0x8769)
        f35 = sub.get(0xA405) or exif.get(0xA405)
        if f35:
            return float(f35) * np.hypot(width, height) / 43.27, f"EXIF 35mm-equiv {f35} mm"
    except Exception:
        pass
    return None, None


def build_K(args, path, w, h):
    if args.calib:
        d = np.load(args.calib)
        K = d["K"].astype(np.float64)
        dist = d["dist"].astype(np.float64) if "dist" in d else np.zeros(5)
        sw, sh = (d["size"] if "size" in d else (w, h))
        K[0] *= w / sw
        K[1] *= h / sh
        return K, dist, "calibration file"
    if args.focal_px:
        f, src = args.focal_px, "--focal-px"
    else:
        f, src = exif_focal_px(path, w, h)
        if f is None:
            f, src = 26.0 * np.hypot(w, h) / 43.27, "fallback (26 mm equiv, typical phone)"
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]], np.float64)
    return K, np.zeros(5), src


def load_image(path, max_side):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {path}")
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w)) if max_side else 1.0
    return img, s


# ---------------------------------------------------------------- page

def find_page_quad(img):
    """Largest bright region, approximated by a quadrilateral (tl, tr, br, bl)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small_s = 1000 / max(g.shape)
    sm = cv2.resize(g, None, fx=small_s, fy=small_s, interpolation=cv2.INTER_AREA)
    # close text strokes so the page becomes one blob
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    _, th = cv2.threshold(cv2.GaussianBlur(sm, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.1 * sm.size:
        return None, None
    mask = np.zeros_like(sm)
    cv2.drawContours(mask, [c], -1, 255, -1)
    mask = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)

    hull = cv2.convexHull(c)
    quad = None
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)
    quad /= small_s
    # order tl, tr, br, bl
    ssum, sdiff = quad.sum(1), quad[:, 1] - quad[:, 0]
    quad = np.array([quad[np.argmin(ssum)], quad[np.argmin(sdiff)],
                     quad[np.argmax(ssum)], quad[np.argmax(sdiff)]])
    return quad, mask


def refine_quad(gray, quad, half=25, n=40):
    """Subpixel corners: fit lines to the strongest edge along short normal profiles
    near each corner, then intersect. Only the part of each edge near the corner is
    used, so bent page edges don't bias the result."""
    from scipy.ndimage import map_coordinates
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    out = quad.copy()
    offs = np.arange(-half, half + 1, dtype=float)

    def edge_pts(a, b, t0, t1):
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        pts = []
        for tt in np.linspace(t0, t1, n):
            p = a + tt * (b - a)
            xy = p[None] + offs[:, None] * nrm[None]
            prof = map_coordinates(g, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
            gr = np.abs(np.gradient(prof))
            k = int(np.argmax(gr))
            if not 0 < k < len(gr) - 1 or gr[k] < 5:
                continue
            den = gr[k - 1] - 2 * gr[k] + gr[k + 1]
            sub = 0.5 * (gr[k - 1] - gr[k + 1]) / den if abs(den) > 1e-9 else 0.0
            pts.append(p + (offs[k] + sub) * nrm)
        return np.array(pts)

    for i in range(4):
        prv, cur, nxt = quad[i - 1], quad[i], quad[(i + 1) % 4]
        e1 = edge_pts(prv, cur, 0.72, 0.97)
        e2 = edge_pts(cur, nxt, 0.03, 0.28)
        if len(e1) < 8 or len(e2) < 8:
            continue
        lines = []
        for e in (e1, e2):
            vx, vy, x0, y0 = cv2.fitLine(e.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        (p, r), (q, s) = lines
        A = np.array([r, -s]).T
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        lam = np.linalg.solve(A, q - p)[0]
        c = p + lam * r
        if np.linalg.norm(c - cur) < 3 * half:
            out[i] = c
    return out


# ---------------------------------------------------------------- matching

def match_features(g1, g2, mask1, n_features):
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    k1, d1 = sift.detectAndCompute(g1, mask1)
    k2, d2 = sift.detectAndCompute(g2, None)
    log(f"  keypoints: {len(k1)} / {len(k2)}")
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    m12 = flann.knnMatch(d1, d2, k=2)
    m21 = flann.knnMatch(d2, d1, k=2)
    best21 = {m[0].queryIdx: m[0].trainIdx for m in m21 if len(m) == 2 and m[0].distance < 0.85 * m[1].distance}
    p1, p2 = [], []
    for m in m12:
        if len(m) < 2 or m[0].distance > 0.85 * m[1].distance:
            continue
        if best21.get(m[0].trainIdx) != m[0].queryIdx:
            continue
        p1.append(k1[m[0].queryIdx].pt)
        p2.append(k2[m[0].trainIdx].pt)
    return np.float64(p1), np.float64(p2)


def refine_subpixel(g1, g2, p1, p2, H_local=None, win=15, search=4):
    """NCC template matching of img1 patches (pre-warped by the local affine A)
    in a +-search window around p2, with parabolic subpixel peak."""
    out = p2.copy()
    ok = np.ones(len(p1), bool)
    r = win // 2
    h1, w1 = g1.shape
    h2, w2 = g2.shape
    g1f = g1.astype(np.float32)
    g2f = g2.astype(np.float32)
    for i, (a, b) in enumerate(zip(p1, p2)):
        x1, y1 = a
        x2, y2 = b
        if not (r + 1 <= x1 < w1 - r - 1 and r + 1 <= y1 < h1 - r - 1 and
                r + search + 1 <= x2 < w2 - r - search - 1 and r + search + 1 <= y2 < h2 - r - search - 1):
            ok[i] = False
            continue
        if H_local is not None:
            A = H_local[i]
            M = np.hstack([A, (np.array([x2, y2]) - A @ np.array([x1, y1]))[:, None]])
            # template = patch of img1 mapped into img2 geometry around (x2, y2)
            Minv = cv2.invertAffineTransform(M)
            T = np.float32([[1, 0, x2 - r], [0, 1, y2 - r]])
            full = np.vstack([Minv, [0, 0, 1]]) @ np.vstack([T, [0, 0, 1]])
            tpl = cv2.warpAffine(g1f, full[:2], (win, win), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        else:
            tpl = cv2.getRectSubPix(g1f, (win, win), (x1, y1))
        reg = cv2.getRectSubPix(g2f, (win + 2 * search, win + 2 * search), (x2, y2))
        if tpl.std() < 4:
            ok[i] = False
            continue
        res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (cx, cy) = cv2.minMaxLoc(res)
        if mx < 0.85 or not (0 < cx < res.shape[1] - 1 and 0 < cy < res.shape[0] - 1):
            ok[i] = False
            continue
        def para(l, c, rr):
            d = l - 2 * c + rr
            return 0.0 if abs(d) < 1e-9 else 0.5 * (l - rr) / d
        dx = para(res[cy, cx - 1], res[cy, cx], res[cy, cx + 1])
        dy = para(res[cy - 1, cx], res[cy, cx], res[cy + 1, cx])
        out[i] = (x2 - search + cx + dx, y2 - search + cy + dy)
    return out, ok


def local_affines(p1, p2, q=None, k=12):
    """Affine map img1->img2 around each query point, fitted to the k nearest matches.
    Returns (2x2 linear parts, predicted positions of q in img2)."""
    from scipy.spatial import cKDTree
    q = p1 if q is None else q
    tree = cKDTree(p1)
    _, idx = tree.query(q, k=min(k, len(p1)))
    A = np.empty((len(q), 2, 2))
    pred = np.empty((len(q), 2))
    for i, nb in enumerate(idx):
        m1, m2 = p1[nb].mean(0), p2[nb].mean(0)
        sol, *_ = np.linalg.lstsq(p1[nb] - m1, p2[nb] - m2, rcond=None)
        A[i] = sol.T
        pred[i] = m2 + A[i] @ (q[i] - m1)
    return A, pred


def densify(g1, g2, mask, p1, p2, max_pts, min_dist, search):
    """Guided matching: corners in img1, position in img2 predicted from nearby
    sparse matches, then refined by NCC on affine-pre-warped patches."""
    q = cv2.goodFeaturesToTrack(g1, max_pts, 0.005, min_dist, mask=mask, blockSize=5)
    if q is None:
        return np.empty((0, 2)), np.empty((0, 2))
    q = q.reshape(-1, 2).astype(np.float64)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(p1).query(q)
    q = q[d < 200]  # only predict near existing matches
    A, pred = local_affines(p1, p2, q)
    r, ok = refine_subpixel(g1, g2, q, pred, A, win=21, search=search)
    ok &= np.linalg.norm(r - pred, axis=1) < search - 1
    return q[ok], r[ok]


# ---------------------------------------------------------------- geometry

def rodrigues(r):
    return cv2.Rodrigues(r.reshape(3, 1))[0]


def project(K, R, t, X):
    Xc = X @ R.T + t
    x = Xc @ K.T
    return x[:, :2] / x[:, 2:3], Xc[:, 2]


def triangulate(K, R, t, p1, p2):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    return (Xh[:3] / Xh[3]).T


def bundle_adjust(K, R, t, X, p1, p2, refine_focal):
    n = len(X)
    t = t / np.linalg.norm(t)
    f0 = K[0, 0]

    def unpack(x):
        r = x[:3]
        tt = x[3:6]
        df = x[6] if refine_focal else 0.0
        Xs = x[6 + refine_focal:].reshape(n, 3)
        Kc = K.copy()
        Kc[0, 0] = Kc[1, 1] = f0 * (1 + df)
        return rodrigues(r), tt, Kc, Xs

    def resid(x):
        Rr, tt, Kc, Xs = unpack(x)
        a, _ = project(Kc, np.eye(3), np.zeros(3), Xs)
        b, _ = project(Kc, Rr, tt, Xs)
        return np.concatenate([(a - p1).ravel(), (b - p2).ravel(),
                               [1e3 * (tt @ tt - 1.0)]])

    nc = 6 + refine_focal
    S = sparse.lil_matrix((4 * n + 1, nc + 3 * n), dtype=int)
    for i in range(n):
        for rr in (2 * i, 2 * i + 1):
            S[rr, nc + 3 * i:nc + 3 * i + 3] = 1
            if refine_focal:
                S[rr, 6] = 1
        for rr in (2 * n + 2 * i, 2 * n + 2 * i + 1):
            S[rr, :nc] = 1
            S[rr, nc + 3 * i:nc + 3 * i + 3] = 1
    S[4 * n, 3:6] = 1

    x0 = np.concatenate([cv2.Rodrigues(R)[0].ravel(), t, [0.0] * refine_focal, X.ravel()])
    res = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=1.0,
                        x_scale="jac", method="trf", max_nfev=60, verbose=0)
    Rr, tt, Kc, Xs = unpack(res.x)
    return Rr, tt, Kc, Xs


# ---------------------------------------------------------------- page frame

def fit_plane(X):
    c = np.median(X, 0)
    for _ in range(3):
        _, _, vt = np.linalg.svd(X - c, full_matrices=False)
        n = vt[2]
        d = (X - c) @ n
        keep = np.abs(d) < 3 * np.median(np.abs(d)) + 1e-12
        c = X[keep].mean(0)
    _, _, vt = np.linalg.svd(X[keep] - c, full_matrices=False)
    n = vt[2]
    if n @ c > 0:  # normal points towards camera 1 (origin)
        n = -n
    return c, n


def ray_plane(K, px, c, n):
    rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T
    s = (c @ n) / (rays @ n)
    return rays * s[:, None]


def page_frame(K, quad, c, n, page_w, page_h):
    """Origin at page corner, eu along short edge, ev along long edge, en = normal.
    Returns frame and mm-per-unit scale (from the measured page outline)."""
    C = ray_plane(K, quad, c, n)  # tl, tr, br, bl
    top, right = np.linalg.norm(C[1] - C[0]), np.linalg.norm(C[2] - C[1])
    bottom, left = np.linalg.norm(C[2] - C[3]), np.linalg.norm(C[3] - C[0])
    horiz, vert = (top + bottom) / 2, (left + right) / 2
    if horiz <= vert:          # portrait in image: u = image-x direction
        o, ue, ve = C[0], C[1] - C[0], C[3] - C[0]
        short, long_ = horiz, vert
    else:                      # landscape in image: rotate so u is the short edge
        o, ue, ve = C[3], C[0] - C[3], C[2] - C[3]
        short, long_ = vert, horiz
    eu = ue - (ue @ n) * n
    eu /= np.linalg.norm(eu)
    ev = np.cross(n, eu)
    if ev @ ve < 0:
        n = -n
        ev = np.cross(n, eu)
    # page outline measured in the plane is the projected size (slightly short
    # where the sheet is tilted); good to ~1 % for typical folds
    scale = 0.5 * (page_w / short + page_h / long_)
    return dict(o=o, eu=eu, ev=ev, en=n, scale=scale,
                measured_mm=(short * scale, long_ * scale), corners=C)


def to_page(F, X):
    d = X - F["o"]
    return np.c_[d @ F["eu"], d @ F["ev"], d @ F["en"]] * F["scale"]


def from_page(F, U):
    U = U / F["scale"]
    return F["o"] + U[..., 0:1] * F["eu"] + U[..., 1:2] * F["ev"] + U[..., 2:3] * F["en"]


# ---------------------------------------------------------------- height field

def fit_heightfield(uvz, W, H, step, lam, n_irls=6):
    """Robust smooth z(u,v) on a grid: bilinear data term + second-difference
    regularizer, both IRLS-Huber so creases stay sharp-ish and outliers drop out."""
    gu = np.arange(-step, W + 2 * step, step)
    gv = np.arange(-step, H + 2 * step, step)
    nu, nv = len(gu), len(gv)
    N = nu * nv
    idx = lambda i, j: j * nu + i

    u, v, z = uvz.T
    fi = (u - gu[0]) / step
    fj = (v - gv[0]) / step
    inside = (fi >= 0) & (fi < nu - 1) & (fj >= 0) & (fj < nv - 1)
    u, v, z, fi, fj = u[inside], v[inside], z[inside], fi[inside], fj[inside]
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0
    rows = np.repeat(np.arange(len(z)), 4)
    cols = np.stack([idx(i0, j0), idx(i0 + 1, j0), idx(i0, j0 + 1), idx(i0 + 1, j0 + 1)], 1).ravel()
    vals = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], 1).ravel()
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(z), N))

    r, c, vv = [], [], []
    k = 0
    for j in range(nv):
        for i in range(1, nu - 1):
            r += [k] * 3; c += [idx(i - 1, j), idx(i, j), idx(i + 1, j)]; vv += [1, -2, 1]; k += 1
    for j in range(1, nv - 1):
        for i in range(nu):
            r += [k] * 3; c += [idx(i, j - 1), idx(i, j), idx(i, j + 1)]; vv += [1, -2, 1]; k += 1
    D = sparse.csr_matrix((vv, (r, c)), shape=(k, N))

    # lam is per unit area; scale so it is independent of point density
    dens = len(z) / (W * H) * step * step
    wd = np.ones(len(z))
    wr = np.ones(k) * lam * dens
    zz = np.zeros(N)
    for _ in range(n_irls):
        Wd = sparse.diags(wd)
        Wr = sparse.diags(wr)
        M = A.T @ Wd @ A + D.T @ Wr @ D + 1e-9 * sparse.eye(N)
        zz = spsolve(M.tocsc(), A.T @ (wd * z))
        rd = A @ zz - z
        sd = 1.4826 * np.median(np.abs(rd)) + 1e-6
        wd = np.minimum(1.0, 2 * sd / (np.abs(rd) + 1e-12))
        rr = D @ zz
        sr = 1.4826 * np.median(np.abs(rr)) + 1e-6
        wr = lam * dens * np.minimum(1.0, 3 * sr / (np.abs(rr) + 1e-12))
    Z = zz.reshape(nv, nu)
    resid = A @ zz - z
    return gu, gv, Z, resid


def sample_heightfield(gu, gv, Z, U, V):
    step = gu[1] - gu[0]
    fi = np.clip((U - gu[0]) / step, 0, len(gu) - 1.001)
    fj = np.clip((V - gv[0]) / step, 0, len(gv) - 1.001)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0
    return ((1 - a) * (1 - b) * Z[j0, i0] + a * (1 - b) * Z[j0, i0 + 1]
            + (1 - a) * b * Z[j0 + 1, i0] + a * b * Z[j0 + 1, i0 + 1])


def ray_surface(K, px, F, gu, gv, Z, c, n):
    """Intersect camera-1 rays with the measured surface (bisection along the ray)."""
    rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T
    lam0 = (c @ n) / (rays @ n)
    out = []
    for r, l0 in zip(rays, lam0):
        def f(lam):
            uvz = to_page(F, (lam * r)[None])[0]
            return uvz[2] - sample_heightfield(gu, gv, Z, uvz[:1], uvz[1:2])[0]
        lo, hi = 0.85 * l0, 1.15 * l0
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            out.append(l0 * r)
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        out.append(0.5 * (lo + hi) * r)
    return np.array(out)


def flatten_grid(gu, gv, Z):
    """Unroll the height-field mesh into the plane, preserving 3D edge lengths
    (paper does not stretch). Returns flat (s, t) for every grid vertex, in mm."""
    U, V = np.meshgrid(gu, gv)
    P = np.dstack([U, V, Z]).reshape(-1, 3)
    nv_, nu_ = Z.shape
    idx = np.arange(nu_ * nv_).reshape(nv_, nu_)
    E = np.concatenate([
        np.c_[idx[:, :-1].ravel(), idx[:, 1:].ravel()],
        np.c_[idx[:-1, :].ravel(), idx[1:, :].ravel()],
        np.c_[idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()],
        np.c_[idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()],
    ])
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    x0 = P[:, :2].ravel().copy()
    anchor = 1e-3

    def resid(x):
        q = x.reshape(-1, 2)
        d = np.linalg.norm(q[E[:, 0]] - q[E[:, 1]], axis=1)
        return np.concatenate([d - L, anchor * (x - x0)])

    m = len(E)
    S = sparse.lil_matrix((m + len(x0), len(x0)), dtype=int)
    for k, (i, j) in enumerate(E):
        S[k, 2 * i:2 * i + 2] = 1
        S[k, 2 * j:2 * j + 2] = 1
    S[m:, :] = sparse.eye(len(x0), dtype=int)
    res = least_squares(resid, x0, jac_sparsity=S, method="trf", x_scale=1.0, max_nfev=200)
    st = res.x.reshape(nv_, nu_, 2)
    strain = np.abs(resid(res.x)[:m]) / L
    return st, float(np.median(strain)), float(strain.max())


def align_flat(st, corners_st, W, H):
    """Rigid 2D transform taking the flat page corners to the W x H rectangle."""
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    a, b = corners_st - corners_st.mean(0), dst - dst.mean(0)
    U_, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, np.sign(np.linalg.det(U_ @ Vt))])
    R = U_ @ D @ Vt
    t = dst.mean(0) - R @ corners_st.mean(0)
    return st @ R.T + t, corners_st @ R.T + t


def flat_to_uv(st_grid, gu, gv, W, H, res_mm=0.5):
    """Invert the grid map (u, v) -> (s, t) on a regular (s, t) raster."""
    from scipy.interpolate import griddata
    S, T = np.meshgrid(np.arange(0, W + res_mm, res_mm), np.arange(0, H + res_mm, res_mm))
    U, V = np.meshgrid(gu, gv)
    pts = st_grid.reshape(-1, 2)
    u = griddata(pts, U.ravel(), (S, T), method="linear")
    v = griddata(pts, V.ravel(), (S, T), method="linear")
    return u.astype(np.float32), v.astype(np.float32), res_mm


# ---------------------------------------------------------------- outputs

def write_ply(path, P, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(P)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(P, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[2]} {c[1]} {c[0]}\n")


def save_heightmap_png(path, gu, gv, Z, uvz, W, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 7.5), constrained_layout=True)
    m = (gu >= 0) & (gu <= W)
    n = (gv >= 0) & (gv <= H)
    Zc = Z[np.ix_(n, m)]
    lim = np.nanmax(np.abs(Zc))
    im = ax[0].imshow(Zc, extent=(gu[m][0], gu[m][-1], gv[n][-1], gv[n][0]),
                      cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[0].contour(gu[m], gv[n], Zc, levels=15, colors="k", linewidths=0.4, alpha=0.6)
    ax[0].set_title("fitted height z(u, v) [mm]")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    sc = ax[1].scatter(uvz[:, 0], uvz[:, 1], c=uvz[:, 2], s=1, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1].set_xlim(0, W); ax[1].set_ylim(H, 0); ax[1].set_aspect("equal")
    ax[1].set_title(f"measured points (n={len(uvz)})")
    fig.colorbar(sc, ax=ax[1], shrink=0.8)
    for a in ax:
        a.set_xlabel("u [mm]"); a.set_ylabel("v [mm]")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_flat(img, K, dist, F, gu, gv, Z, W, H, dpi, inv=None, R=None, t=None):
    """Output pixel (s, t) on the flat page -> surface point (u, v, z(u, v)) ->
    pixel in the photo. Without `inv`, (s, t) = (u, v), i.e. plane projection only."""
    ppm = dpi / 25.4
    ow, oh = int(round(W * ppm)), int(round(H * ppm))
    if inv is None:
        U, V = np.meshgrid((np.arange(ow) + 0.5) / ppm, (np.arange(oh) + 0.5) / ppm)
    else:
        u_r, v_r, res_mm = inv
        sx = ((np.arange(ow) + 0.5) / ppm / res_mm).astype(np.float32)
        sy = ((np.arange(oh) + 0.5) / ppm / res_mm).astype(np.float32)
        MX, MY = np.meshgrid(sx, sy)
        U = cv2.remap(u_r, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        V = cv2.remap(v_r, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bad = ~np.isfinite(U) | ~np.isfinite(V)
        U[bad], V[bad] = -1e4, -1e4
    Zs = sample_heightfield(gu, gv, Z, U.ravel(), V.ravel())
    X = from_page(F, np.c_[U.ravel(), V.ravel(), Zs])
    if R is None:
        R, t = np.eye(3), np.zeros(3)
    rv = cv2.Rodrigues(R)[0]
    px, _ = cv2.projectPoints(X.reshape(-1, 1, 3), rv, t.reshape(3, 1), K, dist)
    mx = px[:, 0, 0].reshape(oh, ow).astype(np.float32)
    my = px[:, 0, 1].reshape(oh, ow).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image1", help="reference view (should show the whole page)")
    ap.add_argument("image2", help="second view, ~10-20 cm sideways from the first")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--page-size", default="210x297", help="page W x H in mm (default A4)")
    ap.add_argument("--focal-px", type=float, help="focal length in pixels at full resolution")
    ap.add_argument("--calib", help="npz with K, dist (and optionally size=(w,h)) from cv2.calibrateCamera")
    ap.add_argument("--fixed-focal", action="store_true", help="do not refine the focal length in bundle adjustment")
    ap.add_argument("--max-side", type=int, default=4000, help="downscale longer side to this many px")
    ap.add_argument("--features", type=int, default=60000)
    ap.add_argument("--ba-points", type=int, default=6000, help="points used in bundle adjustment")
    ap.add_argument("--grid", type=float, default=4.0, help="height-field grid spacing in mm")
    ap.add_argument("--smooth", type=float, default=30.0, help="height-field smoothness weight")
    ap.add_argument("--dpi", type=int, default=200, help="dpi of flattened preview")
    ap.add_argument("--no-subpixel", action="store_true")
    ap.add_argument("--no-densify", action="store_true")
    ap.add_argument("--dense", type=int, default=40000, help="max candidate points for densification")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    W, H = map(float, args.page_size.lower().split("x"))

    img1, s1 = load_image(args.image1, args.max_side)
    img2, s2 = load_image(args.image2, args.max_side)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    K1, dist1, ksrc = build_K(args, args.image1, w1, h1)
    K2, dist2, _ = build_K(args, args.image2, w2, h2)
    if (h1, w1) != (h2, w2) or not np.allclose(K1, K2):
        sys.exit("both images must come from the same camera at the same resolution")
    K, dist = K1, dist1
    log(f"focal: {K[0,0]:.1f} px ({ksrc})")

    if np.any(dist):
        img1 = cv2.undistort(img1, K, dist)
        img2 = cv2.undistort(img2, K, dist)
        dist = np.zeros(5)
    full1 = img1
    if s1 < 1:
        img1 = cv2.resize(img1, None, fx=s1, fy=s1, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
        K = K.copy(); K[:2] *= s1
        log(f"processing at {img1.shape[1]}x{img1.shape[0]}")

    log("page outline ...")
    quad, mask1 = find_page_quad(img1)
    if quad is None:
        sys.exit("could not find the page in image1 (needs a page brighter than the background)")
    q0 = quad.copy()
    quad = refine_quad(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), quad)
    log(f"  corners refined by {np.linalg.norm(quad - q0, axis=1).round(1).tolist()} px")
    mask_er = cv2.erode(mask1, np.ones((15, 15), np.uint8))

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    log("matching ...")
    p1, p2 = match_features(g1, g2, mask_er, args.features)
    log(f"  mutual matches: {len(p1)}")
    if len(p1) < 50:
        sys.exit("too few matches")

    E, inl = cv2.findEssentialMat(p1, p2, K, method=cv2.USAC_MAGSAC, prob=0.99999, threshold=1.0)
    inl = inl.ravel().astype(bool)
    p1, p2 = p1[inl], p2[inl]
    _, R, t, _ = cv2.recoverPose(E, p1, p2, K)
    t = t.ravel()
    log(f"  essential-matrix inliers: {len(p1)}")

    if not args.no_subpixel:
        log("subpixel refinement ...")
        A, _ = local_affines(p1, p2)
        p2r, ok = refine_subpixel(g1, g2, p1, p2, A)
        moved = np.linalg.norm(p2r - p2, axis=1)
        ok &= moved < 1.5
        p1, p2 = p1[ok], p2r[ok]
        log(f"  kept {len(p1)} (median shift {np.median(moved[ok]):.2f} px)")

    X = triangulate(K, R, t, p1, p2)
    a, z1 = project(K, np.eye(3), np.zeros(3), X)
    b, z2 = project(K, R, t, X)
    err = np.maximum(np.linalg.norm(a - p1, axis=1), np.linalg.norm(b - p2, axis=1))
    good = (z1 > 0) & (z2 > 0) & (err < 2.0)
    p1, p2, X = p1[good], p2[good], X[good]

    log("bundle adjustment ...")
    rng = np.random.default_rng(0)
    sel = rng.choice(len(X), min(args.ba_points, len(X)), replace=False)
    R, t, K, _ = bundle_adjust(K, R, t, X[sel], p1[sel], p2[sel], int(not args.fixed_focal))
    X = triangulate(K, R, t, p1, p2)
    a, z1 = project(K, np.eye(3), np.zeros(3), X)
    b, z2 = project(K, R, t, X)
    e1 = np.linalg.norm(a - p1, axis=1)
    e2 = np.linalg.norm(b - p2, axis=1)
    err = np.maximum(e1, e2)
    good = (z1 > 0) & (z2 > 0) & (err < 1.0)
    p1, p2, X, err = p1[good], p2[good], X[good], err[good]

    cam2 = -R.T @ t
    ang = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", X / np.linalg.norm(X, axis=1, keepdims=True),
                  (X - cam2) / np.linalg.norm(X - cam2, axis=1, keepdims=True)), -1, 1)))
    log(f"  points: {len(X)}, reprojection rms {np.sqrt(np.mean(e1[good]**2 + e2[good]**2) / 2):.3f} px, "
        f"triangulation angle median {np.median(ang):.1f} deg")
    if np.median(ang) < 4:
        log("  WARNING: small baseline; move the phone further between the two shots")

    c, n = fit_plane(X)
    F = page_frame(K, quad, c, n, W, H)
    uvz = to_page(F, X)

    if not args.no_densify:
        log("densifying ...")
        gu0, gv0, Z0, _ = fit_heightfield(uvz, W, H, 2 * args.grid, args.smooth)
        mind = max(4, int(round(min(g1.shape) / 400)))
        q1, q2 = densify(g1, g2, mask_er, p1, p2, args.dense, mind, search=6)
        Xd = triangulate(K, R, t, q1, q2)
        a, za = project(K, np.eye(3), np.zeros(3), Xd)
        b, zb = project(K, R, t, Xd)
        ed = np.maximum(np.linalg.norm(a - q1, axis=1), np.linalg.norm(b - q2, axis=1))
        uvzd = to_page(F, Xd)
        dz = uvzd[:, 2] - sample_heightfield(gu0, gv0, Z0, uvzd[:, 0], uvzd[:, 1])
        # wrong-letter matches slide along the epipolar line and show up as depth outliers
        keep = (za > 0) & (zb > 0) & (ed < 1.0) & (np.abs(dz) < 1.0)
        log(f"  candidates {len(q1)}, kept {keep.sum()}")
        p1 = np.r_[p1, q1[keep]]
        p2 = np.r_[p2, q2[keep]]
        X = np.r_[X, Xd[keep]]
        err = np.r_[err, ed[keep]]
        uvz = np.r_[uvz, uvzd[keep]]
        dup = np.unique(np.round(p1 / 2.0), axis=0, return_index=True)[1]
        p1, p2, X, err, uvz = p1[dup], p2[dup], X[dup], err[dup], uvz[dup]
        log(f"  total points {len(X)}")

    log(f"  page outline in plane: {F['measured_mm'][0]:.1f} x {F['measured_mm'][1]:.1f} mm (after scaling)")
    log(f"  baseline: {np.linalg.norm(cam2) * F['scale']:.0f} mm, "
        f"distance to page: {np.median(X[:, 2]) * F['scale']:.0f} mm")

    # Scale and origin come from the page outline. Corners first come from the
    # plane; afterwards they are re-intersected with the measured surface and the
    # outline is measured after flattening (true arc length), then everything is redone.
    for it in range(2):
        log(f"height field + flattening (pass {it + 1}) ...")
        gu, gv, Z, resid = fit_heightfield(uvz, W, H, args.grid, args.smooth)
        st, strain_med, strain_max = flatten_grid(gu, gv, Z)
        Cs = ray_surface(K, quad, F, gu, gv, Z, c, n)
        cuvz = to_page(F, Cs)
        cst = np.stack([
            cv2.remap(st[..., k].astype(np.float32),
                      ((cuvz[:, 0] - gu[0]) / (gu[1] - gu[0])).astype(np.float32)[:, None],
                      ((cuvz[:, 1] - gv[0]) / (gv[1] - gv[0])).astype(np.float32)[:, None],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0] for k in range(2)], 1)
        # corner order in the page frame: (0,0), (W,0), (W,H), (0,H)
        order = [int(np.argmin(np.linalg.norm(cuvz[:, :2] - q, axis=1)))
                 for q in ([0, 0], [W, 0], [W, H], [0, H])]
        cst = cst[order]
        short = 0.5 * (np.linalg.norm(cst[1] - cst[0]) + np.linalg.norm(cst[2] - cst[3]))
        long_ = 0.5 * (np.linalg.norm(cst[3] - cst[0]) + np.linalg.norm(cst[2] - cst[1]))
        k = 0.5 * (W / short + H / long_)
        log(f"  flat outline {short:.2f} x {long_:.2f} mm -> rescale {k:.4f}")
        if it == 0:
            F["scale"] *= k
            o_uv = cuvz[order[0], :2] * k
            F["o"] = from_page(F, np.array([o_uv[0], o_uv[1], 0.0]))
            uvz = to_page(F, X)
    st, cst = align_flat(st, cst, W, H)
    corner_err = np.linalg.norm(cst - [[0, 0], [W, 0], [W, H], [0, H]], axis=1)
    log(f"  z range {uvz[:,2].min():.2f} .. {uvz[:,2].max():.2f} mm, "
        f"point residual to surface (MAD) {1.4826*np.median(np.abs(resid)):.3f} mm")
    log(f"  flattening strain median {strain_med*100:.3f} %, max {strain_max*100:.2f} %, "
        f"corner fit {corner_err.max():.2f} mm")
    sx = (uvz[:, 0] - gu[0]) / (gu[1] - gu[0])
    sy = (uvz[:, 1] - gv[0]) / (gv[1] - gv[0])
    pst = np.stack([cv2.remap(st[..., j].astype(np.float32), sx.astype(np.float32)[:, None],
                              sy.astype(np.float32)[:, None], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)[:, 0] for j in range(2)], 1)

    # outputs
    cols = img1[np.clip(p1[:, 1].astype(int), 0, h1 - 1), np.clip(p1[:, 0].astype(int), 0, w1 - 1)]
    np.savetxt(os.path.join(args.out, "points.csv"),
               np.c_[uvz, pst, p1 / s1, p2 / s2, err],
               delimiter=",", fmt="%.4f", header="u_mm,v_mm,z_mm,s_mm,t_mm,x1,y1,x2,y2,reproj_px", comments="")
    write_ply(os.path.join(args.out, "points.ply"), uvz, cols)
    np.savez(os.path.join(args.out, "heightfield.npz"), u=gu, v=gv, z=Z, st=st,
             K=K, R=R, t=t, origin=F["o"], eu=F["eu"], ev=F["ev"], en=F["en"],
             scale=F["scale"], proc_scale=s1)
    save_heightmap_png(os.path.join(args.out, "heightfield.png"), gu, gv, Z, uvz, W, H)

    dbg = img1.copy()
    zl = np.percentile(np.abs(uvz[:, 2]), 99) + 1e-6
    zc = np.clip((uvz[:, 2] / zl + 1) * 127.5, 0, 255).astype(np.uint8)
    cm = cv2.applyColorMap(zc.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    for (x, y), cc in zip(p1.astype(int), cm):
        cv2.circle(dbg, (x, y), 2, tuple(int(v) for v in cc), -1)
    cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(args.out, "overlay.jpg"), dbg, [cv2.IMWRITE_JPEG_QUALITY, 85])

    Kfull = K.copy(); Kfull[:2] /= s1
    inv = flat_to_uv(st, gu, gv, W, H)
    flat = render_flat(full1, Kfull, dist, F, gu, gv, Z, W, H, args.dpi, inv=inv)
    cv2.imwrite(os.path.join(args.out, "flat.png"), flat)

    report = dict(points=int(len(X)), focal_px=float(Kfull[0, 0]), focal_source=ksrc,
                  reproj_rms_px=float(np.sqrt(np.mean(err ** 2))),
                  triangulation_angle_median_deg=float(np.median(ang)),
                  baseline_mm=float(np.linalg.norm(cam2) * F["scale"]),
                  distance_mm=float(np.median(X[:, 2]) * F["scale"]),
                  page_measured_mm=[float(v) for v in F["measured_mm"]],
                  z_min_mm=float(uvz[:, 2].min()), z_max_mm=float(uvz[:, 2].max()),
                  surface_residual_mad_mm=float(1.4826 * np.median(np.abs(resid))),
                  flatten_strain_median=strain_med, flatten_strain_max=strain_max,
                  corner_fit_max_mm=float(corner_err.max()))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"done -> {args.out}/")


if __name__ == "__main__":
    main()
