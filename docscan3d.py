#!/usr/bin/env python3
"""Multi-view 3D reconstruction and flattening of a curved/folded document page.

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


# ---------------------------------------------------------------- camera model
# One shared pinhole camera with radial distortion (k1, k2) for all views.
# View 0 is the reference: R = I, t = 0. World frame = reference camera frame.

def cam_K(cam):
    return np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1]])


def cam_dist(cam):
    return np.array([cam["k1"], cam["k2"], 0, 0, 0], float)


def distort_norm(xn, cam):
    r2 = (xn ** 2).sum(-1, keepdims=True)
    return xn * (1 + cam["k1"] * r2 + cam["k2"] * r2 * r2)


def norm_to_px(xn, cam):
    xd = distort_norm(xn, cam)
    return np.c_[cam["f"] * xd[:, 0] + cam["cx"], cam["f"] * xd[:, 1] + cam["cy"]]


def px_to_norm(px, cam):
    return cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), cam_K(cam), cam_dist(cam),
                               criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)).reshape(-1, 2)


def project(X, R, t, cam):
    Xc = X @ R.T + t
    return norm_to_px(Xc[:, :2] / Xc[:, 2:3], cam), Xc[:, 2]


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


def triangulate_multi(obs, Rs, ts, cam):
    """N-view DLT. obs: (N, V, 2) pixels with NaN for missing observations."""
    N, V, _ = obs.shape
    A = np.zeros((N, 2 * V, 4))
    for v in range(V):
        ok = np.isfinite(obs[:, v, 0])
        if not ok.any():
            continue
        xn = px_to_norm(obs[ok, v], cam)
        P = np.hstack([Rs[v], ts[v].reshape(3, 1)])
        A[ok, 2 * v] = xn[:, 0:1] * P[2] - P[0]
        A[ok, 2 * v + 1] = xn[:, 1:2] * P[2] - P[1]
    _, _, vt = np.linalg.svd(A)
    Xh = vt[:, -1]
    return Xh[:, :3] / Xh[:, 3:4]


def reproj_errors(X, obs, Rs, ts, cam):
    N, V, _ = obs.shape
    err = np.full((N, V), np.nan)
    depth_ok = np.ones(N, bool)
    for v in range(V):
        p, z = project(X, Rs[v], ts[v], cam)
        depth_ok &= z > 0
        ok = np.isfinite(obs[:, v, 0])
        err[ok, v] = np.linalg.norm(p[ok] - obs[ok, v], axis=1)
    return err, depth_ok


def bundle_adjust(cam, Rs, ts, obs, refine_f, n_dist, max_nfev=200):
    """Reduced bundle adjustment: shared intrinsics and poses of views 1.. are the
    only parameters; points are re-triangulated inside every evaluation.
    Gauge: view 0 fixed, |t_1| fixed."""
    V = len(Rs)
    n_in = int(refine_f) + n_dist
    t1n = np.linalg.norm(ts[1])
    has = np.isfinite(obs[:, :, 0])

    def unpack(x):
        c = dict(cam)
        k = 0
        if refine_f:
            c["f"] = cam["f"] * (1 + x[0]); k = 1
        if n_dist >= 1:
            c["k1"] = x[k]; k += 1
        if n_dist >= 2:
            c["k2"] = x[k]; k += 1
        Rv, tv = [np.eye(3)], [np.zeros(3)]
        for j in range(V - 1):
            o = n_in + 6 * j
            Rv.append(rodrigues(x[o:o + 3])); tv.append(x[o + 3:o + 6])
        return c, Rv, tv

    def resid(x):
        c, Rv, tv = unpack(x)
        X = triangulate_multi(obs, Rv, tv, c)
        r = []
        for v in range(V):
            p, z = project(X, Rv[v], tv[v], c)
            d = p - obs[:, v]
            d[~has[:, v] | (z <= 0)] = 0
            r.append(d.ravel())
        return np.r_[np.concatenate(r), 1e3 * (np.linalg.norm(tv[1]) - t1n) / t1n]

    x0 = [0.0] * int(refine_f) + [cam["k1"], cam["k2"]][:n_dist]
    for j in range(1, V):
        x0 += list(cv2.Rodrigues(Rs[j])[0].ravel()) + list(ts[j])
    x0 = np.array(x0, float)
    xs = np.r_[[0.01] * n_in, np.tile([0.01, 0.01, 0.01, 0.01 * t1n, 0.01 * t1n, 0.01 * t1n], V - 1)]
    res = least_squares(resid, x0, loss="huber", f_scale=1.0, x_scale=xs, diff_step=1e-7,
                        method="trf", max_nfev=max_nfev)
    if os.environ.get("BA_DEBUG"):
        r0 = resid(x0)[:-1]
        log(f"    BA: status {res.status} nfev {res.nfev}, rms {np.sqrt(np.mean(r0[r0 != 0]**2)):.3f} -> "
            f"{np.sqrt(np.mean(res.fun[:-1][res.fun[:-1] != 0]**2)):.3f}, intrinsics {res.x[:n_in]}")
    return unpack(res.x)


# ---------------------------------------------------------------- page outline

def page_outline(gray, mask, step=4.0, half=18):
    """Closed, ordered, subpixel page boundary in pixels (strongest edge along the
    normal of the smoothed mask contour)."""
    from scipy.ndimage import map_coordinates, gaussian_filter1d
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    tt = np.arange(0, cum[-1], step)
    cc = np.vstack([c, c[:1]])
    p = np.c_[np.interp(tt, cum, cc[:, 0]), np.interp(tt, cum, cc[:, 1])]
    p = np.c_[gaussian_filter1d(p[:, 0], 4, mode="wrap"), gaussian_filter1d(p[:, 1], 4, mode="wrap")]
    tan = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    nrm = np.c_[-tan[:, 1], tan[:, 0]]
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    offs = np.arange(-half, half + 1, dtype=float)
    xy = p[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    prof = map_coordinates(g, [xy[..., 1].ravel(), xy[..., 0].ravel()], order=1, mode="nearest").reshape(len(p), -1)
    gr = np.abs(np.gradient(prof, axis=1))
    k = np.argmax(gr, axis=1)
    ok = (k > 0) & (k < len(offs) - 1) & (gr[np.arange(len(p)), np.clip(k, 0, len(offs) - 1)] > 4)
    kk = np.clip(k, 1, len(offs) - 2)
    l, m, r = gr[np.arange(len(p)), kk - 1], gr[np.arange(len(p)), kk], gr[np.arange(len(p)), kk + 1]
    den = l - 2 * m + r
    sub = np.where(np.abs(den) > 1e-9, 0.5 * (l - r) / np.where(np.abs(den) > 1e-9, den, 1), 0)
    out = p + (offs[kk] + sub)[:, None] * nrm
    return out, ok


def assign_edges(pts, quad, corner_excl=0.06):
    """Edge index per outline point (0 top, 1 right, 2 bottom, 3 left in the image
    quad order tl,tr,br,bl); -1 near corners. Uses the order along the contour
    between the corners, so kinked edges of folded pages are kept."""
    n = len(pts)
    ci = [int(np.argmin(np.linalg.norm(pts - q, axis=1))) for q in quad]
    fwd = lambda a, b: (b - a) % n
    # contour direction: forward from corner 0, corner 1 must come before corners 2, 3
    forward = fwd(ci[0], ci[1]) < fwd(ci[0], ci[2]) and fwd(ci[0], ci[1]) < fwd(ci[0], ci[3])
    eid = np.full(n, -1)
    for k in range(4):
        a, b = ci[k], ci[(k + 1) % 4]
        L = fwd(a, b) if forward else fwd(b, a)
        for m in range(L + 1):
            if corner_excl * L < m < (1 - corner_excl) * L:
                eid[(a + m) % n if forward else (a - m) % n] = k
    return eid


def triangulate_outline(out0, outs, cam, Rs, ts, X_pred, min_sin=0.35, max_px=40.0):
    """Match reference-outline points to other views' outlines along epipolar lines.
    X_pred: rough 3D for each reference outline point (to pick the right crossing)."""
    V = len(Rs)
    obs = np.full((len(out0), V, 2), np.nan)
    obs[:, 0] = out0
    x0n = np.c_[px_to_norm(out0, cam), np.ones(len(out0))]
    for j in range(1, V):
        if outs[j] is None:
            continue
        pj = outs[j]
        xjn = np.c_[px_to_norm(pj, cam), np.ones(len(pj))]
        tx = np.array([[0, -ts[j][2], ts[j][1]], [ts[j][2], 0, -ts[j][0]], [-ts[j][1], ts[j][0], 0]])
        Ej = tx @ Rs[j]
        lines = x0n @ Ej.T
        pred, _ = project(X_pred, Rs[j], ts[j], cam)
        nxt = np.roll(xjn, -1, 0)
        seg = nxt[:, :2] - xjn[:, :2]
        segn = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12)
        for i, l in enumerate(lines):
            va = xjn @ l
            vb = nxt @ l
            cross = np.nonzero(va * vb < 0)[0]
            if len(cross) == 0:
                continue
            tpar = va[cross] / (va[cross] - vb[cross])
            pts = xjn[cross, :2] + tpar[:, None] * seg[cross]
            ppx = norm_to_px(pts, cam)
            d = np.linalg.norm(ppx - pred[i], axis=1)
            k = int(np.argmin(d))
            if d[k] > max_px:
                continue
            ldir = np.array([l[1], -l[0]]) / (np.hypot(l[0], l[1]) + 1e-12)
            if abs(cross2(ldir, segn[cross[k]])) < min_sin:
                continue
            obs[i, j] = ppx[k]
    has = np.isfinite(obs[:, 1:, 0]).any(1)
    X = np.full((len(out0), 3), np.nan)
    if has.any():
        X[has] = triangulate_multi(obs[has], Rs, ts, cam)
    return X, obs, has


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
    if n @ c > 0:  # normal points towards the reference camera (origin)
        n = -n
    return c, n


def rays_of(px, cam):
    return np.c_[px_to_norm(px, cam), np.ones(len(px))]


def ray_plane(rays, c, n):
    s = (c @ n) / (rays @ n)
    return rays * s[:, None]


def page_frame(cam, quad, c, n, page_w, page_h):
    """Origin at page corner, eu along short edge, ev along long edge, en = normal.
    Initial mm-per-unit scale from the outline projected onto the plane."""
    C = ray_plane(rays_of(quad, cam), c, n)  # tl, tr, br, bl
    top, right = np.linalg.norm(C[1] - C[0]), np.linalg.norm(C[2] - C[1])
    bottom, left = np.linalg.norm(C[2] - C[3]), np.linalg.norm(C[3] - C[0])
    horiz, vert = (top + bottom) / 2, (left + right) / 2
    if horiz <= vert:
        o, ue, ve = C[0], C[1] - C[0], C[3] - C[0]
        short, long_ = horiz, vert
        edge_map = {0: "top", 1: "right", 2: "bottom", 3: "left"}
    else:
        o, ue, ve = C[3], C[0] - C[3], C[2] - C[3]
        short, long_ = vert, horiz
        edge_map = {3: "top", 0: "right", 1: "bottom", 2: "left"}
    eu = ue - (ue @ n) * n
    eu /= np.linalg.norm(eu)
    ev = np.cross(n, eu)
    if ev @ ve < 0:
        n = -n
        ev = np.cross(n, eu)
    scale = 0.5 * (page_w / short + page_h / long_)
    return dict(o=o, eu=eu, ev=ev, en=n, scale=scale, edge_map=edge_map,
                measured_mm=(short * scale, long_ * scale))


def to_page(F, X):
    d = X - F["o"]
    return np.c_[d @ F["eu"], d @ F["ev"], d @ F["en"]] * F["scale"]


def from_page(F, U):
    U = U / F["scale"]
    return F["o"] + U[..., 0:1] * F["eu"] + U[..., 1:2] * F["ev"] + U[..., 2:3] * F["en"]


# ---------------------------------------------------------------- height field

def grid_bounds(uv, W, H, margin=15.0):
    """Height-field extent: the nominal page plus whatever the data covers. With
    steep folds the page does not sit at 0..W x 0..H in the plane frame."""
    lo = np.percentile(uv, 0.5, axis=0)
    hi = np.percentile(uv, 99.5, axis=0)
    u0 = min(-margin, lo[0] - margin); u1 = max(W + margin, hi[0] + margin)
    v0 = min(-margin, lo[1] - margin); v1 = max(H + margin, hi[1] + margin)
    # guard against outliers blowing up the grid (relative to where the data is)
    mu, mv = np.median(uv, axis=0)
    return (max(u0, mu - 1.2 * W), min(u1, mu + 1.2 * W), max(v0, mv - 1.2 * H), min(v1, mv + 1.2 * H))


def fold_direction(gu, gv, Z):
    """Dominant bending direction from the height field's curvature. Returns the
    in-plane angle of the fold lines (rulings) and an anisotropy in [0, 1]."""
    step = gu[1] - gu[0]
    zv, zu = np.gradient(Z, step)
    zuv, zuu = np.gradient(zu, step)
    zvv, _ = np.gradient(zv, step)
    T = np.zeros((2, 2))
    for a, b, c in zip(zuu.ravel(), zuv.ravel(), zvv.ravel()):
        Hm = np.array([[a, b], [b, c]])
        ev, evec = np.linalg.eigh(Hm)
        k = int(np.argmax(np.abs(ev)))
        wgt = abs(ev[k]) - abs(ev[1 - k])
        T += wgt * np.outer(evec[:, k], evec[:, k])
    ev, evec = np.linalg.eigh(T)
    across = evec[:, 1]
    aniso = (ev[1] - ev[0]) / (ev[1] + ev[0] + 1e-12)
    ruling = np.array([-across[1], across[0]])
    ang = float(np.arctan2(ruling[1], ruling[0]))
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2  # a fold line has no sign
    return ang, float(aniso)


def fit_heightfield(uvz, bounds, step, lam, n_irls=6, w=None, ruling_w=1.0):
    """Robust smooth z(u,v) on a grid: bilinear data term + second-difference
    regularizer, both IRLS-Huber so creases stay sharp-ish and outliers drop out."""
    u0, u1, v0, v1 = bounds
    gu = np.arange(u0, u1 + step, step)
    gv = np.arange(v0, v1 + step, step)
    nu, nv = len(gu), len(gv)
    N = nu * nv
    idx = np.arange(N).reshape(nv, nu)

    u, v, z = uvz.T
    w = np.ones(len(z)) if w is None else w
    fi = (u - gu[0]) / step
    fj = (v - gv[0]) / step
    inside = (fi >= 0) & (fi < nu - 1) & (fj >= 0) & (fj < nv - 1)
    z, fi, fj, w = z[inside], fi[inside], fj[inside], w[inside]
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0
    rows = np.repeat(np.arange(len(z)), 4)
    cols = np.stack([idx[j0, i0], idx[j0, i0 + 1], idx[j0 + 1, i0], idx[j0 + 1, i0 + 1]], 1).ravel()
    vals = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], 1).ravel()
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(z), N))

    tri = []
    tri.append(np.stack([idx[:, :-2].ravel(), idx[:, 1:-1].ravel(), idx[:, 2:].ravel()], 1))
    tri.append(np.stack([idx[:-2, :].ravel(), idx[1:-1, :].ravel(), idx[2:, :].ravel()], 1))
    n_u = len(tri[0])
    tri = np.concatenate(tri)
    k = len(tri)
    # extra stiffness along u when u is the fold (ruling) direction: paper stays straight there
    dir_w = np.r_[np.full(n_u, ruling_w), np.ones(k - n_u)]
    D = sparse.csr_matrix((np.tile([1.0, -2.0, 1.0], k), (np.repeat(np.arange(k), 3), tri.ravel())), shape=(k, N))

    area = (u1 - u0) * (v1 - v0)
    dens = w.sum() / area * step * step
    wd = w.copy()
    wr = dir_w * lam * dens
    for _ in range(n_irls):
        M = A.T @ sparse.diags(wd) @ A + D.T @ sparse.diags(wr) @ D + 1e-9 * sparse.eye(N)
        zz = spsolve(M.tocsc(), A.T @ (wd * z))
        rd = A @ zz - z
        sd = 1.4826 * np.median(np.abs(rd)) + 1e-6
        wd = w * np.minimum(1.0, 2 * sd / (np.abs(rd) + 1e-12))
        rr = D @ zz
        sr = 1.4826 * np.median(np.abs(rr)) + 1e-6
        # heavy-tailed (Cauchy) so genuine creases are not smoothed away
        wr = dir_w * lam * dens / (1 + (rr / (3 * sr)) ** 2)
    return gu, gv, zz.reshape(nv, nu), A @ zz - z


def sample_heightfield(gu, gv, Z, U, V):
    step = gu[1] - gu[0]
    fi = np.clip((U - gu[0]) / step, 0, len(gu) - 1.001)
    fj = np.clip((V - gv[0]) / step, 0, len(gv) - 1.001)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0
    return ((1 - a) * (1 - b) * Z[j0, i0] + a * (1 - b) * Z[j0, i0 + 1]
            + (1 - a) * b * Z[j0 + 1, i0] + a * b * Z[j0 + 1, i0 + 1])


def ray_surface(rays, F, gu, gv, Z, c, n, iters=40, n_scan=41):
    """Intersect reference-camera rays with the measured surface. The ray is scanned
    for all crossings and the one nearest the plane estimate is refined by bisection
    (the extrapolated surface beyond the page can produce a second, wrong crossing)."""
    lam0 = (c @ n) / (rays @ n)

    def f(lam):
        uvz = to_page(F, rays * lam[:, None])
        return uvz[:, 2] - sample_heightfield(gu, gv, Z, uvz[:, 0], uvz[:, 1])

    fr = np.linspace(0.6, 1.4, n_scan)
    vals = np.stack([f(lam0 * k) for k in fr], 1)
    cross = vals[:, :-1] * vals[:, 1:] <= 0
    ok = cross.any(1)
    centre = np.abs(0.5 * (fr[:-1] + fr[1:]) - 1.0)
    pick = np.argmin(np.where(cross, centre[None, :], np.inf), axis=1)
    lo = lam0 * fr[pick]
    hi = lam0 * fr[np.minimum(pick + 1, n_scan - 1)]
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        left = flo * fm <= 0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        flo = np.where(left, flo, fm)
    lam = np.where(ok, 0.5 * (lo + hi), lam0)
    return rays * lam[:, None], ok


def sample_grid2(st, gu, gv, U, V):
    sx = ((U - gu[0]) / (gu[1] - gu[0])).astype(np.float32)[:, None]
    sy = ((V - gv[0]) / (gv[1] - gv[0])).astype(np.float32)[:, None]
    return np.stack([cv2.remap(st[..., k].astype(np.float32), sx, sy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)[:, 0] for k in range(2)], 1).astype(float)


def fit_rectangle(st, side, W, H, x0):
    """Similarity (k, theta, tx, ty) putting flattened outline points on the W x H
    rectangle; side: 0 top, 1 right, 2 bottom, 3 left (page frame)."""
    def xf(x, p):
        k, th, tx, ty = x
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return k * p @ R.T + [tx, ty]

    target = np.array([0.0, W, H, 0.0])
    axis = np.array([1, 0, 1, 0])

    def resid(x):
        q = xf(x, st)
        return q[np.arange(len(q)), axis[side]] - target[side]

    res = least_squares(resid, x0, loss="soft_l1", f_scale=0.3)
    r = resid(res.x)
    return res.x, xf, r


def local_line(a, r, ok, q, half=10.0, k_min=8):
    """Value at each q of a straight line fitted to the ok samples (a, r) within +-half
    of q (at least the k_min nearest ones)."""
    a, r = a[ok], r[ok]
    out = np.zeros(len(q))
    if len(a) == 0:
        return out
    for i, x in enumerate(q):
        d = np.abs(a - x)
        m = d <= half
        if m.sum() < k_min:
            m = d <= np.sort(d)[min(k_min, len(d)) - 1]
        aa, rr = a[m], r[m]
        if np.ptp(aa) < 1.0:
            out[i] = np.mean(rr)
        else:
            cf = np.polyfit(aa - x, rr, 1)
            out[i] = cf[1]
    return out


def detect_lines(flat_gray, W, H, min_mm=8.0, max_dev_deg=3.0, border_mm=4.0):
    """Near-axis line segments in a flat render, in flat mm: (segments, is_horizontal)."""
    oh, ow = flat_gray.shape
    L = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(flat_gray)[0]
    if L is None:
        return np.empty((0, 4)), np.empty(0, bool)
    L = L.reshape(-1, 4).astype(float) * (W / ow)
    d = L[:, 2:] - L[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    dev = (ang + 45) % 90 - 45
    horiz = np.abs(((ang + 90) % 180) - 90) < 45
    inside = lambda p: (p[:, 0] > border_mm) & (p[:, 0] < W - border_mm) & (p[:, 1] > border_mm) & (p[:, 1] < H - border_mm)
    keep = (length >= min_mm) & (np.abs(dev) < max_dev_deg) & inside(L[:, :2]) & inside(L[:, 2:])
    return L[keep], horiz[keep]


def fit_correction(W, H, edges, lines, anchors, step=4.0, lam=30.0, w_edge=1.0, w_line=1.0, w_anchor=0.1):
    """One smooth 2D displacement field on the flat page, per axis, from
      edges:   (points, axis, target)  -> point + D = target   (paper edge on the rectangle)
      lines:   (segments, is_horizontal) -> all samples of a segment share one coordinate
               along the cross axis (straight, axis-parallel; position left free)
      anchors: points that should stay put (measured content)
    plus second-difference smoothness. Returns D(q) -> (n, 2) in mm."""
    gs_ = np.arange(-2 * step, W + 3 * step, step)
    gt_ = np.arange(-2 * step, H + 3 * step, step)
    ns, nt = len(gs_), len(gt_)
    N = ns * nt
    idx = np.arange(N).reshape(nt, ns)

    def bil(P):
        fi = np.clip((P[:, 0] - gs_[0]) / step, 0, ns - 1.001)
        fj = np.clip((P[:, 1] - gt_[0]) / step, 0, nt - 1.001)
        i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
        a, b = fi - i0, fj - j0
        cols = np.stack([idx[j0, i0], idx[j0, i0 + 1], idx[j0 + 1, i0], idx[j0 + 1, i0 + 1]], 1)
        vals = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], 1)
        return cols, vals

    tri = np.concatenate([np.stack([idx[:, :-2].ravel(), idx[:, 1:-1].ravel(), idx[:, 2:].ravel()], 1),
                          np.stack([idx[:-2, :].ravel(), idx[1:-1, :].ravel(), idx[2:, :].ravel()], 1)])
    Dm = sparse.csr_matrix((np.tile([1.0, -2.0, 1.0], len(tri)), (np.repeat(np.arange(len(tri)), 3), tri.ravel())),
                           shape=(len(tri), N))
    fields = []
    for axis in (0, 1):
        rows, cols, vals, rhs, wts = [], [], [], [], []
        r = 0
        n_c = 0
        # edges
        for P, ax, tg in edges:
            if ax != axis or len(P) == 0:
                continue
            c, v = bil(P)
            rr = r + np.arange(len(P))
            rows.append(np.repeat(rr, 4)); cols.append(c.ravel()); vals.append(v.ravel())
            rhs.append(tg - P[:, axis]); wts.append(np.full(len(P), w_edge)); r += len(P)
        # lines: horizontal segments constrain t (axis 1), vertical ones s (axis 0)
        L, hz = lines
        sel = L[hz] if axis == 1 else L[~hz]
        for seg in sel:
            n = max(int(np.hypot(*(seg[2:] - seg[:2])) / 2.0), 2)
            P = seg[:2] + np.linspace(0, 1, n)[:, None] * (seg[2:] - seg[:2])
            c, v = bil(P)
            rr = r + np.arange(n)
            # P_axis + D(P) - c_k = 0, c_k unknown (column N + k)
            rows.append(np.repeat(rr, 5))
            cols.append(np.c_[c, np.full(n, N + n_c)].ravel())
            vals.append(np.c_[v, -np.ones(n)].ravel())
            rhs.append(-P[:, axis]); wts.append(np.full(n, w_line / np.sqrt(n / 10.0 + 1))); r += n
            n_c += 1
        # anchors
        if len(anchors):
            c, v = bil(anchors)
            rr = r + np.arange(len(anchors))
            rows.append(np.repeat(rr, 4)); cols.append(c.ravel()); vals.append(v.ravel())
            rhs.append(np.zeros(len(anchors))); wts.append(np.full(len(anchors), w_anchor)); r += len(anchors)
        nu = N + n_c
        if r == 0:
            fields.append(np.zeros(N))
            continue
        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nu))
        w = np.concatenate(wts)
        Wd = sparse.diags(w ** 2)
        Dfull = sparse.hstack([Dm, sparse.csr_matrix((Dm.shape[0], n_c))]).tocsr()
        M = A.T @ Wd @ A + lam * (Dfull.T @ Dfull) + 1e-6 * sparse.eye(nu)
        x = spsolve(M.tocsc(), A.T @ (w ** 2 * np.concatenate(rhs)))
        fields.append(x[:N])

    def disp(q):
        c, v = bil(q)
        return np.stack([(f[c] * v).sum(1) for f in fields], 1)
    return disp


def corners_init(st_c, dst):
    a, b = st_c - st_c.mean(0), dst - dst.mean(0)
    U_, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, np.sign(np.linalg.det(U_ @ Vt))])
    R = U_ @ D @ Vt
    t = dst.mean(0) - R @ st_c.mean(0)
    return np.array([1.0, np.arctan2(R[1, 0], R[0, 0]), t[0], t[1]])


def flatten_grid(gu, gv, Z, support=None):
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
    # cells far from any measured data carry extrapolated heights; their 3D lengths
    # must not distort the unrolling of the measured page
    we = np.ones(len(E))
    if support is not None and len(support):
        from scipy.spatial import cKDTree
        mid = 0.5 * (P[E[:, 0], :2] + P[E[:, 1], :2])
        dd = cKDTree(support).query(mid)[0]
        we = np.clip(1.0 - (dd - 10.0) / 20.0, 0.05, 1.0)

    def resid(x):
        q = x.reshape(-1, 2)
        d = np.linalg.norm(q[E[:, 0]] - q[E[:, 1]], axis=1)
        return np.concatenate([we * (d - L), anchor * (x - x0)])

    m = len(E)
    n2 = len(x0)
    rows = np.r_[np.repeat(np.arange(m), 4), m + np.arange(n2)]
    cols = np.r_[np.stack([2 * E[:, 0], 2 * E[:, 0] + 1, 2 * E[:, 1], 2 * E[:, 1] + 1], 1).ravel(), np.arange(n2)]
    S = sparse.csr_matrix((np.ones(len(rows), int), (rows, cols)), shape=(m + n2, n2))
    res = least_squares(resid, x0, jac_sparsity=S, method="trf", x_scale=1.0, max_nfev=200)
    st = res.x.reshape(nv_, nu_, 2)
    strain = np.abs(resid(res.x)[:m]) / we / L
    return st, float(np.median(strain)), float(strain.max())


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


# ---------------------------------------------------------------- photometric

def flat_geometry(cam, F, gu, gv, Z, W, H, dpi, inv):
    """For every output pixel: 3D point (reference frame), page-frame normal, page mask."""
    ppm = dpi / 25.4
    ow, oh = int(round(W * ppm)), int(round(H * ppm))
    u_r, v_r, res_mm = inv
    sx = ((np.arange(ow) + 0.5) / ppm / res_mm).astype(np.float32)
    sy = ((np.arange(oh) + 0.5) / ppm / res_mm).astype(np.float32)
    MX, MY = np.meshgrid(sx, sy)
    U = cv2.remap(u_r, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    V = cv2.remap(v_r, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    page = np.isfinite(U) & np.isfinite(V)
    U[~page], V[~page] = -1e4, -1e4
    Zs = sample_heightfield(gu, gv, Z, U.ravel(), V.ravel())
    X = from_page(F, np.c_[U.ravel(), V.ravel(), Zs])
    step = gu[1] - gu[0]
    zv, zu = np.gradient(Z, step)
    nu = -sample_heightfield(gu, gv, zu, U.ravel(), V.ravel()).reshape(oh, ow)
    nv = -sample_heightfield(gu, gv, zv, U.ravel(), V.ravel()).reshape(oh, ow)
    return X, (oh, ow), nu, nv, page


def render_view(img, X, shape, cam, R, t):
    oh, ow = shape
    px, z = project(X, R, t, cam)
    mx = px[:, 0].reshape(oh, ow).astype(np.float32)
    my = px[:, 1].reshape(oh, ow).astype(np.float32)
    h, w = img.shape[:2]
    valid = (z.reshape(oh, ow) > 0) & (mx >= 2) & (my >= 2) & (mx < w - 3) & (my < h - 3)
    out = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    # detail captured per output pixel: photo pixels per output pixel (area ratio)
    gxx, gxy = np.gradient(mx, axis=1), np.gradient(mx, axis=0)
    gyx, gyy = np.gradient(my, axis=1), np.gradient(my, axis=0)
    mag = np.abs(gxx * gyy - gxy * gyx)
    return out, valid, mag


def remove_shading(img, valid, nu, nv, sub=4):
    """Divide out illumination. log-luminance of the paper is modelled as a low-order
    function of position and surface normal (fold shading), fitted to the upper
    envelope (ink and printed areas are darker and get down-weighted), then a
    medium-scale residual correction catches what the model misses."""
    oh, ow = valid.shape
    L = np.log(np.clip(img.mean(2), 1.0, None))
    ys, xs = np.mgrid[0:oh:sub, 0:ow:sub]
    m = valid[ys, xs]
    ys, xs = ys[m], xs[m]
    s = xs / ow * 2 - 1
    t = ys / oh * 2 - 1
    a, b = nu[ys, xs], nv[ys, xs]

    def feats(s, t, a, b):
        return np.stack([np.ones_like(s), s, t, s * s, s * t, t * t, s ** 3, s * s * t, s * t * t, t ** 3,
                         a, b, a * a, b * b, a * b], -1)

    Phi = feats(s, t, a, b)
    y = L[ys, xs]
    w = np.ones(len(y))
    beta = np.zeros(Phi.shape[1])
    for it in range(12):
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(Phi * sw[:, None], y * sw, rcond=None)[0]
        r = y - Phi @ beta
        scale_neg = 0.25 if it < 3 else 0.08  # start tolerant, then hug the paper level
        w = np.where(r < 0, 1 / (1 + (r / scale_neg) ** 2), 1 / (1 + (r / 0.05) ** 2))

    yy, xx = np.mgrid[0:oh, 0:ow]
    S = np.empty((oh, ow), np.float32)
    for r0 in range(0, oh, 256):  # chunked to keep memory flat
        sl = slice(r0, min(r0 + 256, oh))
        Pf = feats(xx[sl] / ow * 2 - 1, yy[sl] / oh * 2 - 1, nu[sl], nv[sl])
        S[sl] = Pf @ beta

    # residual correction on paper pixels (normalized convolution)
    res = L - S
    paper = valid & (res > -0.10) & (res < 0.10)
    sig = max(oh, ow) / 200.0
    num = cv2.GaussianBlur(np.where(paper, res, 0).astype(np.float32), (0, 0), sig)
    den = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), sig)
    corr = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0)
    # wide fallback where paper evidence is sparse (inside large printed areas)
    num2 = cv2.GaussianBlur(np.where(paper, res, 0).astype(np.float32), (0, 0), 6 * sig)
    den2 = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), 6 * sig)
    corr2 = np.where(den2 > 1e-3, num2 / np.maximum(den2, 1e-6), 0)
    blend = np.clip((den - 0.05) / 0.25, 0, 1)
    S = S + blend * corr + (1 - blend) * corr2

    alb = img / np.exp(S)[..., None]
    pw = valid & paper
    gains = np.array([np.median(alb[..., c][pw]) for c in range(3)])
    return alb / gains


def fuse_views(albs, valids, mags, ref=0, cap=16, ref_bonus=3.0, power=4.0):
    """Align every view to the reference with dense flow, then blend per pixel.
    Weights: captured detail, local agreement with the reference (bad alignment
    gets no say), outlier rejection against the per-pixel median (glare, local
    shadows). The power makes it a soft best-view selection instead of an average,
    which keeps edges sharp."""
    V = len(albs)
    g = [np.clip(a.mean(2) * 255, 0, 255).astype(np.uint8) for a in albs]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    oh, ow = g[0].shape
    gx, gy = np.meshgrid(np.arange(ow, dtype=np.float32), np.arange(oh, dtype=np.float32))
    gr = g[ref].astype(np.float32)

    def lncc(a, b, k=15):
        a = a - cv2.blur(a, (k, k)); b = b - cv2.blur(b, (k, k))
        va, vb = cv2.blur(a * a, (k, k)), cv2.blur(b * b, (k, k))
        return cv2.blur(a * b, (k, k)) / np.sqrt(va * vb + 1e-3), np.minimum(va, vb)

    aligned, vals, ws = [], [], []
    for j in range(V):
        a, v = albs[j], valids[j]
        conf = np.ones((oh, ow), np.float32)
        if j != ref:
            flow = dis.calc(g[ref], g[j], None)
            flow[np.linalg.norm(flow, axis=2) > cap] = 0
            a = cv2.remap(a, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            v = cv2.remap(v.astype(np.uint8), gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_NEAREST) > 0
            n, var = lncc(gr, a.mean(2).astype(np.float32) * 255)
            conf = np.where(var > 20, np.clip((n - 0.6) / 0.3, 0, 1), 1.0).astype(np.float32)
            conf = cv2.erode(conf, np.ones((7, 7), np.uint8))
        edge = cv2.distanceTransform(v.astype(np.uint8), cv2.DIST_L2, 3)
        w = np.clip(mags[j], 0, 9) * np.clip(edge / 40.0, 0, 1) * conf * (ref_bonus if j == ref else 1.0)
        aligned.append(a); vals.append(v); ws.append(w)
    G = np.stack([a.mean(2) for a in aligned])
    med = np.nanmedian(np.where(np.stack(vals), G, np.nan), axis=0)
    out = np.zeros_like(aligned[0])
    wsum = np.zeros((oh, ow), np.float32)
    for j in range(V):
        w = ws[j] * vals[j] / (1 + ((G[j] - med) / 0.06) ** 2)
        w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 1.5) ** power
        out += aligned[j] * w[..., None]
        wsum += w
    fallback = wsum < 1e-12
    out = out / np.maximum(wsum, 1e-12)[..., None]
    out[fallback] = aligned[ref][fallback]
    return out


def to_uint8(alb, white=0.95):
    """Paper to `white` (not 1.0: clipping the paper flattens the bright side of
    every edge); near-neutral pixels made neutral, real colours kept."""
    gray = alb.mean(2, keepdims=True)
    chroma = alb - gray
    sat = np.abs(chroma).max(2, keepdims=True)
    keep = np.clip((sat - 0.03) / 0.05, 0, 1)
    return np.clip((gray + chroma * keep) * 255 * white, 0, 255).astype(np.uint8)


def clean_paper(img, nlm_h=4.0, near=0.10):
    """Scanner-like paper: denoise, even out small paper-level variations (texture,
    show-through), then push the paper to white with a ramp (not a hard clip, so the
    bright side of edges keeps its gradient). Larger deviations such as printed grey
    areas are left alone. Pages without real colour come back single-channel."""
    gray = img.mean(2)
    chroma = np.abs(img.astype(np.float32) - gray[..., None]).max(2)
    colour = (cv2.medianBlur((chroma > 15).astype(np.uint8), 5).mean() > 1e-3)
    g = cv2.fastNlMeansDenoising(np.clip(gray, 0, 255).astype(np.uint8), None, nlm_h, 7, 21).astype(np.float32)

    def paper_level(x):
        x8 = np.clip(x, 0, 255).astype(np.uint8)
        bg = cv2.medianBlur(x8, 21)
        p = (bg > 180) & (np.abs(x8.astype(np.int16) - bg) < 25)
        if p.sum() < 1000:
            return None
        P = float(np.median(x[p]))
        return P, 1.4826 * float(np.median(np.abs(x[p] - P)))

    st = paper_level(g)
    if st is None:
        return img if colour else np.clip(gray, 0, 255).astype(np.uint8)
    # local paper level: ink removed by a closing, then smoothed
    bg = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), (0, 0), 6.0)
    ratio = bg / st[0]
    w = np.clip((ratio - (1 - near)) / (near / 2), 0, 1)
    e = g * (1 + w * (1 / np.maximum(ratio, 1e-3) - 1))
    P, s = paper_level(e) or st
    s = max(s, 2.0)
    lo, hi = P - 4 * s - 4, P - s
    y = np.where(e <= lo, e, lo + (e - lo) * (255 - lo) / max(hi - lo, 1e-3))
    y = np.clip(y, 0, 255)
    if not colour:
        return y.astype(np.uint8)
    return np.clip(img * (y / np.maximum(gray, 1.0))[..., None], 0, 255).astype(np.uint8)


def tilt_sims(g, mask, tilts, sift):
    """ASIFT-style affine simulations: keypoints of tilted/rotated copies of the
    image, mapped back to original coordinates."""
    out = []
    h, w = g.shape
    for t in tilts:
        for phi in ([0.0] if t == 1 else np.arange(0, 180, 72 / t)):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), phi, 1.0)
            c = np.abs(M[:, :2])
            nw, nh = int(h * c[0, 1] + w * c[0, 0]), int(h * c[1, 1] + w * c[1, 0])
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            T = np.diag([1 / t, 1.0])
            A = np.c_[T @ M[:, :2], T @ M[:, 2]]
            src = cv2.GaussianBlur(g, (0, 0), 0.8 * np.sqrt(t * t - 1)) if t > 1 else g
            gi = cv2.warpAffine(src, A, (int(np.ceil(nw / t)), nh), flags=cv2.INTER_LINEAR)
            mi = cv2.warpAffine(mask, A, (int(np.ceil(nw / t)), nh), flags=cv2.INTER_NEAREST) if mask is not None else None
            kp, ds = sift.detectAndCompute(gi, mi)
            if ds is None or len(kp) < 2:
                continue
            P = np.float64([k.pt for k in kp])
            out.append((np.c_[P, np.ones(len(P))] @ cv2.invertAffineTransform(A).T, ds))
    return out


def asift_match(sims0, simsj, ratio=0.8):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    P0, Pj = [], []
    for p0, d0 in sims0:
        for pj, dj in simsj:
            m = flann.knnMatch(d0, dj, k=2)
            m2 = flann.knnMatch(dj, d0, k=2)
            back = {x[0].queryIdx: x[0].trainIdx for x in m2 if len(x) == 2 and x[0].distance < ratio * x[1].distance}
            for x in m:
                if len(x) == 2 and x[0].distance < ratio * x[1].distance and back.get(x[0].trainIdx) == x[0].queryIdx:
                    P0.append(p0[x[0].queryIdx]); Pj.append(pj[x[0].trainIdx])
    if not P0:
        return np.empty((0, 2)), np.empty((0, 2))
    P0, Pj = np.array(P0), np.array(Pj)
    _, u = np.unique(np.round(np.c_[P0, Pj] / 2), axis=0, return_index=True)
    return P0[u], Pj[u]


def page_coverage(pts, mask, n=8):
    """Share of page cells (n x n over the page bbox) containing >= 3 points."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cells = 0
    hit = 0
    hist = np.histogram2d(pts[:, 1], pts[:, 0], [n, n], [[y0, y1], [x0, x1]])[0] if len(pts) else np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub = mask[y0 + (y1 - y0) * i // n:y0 + (y1 - y0) * (i + 1) // n, x0 + (x1 - x0) * j // n:x0 + (x1 - x0) * (j + 1) // n]
            if sub.size and (sub > 0).mean() > 0.5:
                cells += 1
                hit += hist[i, j] >= 3
    return hit / max(cells, 1)


def parallax_deg(q0, qj, R, cam):
    """Median angle between corresponding rays after removing the rotation:
    ~0 when the camera did not move (no depth information)."""
    d0 = rays_of(q0, cam)
    dj = rays_of(qj, cam) @ R
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    dj /= np.linalg.norm(dj, axis=1, keepdims=True)
    return float(np.degrees(np.median(np.arccos(np.clip((d0 * dj).sum(1), -1, 1)))))


def match_ref(kp0, d0, g, mask, n_features):
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    kp, d = sift.detectAndCompute(g, mask)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    m01 = flann.knnMatch(d0, d, k=2)
    m10 = flann.knnMatch(d, d0, k=2)
    back = {m[0].queryIdx: m[0].trainIdx for m in m10 if len(m) == 2 and m[0].distance < 0.85 * m[1].distance}
    i0, pj = [], []
    for m in m01:
        if len(m) == 2 and m[0].distance < 0.85 * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx:
            i0.append(m[0].queryIdx)
            pj.append(kp[m[0].trainIdx].pt)
    return np.array(i0, int), np.float64(pj), len(kp)


def densify_multi(g0, gs, mask, p_ref, X, cam, Rs, ts, max_pts, min_dist, search):
    """Guided matching from reference corners into every other view: position
    predicted by projecting nearby reconstructed points, refined by NCC."""
    from scipy.spatial import cKDTree
    q = cv2.goodFeaturesToTrack(g0, max_pts, 0.005, min_dist, mask=mask, blockSize=5)
    if q is None:
        return np.empty((0, 2)), np.empty((0, len(Rs), 2))
    q = q.reshape(-1, 2).astype(np.float64)
    d, _ = cKDTree(p_ref).query(q)
    q = q[d < 200]
    obs = np.full((len(q), len(Rs), 2), np.nan)
    obs[:, 0] = q
    for j in range(1, len(Rs)):
        pj, _ = project(X, Rs[j], ts[j], cam)
        A, pred = local_affines(p_ref, pj, q)
        r, ok = refine_subpixel(g0, gs[j], q, pred, A, win=21, search=search)
        ok &= np.linalg.norm(r - pred, axis=1) < search - 1
        obs[ok, j] = r[ok]
    keep = np.isfinite(obs[:, 1:, 0]).any(1)
    return q[keep], obs[keep]


def clean_obs(X, obs, Rs, ts, cam, thr):
    """Drop single observations with reprojection error > thr, retriangulate,
    and keep points still seen in the reference and at least one other view."""
    for _ in range(2):
        err, dok = reproj_errors(X, obs, Rs, ts, cam)
        bad = err > thr
        obs = obs.copy()
        obs[bad] = np.nan
        keep = dok & np.isfinite(obs[:, 0, 0]) & np.isfinite(obs[:, 1:, 0]).any(1)
        obs, X = obs[keep], X[keep]
        X = triangulate_multi(obs, Rs, ts, cam)
    err, dok = reproj_errors(X, obs, Rs, ts, cam)
    keep = dok & ~(err > thr).any(1)
    return X[keep], obs[keep], err[keep]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="reference view first, then 1-3 more views (left/right/up/down)")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--page-size", default="210x297", help="page W x H in mm (default A4)")
    ap.add_argument("--focal-px", type=float, help="focal length in pixels at full resolution")
    ap.add_argument("--calib", help="npz with K, dist (and optionally size=(w,h)) from cv2.calibrateCamera")
    ap.add_argument("--fixed-focal", action="store_true", help="do not refine the focal length")
    ap.add_argument("--distortion", type=int, default=0, choices=(0, 1, 2),
                    help="radial distortion terms refined in bundle adjustment (0, k1, k1+k2)")
    ap.add_argument("--max-side", type=int, default=4100, help="downscale longer side to this many px")
    ap.add_argument("--features", type=int, default=60000)
    ap.add_argument("--ba-points", type=int, default=5000, help="points used in bundle adjustment")
    ap.add_argument("--grid", type=float, help="height-field grid spacing in mm (default 2.5 with 3+ views, else 4)")
    ap.add_argument("--smooth", type=float, help="height-field smoothness (default 10 with 3+ views, else 30)")
    ap.add_argument("--dpi", type=int, default=200, help="dpi of the flattened output")
    ap.add_argument("--no-densify", action="store_true")
    ap.add_argument("--no-outline", action="store_true", help="do not triangulate the page outline")
    ap.add_argument("--no-line-fix", action="store_true",
                    help="do not straighten printed horizontal/vertical lines")
    ap.add_argument("--anchor-w", type=float, default=0.1,
                    help="how strongly measured content resists the edge/line correction")
    ap.add_argument("--no-clean", action="store_true",
                    help="keep paper grain and colour in flat.png (no denoise / white paper)")
    ap.add_argument("--no-edge-fix", action="store_true",
                    help="do not force the detected page outline onto the rectangle")
    ap.add_argument("--fuse", action="store_true",
                    help="blend all views (against glare/shadows in single shots; may soften geometry slightly)")
    ap.add_argument("--no-shading", action="store_true", help="keep the original illumination")
    ap.add_argument("--dense", type=int, default=40000, help="max candidate points for densification")
    ap.add_argument("--ruling", type=float, default=20.0,
                    help="stiffness along the fold direction when the page is folded one way (1 = off)")
    ap.add_argument("--asift", choices=("auto", "on", "off"), default="auto",
                    help="tilt-simulated matching for steep folds (auto: when plain matches leave the page uncovered)")
    args = ap.parse_args()
    if len(args.images) < 2:
        sys.exit("need at least two images")

    os.makedirs(args.out, exist_ok=True)
    W, H = map(float, args.page_size.lower().split("x"))
    V = len(args.images)
    if args.grid is None:
        args.grid = 2.5 if V >= 3 else 4.0
    if args.smooth is None:
        args.smooth = 10.0 if V >= 3 else 30.0

    imgs, scales = zip(*[load_image(p, args.max_side) for p in args.images])
    if len({im.shape for im in imgs}) != 1:
        sys.exit("all images must come from the same camera at the same resolution")
    h, w = imgs[0].shape[:2]
    K, dist, ksrc = build_K(args, args.images[0], w, h)
    s = scales[0]
    cam = dict(f=K[0, 0] * s, cx=(K[0, 2] + 0.5) * s - 0.5, cy=(K[1, 2] + 0.5) * s - 0.5,
               k1=float(dist[0]), k2=float(dist[1]))
    if args.calib:
        args.fixed_focal, n_dist = True, 0
    else:
        n_dist = args.distortion
    if V == 2 and not args.fixed_focal:
        # two views do not determine the focal length; refining it only adds error
        args.fixed_focal = True
        log("2 views: focal length kept fixed (use --focal-px if EXIF is off, e.g. 16:9 crops)")
    fulls = list(imgs)
    imgs = [cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im for im in imgs]
    gs = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    log(f"{V} views, focal start {cam['f'] / s:.1f} px ({ksrc})")

    log("page outlines ...")
    quads, masks = [], []
    for j, im in enumerate(imgs):
        q, m = find_page_quad(im)
        if q is None:
            if j == 0:
                sys.exit("could not find the page in the reference image")
            quads.append(None); masks.append(None)
            continue
        quads.append(refine_quad(gs[j], q)); masks.append(m)
    quad = quads[0]
    mask_er = cv2.erode(masks[0], np.ones((15, 15), np.uint8))

    log("matching ...")
    sift = cv2.SIFT_create(nfeatures=args.features, contrastThreshold=0.02)
    kp0, d0 = sift.detectAndCompute(gs[0], mask_er)
    P0 = np.float64([k.pt for k in kp0])
    obs = np.full((len(kp0), V, 2), np.nan)
    obs[:, 0] = P0
    Rs, ts = [np.eye(3)], [np.zeros(3)]
    Ks = cam_K(cam)
    extra = []  # rows from tilt-simulated matching (not tied to reference keypoints)
    sims0 = None
    for j in range(1, V):
        mj = cv2.dilate(masks[j], np.ones((41, 41), np.uint8)) if masks[j] is not None else None
        i0, pj, nk = match_ref(kp0, d0, gs[j], mj, args.features)
        E, inl = cv2.findEssentialMat(P0[i0], pj, Ks, method=cv2.USAC_MAGSAC, prob=0.99999, threshold=1.0)
        inl = inl.ravel().astype(bool)
        i0, pj = i0[inl], pj[inl]
        q0, qj = P0[i0], pj
        cov = page_coverage(q0, mask_er)
        used_asift = False
        if args.asift == "on" or (args.asift == "auto" and cov < 0.4):
            # steep panels look too different between views for plain SIFT
            sa = min(1.0, 2000.0 / max(gs[0].shape))
            small = lambda im: cv2.resize(im, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA) if sa < 1 else im
            sift_s = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
            if sims0 is None:
                sims0 = tilt_sims(small(gs[0]), small(mask_er), (1, 2 ** 0.5, 2), sift_s)
            simsj = tilt_sims(small(gs[j]), small(mj) if mj is not None else None, (1, 2 ** 0.5, 2, 2 * 2 ** 0.5), sift_s)
            a0, aj = asift_match(sims0, simsj)
            a0, aj = (a0 + 0.5) / sa - 0.5, (aj + 0.5) / sa - 0.5
            q0, qj = np.r_[q0, a0], np.r_[qj, aj]
            E, inl = cv2.findEssentialMat(q0, qj, Ks, method=cv2.USAC_MAGSAC, prob=0.99999, threshold=1.5)
            inl = inl.ravel().astype(bool)
            q0, qj = q0[inl], qj[inl]
            used_asift = True
        if E is None or len(q0) < 8:
            sys.exit(f"view {j}: no usable matches with the reference view")
        E = E[:3]  # degenerate input can return several stacked solutions
        _, R, t, _ = cv2.recoverPose(E, q0, qj, Ks)
        par = parallax_deg(q0, qj, R, cam)
        if par < 1.5:
            sys.exit(f"view {j}: almost no viewpoint change between the photos (parallax {par:.2f} deg). "
                     "Move the phone 10-20 cm between shots, and check that two different images were passed.")
        A, _ = local_affines(q0, qj)
        pr, ok = refine_subpixel(gs[0], gs[j], q0, qj, A)
        ok &= np.linalg.norm(pr - qj, axis=1) < (3.0 if used_asift else 1.5)
        n_plain = len(i0)
        okp = ok[:n_plain]
        obs[i0[okp], j] = pr[:n_plain][okp]
        if used_asift:
            oka = ok[n_plain:]
            e = np.full((oka.sum(), V, 2), np.nan)
            e[:, 0] = q0[n_plain:][oka]
            e[:, j] = pr[n_plain:][oka]
            extra.append(e)
        Rs.append(R); ts.append(t.ravel())
        log(f"  view {j}: {nk} keypoints, {len(q0)} inliers, {ok.sum()} refined, page coverage {cov:.2f}"
            + (f" -> {page_coverage(q0[ok], mask_er):.2f} with tilt simulation" if used_asift else ""))

    keep = np.isfinite(obs[:, 1:, 0]).any(1)
    obs = np.concatenate([obs[keep]] + extra)
    # put all views on the scale of view 1: depth ratios at nearby reference pixels
    from scipy.spatial import cKDTree
    base = np.isfinite(obs[:, 1, 0])
    X1 = triangulate_multi(obs[base][:, [0, 1]], Rs[:2], ts[:2], cam)
    tree1 = cKDTree(obs[base][:, 0])
    for j in range(2, V):
        mj_ = np.isfinite(obs[:, j, 0])
        Xj = triangulate_multi(obs[mj_][:, [0, j]], [Rs[0], Rs[j]], [ts[0], ts[j]], cam)
        dd, nn = tree1.query(obs[mj_][:, 0])
        close = dd < 15
        if close.sum() < 20:
            sys.exit(f"view {j} overlaps too little with view 1")
        ts[j] = ts[j] * np.median(X1[nn[close]][:, 2] / Xj[close][:, 2])

    X = triangulate_multi(obs, Rs, ts, cam)
    X, obs, _ = clean_obs(X, obs, Rs, ts, cam, 3.0)

    log("bundle adjustment ...")
    for it in range(2):
        rng = np.random.default_rng(it)
        sel = rng.choice(len(X), min(args.ba_points, len(X)), replace=False)
        cam, Rs, ts = bundle_adjust(cam, Rs, ts, obs[sel], not args.fixed_focal, n_dist)
        X = triangulate_multi(obs, Rs, ts, cam)
        X, obs, err = clean_obs(X, obs, Rs, ts, cam, 1.0 if it else 2.0)
    if len(X) < 50:
        sys.exit(f"only {len(X)} reliable 3D points; the photos do not overlap or differ too little in viewpoint")
    rms = np.sqrt(np.nanmean(err ** 2))
    cams_c = [-R.T @ t for R, t in zip(Rs, ts)]
    log(f"  points {len(X)}, reprojection rms {rms:.3f} px, focal {cam['f'] / s:.1f} px, "
        f"k1 {cam['k1']:.4f}, k2 {cam['k2']:.4f}")

    c, n = fit_plane(X)
    F = page_frame(cam, quad, c, n, W, H)
    uvz = to_page(F, X)
    bnd = grid_bounds(uvz[:, :2], W, H)

    if not args.no_densify:
        log("densifying ...")
        gu0, gv0, Z0, _ = fit_heightfield(uvz, bnd, 2 * args.grid, args.smooth)
        mind = max(4, int(round(min(gs[0].shape) / 400)))
        q, dobs = densify_multi(gs[0], gs, mask_er, obs[:, 0], X, cam, Rs, ts, args.dense, mind, search=6)
        Xd = triangulate_multi(dobs, Rs, ts, cam)
        Xd, dobs, derr = clean_obs(Xd, dobs, Rs, ts, cam, 1.0)
        uvzd = to_page(F, Xd)
        dz = uvzd[:, 2] - sample_heightfield(gu0, gv0, Z0, uvzd[:, 0], uvzd[:, 1])
        ok = np.abs(dz) < 1.0
        log(f"  candidates {len(q)}, kept {ok.sum()}")
        X = np.r_[X, Xd[ok]]
        obs = np.r_[obs, dobs[ok]]
        err = np.r_[err, derr[ok]]
        dup = np.unique(np.round(obs[:, 0] / 2.0), axis=0, return_index=True)[1]
        X, obs, err = X[dup], obs[dup], err[dup]
        uvz = to_page(F, X)
        log(f"  total points {len(X)}")

    n_views = np.isfinite(obs[:, :, 0]).sum(1)
    ang = []
    for j in range(1, V):
        m = np.isfinite(obs[:, j, 0])
        a = X[m] / np.linalg.norm(X[m], axis=1, keepdims=True)
        b = (X[m] - cams_c[j]) / np.linalg.norm(X[m] - cams_c[j], axis=1, keepdims=True)
        ang.append(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))))
    ang_med = float(np.median(np.concatenate(ang)))
    log(f"  views per point: {np.bincount(n_views)[2:].tolist()} (2, 3, ...), "
        f"triangulation angle median {ang_med:.1f} deg")

    # page outline in the reference view, and its 3D where triangulable
    out0, ok0 = page_outline(gs[0], masks[0])
    out0 = out0[ok0]
    eid = assign_edges(out0, quad)
    side = np.array([{"top": 0, "right": 1, "bottom": 2, "left": 3}[F["edge_map"][e]] if e >= 0 else -1
                     for e in eid])
    Xb = np.empty((0, 3))
    bidx = np.empty(0, int)
    if not args.no_outline:
        log("outline ...")
        outs = [None] + [page_outline(gs[j], masks[j])[0] if masks[j] is not None else None for j in range(1, V)]
        gu0, gv0, Z0, _ = fit_heightfield(uvz, bnd, 2 * args.grid, args.smooth)
        Xp, _ = ray_surface(rays_of(out0, cam), F, gu0, gv0, Z0, c, n)
        Xb, bobs, has = triangulate_outline(out0, outs, cam, Rs, ts, Xp)
        bobs, bside = bobs[has], side[has]
        Xb = Xb[has]
        bidx = np.nonzero(has)[0]
        if len(Xb):
            # drop the worst inconsistent observation per point until the rest agree
            for _ in range(V - 1):
                berr, _ = reproj_errors(Xb, bobs, Rs, ts, cam)
                worst = np.nanargmax(np.where(np.isfinite(berr[:, 1:]), berr[:, 1:], -1), axis=1) + 1
                bad = np.nanmax(berr, 1) >= 1.5
                if not bad.any():
                    break
                bobs[np.nonzero(bad)[0], worst[bad]] = np.nan
                still = np.isfinite(bobs[:, 1:, 0]).any(1)
                bobs, bside, Xb, bidx = bobs[still], bside[still], Xb[still], bidx[still]
                Xb = triangulate_multi(bobs, Rs, ts, cam)
            berr, bdok = reproj_errors(Xb, bobs, Rs, ts, cam)
            uvzb = to_page(F, Xb)
            dzb = uvzb[:, 2] - sample_heightfield(gu0, gv0, Z0, uvzb[:, 0], uvzb[:, 1])
            okb = bdok & (np.nanmax(berr, 1) < 1.5) & (np.abs(dzb) < 8.0) & (bside >= 0)
            if os.environ.get("BA_DEBUG"):
                for sd in range(4):
                    m = bside == sd
                    log(f"    side {sd}: {m.sum()} matched, depth ok {(bdok & m).sum()}, "
                        f"reproj ok {(m & (np.nanmax(berr, 1) < 1.5)).sum()}, dz ok {(m & (np.abs(dzb) < 8)).sum()}, "
                        f"dz median {np.median(np.abs(dzb[m])) if m.any() else 0:.1f} mm")
            Xb, bside, bidx = Xb[okb], bside[okb], bidx[okb]
        counts = np.bincount(bside, minlength=4) if len(Xb) else np.zeros(4, int)
        log(f"  outline points triangulated: top {counts[0]}, right {counts[1]}, "
            f"bottom {counts[2]}, left {counts[3]} (of {len(out0)})")

    if len(Xb):
        bnd = grid_bounds(np.r_[uvz[:, :2], to_page(F, Xb)[:, :2]], W, H)

    # Letter folds are parallel creases: if the surface clearly bends in one
    # direction only, turn the frame so u runs along the folds and keep the
    # paper straight along them (developable prior). Otherwise stay isotropic.
    ruling_w = 1.0
    if args.ruling > 1:
        data = np.r_[uvz, to_page(F, Xb)] if len(Xb) else uvz
        gu0, gv0, Z0, _ = fit_heightfield(data, bnd, 2 * args.grid, args.smooth)
        ang, aniso = fold_direction(gu0, gv0, Z0)
        log(f"  fold direction {np.degrees(ang):.1f} deg, anisotropy {aniso:.2f}")
        if aniso > 0.6:
            ca, sa_ = np.cos(ang), np.sin(ang)
            eu = ca * F["eu"] + sa_ * F["ev"]
            ev = -sa_ * F["eu"] + ca * F["ev"]
            F["eu"], F["ev"] = eu, ev
            uvz = to_page(F, X)
            bnd = grid_bounds(np.r_[uvz[:, :2], to_page(F, Xb)[:, :2]] if len(Xb) else uvz[:, :2], W, H)
            ruling_w = args.ruling
    # physical page corners: image quad corner k joins image edges k-1 and k
    pos = {frozenset(("top", "left")): (0, 0), frozenset(("top", "right")): (W, 0),
           frozenset(("bottom", "right")): (W, H), frozenset(("bottom", "left")): (0, H)}
    cidx = [int(np.argmin(np.linalg.norm(out0 - quad[k], axis=1))) for k in range(4)]
    ctarget = np.array([pos[frozenset((F["edge_map"][(k - 1) % 4], F["edge_map"][k]))] for k in range(4)], float)
    # Height field, isometric unrolling, then scale/placement from the whole
    # flattened outline (all reference outline points, intersected with the surface).
    rect = None
    for it in range(2):
        log(f"height field + flattening (pass {it + 1}), grid u {bnd[0]:.0f}..{bnd[1]:.0f} v {bnd[2]:.0f}..{bnd[3]:.0f} mm ...")
        data = np.r_[uvz, to_page(F, Xb)] if len(Xb) else uvz
        gu, gv, Z, resid = fit_heightfield(data, bnd, args.grid, args.smooth, ruling_w=ruling_w)
        st, strain_med, strain_max = flatten_grid(gu, gv, Z)
        Xo, hit = ray_surface(rays_of(out0, cam), F, gu, gv, Z, c, n)
        if len(Xb):
            # where the outline was triangulated, its 3D position is measured
            Xo[bidx], hit[bidx] = Xb, True
        uvo = to_page(F, Xo)
        sto = sample_grid2(st, gu, gv, uvo[:, 0], uvo[:, 1])
        use = hit & (side >= 0)
        x0 = corners_init(sto[cidx], ctarget) if rect is None else rect
        rect, xf, rres = fit_rectangle(sto[use], side[use], W, H, x0)
        k = rect[0]
        log(f"  outline fit: scale {k:.4f}, rms {np.sqrt(np.mean(rres ** 2)):.3f} mm "
            f"({use.sum()} outline points)")
        if it == 0:
            F["scale"] *= k
            uvz = to_page(F, X)
            rect = rect.copy(); rect[0] = 1.0
    st = xf(rect, st.reshape(-1, 2)).reshape(st.shape)
    sto_f = xf(rect, sto)
    if not (args.no_line_fix and args.no_edge_fix):
        # One smooth correction of the flat map: the paper outline seen in the photo
        # must lie on the rectangle, printed lines must be straight and axis-parallel,
        # measured content should stay where the 3D reconstruction put it.
        okc = hit[cidx]
        edges = []
        if not args.no_edge_fix:
            # Outline points must agree with their neighbours along the edge: a smooth
            # offset of a whole edge is real (extrapolated margin) and gets corrected,
            # isolated jumps are detection errors (shadows, background) and are dropped.
            # Exception: an extrapolated page end can turn fast (5-8 mm over the last 30 mm,
            # no measured data there). Points at the ends of an edge that the whole-edge fit
            # rejects are restored if they agree with their local neighbours. Bumps in the
            # middle (fold creases) stay rejected.
            edge_fit = {}
            for sd, ax, tg in ((3, 0, 0.0), (1, 0, W), (0, 1, 0.0), (2, 1, H)):
                m = use & (side == sd)
                if m.sum() < 10:
                    continue
                P_ = sto_f[m]
                along = P_[:, 1 - ax]
                r = tg - P_[:, ax]
                ok = np.abs(r) < 25.0
                for _ in range(3):
                    cf = np.polyfit(along[ok], r[ok], 2)
                    dev = r - np.polyval(cf, along)
                    mad = 1.4826 * np.median(np.abs(dev[ok])) + 1e-6
                    ok = (np.abs(r) < 25.0) & (np.abs(dev) < max(1.0, 4 * mad))
                ok_l = np.abs(r) < 25.0
                for _ in range(3):
                    dev = r - local_line(along, r, ok_l, along)
                    mad = 1.4826 * np.median(np.abs(dev[ok_l])) + 1e-6
                    ok_l = (np.abs(r) < 25.0) & (np.abs(dev) < max(1.0, 4 * mad))
                o = np.argsort(along)
                kept = np.nonzero(ok[o])[0]
                if len(kept):
                    ends = np.zeros(len(o), bool)
                    ends[:kept[0]] = True
                    ends[kept[-1] + 1:] = True
                    ok[o[ends & ok_l[o]]] = True
                edge_fit[sd] = (ax, along[ok], r[ok])
                edges.append((P_[ok], ax, tg))
                if os.environ.get("BA_DEBUG"):
                    np.savez(os.path.join(args.out, f"dbg_edge{sd}.npz"), img=out0[m], uvo=uvo[m], st=P_, r=r, ok=ok)
                log(f"    edge {('top', 'right', 'bottom', 'left')[sd]}: offset median {np.median(r):+.2f} mm, "
                    f"range {np.percentile(r, 5):+.2f}..{np.percentile(r, 95):+.2f}, kept {ok.sum()}/{len(r)}")
            # corners: keep only if consistent with the trend of the adjacent edge near them
            Pc, Tc = sto_f[cidx], ctarget
            for ax in (0, 1):
                keepc = okc.copy()
                for k in range(4):
                    for sd, (eax, ea, er) in edge_fit.items():
                        if eax == ax and abs(Tc[k, ax] - (0.0 if sd in (3, 0) else (W if sd == 1 else H))) < 1e-6:
                            pred = local_line(ea, er, np.ones(len(ea), bool), Pc[k, 1 - ax:2 - ax], half=30.0)[0]
                            if abs((Tc[k, ax] - Pc[k, ax]) - pred) > 3.0:
                                keepc[k] = False
                edges.append((Pc[keepc], ax, Tc[keepc][:, ax]))
        anchors = sample_grid2(st, gu, gv, uvz[:, 0], uvz[:, 1])
        if len(anchors) > 4000:
            anchors = anchors[np.random.default_rng(0).choice(len(anchors), 4000, replace=False)]
        camp = dict(cam, f=cam["f"] / s, cx=(cam["cx"] + 0.5) / s - 0.5, cy=(cam["cy"] + 0.5) / s - 0.5)
        no_lines = (np.empty((0, 4)), np.empty(0, bool))
        # 1) edges: measured content holds firmly, so the correction stays in the margins
        if edges:
            disp_e = fit_correction(W, H, edges, no_lines, anchors, w_anchor=1.0)
            st = st + disp_e(st.reshape(-1, 2)).reshape(st.shape)
            sto_f = sto_f + disp_e(sto_f)
            anchors = anchors + disp_e(anchors)
            dd = np.linalg.norm(disp_e(anchors), axis=1)
            log(f"  edge correction: shift at measured content median {np.median(dd):.3f} mm")
        # 2) lines: printed lines may override the measurement (weak hold); the
        #    corrected outline stays where it is
        if not args.no_line_fix:
            hold = [(sto_f[use], 0, sto_f[use][:, 0]), (sto_f[use], 1, sto_f[use][:, 1])] if edges else []
            st1, sto1 = st.copy(), sto_f.copy()
            disp_l = None
            for it in range(2):
                inv0 = flat_to_uv(st, gu, gv, W, H)
                Xp_, shp, _, _, page_p = flat_geometry(camp, F, gu, gv, Z, W, H, 150, inv0)
                g8 = np.clip(render_view(fulls[0], Xp_, shp, camp, Rs[0], ts[0])[0].mean(2), 0, 255).astype(np.uint8)
                g8[~page_p] = 255
                Lc, hz = detect_lines(g8, W, H)
                if disp_l is not None and len(Lc):
                    back = lambda P: P - disp_l(P)
                    Lc = np.c_[back(Lc[:, :2]), back(Lc[:, 2:])]
                disp_l = fit_correction(W, H, hold, (Lc, hz), anchors, w_anchor=args.anchor_w)
                st = st1 + disp_l(st1.reshape(-1, 2)).reshape(st1.shape)
                sto_f = sto1 + disp_l(sto1)
                dd = np.linalg.norm(disp_l(st1.reshape(-1, 2)), axis=1)
                log(f"  line correction {it + 1}: {len(Lc)} line segments, shift median {np.median(dd):.3f} mm, "
                    f"max {dd.max():.2f} mm")
        if not args.no_edge_fix:
            ax_ = np.array([1, 0, 1, 0])[side[use]]
            tg_ = np.array([0.0, W, H, 0.0])[side[use]]
            rr = sto_f[use][np.arange(use.sum()), ax_] - tg_
            log(f"  outline on rectangle: rms {np.sqrt(np.mean(rres ** 2)):.3f} -> {np.sqrt(np.mean(rr ** 2)):.3f} mm")
    outline_rms = float(np.sqrt(np.mean(rres ** 2)))
    pst = sample_grid2(st, gu, gv, uvz[:, 0], uvz[:, 1])
    log(f"  z range {uvz[:, 2].min():.2f} .. {uvz[:, 2].max():.2f} mm, "
        f"point residual to surface (MAD) {1.4826 * np.median(np.abs(resid)):.3f} mm, "
        f"flattening strain median {strain_med * 100:.3f} %")

    # outputs
    sf = s
    cols = imgs[0][np.clip(obs[:, 0, 1].astype(int), 0, imgs[0].shape[0] - 1),
                   np.clip(obs[:, 0, 0].astype(int), 0, imgs[0].shape[1] - 1)]
    obs_full = obs / sf
    hdr = "u_mm,v_mm,z_mm,s_mm,t_mm," + ",".join(f"x{j},y{j}" for j in range(V)) + ",reproj_px"
    np.savetxt(os.path.join(args.out, "points.csv"),
               np.c_[uvz, pst, obs_full.reshape(len(obs), -1), np.nanmax(err, 1)],
               delimiter=",", fmt="%.4f", header=hdr, comments="")
    if len(Xb):
        np.savetxt(os.path.join(args.out, "outline.csv"), to_page(F, Xb), delimiter=",", fmt="%.4f",
                   header="u_mm,v_mm,z_mm", comments="")
    write_ply(os.path.join(args.out, "points.ply"), uvz, cols)
    np.savez(os.path.join(args.out, "heightfield.npz"), u=gu, v=gv, z=Z, st=st,
             f=cam["f"] / sf, k1=cam["k1"], k2=cam["k2"], R=np.array(Rs), t=np.array(ts),
             origin=F["o"], eu=F["eu"], ev=F["ev"], en=F["en"], scale=F["scale"])
    allpts = np.r_[uvz, to_page(F, Xb)] if len(Xb) else uvz
    save_heightmap_png(os.path.join(args.out, "heightfield.png"), gu, gv, Z, allpts, W, H)

    dbg = imgs[0].copy()
    zl = np.percentile(np.abs(uvz[:, 2]), 99) + 1e-6
    zc = np.clip((uvz[:, 2] / zl + 1) * 127.5, 0, 255).astype(np.uint8)
    cm = cv2.applyColorMap(zc.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    for (x, y), cc in zip(obs[:, 0].astype(int), cm):
        cv2.circle(dbg, (x, y), 2, tuple(int(v) for v in cc), -1)
    cv2.polylines(dbg, [out0.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.out, "overlay.jpg"), dbg, [cv2.IMWRITE_JPEG_QUALITY, 85])

    camf = dict(cam, f=cam["f"] / sf, cx=(cam["cx"] + 0.5) / sf - 0.5, cy=(cam["cy"] + 0.5) / sf - 0.5)
    inv = flat_to_uv(st, gu, gv, W, H)
    log("rendering ...")
    Xr, shape, nu_m, nv_m, page = flat_geometry(camf, F, gu, gv, Z, W, H, args.dpi, inv)
    views = list(range(V)) if args.fuse else [0]
    rend = [render_view(fulls[j], Xr, shape, camf, Rs[j], ts[j]) for j in views]
    cv2.imwrite(os.path.join(args.out, "flat_raw.png"), np.clip(rend[0][0], 0, 255).astype(np.uint8))
    if args.no_shading:
        albs = [r[0] / 255.0 for r in rend]
    else:
        albs = [remove_shading(r[0], r[1] & page, nu_m, nv_m) for r in rend]
    if len(views) > 1:
        flat = fuse_views(albs, [r[1] & page for r in rend], [r[2] for r in rend])
    else:
        flat = albs[0]
    out8 = to_uint8(flat)
    if not args.no_clean:
        out8 = clean_paper(out8)
    cv2.imwrite(os.path.join(args.out, "flat.png"), out8, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    report = dict(views=V, points=int(len(X)), outline_points=int(len(Xb)),
                  focal_px=float(camf["f"]), focal_source=ksrc, k1=cam["k1"], k2=cam["k2"],
                  reproj_rms_px=float(rms), triangulation_angle_median_deg=ang_med,
                  distance_mm=float(np.median(X[:, 2]) * F["scale"]),
                  baselines_mm=[float(np.linalg.norm(cc) * F["scale"]) for cc in cams_c[1:]],
                  z_min_mm=float(uvz[:, 2].min()), z_max_mm=float(uvz[:, 2].max()),
                  surface_residual_mad_mm=float(1.4826 * np.median(np.abs(resid))),
                  flatten_strain_median=strain_med, flatten_strain_max=strain_max,
                  outline_fit_rms_mm=outline_rms)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"done -> {args.out}/")


if __name__ == "__main__":
    main()
