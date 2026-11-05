# docscan3d

2–4 photos of a folded/curved page → measured 3D surface → flat, scanner-like image.

```
pip install opencv-python numpy scipy pillow matplotlib
python3 docscan3d.py ref.jpg right.jpg up.jpg down.jpg -o out   # recommended: 4 views
python3 docscan3d.py ref.jpg right.jpg -o out --focal-px 3030   # 2 views: focal is kept fixed, pass it
```

The first image is the reference that gets rendered. With 4 views the focal length is self-calibrated
(no checkerboard needed); with 2 views it is not observable, so pass it (Pixel 8, 16:9 mode: ~3030 px).

## Capture

- Same phone, same lens, same zoom, focus locked if possible.
- View 1 shows the whole page; it is the one that gets rendered.
- View 2: move the phone 10–20 cm sideways at the same height (aim for a reported triangulation angle > 8°).
- Page brighter than the background (dark table).
- If the phone doesn't write `FocalLengthIn35mmFilm` to EXIF, it assumes 26 mm equiv; the focal is refined in bundle adjustment anyway.

## Pipeline

1. Page outline in view 1, corners refined to subpixel on the paper edge.
2. SIFT + mutual ratio test, essential matrix (MAGSAC), pose.
3. NCC refinement of matches on affine pre-warped patches.
4. Bundle adjustment (pose, points, focal), robust triangulation.
5. Guided densification: corners in view 1, predicted in view 2 from nearby matches, NCC refined; outliers rejected against the sparse surface.
6. Page frame: best-fit plane, axes along the page edges. Scale from the known page size.
7. Robust height field z(u, v) (IRLS, second-difference regularizer).
8. Isometric unrolling of the height-field mesh (edge lengths preserved), corners re-intersected with the surface, scale re-measured on the unrolled outline, then repeated once.
9. Render: flat pixel (s, t) → (u, v) → (u, v, z) → pixel in view 1.

## Outputs (`out/`)

| file | content |
|---|---|
| `points.csv` | `u_mm, v_mm, z_mm, s_mm, t_mm, x1, y1, x2, y2, reproj_px` per measured point |
| `points.ply` | same points in page frame, colored |
| `heightfield.npz` | grid `u`, `v`, `z`, unrolled grid `st`, camera `K`, `R`, `t`, page frame |
| `heightfield.png` | fitted height map + measured points |
| `flat.png` | flattened page at `--dpi` (default 200), shading removed |
| `flat_raw.png` | same, original illumination |
| `overlay.jpg` | view 1 with detected page and points colored by height |
| `report.json` | point count, reprojection error, baseline, distance, strain, ... |

## Output

`flat.png` is the flattened page with the illumination divided out: the paper is modelled as
a smooth function of position and of the measured surface normal (fold shading), fitted to
the paper level (ink is darker and gets down-weighted), so printed grey areas stay grey.
`flat_raw.png` is the same geometry with the original lighting. `--fuse` blends all views
(meant for glare or a phone shadow in single shots); on the test samples it gave no image
gain and slightly worse geometry, so it is off by default.

Paper non-uniformity vs. the raw render (`test/eval_photo.py`, on scan-defined paper areas):
sample 1 10.6 % → 3.0 %, 2 8.2 % → 5.5 % (rest is show-through from the printed back side),
4 6.4 % → 2.3 %, 5 7.0 % → 1.4 %; edge sharpness unchanged.

## Real-photo results (Pixel 8, 5 folded letters, vs. flatbed scan)

`test/batch_real.sh` runs all samples and compares the output with the scans (`test/eval_real.py`).

| | median | p95 | worst panel | coverage |
|---|---|---|---|---|
| 4-corner homography (what apps do) | 1.00 mm | 3.04 mm | 2.03 mm | 0.74 |
| 2 views, known focal | 0.34 mm | 0.96 mm | 1.10 mm (one panel lost) | 0.72 |
| 4 views, self-calibrated (default) | 0.27 mm | 0.63 mm | 0.47 mm | 0.79 |

Extreme fold (2 views, ~45° creases, downscaled chat photos): 0.34 mm median, p95 0.59 mm.

### Steep folds

- Tilt-simulated (ASIFT-style) matching kicks in automatically when plain SIFT matches
  leave the page uncovered (`--asift auto|on|off`); steep panels look too different between
  views otherwise.
- Fold prior: if the surface bends in one direction only (parallel letter folds), the frame
  is turned along the folds and the paper is kept straight along them (`--ruling`, 1 = off).
  This also helps normal letters; curls and diagonal folds keep the isotropic model.
- The height-field extent follows the data (steeply folded pages do not sit at 0..W x 0..H).

## Synthetic test

```
python3 test/make_synthetic.py -o test/synth            # folded A4 letter, 2 views, known geometry
python3 docscan3d.py test/synth/view1.jpg test/synth/view2.jpg -o test/out
python3 test/evaluate.py test/synth test/out
```

Results (3000×4000 px, ~380 mm distance, folds 9° / −7°, curl + bent corner, noise, blur, JPEG):

| | 150 mm baseline | 60 mm baseline | 4-corner homography (what apps do) |
|---|---|---|---|
| 3D surface error (rms) | 0.042 mm | 0.040 mm | – |
| local warp in flat output (median / p95) | 0.066 / 0.17 mm | 0.069 / 0.19 mm | 0.37 / 1.08 mm |
| global scale error | 0.03 % | 0.05 % | 0.6 % |

## Limitations / next steps

- Synthetic only so far: no lens distortion, no rolling shutter. For real phones calibrate once with a checkerboard and pass `--calib`:
  `np.savez("calib.npz", K=K, dist=dist, size=(w, h))` with `K, dist` from `cv2.calibrateCamera`.
- Blank areas have no matches; the height field interpolates there. Page corners usually lie in such areas, which is where the remaining ~0.5 mm global offset comes from.
- Height field regularization rounds sharp creases a bit; a finer `--grid` with a lower `--smooth` keeps them sharper if the point density allows.
- Page detection needs a page brighter than its background; no segmentation network yet.
