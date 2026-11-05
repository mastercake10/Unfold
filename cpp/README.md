# docscan (C++ port)

Minimal C++ port of `docscan3d.py`, as the basis for the Android app. Same pipeline and
defaults; left out: ASIFT fallback for very steep folds (`--asift`), view fusion (`--fuse`), lens
distortion / calibration files, debug outputs and most flags. Output is `flat.png` only.

Dependencies: OpenCV >= 4.5.1 (core, imgproc, imgcodecs, features2d, flann, calib3d, photo) and
Eigen 3.3+ (header-only). Both are available for Android (OpenCV Android SDK includes SIFT).

```
sudo apt install libopencv-dev libeigen3-dev
cmake -S cpp -B cpp/build -G Ninja && ninja -C cpp/build
cpp/build/docscan page.jpg -o out --focal-px 3030                    # 1 photo
cpp/build/docscan ref.jpg right.jpg -o out --focal-px 3030            # 2 views: pass the focal
cpp/build/docscan ref.jpg right.jpg up.jpg down.jpg -o out            # 4 views
```

One photo: there is no 3D measurement. The page is taken as the plane through its four corners
(focal length from the camera; results barely change with a focal 17 % off), then the curved
outline is pulled onto the A4 rectangle and printed lines and text lines are straightened. OCR
reads it as well as 2 or 4 views (tesseract, 5 samples: 92-98 % of the scan's words in all three
modes); geometry is about 2x less exact (median 0.18-0.48 mm vs 0.14-0.25 mm against the scans).
Text lines are used in every mode; `DOCSCAN_TEXTLINES=0` turns them off. They are tracked per
3 mm column (median row of dark ink, underlines removed) and linked across word gaps, so lines
that kink at a crease are kept whole; vertical ones too for landscape pages. With one photo they
weigh 8x more (nothing else measures the shape inside the page): on 4 one-photo samples compared
with the source PDFs, text tilt went from 0.2-0.6 deg rms to 0.07-0.3 deg and tesseract finds
96.6 % of the PDF's words (OCR of the PDF image itself: 97.2 %). The output is turned so the text
is upright (text line direction, ascenders vs descenders); `--no-rotate` turns that off.

Library API (`src/docscan.hpp`): `cv::Mat docscan::process(images, options)`; throws
`std::runtime_error` with a user-facing message (1 to 8 photos). The CLI reads the focal length from EXIF
(`FocalLengthIn35mmFilm`); in the app, pass `Options::focal_px` from the camera characteristics.
For capture apps, `docscan::Session` splits the work: `add(photo)` does the per-photo part (page
outline; from the second photo on SIFT and matching with photo 1) and returns a message if the
photo is unusable; `finish()` does the rest. Photos that fail matching or the scale check are
left out (down to 1 photo).

GPU SIFT (`src/gpu_sift.cpp`, `src/shaders/`): Vulkan compute port of OpenCV's SIFT (same
parameters, same tiles), used when the build finds Vulkan and `glslc`; falls back to OpenCV on
any GPU error. `--no-gpu` or `DOCSCAN_GPU=0` disables it, `DOCSCAN_GPU=cpu` selects a software
Vulkan device for testing. The same device runs non-local means denoising (`shaders/nlm.comp`)
on the top part of the page while OpenCV does the rest; the split follows the measured speeds.
Parallel loops use an own pool of persistent threads (`parallel_for` in `common`), because
OpenCV runs a `parallel_for_` serially while another thread is inside one.

Layout: `common` (types, kd-tree, robust Levenberg-Marquardt), `camera` (triangulation, bundle
adjustment with exact Jacobian), `matching` (page quad/outline, tiled SIFT, FLANN, essential
matrix, NCC refinement, densification), `surface` (page frame, height field, unrolling, outline,
edge/line correction), `render` (rendering, shading removal, paper cleanup), `pipeline` (main flow).

Differences to the Python version: own sparse/dense solvers instead of scipy (same objectives),
the inverse flat map is rasterised from the mesh triangles instead of `griddata`, exact bilinear
sampling instead of `cv2.remap` fixed point in a few places, different random subsets for bundle
adjustment. Results agree within the run-to-run noise of the pipeline.

## Results (5 samples, 16-core desktop)

| | Python | C++ |
|---|---|---|
| 4 views, all 5 samples | 75 s | 51 s (~10 s per page) |
| 2 views, all 5 samples | 48 s | 31 s |
| peak memory (4 views) | ~1.1 GB | 0.72 GB |

One page, 4 views: 19.9 s on 1 core, 11.7 s on 4 cores. Stage times on 4 cores: matching (SIFT
+ FLANN) 6.4 s, rendering + cleanup 2.7 s, height field + unrolling 1.6 s, bundle adjustment 0.1 s.
Accuracy vs. flatbed scans and the synthetic test: same as Python within noise.
