# docscan (C++ port)

Minimal C++ port of `docscan3d.py`, as the basis for the Android app. Same pipeline and
defaults; left out: ASIFT fallback for very steep folds (`--asift`), view fusion (`--fuse`), lens
distortion / calibration files, debug outputs and most flags. Output is `flat.png` only.

Dependencies: OpenCV >= 4.5.1 (core, imgproc, imgcodecs, features2d, flann, calib3d, photo) and
Eigen 3.3+ (header-only). Both are available for Android (OpenCV Android SDK includes SIFT).

```
sudo apt install libopencv-dev libeigen3-dev
cmake -S cpp -B cpp/build -G Ninja && ninja -C cpp/build
cpp/build/docscan ref.jpg right.jpg up.jpg down.jpg -o out            # 4 views
cpp/build/docscan ref.jpg right.jpg -o out --focal-px 3030            # 2 views: pass the focal
```

Library API (`src/docscan.hpp`): `cv::Mat docscan::process(images, options)`; throws
`std::runtime_error` with a user-facing message. The CLI reads the focal length from EXIF
(`FocalLengthIn35mmFilm`); in the app, pass `Options::focal_px` from the camera characteristics.
For capture apps, `docscan::Session` splits the work: `add(photo)` does the per-photo part (page
outline, SIFT, matching with photo 1) and returns a message if the photo is unusable; `finish()`
does the rest. Photos that fail matching or the scale check are left out if at least 2 remain.

GPU SIFT (`src/gpu_sift.cpp`, `src/shaders/`): Vulkan compute port of OpenCV's SIFT (same
parameters, same tiles), used when the build finds Vulkan and `glslc`; falls back to OpenCV on
any GPU error. `--no-gpu` or `DOCSCAN_GPU=0` disables it, `DOCSCAN_GPU=cpu` selects a software
Vulkan device for testing.

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
