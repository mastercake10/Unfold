# Unfold – Document Scanner

An Android scanner app for folded and curved letters: one photo per page → a flat, scanner-like
page, all pages of a document → one PDF. Everything runs on the phone, offline.

![One photo of a curved page → flat, scanner-like page](docs/example.jpg)

## Why it is different

Most scanner apps find the four page corners and apply a perspective transform (a homography).
That fixes the camera angle but not the paper: folds and curls stay bent, text lines wave, and the
fold shading stays in the image. Unfold corrects the page itself:

- **Folds and curls are straightened.** The curved paper edge is measured to subpixel and pulled
  onto the A4 rectangle; printed lines and text lines are made straight, with the content between
  them holding the correction in place. Against flatbed scans of folded letters: median error
  0.18–0.48 mm per page, vs 1.00 mm (p95 3.04 mm) for the four-corner method.
- **Fold shading is removed, not just brightened.** The paper's brightness is modelled across the
  page, so shadows at creases go away while printed grey areas stay grey.
- **Text that OCR reads.** "Clean look" turns ink black and paper white, with the black point
  following the darkest ink nearby, so thin print the camera blurred to grey is darkened rather
  than faded. Tesseract finds 96.6 % of the source PDF's words in the output (OCR of the PDF image
  itself: 97.2 %).
- **Upright and to scale.** The page comes out turned so the text is upright, as A4 at 200 dpi.
- **Private.** No cloud, no account: the processing is C++ on the phone (GPU-accelerated where the
  phone allows).

## The app

- Camera screen with a live page outline (green when the whole page is in view), flash toggle and
  the shutter on the preview.
- One photo per page. Each page is processed in the background, so you can take the next page right
  away; the pages of the current scan are shown in a strip below the preview.
- Save the scan as one PDF to `Download/Unfold`, or share it (e.g. to Paperless).
- All scans stay in the app's list until you delete them; view, save, share or delete them later.

**Download:** the APK is on the [releases page](../../releases/latest). Android 10 or newer,
64-bit ARM phones. To build it yourself (no Android Studio needed): [android/README.md](android/README.md).

Tips: a darker or coloured surface works best (a light wooden desk is fine; a white desk often is
not), fill the frame with the page and keep all of it in view.

## How it works

1. **Page outline.** The page is found against the background; its corners and edges are refined to
   subpixel on the paper edge.
2. **Page plane.** The four corners give the page's position and tilt; the focal length comes from
   the camera's own data, so no calibration is needed.
3. **Flattening.** The measured outline is pulled onto the A4 rectangle. Printed lines and text lines
   are straightened: text lines are tracked in 3 mm columns and linked across word gaps, so a line
   that kinks at a crease stays one line.
4. **Clean-up.** Shading is divided out, the paper is denoised and evened out, the result is turned
   upright.

One photo has no 3D measurement inside the page; the shape between the edges comes from the
printed content. That is enough for normal letters, and weaker on a strongly curled sheet or a page
without text or lines. Details: [cpp/README.md](cpp/README.md).

## Repository

| dir | what |
|---|---|
| [android/](android/README.md) | the app (Kotlin, Compose, CameraX; JNI to the C++ pipeline) |
| [cpp/](cpp/README.md) | the pipeline in C++ (OpenCV + Eigen), also as a command-line tool: `docscan page.jpg -o out` |
| [python/](python/README.md) | the reference implementation: 2–4 photos with a measured 3D surface (about 2× more exact geometry than one photo, same OCR), used to develop and benchmark the pipeline, plus the evaluation tools |

## License

MIT, see [LICENSE](LICENSE). The app bundles OpenCV (Apache-2.0) and Eigen (MPL-2.0).
