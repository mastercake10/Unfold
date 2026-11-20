# Unfold for Android

One photo per page, then the C++ pipeline (`../cpp/src`, via JNI) on the phone. The pages of a
scan (one document) are saved to `Download/Unfold` as one PDF or shared as one PDF (e.g. to the
Paperless app).

- Camera: back camera at 1x, 4:3 full resolution, shutter on the preview. Each photo is flattened
  in the background (one native session per page), so the next page can be taken right away; the
  pages of the current scan show as a strip below the preview (spinner while processing, tap to
  view or delete). "New scan" starts the next document; every page is already in the scan list. The focal length
  comes from the camera characteristics (lens mm / pixel pitch); with one photo it is not refined
  (a focal 17 % off barely changes the result).
- Flash button on the preview (bolt, remembered). Off by default: a phone flash easily blows out
  paper; useful in dim rooms.
- Live page outline (green: whole page in view). A photo that cannot be used (no page) is
  dropped from the strip with a message, and the user takes it again (it is added at the end).
- Page view: pinch to zoom, drag to pan, double tap to toggle fit / 2.5x; ‹ › step through pages.
- "Scans" on the camera screen lists every scan in the app, newest first. Tap one to view it
  (save PDF, share PDF, delete), or Select several and save or share them (one PDF each). Delete
  removes the app's copy; PDFs already in Download stay.
- The output is turned so the text is upright.
- The page analysis runs on the photo downscaled to 3200 px (the same page pixels/mm as the
  full-size 16:9 photos the pipeline was tuned on); rendering uses the full-size photo.
- Settings (gear on the camera preview): "Clean look" (on by default, remembered): black text on white paper (`Options::doc_tone`);
  turn it off for pages with light grey print, which it fades.
- "Send report" (in Settings) shares the photos and the timed log (`log.txt`) of the current scan
  for debugging.
- Output: one A4, 200 dpi page image per page (PNG in the app, embedded losslessly in the PDF,
  595 pt wide). arm64-v8a only.

## Build (no Android Studio needed)

Needs JDK 17, Android SDK command-line tools with `platforms;android-35`, `build-tools;35.0.0`,
`ndk;27.2.12479018`, `cmake;3.22.1`, the OpenCV Android SDK 4.10.0 and Eigen 3.4 headers.
`local.properties` (not in git):

```
sdk.dir=/opt/android-sdk
opencv.dir=/opt/opencv-android/sdk/native/jni
eigen.dir=/opt/eigen-3.4.0
```

```
./gradlew assembleRelease        # app/build/outputs/apk/release/app-release.apk
adb install -r app/build/outputs/apk/release/app-release.apk
```

The release APK is signed with the debug key (no release keystore yet), so it installs directly
but cannot be uploaded to Play.
