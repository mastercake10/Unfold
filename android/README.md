# DocScan for Android

One photo of the page (a second one is optional, for more exact geometry), then the C++ pipeline (`../cpp/src`, via JNI) on the phone; the
flat page is saved to `Pictures/DocScan` or shared (e.g. to the Paperless app).

- Camera: back camera at 1x, 4:3 full resolution. Focus is measured on the first photo and
  then locked, so all views share one focal length. The start value for the focal length comes
  from the camera characteristics (lens mm / pixel pitch); bundle adjustment refines it.
- Live page outline (green: whole page in view). Shot 1: the page (rendered). "Process" works
  from 1 shot. Optional shot 2: 10–20 cm to the right of the first position, for folded pages
  (the 3D part then measures the folds; focal kept fixed).
- Each photo is analysed right after it is taken (page outline; with a second photo features and
  matching with photo 1, via `docscan::Session`). A photo that cannot be used (no page, phone not
  moved, no match) is rejected at once with a message, and the user retakes it.
- The output is turned so the text is upright.
- The 3D part runs on photos downscaled to 3200 px (same page pixels/mm as the full-size 16:9
  photos the pipeline was tuned on); the page is rendered from the full-size first photo.
- "Clean look" switch (on by default, remembered): black text on white paper (`Options::doc_tone`);
  turn it off for pages with light grey print, which it fades.
- "Send report" shares the photos and the timed log (`log.txt`) for debugging.
- Output: A4, 200 dpi PNG. arm64-v8a only.

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
