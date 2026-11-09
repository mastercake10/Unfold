# DocScan for Android

One photo of the page, then the C++ pipeline (`../cpp/src`, via JNI) on the phone; the flat page
is saved to `Pictures/DocScan` or shared (e.g. to the Paperless app).

- Camera: back camera at 1x, 4:3 full resolution. The photo is analysed right after it is taken
  (page outline, features) and the result appears as soon as it is accepted. The focal length
  comes from the camera characteristics (lens mm / pixel pitch); bundle adjustment refines it.
- Flash button on the preview (bolt, remembered). Off by default: a phone flash easily blows out
  paper; useful in dim rooms.
- Live page outline (green: whole page in view). A photo that cannot be used (no page) is
  rejected at once with a message, and the user takes it again.
- Result screen: pinch to zoom, drag to pan, double tap to toggle fit / 2.5x.
- The output is turned so the text is upright.
- The 3D part runs on photos downscaled to 3200 px (same page pixels/mm as the full-size 16:9
  photos the pipeline was tuned on); the page is rendered from the full-size first photo.
- Settings (gear on the camera preview): "Clean look" (on by default, remembered): black text on white paper (`Options::doc_tone`);
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
