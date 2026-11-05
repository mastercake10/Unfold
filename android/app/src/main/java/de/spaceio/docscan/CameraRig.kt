package de.spaceio.docscan

import android.content.Context
import android.graphics.BitmapFactory
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Back camera at 1x with 4:3 full-resolution stills, a 4:3 preview and a small analysis stream for
 * the live page outline. Focus is measured once for the first photo and then locked, so all views
 * share the same focal length.
 */
@OptIn(ExperimentalCamera2Interop::class)
class CameraRig(private val context: Context) {
    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var capture: ImageCapture? = null
    private val analysisExec = Executors.newSingleThreadExecutor()
    private var released = false

    @Volatile private var focusDistance: Float? = null
    var focusLocked = false
        private set

    /** Called on the analysis thread with the page corners (normalised, upright) or null. */
    var onQuad: (FloatArray?) -> Unit = {}

    fun bind(owner: LifecycleOwner, view: PreviewView) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (released) return@addListener
            val provider = future.get()
            val ratio43 = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY

            val previewBuilder = Preview.Builder().setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(ratio43).build()
            )
            // remember where autofocus ended up, to lock it there later
            Camera2Interop.Extender(previewBuilder).setSessionCaptureCallback(
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        s: CameraCaptureSession, r: CaptureRequest, result: TotalCaptureResult,
                    ) {
                        if (!focusLocked) result.get(CaptureResult.LENS_FOCUS_DISTANCE)?.let { focusDistance = it }
                    }
                })
            val preview = previewBuilder.build().also { it.setSurfaceProvider(view.surfaceProvider) }

            val cap = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setJpegQuality(95)
                .setFlashMode(ImageCapture.FLASH_MODE_OFF)
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(ratio43)
                        .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                        .build()
                ).build()

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(ratio43)
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                            )
                        ).build()
                ).build()
            val quad = FloatArray(8)
            var lastDetect = 0L
            analysis.setAnalyzer(analysisExec) { img ->
                img.use {
                    // a few outlines per second are enough; leave the CPU to the photo analysis
                    val now = android.os.SystemClock.elapsedRealtime()
                    if (now - lastDetect < 200) return@use
                    lastDetect = now
                    val y = it.planes[0]
                    val found = Native.detectPage(
                        y.buffer, it.width, it.height, y.rowStride, it.imageInfo.rotationDegrees, quad
                    )
                    onQuad(if (found) quad.copyOf() else null)
                }
            }

            this.provider = provider
            provider.unbindAll()
            val cam = provider.bindToLifecycle(
                owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, cap, analysis
            )
            cam.cameraControl.setZoomRatio(1f)
            if (focusLocked) applyFocusLock(cam)
            camera = cam
            capture = cap
        }, ContextCompat.getMainExecutor(context))
    }

    /** Focal length in pixels for an image of the given size, from the camera characteristics. */
    fun focalPx(imgW: Int, imgH: Int): Double? {
        val info = camera?.let { Camera2CameraInfo.from(it.cameraInfo) } ?: return null
        return focalPx(
            info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
                ?: return null,
            info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null,
            info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE) ?: return null,
            info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return null,
            imgW, imgH,
        )
    }

    /**
     * Takes a photo into [file]. For the first photo ([lockAfter]) autofocus runs on the page centre
     * first and focus is locked afterwards.
     */
    fun shoot(file: File, lockAfter: Boolean, done0: (Result<Size>) -> Unit) {
        // report once; if the camera never answers, give the shutter back after a while
        val handler = android.os.Handler(android.os.Looper.getMainLooper())
        var reported = false
        val done = { r: Result<Size> ->
            if (!reported) {
                reported = true
                handler.removeCallbacksAndMessages(null)
                done0(r)
            }
        }
        handler.postDelayed({
            Log.w("docscan", "photo: no answer from the camera")
            done(Result.failure(IllegalStateException("the camera did not respond, please try again")))
        }, 8000)
        val cam = camera ?: return done(Result.failure(IllegalStateException("camera not ready")))
        val cap = capture ?: return done(Result.failure(IllegalStateException("camera not ready")))
        val main = ContextCompat.getMainExecutor(context)
        val take = {
            cap.takePicture(
                ImageCapture.OutputFileOptions.Builder(file).build(), main,
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(r: ImageCapture.OutputFileResults) {
                        if (lockAfter) lockFocus()
                        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                        BitmapFactory.decodeFile(file.path, o)
                        done(Result.success(Size(o.outWidth, o.outHeight)))
                    }

                    override fun onError(e: ImageCaptureException) = done(Result.failure(e))
                })
        }
        if (focusLocked) {
            take()
            return
        }
        val point = SurfaceOrientedMeteringPointFactory(1f, 1f).createPoint(0.5f, 0.5f)
        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
            .setAutoCancelDuration(3, TimeUnit.SECONDS).build()
        val f = cam.cameraControl.startFocusAndMetering(action)
        f.addListener({ take() }, main)  // take the photo even if focusing failed
    }

    private fun lockFocus() {
        val cam = camera ?: return
        val minFocus = Camera2CameraInfo.from(cam.cameraInfo)
            .getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        if (minFocus <= 0f || focusDistance == null) return  // fixed-focus lens or no AF result
        focusLocked = true
        cam.cameraControl.cancelFocusAndMetering()
        applyFocusLock(cam)
    }

    private fun applyFocusLock(cam: Camera) {
        val d = focusDistance ?: return
        Log.i("docscan", "focus locked at %.2f dpt".format(d))
        Camera2CameraControl.from(cam.cameraControl).setCaptureRequestOptions(
            CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, d)
                .build()
        )
    }

    fun unlockFocus() {
        focusLocked = false
        focusDistance = null
        camera?.let { Camera2CameraControl.from(it.cameraControl).clearCaptureRequestOptions() }
    }

    /** Closes the camera (the screen goes away while the activity keeps running). */
    fun release() {
        released = true
        provider?.unbindAll()
        camera = null
        analysisExec.shutdown()
    }
}
