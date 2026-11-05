package de.spaceio.docscan

import java.nio.ByteBuffer

/** The C++ pipeline (cpp/src) via JNI. */
object Native {
    init {
        System.loadLibrary("docscan_jni")
    }

    // Session calls must all come from one worker thread, in order (see ScanModel).

    /** Analyses a photo; the first one starts a new scan. Null if usable, else a message for the user. */
    @JvmStatic
    external fun add(path: String, focalPx: Double): String?

    /** Prepares the GPU so the first photo does not wait for it. */
    @JvmStatic
    external fun warmup()

    @JvmStatic
    external fun removeLast()

    @JvmStatic
    external fun reset()

    /**
     * Computes the flat page from the added photos and writes a PNG. Null on success, else the error.
     * docTone: ink to black and light grey to white (cleaner, but fades light grey print).
     */
    @JvmStatic
    external fun finish(outPath: String, docTone: Boolean): String?

    /** Last progress message. */
    @JvmStatic
    external fun progress(): String

    /** All messages of the current scan, with seconds since its first photo. */
    @JvmStatic
    external fun log(): String

    /** Page corners in a camera frame (Y plane), as 8 floats normalised to the upright frame. */
    @JvmStatic
    external fun detectPage(
        y: ByteBuffer, width: Int, height: Int, rowStride: Int, rotation: Int, out: FloatArray,
    ): Boolean
}
