package de.spaceio.docscan

import android.graphics.Rect
import android.util.Size
import android.util.SizeF
import kotlin.math.max
import kotlin.math.min

/**
 * Focal length in pixels of a captured image, from the lens focal length (mm) and the sensor size.
 * The image is assumed to be the largest crop of the active array with the image's aspect ratio,
 * scaled to the image size. Only a start value: with 3+ views bundle adjustment refines it.
 */
fun focalPx(focalMm: Float, physical: SizeF, pixelArray: Size, active: Rect, imgW: Int, imgH: Int): Double {
    val pitch = physical.width.toDouble() / pixelArray.width  // mm per sensor pixel
    val aw = max(active.width(), active.height()).toDouble()
    val ah = min(active.width(), active.height()).toDouble()
    val il = max(imgW, imgH).toDouble()
    val ish = min(imgW, imgH).toDouble()
    val cropW = if (il / ish >= aw / ah) aw else ah * il / ish
    return focalMm / pitch * il / cropW
}
