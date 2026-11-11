package de.spaceio.docscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** One flattened page kept in the app's scan list. */
data class ScanItem(val file: File, val date: Date)

/**
 * Finished scans live in the app's private files dir, one PNG per page, so they stay until the user
 * deletes them. Export (Pictures / share) copies them out; the original stays in the list.
 */
object ScanStore {
    private val stampFmt = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
    private val nameRe = Regex("""docscan_(\d{8}_\d{6})(?:_\d+)?\.png""")

    fun dir(context: Context): File = File(context.filesDir, "scans").apply { mkdirs() }

    /** New, unused file for a scan finished at [stamp]. */
    fun newFile(context: Context, stamp: String): File {
        val d = dir(context)
        var f = File(d, "docscan_$stamp.png")
        var n = 2
        while (f.exists()) f = File(d, "docscan_${stamp}_${n++}.png")
        return f
    }

    fun list(context: Context): List<ScanItem> {
        importCache(context)
        return dir(context).listFiles { f -> f.isFile && f.name.endsWith(".png", true) }
            ?.map { ScanItem(it, dateOf(it)) }
            ?.sortedByDescending { it.date }
            ?: emptyList()
    }

    /** Keep finished pages that an older build left in the cache (before the scan list existed). */
    private fun importCache(context: Context) {
        val old = File(context.cacheDir, "out")
        val re = Regex("""docscan_\d{8}_\d{6}(?:_\d+)?\.png""")
        old.listFiles { f -> f.isFile && re.matches(f.name) }?.forEach { f ->
            val dst = File(dir(context), f.name)
            if (dst.exists()) {
                f.delete()  // duplicate of a page already in the list
                return@forEach
            }
            runCatching { if (!f.renameTo(dst)) f.copyTo(dst, overwrite = false) }
        }
    }

    fun dateOf(f: File): Date {
        val m = nameRe.matchEntire(f.name)
        if (m != null) runCatching { return stampFmt.parse(m.groupValues[1])!! }
        return Date(f.lastModified())
    }

    /** Downsampled page for the grid; null if the file cannot be decoded. */
    fun thumb(file: File, maxSide: Int = 1200): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > maxSide) sample *= 2
        val o = BitmapFactory.Options().apply { inSampleSize = sample }
        return BitmapFactory.decodeFile(file.path, o)
    }

    fun full(file: File): Bitmap? = BitmapFactory.decodeFile(file.path)
}
