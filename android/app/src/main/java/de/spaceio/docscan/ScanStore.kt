package de.spaceio.docscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** One scan (a document): its folder and its flattened pages in order. */
data class ScanItem(val dir: File, val date: Date, val pages: List<File>)

/**
 * Finished scans live in the app's private files dir, one folder per scan with one PNG per page, so
 * they stay until the user deletes them. Export (PDF to Downloads / share) leaves them in the list.
 */
object ScanStore {
    private val stampRe = Regex("""(\d{8}_\d{6})(?:_\d+)?""")
    private val pageRe = Regex("""page(\d+)\.png""")
    // single pages of older builds: in the scans dir, or even older, in the cache
    private val oldPageRe = Regex("""docscan_(\d{8}_\d{6}(?:_\d+)?)\.png""")

    private fun stampFmt() = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)  // not thread-safe: one per call

    fun dir(context: Context): File = File(context.filesDir, "scans").apply { mkdirs() }

    /** New, empty folder for a scan started now. */
    fun newScanDir(context: Context): File {
        val stamp = stampFmt().format(Date())
        var d = File(dir(context), stamp)
        var n = 2
        while (d.exists()) d = File(dir(context), "${stamp}_${n++}")
        d.mkdirs()
        return d
    }

    /** File for page number [n] (only the order of the numbers matters, gaps are fine). */
    fun pageFile(scanDir: File, n: Int) = File(scanDir, "page%03d.png".format(n))

    fun pages(scanDir: File): List<File> =
        scanDir.listFiles { f -> f.isFile && pageRe.matches(f.name) }
            ?.sortedBy { pageRe.matchEntire(it.name)!!.groupValues[1].toInt() }
            ?: emptyList()

    /** All scans with at least one page, newest first. */
    fun list(context: Context): List<ScanItem> {
        migrate(context)
        return dir(context).listFiles { f -> f.isDirectory }
            ?.map { ScanItem(it, dateOf(it), pages(it)) }
            ?.filter { it.pages.isNotEmpty() }
            ?.sortedByDescending { it.date }
            ?: emptyList()
    }

    /** Older builds kept loose single pages: each becomes a one-page scan. */
    private fun migrate(context: Context) {
        for (src in listOf(dir(context), File(context.cacheDir, "out"))) {
            src.listFiles { f -> f.isFile && oldPageRe.matches(f.name) }?.forEach { f ->
                val dst = pageFile(File(dir(context), oldPageRe.matchEntire(f.name)!!.groupValues[1]), 1)
                if (dst.exists()) {
                    f.delete()  // already moved before
                    return@forEach
                }
                dst.parentFile!!.mkdirs()
                runCatching { if (!f.renameTo(dst)) { f.copyTo(dst); f.delete() } }
            }
        }
    }

    fun dateOf(scanDir: File): Date {
        val m = stampRe.matchEntire(scanDir.name)
        if (m != null) runCatching { return stampFmt().parse(m.groupValues[1])!! }
        return Date(scanDir.lastModified())
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
