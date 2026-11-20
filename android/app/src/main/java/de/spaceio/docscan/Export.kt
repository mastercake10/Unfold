package de.spaceio.docscan

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.pdf.PdfDocument
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.io.OutputStream
import kotlin.math.roundToInt

private fun pdfName(scan: ScanItem) = "Unfold_${scan.dir.name}.pdf"

/** One PDF page per image, 595 pt (A4) wide; the image keeps its full resolution inside. */
fun writePdf(pages: List<File>, out: OutputStream) {
    val doc = PdfDocument()
    try {
        pages.forEachIndexed { i, f ->
            val bmp = BitmapFactory.decodeFile(f.path) ?: throw IOException("cannot read page ${i + 1}")
            val w = 595
            val h = (w * bmp.height.toFloat() / bmp.width).roundToInt()
            val page = doc.startPage(PdfDocument.PageInfo.Builder(w, h, i + 1).create())
            page.canvas.drawBitmap(bmp, null, Rect(0, 0, w, h), Paint(Paint.FILTER_BITMAP_FLAG))
            doc.finishPage(page)
        }
        doc.writeTo(out)
    } finally {
        doc.close()
    }
}

/** Writes the scan as Download/Unfold/<name>.pdf (no storage permission needed from Android 10). */
private fun savePdf(context: Context, scan: ScanItem): Boolean {
    val values = ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, pdfName(scan))
        put(MediaStore.MediaColumns.MIME_TYPE, "application/pdf")
        put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/Unfold")
    }
    val resolver = context.contentResolver
    val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return false
    return runCatching { resolver.openOutputStream(uri)!!.use { writePdf(scan.pages, it) } }
        .onFailure { resolver.delete(uri, null, null) }
        .isSuccess
}

suspend fun saveScans(context: Context, scans: List<ScanItem>) {
    val n = withContext(Dispatchers.IO) { scans.count { savePdf(context, it) } }
    val msg = when {
        n == scans.size && n == 1 -> "Saved to Download/Unfold"
        n == scans.size -> "Saved $n PDFs to Download/Unfold"
        else -> "Saved $n of ${scans.size} PDFs"
    }
    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
}

/** Shares each scan as one PDF (the Paperless app accepts one or several). */
suspend fun shareScans(context: Context, scans: List<ScanItem>) {
    val files = withContext(Dispatchers.IO) {
        runCatching {
            val dir = File(context.cacheDir, "export").apply { mkdirs() }
            scans.map { s -> File(dir, pdfName(s)).also { f -> f.outputStream().use { writePdf(s.pages, it) } } }
        }
    }.getOrElse {
        Toast.makeText(context, "PDF failed: ${it.message}", Toast.LENGTH_LONG).show()
        return
    }
    val uris = ArrayList(files.map { FileProvider.getUriForFile(context, "${context.packageName}.files", it) })
    val send = if (uris.size == 1) {
        Intent(Intent.ACTION_SEND).putExtra(Intent.EXTRA_STREAM, uris[0])
    } else {
        Intent(Intent.ACTION_SEND_MULTIPLE).putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
    }
    send.type = "application/pdf"
    send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    context.startActivity(Intent.createChooser(send, if (uris.size == 1) "Share scan" else "Share scans"))
}
