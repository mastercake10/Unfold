package de.spaceio.docscan

import android.app.Application
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import java.io.File
import java.util.concurrent.Executors

/** What to tell the user before the first page. */
const val CAPTURE_HINT =
    "Hold the phone above the page, page large and centred, and take the photo. " +
        "Then the next page: all pages go into this scan."

/** A page of the current scan; [file] is null while its photo is still being processed. */
data class Page(val id: Int, val file: File? = null)

/**
 * The current scan (one document, one or more pages). Each photo is flattened in the background
 * (one native session per page), so the next page can be taken right away. All native calls run in
 * order on one worker thread; finished pages go straight into the scan's folder in [ScanStore].
 */
class ScanModel(app: Application) : AndroidViewModel(app) {
    private val shotDir = File(app.cacheDir, "shots").apply { mkdirs() }
    private val outDir = File(app.cacheDir, "out").apply { mkdirs() }
    private val main = Handler(Looper.getMainLooper())

    // the pipeline needs more stack than a default thread has
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(null, r, "docscan", 64L shl 20) }
    private var scanDir: File? = null  // folder of the current scan, made with its first page; worker thread only
    private val prefs = app.getSharedPreferences("docscan", android.content.Context.MODE_PRIVATE)
    private var counter = 0  // never reset, so a page still in work from the last scan cannot be mixed up

    val pages = mutableStateListOf<Page>()

    /** Photos still being processed. */
    val pending get() = pages.count { it.file == null }

    /** Why the last photo was rejected (shown on the camera screen until the next photo). */
    var notice by mutableStateOf<String?>(null)

    /** Clean look: ink to black, light grey to white (better OCR on most letters, fades light grey print). */
    var docTone by mutableStateOf(prefs.getBoolean("docTone", true))
        private set

    /** Flash on the photo (remembered). */
    var flashOn by mutableStateOf(prefs.getBoolean("flashOn", false))
        private set

    fun changeFlash(on: Boolean) {
        flashOn = on
        prefs.edit().putBoolean("flashOn", on).apply()
    }

    fun changeDocTone(on: Boolean) {
        docTone = on
        prefs.edit().putBoolean("docTone", on).apply()
    }

    init {
        worker.execute { Native.warmup() }
    }

    /** Log of the current scan, shared with the photos by "Send report". */
    val logFile = File(outDir, "log.txt")

    fun reportFiles(): List<File> = (shotDir.listFiles()?.sorted() ?: emptyList()) + listOf(logFile).filter { it.exists() }

    fun nextShotFile() = File(shotDir, "photo${++counter}.jpg")

    fun addShot(f: File, focalPx: Double?) {
        val id = ++counter
        pages.add(Page(id))
        notice = null
        val tone = docTone
        worker.execute {
            val t0 = SystemClock.elapsedRealtime()
            val tmp = File(outDir, "page.png")
            val err = Native.add(f.path, focalPx ?: 0.0) ?: Native.finish(tmp.path, tone)
            val log = Native.log()
            Native.reset()  // one photo per page: free it before the next one
            val out = if (err != null) null else {
                val d = scanDir ?: ScanStore.newScanDir(getApplication()).also { scanDir = it }
                ScanStore.pageFile(d, id).takeIf { moveOrCopy(tmp, it) }
            }
            // 0 = none ... 3 = severe: the CPU is slowed down to cool the phone
            val thermal = getApplication<Application>().getSystemService(PowerManager::class.java)
                ?.currentThermalStatus ?: -1
            val sec = (SystemClock.elapsedRealtime() - t0) / 1000.0
            runCatching { logFile.appendText("== ${f.name}\n$log\npage: %.1f s, thermal status %d\n".format(sec, thermal)) }
            main.post {
                val i = pages.indexOfFirst { it.id == id }
                if (i < 0) return@post  // the scan was closed meanwhile; the page is in its folder
                if (out != null) {
                    pages[i] = Page(id, out)
                } else {
                    pages.removeAt(i)
                    notice = "Page ${i + 1}: ${err ?: "cannot store the page"}\nPlease take it again."
                }
            }
        }
    }

    fun deletePage(file: File) {
        pages.removeAll { it.file == file }
        worker.execute {
            file.delete()
            scanDir?.let { if (ScanStore.pages(it).isEmpty()) { it.delete(); scanDir = null } }
        }
    }

    /** The finished pages as a scan, null if there are none yet. */
    fun current(): ScanItem? {
        val files = pages.mapNotNull { it.file }
        val dir = files.firstOrNull()?.parentFile ?: return null
        return ScanItem(dir, ScanStore.dateOf(dir), files)
    }

    /** Pages deleted in the scan list (the whole scan or not) leave the current scan too. */
    fun refresh() {
        pages.removeAll { it.file != null && !it.file.exists() }
    }

    /** Starts a new scan; the finished one stays in the scan list. */
    fun newScan() {
        pages.clear()
        notice = null
        worker.execute {
            scanDir = null
            shotDir.listFiles()?.forEach { it.delete() }
            outDir.listFiles()?.forEach { it.delete() }
        }
    }

    private fun moveOrCopy(src: File, dst: File): Boolean {
        if (src.renameTo(dst)) return true
        return runCatching { src.copyTo(dst, overwrite = true); src.delete(); true }.getOrDefault(false)
    }

    override fun onCleared() {
        worker.execute { Native.reset() }
        worker.shutdown()
    }
}
