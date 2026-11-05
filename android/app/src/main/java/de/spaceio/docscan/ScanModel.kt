package de.spaceio.docscan

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/** The guided shots, in pipeline order: the first one is rendered, the second is optional. */
val STEPS = listOf(
    "Page" to "Hold the phone above the page, page large and centred, and take the photo.",
    "Extra" to "Optional, for folded pages: move the phone 10–20 cm to the right at the same height and " +
        "take a second photo. Or press Process.",
)

sealed interface Phase {
    data object Capture : Phase
    data object Processing : Phase
    data class Done(val file: File, val bitmap: Bitmap, val seconds: Double, val notes: List<String>) : Phase
    data class Failed(val message: String) : Phase
}

/**
 * Each photo is analysed natively right after it is taken (features, matching with photo 1), so
 * after the last photo only the 3D and rendering part is left. All native calls run in order on
 * one worker thread.
 */
class ScanModel(app: Application) : AndroidViewModel(app) {
    private val shotDir = File(app.cacheDir, "shots").apply { mkdirs() }
    private val outDir = File(app.cacheDir, "out").apply { mkdirs() }
    private val main = Handler(Looper.getMainLooper())

    // the pipeline needs more stack than a default thread has
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(null, r, "docscan", 64L shl 20) }
    private val accepted = ArrayList<File>()  // photos the native session holds; worker thread only
    // Result for photo 1 alone, made right after it is accepted (most scans have one photo), so
    // "Process" can show it at once. Worker thread only; valid while photo 1 is the only photo.
    private var early: File? = null
    private var earlyErr: String? = null
    private var earlyFor: File? = null
    private var earlyTone = true
    private val prefs = app.getSharedPreferences("docscan", android.content.Context.MODE_PRIVATE)
    private var counter = 0

    val shots = mutableStateListOf<File>()
    var phase by mutableStateOf<Phase>(Phase.Capture)
        private set
    var progress by mutableStateOf("")
        private set
    /** Photos still being analysed. */
    var pending by mutableIntStateOf(0)
        private set
    /** Why the last photo was rejected (shown on the camera screen until the next photo). */
    var notice by mutableStateOf<String?>(null)
    /** Clean look: ink to black, light grey to white (better OCR on most letters, fades light grey print). */
    var docTone by mutableStateOf(prefs.getBoolean("docTone", true))
        private set

    fun changeDocTone(on: Boolean) {
        docTone = on
        prefs.edit().putBoolean("docTone", on).apply()
        // redo the early one-photo result with the new look
        worker.execute { if (accepted.size == 1 && early != null && earlyTone != on) makeEarly(accepted[0], on) }
    }

    private fun makeEarly(f: File, tone: Boolean) {
        val out = File(outDir, "early.png")
        earlyErr = Native.finish(out.path, tone)
        early = out
        earlyFor = f
        earlyTone = tone
    }

    init {
        worker.execute { Native.warmup() }
    }

    /** Log of the current scan, shared with the photos by "Send report". */
    val logFile = File(outDir, "log.txt")

    fun nextShotFile() = File(shotDir, "photo${++counter}.jpg")

    fun addShot(f: File, focalPx: Double?) {
        shots.add(f)
        notice = null
        pending++
        worker.execute {
            val err = Native.add(f.path, focalPx ?: 0.0)
            if (err == null) accepted.add(f)
            main.post {
                pending--
                if (err != null) {
                    shots.remove(f)
                    f.delete()
                    notice = "$err\nPlease take this photo again."
                }
            }
            if (err == null && accepted.size == 1) makeEarly(f, docTone)
        }
    }

    fun undo() {
        val f = shots.removeLastOrNull() ?: return
        notice = null
        worker.execute {
            if (accepted.lastOrNull() == f) {
                accepted.removeAt(accepted.size - 1)
                Native.removeLast()
            }
            if (accepted.size != 1 || accepted[0] != earlyFor) early = null
            f.delete()
        }
    }

    fun reset() {
        shots.clear()
        notice = null
        phase = Phase.Capture
        worker.execute {
            Native.reset()
            accepted.clear()
            early = null
            shotDir.listFiles()?.forEach { it.delete() }
            outDir.listFiles()?.forEach { it.delete() }
        }
    }

    fun process() {
        if (shots.isEmpty() || phase == Phase.Processing) return
        phase = Phase.Processing
        progress = ""
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val out = File(outDir, "docscan_$stamp.png")
        val t0 = SystemClock.elapsedRealtime()  // what the user waits for, including photos still in analysis
        val tone = docTone
        worker.execute {
            val e = early
            val err = when {
                accepted.isEmpty() -> "No usable photo."
                accepted.size == 1 && e != null && earlyFor == accepted[0] && earlyTone == tone ->
                    earlyErr ?: if (e.renameTo(out)) null else Native.finish(out.path, tone)
                else -> Native.finish(out.path, tone)
            }
            early = null
            val sec = (SystemClock.elapsedRealtime() - t0) / 1000.0
            val log = Native.log()
            // 0 = none ... 3 = severe: the CPU is slowed down to cool the phone
            val thermal = getApplication<Application>().getSystemService(PowerManager::class.java)
                ?.currentThermalStatus ?: -1
            runCatching {
                logFile.writeText(log + "\nwait after last photo: %.1f s, thermal status %d\n".format(sec, thermal))
            }
            val notes = log.lines().filter { "leaving out" in it }.map { it.substringAfter("] ") }
            val result = if (err != null) Phase.Failed(err)
            else BitmapFactory.decodeFile(out.path)?.let { Phase.Done(out, it, sec, notes) }
                ?: Phase.Failed("cannot read the result")
            main.post { phase = result }
        }
    }

    fun pollProgress() {
        progress = Native.progress()
    }

    override fun onCleared() {
        worker.execute { Native.reset() }
        worker.shutdown()
    }
}
