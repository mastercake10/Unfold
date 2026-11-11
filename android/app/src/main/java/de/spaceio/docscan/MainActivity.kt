package de.spaceio.docscan

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.delay
import java.io.File
import kotlin.math.min

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Box(Modifier.fillMaxSize().background(Color.Black).safeDrawingPadding()) { App() }
            }
        }
    }
}

@Composable
private fun App(model: ScanModel = viewModel()) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) ask.launch(Manifest.permission.CAMERA) }

    var library by remember { mutableStateOf(false) }
    if (library) {
        LibraryScreen(onClose = { library = false })
        return
    }

    when (val p = model.phase) {
        Phase.Capture ->
            if (granted) CaptureScreen(model) { library = true }
            else Centered {
                Text("DocScan needs the camera to photograph the page.", textAlign = TextAlign.Center)
                Spacer(Modifier.height(16.dp))
                Button(onClick = { ask.launch(Manifest.permission.CAMERA) }) { Text("Allow camera") }
            }

        Phase.Processing -> ProcessingScreen(model)
        is Phase.Done -> ResultScreen(
            p, onNew = { model.reset() }, onReport = { report(context, model) },
            onLibrary = { library = true },
        )
        is Phase.Failed -> {
            BackHandler { model.reset() }
            Centered {
                Text("Could not process the photo", style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.height(12.dp))
                Text(p.message, textAlign = TextAlign.Center)
                Spacer(Modifier.height(24.dp))
                Button(onClick = { model.reset() }) { Text("Take a new photo") }
                Spacer(Modifier.height(8.dp))
                TextButton(onClick = { report(context, model) }) { Text("Send report (photos + log)") }
            }
        }
    }
}

@Composable
private fun Centered(content: @Composable () -> Unit) =
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally,
    ) { content() }

@Composable
private fun ProcessingScreen(model: ScanModel) {
    KeepScreenOn()
    BackHandler { }  // the native run cannot be cancelled
    LaunchedEffect(Unit) {
        while (true) {
            model.pollProgress()
            delay(250)
        }
    }
    Centered {
        CircularProgressIndicator()
        Spacer(Modifier.height(24.dp))
        Text("Measuring the page in 3D and flattening it …", textAlign = TextAlign.Center)
        Spacer(Modifier.height(12.dp))
        Text(model.progress.trim(), color = Color.Gray, textAlign = TextAlign.Center)
    }
}

@Composable
private fun KeepScreenOn() {
    val window = (LocalContext.current as? ComponentActivity)?.window
    DisposableEffect(window) {
        window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose { window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
    }
}

@Composable
private fun ResultScreen(
    r: Phase.Done, onNew: () -> Unit, onReport: () -> Unit, onLibrary: () -> Unit,
) {
    val context = LocalContext.current
    BackHandler(onBack = onNew)
    Column(Modifier.fillMaxSize()) {
        ZoomablePage(
            r.bitmap,
            Modifier.weight(1f).fillMaxWidth().background(Color.DarkGray).padding(8.dp),
        )
        r.notes.forEach {
            Text(it, color = Color(0xFFFFB020), modifier = Modifier.padding(horizontal = 16.dp))
        }
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("%.1f s".format(r.seconds), color = Color.Gray)
            Row {
                TextButton(onClick = onLibrary) { Text("All scans") }
                TextButton(onClick = onReport) { Text("Send report") }
            }
        }
        Row(
            Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) {
            OutlinedButton(onClick = onNew) { Text("New scan") }
            OutlinedButton(onClick = { save(context, r.file) }) { Text("Save") }
            Button(onClick = { share(context, r.file) }) { Text("Share") }
        }
    }
}

/**
 * The flattened page, pinch-zoomed and panned; double tap toggles between fit and 2.5x.
 */
@Composable
internal fun ZoomablePage(bitmap: Bitmap, modifier: Modifier = Modifier) {
    val image = remember(bitmap) { bitmap.asImageBitmap() }
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    var box by remember { mutableStateOf(IntSize.Zero) }

    fun fitted(): Pair<Float, Float> {
        if (box.width == 0 || box.height == 0) return 0f to 0f
        val s = min(box.width.toFloat() / bitmap.width, box.height.toFloat() / bitmap.height)
        return bitmap.width * s to bitmap.height * s
    }

    fun clamp() {
        val (w, h) = fitted()
        val mx = ((w * scale - box.width) / 2).coerceAtLeast(0f)
        val my = ((h * scale - box.height) / 2).coerceAtLeast(0f)
        offset = Offset(offset.x.coerceIn(-mx, mx), offset.y.coerceIn(-my, my))
    }

    Box(
        modifier
            .clipToBounds()
            .onSizeChanged { box = it; clamp() }
            .pointerInput(Unit) {
                detectTransformGestures { centroid, pan, zoom, _ ->
                    val s0 = scale
                    scale = (s0 * zoom).coerceIn(1f, 8f)
                    // zoom around the pinch centre, then pan
                    val c = Offset(centroid.x - box.width / 2f, centroid.y - box.height / 2f)
                    offset = c - (c - offset) * (scale / s0) + pan
                    if (scale == 1f) offset = Offset.Zero else clamp()
                }
            }
            .pointerInput(Unit) {
                detectTapGestures(onDoubleTap = {
                    scale = if (scale > 1f) 1f else 2.5f
                    offset = Offset.Zero
                })
            },
        contentAlignment = Alignment.Center,
    ) {
        Image(
            image, "flattened page",
            Modifier.fillMaxSize().graphicsLayer {
                scaleX = scale
                scaleY = scale
                translationX = offset.x
                translationY = offset.y
            },
            contentScale = ContentScale.Fit,
        )
    }
}

/** Copies the PNG to Pictures/DocScan (no storage permission needed from Android 10). */
private fun saveFile(context: Context, file: File): Boolean {
    if (!file.exists()) return false
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, file.name)
        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/DocScan")
    }
    return runCatching {
        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)!!
        context.contentResolver.openOutputStream(uri)!!.use { out -> file.inputStream().use { it.copyTo(out) } }
    }.isSuccess
}

/** Saves several pages to Pictures/DocScan; returns how many were written. */
internal fun saveAll(context: Context, files: List<File>): Int = files.count { saveFile(context, it) }

internal fun save(context: Context, file: File) {
    val ok = saveFile(context, file)
    Toast.makeText(context, if (ok) "Saved to Pictures/DocScan" else "Saving failed", Toast.LENGTH_SHORT).show()
}

/** Photos and log of the last run, e.g. to send to a computer for debugging. */
private fun report(context: Context, model: ScanModel) {
    val files = model.shots.toList() + listOf(model.logFile).filter { it.exists() }
    val uris = ArrayList(files.map { FileProvider.getUriForFile(context, "${context.packageName}.files", it) })
    val send = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
        type = "*/*"
        putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(send, "Send report"))
}

internal fun share(context: Context, file: File) {
    if (!file.exists()) {
        Toast.makeText(context, "The file is gone", Toast.LENGTH_SHORT).show()
        return
    }
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "image/png"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(send, "Share scan"))
}

/** Shares one or several pages (the Paperless app accepts the multi-page intent). */
internal fun shareMany(context: Context, files: List<File>) {
    val ok = files.filter { it.exists() }
    if (ok.isEmpty()) {
        Toast.makeText(context, "The files are gone", Toast.LENGTH_SHORT).show()
        return
    }
    if (ok.size == 1) return share(context, ok[0])
    val uris = ArrayList(ok.map { FileProvider.getUriForFile(context, "${context.packageName}.files", it) })
    val send = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
        type = "image/png"
        putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(send, "Share scans"))
}
