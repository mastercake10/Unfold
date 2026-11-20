package de.spaceio.docscan

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
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
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
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
    var viewing by remember { mutableStateOf<Int?>(null) }  // page of the current scan shown large
    var confirm by remember { mutableStateOf<File?>(null) }
    if (library) {
        LibraryScreen(onClose = { library = false; model.refresh() })
        return
    }

    val done = model.pages.mapNotNull { it.file }
    val v = viewing
    if (v != null && done.isNotEmpty()) {
        PagesViewer("This scan", done, v, onBack = { viewing = null }) { i ->
            TextButton(onClick = { confirm = done[i] }) { Text("Delete page") }
        }
        confirm?.let { f ->
            ConfirmDelete("Delete this page?", "Take it again afterwards if you need it.", onDismiss = { confirm = null }) {
                model.deletePage(f)
                confirm = null
                if (done.size == 1) viewing = null
            }
        }
        return
    }

    if (granted) {
        CaptureScreen(model, onLibrary = { library = true }, onPage = { viewing = it }, onReport = { report(context, model) })
    } else Centered {
        Text("Unfold needs the camera to photograph the page.", textAlign = TextAlign.Center)
        Spacer(Modifier.height(16.dp))
        Button(onClick = { ask.launch(Manifest.permission.CAMERA) }) { Text("Allow camera") }
    }
}

@Composable
private fun Centered(content: @Composable () -> Unit) =
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally,
    ) { content() }

@Composable
internal fun KeepScreenOn() {
    val window = (LocalContext.current as? ComponentActivity)?.window
    DisposableEffect(window) {
        window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose { window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
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

/** Photos and log of the current scan, e.g. to send to a computer for debugging. */
internal fun report(context: Context, model: ScanModel) {
    val uris = ArrayList(model.reportFiles().map { FileProvider.getUriForFile(context, "${context.packageName}.files", it) })
    if (uris.isEmpty()) {
        Toast.makeText(context, "No photos in this scan yet", Toast.LENGTH_SHORT).show()
        return
    }
    val send = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
        type = "*/*"
        putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(send, "Send report"))
}
