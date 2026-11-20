package de.spaceio.docscan

import android.widget.Toast
import androidx.camera.view.PreviewView
import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.runtime.produceState
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner

private val Ok = Color(0xFF4CD964)
private val Warn = Color(0xFFFFB020)

/** Page fully in view: all corners away from the frame border and the page reasonably large. */
private fun pageOk(q: FloatArray): Boolean {
    if ((0 until 8).any { q[it] < 0.015f || q[it] > 0.985f }) return false
    var a = 0f
    for (k in 0 until 4) {
        val j = (k + 1) % 4
        a += q[2 * k] * q[2 * j + 1] - q[2 * j] * q[2 * k + 1]
    }
    return kotlin.math.abs(a) / 2 > 0.2f
}

@Composable
fun CaptureScreen(model: ScanModel, onLibrary: () -> Unit, onPage: (Int) -> Unit, onReport: () -> Unit) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    val rig = remember { CameraRig(context) }
    var quad by remember { mutableStateOf<FloatArray?>(null) }
    var busy by remember { mutableStateOf(false) }
    var exporting by remember { mutableStateOf(false) }
    var settings by remember { mutableStateOf(false) }
    var flashUnit by remember { mutableStateOf(false) }
    if (settings) SettingsDialog(model, onReport) { settings = false }
    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
    }
    DisposableEffect(owner) {
        rig.onQuad = { q -> previewView.post { quad = q } }
        rig.onFlashAvailable = { a -> previewView.post { flashUnit = a } }
        rig.bind(owner, previewView)
        onDispose { rig.release() }
    }

    val pages = model.pages
    val pending = model.pending
    if (pending > 0) KeepScreenOn()
    LaunchedEffect(model.flashOn) { rig.setFlashMode(model.flashOn) }
    val q = quad
    val ok = q != null && pageOk(q)

    Column(Modifier.fillMaxSize().background(Color.Black)) {
        Row(Modifier.fillMaxWidth().padding(start = 16.dp, end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(
                if (pages.isEmpty()) "New scan" else "Scan · ${pages.size} page${if (pages.size == 1) "" else "s"}",
                color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.SemiBold,
                modifier = Modifier.weight(1f),
            )
            TextButton(onClick = onLibrary, enabled = !busy) { Text("Scans") }
        }
        // shrinks on short screens so the page strip and the buttons below stay visible
        Box(
            Modifier.weight(1f, fill = false).aspectRatio(3f / 4f, matchHeightConstraintsFirst = true)
                .align(Alignment.CenterHorizontally)
        ) {
            AndroidView({ previewView }, Modifier.fillMaxSize())
            Canvas(Modifier.fillMaxSize()) {
                if (q != null) {
                    val p = Path()
                    for (k in 0 until 4) {
                        val o = Offset(q[2 * k] * size.width, q[2 * k + 1] * size.height)
                        if (k == 0) p.moveTo(o.x, o.y) else p.lineTo(o.x, o.y)
                    }
                    p.close()
                    drawPath(p, if (ok) Ok else Warn, style = Stroke(width = 3.dp.toPx()))
                }
            }
            Text(
                when {
                    busy -> "Taking the photo …"
                    q == null -> "No page found – dark background helps"
                    !ok -> "Keep the whole page in view"
                    else -> "Page OK"
                },
                color = if (ok) Ok else Warn, fontSize = 14.sp,
                modifier = Modifier.align(Alignment.TopCenter).padding(horizontal = 64.dp, vertical = 12.dp)
                    .background(Color(0x99000000), RoundedCornerShape(6.dp)).padding(8.dp, 4.dp),
            )
            IconButton(
                onClick = { settings = true }, enabled = !busy,
                modifier = Modifier.align(Alignment.TopEnd).padding(8.dp).background(Color(0x99000000), CircleShape),
            ) {
                Icon(Icons.Filled.Settings, contentDescription = "Settings", tint = Color.White)
            }
            if (flashUnit) {
                IconButton(
                    onClick = { model.changeFlash(!model.flashOn) }, enabled = !busy,
                    modifier = Modifier.align(Alignment.TopStart).padding(8.dp).background(Color(0x99000000), CircleShape),
                ) {
                    FlashIcon(model.flashOn)
                }
            }
            Box(
                Modifier.align(Alignment.BottomCenter).padding(bottom = 20.dp)
                    .size(76.dp).clip(CircleShape)
                    .clickable(enabled = !busy) {
                        busy = true
                        val file = model.nextShotFile()
                        // no focus lock: every page is focused on its own
                        rig.shoot(file, lockAfter = false) { r ->
                            busy = false
                            r.onSuccess { sz ->
                                model.addShot(file, rig.focalPx(sz.width, sz.height))
                            }.onFailure {
                                Toast.makeText(context, "Photo failed: ${it.message}", Toast.LENGTH_LONG).show()
                            }
                        }
                    }
                    // the whole button reacts, including the outer ring
                    .border(4.dp, Color.White, CircleShape).padding(8.dp)
                    .clip(CircleShape).background(if (busy) Color.Gray else Color.White)
            )
        }
        model.notice?.let {
            Text(
                it, color = Warn, fontSize = 14.sp,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
            )
        }
        if (pages.isEmpty()) {
            Text(
                CAPTURE_HINT, color = Color(0xFFDDDDDD), fontSize = 15.sp,
                modifier = Modifier.padding(16.dp),
            )
        } else {
            val strip = rememberLazyListState()
            LaunchedEffect(pages.size) { strip.animateScrollToItem(pages.size - 1) }
            LazyRow(
                state = strip,
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                itemsIndexed(pages, key = { _, p -> p.id }) { i, p ->
                    PageThumb(p, i + 1) {
                        // viewer index counts finished pages only
                        onPage(pages.subList(0, i).count { it.file != null })
                    }
                }
            }
        }
        if (pages.isNotEmpty()) {
            val canExport = pending == 0 && !exporting
            fun export(action: suspend (android.content.Context, List<ScanItem>) -> Unit) {
                val scan = model.current() ?: return
                exporting = true
                scope.launch { action(context, listOf(scan)); exporting = false }
            }
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                OutlinedButton(onClick = { model.newScan() }, enabled = !busy) { Text("New scan") }
                OutlinedButton(onClick = { export(::saveScans) }, enabled = canExport) { Text("Save PDF") }
                Button(onClick = { export(::shareScans) }, enabled = canExport) { Text("Share PDF") }
            }
        }
    }
}

/** Page of the current scan in the strip; a spinner while it is processed. */
@Composable
private fun PageThumb(page: Page, number: Int, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            Modifier.height(112.dp).aspectRatio(210f / 297f).clip(RoundedCornerShape(6.dp))
                .background(Color(0xFF222222)).clickable(enabled = page.file != null, onClick = onClick),
            contentAlignment = Alignment.Center,
        ) {
            val f = page.file
            if (f == null) {
                CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp)
            } else {
                val bmp by produceState<Bitmap?>(null, f.path) {
                    value = withContext(Dispatchers.IO) { ScanStore.thumb(f, 400) }
                }
                bmp?.let { Image(it.asImageBitmap(), "page $number", Modifier.fillMaxSize(), contentScale = ContentScale.Fit) }
            }
        }
        Text("$number", color = Color.Gray, fontSize = 12.sp)
    }
}

/** Lightning bolt: yellow when the flash is on, grey with a red slash when off. */
@Composable
private fun FlashIcon(on: Boolean) {
    Canvas(Modifier.size(22.dp)) {
        val p = Path().apply {
            moveTo(size.width * 0.58f, 0f)
            lineTo(size.width * 0.10f, size.height * 0.58f)
            lineTo(size.width * 0.45f, size.height * 0.58f)
            lineTo(size.width * 0.36f, size.height)
            lineTo(size.width * 0.90f, size.height * 0.40f)
            lineTo(size.width * 0.52f, size.height * 0.40f)
            close()
        }
        drawPath(p, if (on) Color(0xFFFFD54A) else Color(0xFF999999))
        if (!on) {
            drawLine(
                Color(0xFFFF5252),
                Offset(size.width * 0.06f, size.height * 0.06f),
                Offset(size.width * 0.94f, size.height * 0.94f),
                strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round,
            )
        }
    }
}
