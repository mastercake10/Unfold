package de.spaceio.docscan

import android.widget.Toast
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
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
fun CaptureScreen(model: ScanModel, onLibrary: () -> Unit) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val rig = remember { CameraRig(context) }
    var quad by remember { mutableStateOf<FloatArray?>(null) }
    var busy by remember { mutableStateOf(false) }
    var settings by remember { mutableStateOf(false) }
    var flashUnit by remember { mutableStateOf(false) }
    if (settings) SettingsDialog(model) { settings = false }
    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
    }
    DisposableEffect(owner) {
        rig.onQuad = { q -> previewView.post { quad = q } }
        rig.onFlashAvailable = { a -> previewView.post { flashUnit = a } }
        rig.bind(owner, previewView)
        onDispose { rig.release() }
    }

    val n = model.shots.size
    // first photo gone (rejected): measure focus again for the next one
    LaunchedEffect(n == 0) { if (n == 0) rig.unlockFocus() }
    LaunchedEffect(model.flashOn) { rig.setFlashMode(model.flashOn) }
    val q = quad
    val ok = q != null && pageOk(q)

    Column(Modifier.fillMaxSize().background(Color.Black)) {
        Row(Modifier.padding(16.dp).height(96.dp)) {
            Column(Modifier.weight(1f)) {
                Text(
                    "Page",
                    color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.SemiBold,
                )
                Spacer(Modifier.height(4.dp))
                Text(CAPTURE_HINT, color = Color(0xFFDDDDDD), fontSize = 15.sp)
            }
            TextButton(onClick = onLibrary, enabled = !busy) { Text("Scans") }
        }
        Box(Modifier.fillMaxWidth().aspectRatio(3f / 4f)) {
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
                    q == null -> "No page found – dark background helps"
                    !ok -> "Keep the whole page in view"
                    rig.focusLocked -> "Page OK · focus locked"
                    else -> "Page OK"
                },
                color = if (ok) Ok else Warn, fontSize = 14.sp,
                modifier = Modifier.align(Alignment.BottomCenter).padding(8.dp)
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
        }
        model.notice?.let {
            Text(
                it, color = Warn, fontSize = 14.sp,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
            )
        }
        if (busy) {
            Text(
                "focusing and taking the photo …", color = Color.Gray, fontSize = 13.sp,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        } else if (model.pending > 0) {
            Text(
                "analysing photo …", color = Color.Gray, fontSize = 13.sp,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }
        Spacer(Modifier.weight(1f))
        Row(
            Modifier.fillMaxWidth().padding(bottom = 24.dp),
            horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier.size(76.dp).clip(CircleShape)
                    .clickable(enabled = !busy && n == 0) {
                        busy = true
                        val file = model.nextShotFile()
                        rig.shoot(file, lockAfter = n == 0) { r ->
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
                    .clip(CircleShape).background(if (busy || n > 0) Color.Gray else Color.White)
            )
        }
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
