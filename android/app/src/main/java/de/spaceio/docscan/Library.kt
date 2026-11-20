package de.spaceio.docscan

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** All scans, newest first; select several to save/share/delete them together. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun LibraryScreen(onClose: () -> Unit) {
    val context = LocalContext.current
    var items by remember { mutableStateOf(ScanStore.list(context)) }
    var selecting by remember { mutableStateOf(false) }
    var selected by remember { mutableStateOf(setOf<File>()) }  // scan folders
    var viewing by remember { mutableStateOf<ScanItem?>(null) }
    var confirmDelete by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    fun reload() {
        items = ScanStore.list(context)
        selected = selected.filter { f -> items.any { it.dir == f } }.toSet()
    }

    val v = viewing
    if (v != null) {
        var confirm by remember { mutableStateOf(false) }
        PagesViewer(fmtDate(v.date), v.pages, 0, onBack = { viewing = null }) {
            OutlinedButton(onClick = { scope.launch { saveScans(context, listOf(v)) } }) { Text("Save PDF") }
            Button(onClick = { scope.launch { shareScans(context, listOf(v)) } }) { Text("Share PDF") }
            TextButton(onClick = { confirm = true }) { Text("Delete") }
        }
        if (confirm) ConfirmDelete(
            "Delete this scan?", "It is removed from the app. PDFs already saved to Download/Unfold stay.",
            onDismiss = { confirm = false },
        ) {
            v.dir.deleteRecursively()
            confirm = false
            viewing = null
            reload()
        }
        return
    }

    BackHandler {
        if (selecting) {
            selecting = false
            selected = emptySet()
        } else {
            onClose()
        }
    }

    Column(Modifier.fillMaxSize()) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextButton(onClick = {
                if (selecting) {
                    selecting = false
                    selected = emptySet()
                } else {
                    onClose()
                }
            }) { Text("Back") }
            Text(
                if (items.isEmpty()) "Scans" else "Scans (${items.size})",
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.padding(start = 4.dp),
            )
            Spacer(Modifier.weight(1f))
            if (items.isNotEmpty()) {
                if (selecting) TextButton(onClick = {
                    selected = if (selected.size == items.size) emptySet() else items.map { it.dir }.toSet()
                }) { Text(if (selected.size == items.size) "None" else "All") }
                TextButton(onClick = {
                    selecting = !selecting
                    if (!selecting) selected = emptySet()
                }) { Text(if (selecting) "Cancel" else "Select") }
            }
        }

        if (items.isEmpty()) {
            Column(
                Modifier.weight(1f).fillMaxWidth().padding(24.dp),
                verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text("No scans yet", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(8.dp))
                Text(
                    "Finished scans are kept here, so you can save or share them later.",
                    textAlign = TextAlign.Center, color = Color.Gray,
                )
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(12.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                items(items, key = { it.dir.path }) { item ->
                    ScanCard(item, selecting, item.dir in selected) {
                        if (selecting) {
                            selected = if (item.dir in selected) selected - item.dir
                            else selected + item.dir
                        } else {
                            viewing = item
                        }
                    }
                }
            }
        }

        if (selecting) {
            Row(
                Modifier.fillMaxWidth().padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("${selected.size} selected", Modifier.weight(1f))
                OutlinedButton(
                    enabled = selected.isNotEmpty(),
                    onClick = {
                        val scans = items.filter { it.dir in selected }
                        selecting = false
                        selected = emptySet()
                        scope.launch { saveScans(context, scans) }
                    },
                ) { Text("Save") }
                Spacer(Modifier.width(8.dp))
                OutlinedButton(
                    enabled = selected.isNotEmpty(),
                    onClick = {
                        val scans = items.filter { it.dir in selected }
                        selecting = false
                        selected = emptySet()
                        scope.launch { shareScans(context, scans) }
                    },
                ) { Text("Share") }
                Spacer(Modifier.width(8.dp))
                TextButton(enabled = selected.isNotEmpty(), onClick = { confirmDelete = true }) {
                    Text("Delete")
                }
            }
        }
    }

    if (confirmDelete) {
        val n = selected.size
        ConfirmDelete(
            "Delete $n scan${if (n == 1) "" else "s"}?",
            "They are removed from the app. PDFs already saved to Download/Unfold stay.",
            onDismiss = { confirmDelete = false },
        ) {
            selected.forEach { it.deleteRecursively() }
            confirmDelete = false
            selecting = false
            selected = emptySet()
            reload()
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ScanCard(item: ScanItem, selecting: Boolean, selected: Boolean, onClick: () -> Unit) {
    Column(Modifier.clickable(onClick = onClick)) {
        Box(
            Modifier.fillMaxWidth().aspectRatio(210f / 297f)
                .clip(RoundedCornerShape(12.dp)).background(Color(0xFF161616)),
        ) {
            val bmp by produceState<Bitmap?>(null, item.dir.path) {
                value = withContext(Dispatchers.IO) { ScanStore.thumb(item.pages[0]) }
            }
            val b = bmp
            if (b != null) {
                Image(
                    b.asImageBitmap(), "scan",
                    Modifier.fillMaxSize().padding(2.dp), contentScale = ContentScale.Fit,
                )
            } else {
                CircularProgressIndicator(
                    Modifier.align(Alignment.Center).size(22.dp), strokeWidth = 2.dp,
                )
            }
            if (selecting) {
                Checkbox(
                    checked = selected,
                    onCheckedChange = { onClick() },
                    modifier = Modifier.align(Alignment.TopEnd).padding(2.dp),
                )
            }
        }
        Text(
            fmtDate(item.date) + if (item.pages.size > 1) " · ${item.pages.size} pages" else "",
            style = MaterialTheme.typography.labelMedium,
            color = Color(0xFFB0B0B0),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(top = 4.dp, start = 2.dp),
        )
    }
}

/** Pages of a scan, one at a time with zoom; ‹ › step through them. [actions] get the page index. */
@Composable
internal fun PagesViewer(
    title: String, pages: List<File>, start: Int, onBack: () -> Unit,
    actions: @Composable RowScope.(Int) -> Unit,
) {
    var index by remember { mutableIntStateOf(start) }
    val i = index.coerceIn(0, pages.lastIndex)  // pages can be deleted while shown
    val file = pages[i]
    val bitmap by produceState<Bitmap?>(null, file) { value = withContext(Dispatchers.IO) { ScanStore.full(file) } }
    BackHandler(onBack = onBack)

    Column(Modifier.fillMaxSize()) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextButton(onClick = onBack) { Text("Back") }
            Text(
                title,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.weight(1f).padding(start = 4.dp),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            if (pages.size > 1) {
                TextButton(onClick = { index = i - 1 }, enabled = i > 0) { Text("‹") }
                Text("${i + 1} / ${pages.size}")
                TextButton(onClick = { index = i + 1 }, enabled = i < pages.lastIndex) { Text("›") }
            }
        }
        Box(Modifier.weight(1f).fillMaxWidth().background(Color.DarkGray).padding(8.dp)) {
            val b = bitmap
            if (b != null) {
                key(file) { ZoomablePage(b, Modifier.fillMaxSize()) }
            } else {
                CircularProgressIndicator(Modifier.align(Alignment.Center))
            }
        }
        Row(
            Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) { actions(i) }
    }
}

@Composable
internal fun ConfirmDelete(title: String, text: String, onDismiss: () -> Unit, onDelete: () -> Unit) =
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(text) },
        confirmButton = { TextButton(onClick = onDelete) { Text("Delete") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )

private val dateFmt = SimpleDateFormat("d MMM yyyy, HH:mm", Locale.getDefault())

private fun fmtDate(d: Date): String = dateFmt.format(d)
