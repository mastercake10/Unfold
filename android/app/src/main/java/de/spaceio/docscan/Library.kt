package de.spaceio.docscan

import android.graphics.Bitmap
import android.widget.Toast
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

/** All finished scans, newest first; select several to save/share them together. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun LibraryScreen(onClose: () -> Unit) {
    val context = LocalContext.current
    var items by remember { mutableStateOf(ScanStore.list(context)) }
    var selecting by remember { mutableStateOf(false) }
    var selected by remember { mutableStateOf(setOf<File>()) }
    var viewing by remember { mutableStateOf<ScanItem?>(null) }
    var confirmDelete by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    fun reload() {
        items = ScanStore.list(context)
        selected = selected.filter { f -> items.any { it.file == f } }.toSet()
    }

    val v = viewing
    if (v != null) {
        ScanViewer(
            v,
            onBack = { viewing = null },
            onDeleted = { viewing = null; reload() },
        )
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
                    selected = if (selected.size == items.size) emptySet() else items.map { it.file }.toSet()
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
                    "Finished pages are kept here, so you can save or share them later.",
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
                items(items, key = { it.file.path }) { item ->
                    ScanCard(item, selecting, item.file in selected) {
                        if (selecting) {
                            selected = if (item.file in selected) selected - item.file
                            else selected + item.file
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
                        val files = selected.toList()
                        scope.launch {
                            val n = withContext(Dispatchers.IO) { saveAll(context, files) }
                            Toast.makeText(
                                context,
                                if (n == files.size) "Saved $n page${if (n == 1) "" else "s"} to Pictures/DocScan"
                                else "Saved $n of ${files.size} pages",
                                Toast.LENGTH_SHORT,
                            ).show()
                            selecting = false
                            selected = emptySet()
                        }
                    },
                ) { Text("Save") }
                Spacer(Modifier.width(8.dp))
                OutlinedButton(
                    enabled = selected.isNotEmpty(),
                    onClick = {
                        shareMany(context, selected.toList())
                        selecting = false
                        selected = emptySet()
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
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete ${n} page${if (n == 1) "" else "s"}?") },
            text = { Text("They are removed from the app. Copies already saved to Pictures/DocScan stay.") },
            confirmButton = {
                TextButton(onClick = {
                    selected.forEach { it.delete() }
                    confirmDelete = false
                    selecting = false
                    selected = emptySet()
                    reload()
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } },
        )
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
            val bmp by produceState<Bitmap?>(null, item.file.path) {
                value = withContext(Dispatchers.IO) { ScanStore.thumb(item.file) }
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
            fmtDate(item.date),
            style = MaterialTheme.typography.labelMedium,
            color = Color(0xFFB0B0B0),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(top = 4.dp, start = 2.dp),
        )
    }
}

@Composable
private fun ScanViewer(item: ScanItem, onBack: () -> Unit, onDeleted: () -> Unit) {
    val context = LocalContext.current
    var bitmap by remember(item.file) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(item.file) {
        bitmap = withContext(Dispatchers.IO) { ScanStore.full(item.file) }
    }
    var confirm by remember { mutableStateOf(false) }
    BackHandler(onBack = onBack)

    Column(Modifier.fillMaxSize()) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextButton(onClick = onBack) { Text("Back") }
            Text(
                fmtDate(item.date),
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.weight(1f).padding(start = 4.dp),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        Box(Modifier.weight(1f).fillMaxWidth().background(Color.DarkGray).padding(8.dp)) {
            val b = bitmap
            if (b != null) {
                ZoomablePage(b, Modifier.fillMaxSize())
            } else {
                CircularProgressIndicator(Modifier.align(Alignment.Center))
            }
        }
        Row(
            Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) {
            OutlinedButton(onClick = { save(context, item.file) }) { Text("Save") }
            Button(onClick = { share(context, item.file) }) { Text("Share") }
            TextButton(onClick = { confirm = true }) { Text("Delete") }
        }
    }

    if (confirm) {
        AlertDialog(
            onDismissRequest = { confirm = false },
            title = { Text("Delete this page?") },
            text = { Text("It is removed from the app. A copy already saved to Pictures/DocScan stays.") },
            confirmButton = {
                TextButton(onClick = {
                    item.file.delete()
                    confirm = false
                    onDeleted()
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { confirm = false }) { Text("Cancel") } },
        )
    }
}

private val dateFmt = SimpleDateFormat("d MMM yyyy, HH:mm", Locale.getDefault())

private fun fmtDate(d: Date): String = dateFmt.format(d)
