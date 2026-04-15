use std::fs::File;
use std::io::{BufRead, BufReader, Stdout, Write};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    self, disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen,
    LeaveAlternateScreen,
};

use crate::trie::{SparseTrieSnapshot, TrieNodeSnapshot, TrieTensorSnapshot, TrieTensorSource};
use crate::visualizer::{
    RuntimeEnvelope, RuntimeEvent, TrieMemoryDiagnostics, TrieReadBatch, TrieReadRecord,
    TrieWriteBatch, TrieWriteRecord,
};

const COMPACT_LEVELS: &[char] = &[' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const HEATMAP_LEVELS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

#[derive(Debug, Clone)]
struct TrieFrame {
    seq: u64,
    emitted_at_ms: u128,
    phase: String,
    step: Option<usize>,
    trie_entries: usize,
    writes: TrieWriteBatch,
    reads: TrieReadBatch,
    memory: Option<TrieMemoryDiagnostics>,
    snapshot: SparseTrieSnapshot,
}

impl TrieFrame {
    fn from_envelope(envelope: RuntimeEnvelope) -> Option<Self> {
        match envelope.event {
            RuntimeEvent::TrieUpdated {
                phase,
                step,
                trie_entries,
                writes,
                reads,
                memory,
                snapshot,
            } => Some(Self {
                seq: envelope.seq,
                emitted_at_ms: envelope.emitted_at_ms,
                phase,
                step,
                trie_entries,
                writes,
                reads,
                memory,
                snapshot,
            }),
            _ => None,
        }
    }
}

#[derive(Default)]
struct NodeOverlayStats {
    write_count: usize,
    read_count: usize,
    hit_count: usize,
    miss_count: usize,
    confidence_sum: f64,
}

impl NodeOverlayStats {
    fn avg_confidence(&self) -> Option<f64> {
        if self.read_count == 0 {
            None
        } else {
            Some(self.confidence_sum / self.read_count as f64)
        }
    }
}

#[derive(Debug)]
struct JsonlFrameSource {
    reader: BufReader<File>,
}

impl JsonlFrameSource {
    fn open(path: &str) -> Result<Self> {
        if path == "-" {
            bail!("visualize requires a file path so stdin remains available for keyboard input");
        }
        let file = File::open(path).with_context(|| format!("opening visualizer input {path}"))?;
        Ok(Self {
            reader: BufReader::new(file),
        })
    }

    fn read_available_frames(&mut self) -> Result<Vec<TrieFrame>> {
        let mut frames = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let envelope: RuntimeEnvelope = serde_json::from_str(trimmed)
                .with_context(|| format!("parsing visualizer event: {trimmed}"))?;
            if let Some(frame) = TrieFrame::from_envelope(envelope) {
                frames.push(frame);
            }
        }
        Ok(frames)
    }
}

struct TerminalSession {
    stdout: Stdout,
}

impl TerminalSession {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide)?;
        Ok(Self { stdout })
    }

    fn draw(&mut self, contents: &str) -> Result<()> {
        execute!(self.stdout, MoveTo(0, 0), Clear(ClearType::All))?;
        self.stdout.write_all(contents.as_bytes())?;
        self.stdout.flush()?;
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = execute!(self.stdout, Show, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

#[derive(Debug)]
struct VisualizerApp {
    input: String,
    follow: bool,
    current_frame: usize,
    selected_address: Option<Vec<u8>>,
    frames: Vec<TrieFrame>,
    source: JsonlFrameSource,
}

impl VisualizerApp {
    fn from_path(path: &str, follow: bool) -> Result<Self> {
        let mut source = JsonlFrameSource::open(path)?;
        let frames = source.read_available_frames()?;
        let mut app = Self {
            input: path.to_string(),
            follow,
            current_frame: frames.len().saturating_sub(1),
            selected_address: None,
            frames,
            source,
        };
        app.sync_selection();
        Ok(app)
    }

    fn refresh(&mut self) -> Result<()> {
        let new_frames = self.source.read_available_frames()?;
        if new_frames.is_empty() {
            return Ok(());
        }
        self.frames.extend(new_frames);
        if self.follow {
            self.current_frame = self.frames.len().saturating_sub(1);
        }
        self.sync_selection();
        Ok(())
    }

    fn current_frame(&self) -> Option<&TrieFrame> {
        self.frames.get(self.current_frame)
    }

    fn selected_index(&self) -> usize {
        self.current_frame()
            .and_then(|frame| {
                self.selected_address.as_ref().and_then(|address| {
                    frame
                        .snapshot
                        .nodes
                        .iter()
                        .position(|node| &node.address == address)
                })
            })
            .unwrap_or(0)
    }

    fn sync_selection(&mut self) {
        let next = self.current_frame().and_then(|frame| {
            if frame.snapshot.nodes.is_empty() {
                return None;
            }
            if let Some(address) = self.selected_address.as_ref() {
                if frame
                    .snapshot
                    .nodes
                    .iter()
                    .any(|node| &node.address == address)
                {
                    return Some(address.clone());
                }
            }
            frame.snapshot.focus.first().cloned().or_else(|| {
                frame
                    .snapshot
                    .nodes
                    .first()
                    .map(|node| node.address.clone())
            })
        });
        self.selected_address = next;
    }

    fn move_selection(&mut self, delta: isize) {
        let Some(frame) = self.current_frame() else {
            return;
        };
        if frame.snapshot.nodes.is_empty() {
            return;
        }
        let current = self.selected_index() as isize;
        let next = (current + delta).clamp(0, frame.snapshot.nodes.len() as isize - 1) as usize;
        self.selected_address = Some(frame.snapshot.nodes[next].address.clone());
    }

    fn move_frame(&mut self, delta: isize) {
        if self.frames.is_empty() {
            return;
        }
        self.follow = false;
        let next =
            (self.current_frame as isize + delta).clamp(0, self.frames.len() as isize - 1) as usize;
        self.current_frame = next;
        self.sync_selection();
    }

    fn jump_to(&mut self, index: usize) {
        if self.frames.is_empty() {
            return;
        }
        self.follow = false;
        self.current_frame = index.min(self.frames.len() - 1);
        self.sync_selection();
    }

    fn toggle_follow(&mut self) {
        self.follow = !self.follow;
        if self.follow && !self.frames.is_empty() {
            self.current_frame = self.frames.len() - 1;
            self.sync_selection();
        }
    }
}

pub fn run(input: &str, follow: bool) -> Result<()> {
    let mut app = VisualizerApp::from_path(input, follow)?;
    if app.frames.is_empty() && !follow {
        bail!("no trie_updated events found in {input}");
    }

    let mut terminal = TerminalSession::enter()?;
    loop {
        app.refresh()?;
        let (width, height) = terminal::size()?;
        terminal.draw(&render_screen(&app, width as usize, height as usize))?;

        if event::poll(Duration::from_millis(120))? {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if handle_key(&mut app, key) {
                break;
            }
        }
    }

    Ok(())
}

fn handle_key(app: &mut VisualizerApp, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Up | KeyCode::Char('k') => {
            app.move_selection(-1);
            false
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.move_selection(1);
            false
        }
        KeyCode::Left | KeyCode::Char('h') => {
            app.move_frame(-1);
            false
        }
        KeyCode::Right | KeyCode::Char('l') => {
            app.move_frame(1);
            false
        }
        KeyCode::Home | KeyCode::Char('g') => {
            app.jump_to(0);
            false
        }
        KeyCode::End | KeyCode::Char('G') => {
            if !app.frames.is_empty() {
                app.jump_to(app.frames.len() - 1);
            }
            false
        }
        KeyCode::Char('f') => {
            app.toggle_follow();
            false
        }
        _ => false,
    }
}

fn render_screen(app: &VisualizerApp, width: usize, height: usize) -> String {
    if width < 48 || height < 12 {
        return render_too_small(width, height);
    }

    let mut lines = Vec::new();
    lines.push(fit_line(
        &format!(
            "Zeno Trie Visualizer  source={}  frames={}  mode={}",
            app.input,
            app.frames.len(),
            if app.follow { "follow" } else { "inspect" }
        ),
        width,
    ));

    match app.current_frame() {
        Some(frame) => {
            lines.push(fit_line(&render_frame_summary(app, frame), width));
            let body_height = height.saturating_sub(4);
            let left_width = (width * 58 / 100).clamp(28, width.saturating_sub(20));
            let right_width = width.saturating_sub(left_width + 3);
            let tree_lines = render_tree_pane(app, frame, left_width, body_height);
            let inspector_lines = render_inspector_pane(app, frame, right_width, body_height);
            for row in 0..body_height {
                let left = tree_lines.get(row).map(String::as_str).unwrap_or("");
                let right = inspector_lines.get(row).map(String::as_str).unwrap_or("");
                lines.push(format!(
                    "{} │ {}",
                    fit_line(left, left_width),
                    fit_line(right, right_width)
                ));
            }
        }
        None => {
            lines.push(fit_line("Waiting for trie_updated events...", width));
            for _ in 0..height.saturating_sub(4) {
                lines.push(" ".repeat(width));
            }
        }
    }

    lines.push(fit_line(
        "keys: ↑↓/j k select node  ←→/h l frame  g/G first/last  f follow  q quit",
        width,
    ));
    lines.join("\n")
}

fn render_too_small(width: usize, height: usize) -> String {
    let mut lines = Vec::new();
    lines.push(fit_line("terminal too small for trie visualizer", width));
    lines.push(fit_line("resize to at least 48x12", width));
    while lines.len() < height {
        lines.push(" ".repeat(width));
    }
    lines.join("\n")
}

fn render_frame_summary(app: &VisualizerApp, frame: &TrieFrame) -> String {
    let memory = frame
        .memory
        .as_ref()
        .map(|diag| {
            format!(
                "  read h/m={}/{} conf={:.2}",
                diag.read_hits, diag.read_misses, diag.avg_confidence
            )
        })
        .unwrap_or_default();
    format!(
        "frame {}/{}  seq={}  phase={}  step={}  entries={}  writes={}  active_reads={}  nodes={}  focus={}{}",
        app.current_frame + 1,
        app.frames.len(),
        frame.seq,
        frame.phase,
        frame
            .step
            .map(|step| step.to_string())
            .unwrap_or_else(|| "-".to_string()),
        frame.trie_entries,
        frame.writes.len(),
        frame.reads.len(),
        frame.snapshot.nodes.len(),
        frame.snapshot.focus.len(),
        memory,
    )
}

fn node_overlay_stats(frame: &TrieFrame, address: &[u8]) -> NodeOverlayStats {
    let mut stats = NodeOverlayStats::default();
    for write in &frame.writes.writes {
        if write.address == address {
            stats.write_count += 1;
        }
    }
    for read in &frame.reads.reads {
        if read.address == address {
            stats.read_count += 1;
            if read.hit {
                stats.hit_count += 1;
            } else {
                stats.miss_count += 1;
            }
            stats.confidence_sum += read.confidence;
        }
    }
    stats
}

fn matching_writes<'a>(frame: &'a TrieFrame, address: &[u8]) -> Vec<&'a TrieWriteRecord> {
    frame
        .writes
        .writes
        .iter()
        .filter(|write| write.address == address)
        .collect()
}

fn matching_reads<'a>(frame: &'a TrieFrame, address: &[u8]) -> Vec<&'a TrieReadRecord> {
    frame
        .reads
        .reads
        .iter()
        .filter(|read| read.address == address)
        .collect()
}

fn render_tree_pane(
    app: &VisualizerApp,
    frame: &TrieFrame,
    width: usize,
    height: usize,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("Trie nodes ({})", frame.snapshot.nodes.len()));
    if frame.snapshot.nodes.is_empty() {
        lines.push("  no populated nodes in this frame".to_string());
        while lines.len() < height {
            lines.push(String::new());
        }
        return lines;
    }

    let selected_index = app.selected_index();
    let available_rows = height.saturating_sub(1);
    let mut start = selected_index.saturating_sub(available_rows / 2);
    let end = (start + available_rows).min(frame.snapshot.nodes.len());
    start = end.saturating_sub(available_rows);

    for (index, node) in frame
        .snapshot
        .nodes
        .iter()
        .enumerate()
        .skip(start)
        .take(available_rows)
    {
        let overlay = node_overlay_stats(frame, &node.address);
        lines.push(render_tree_line(
            node,
            frame
                .snapshot
                .focus
                .iter()
                .any(|focus| focus == &node.address),
            &overlay,
            index == selected_index,
            width,
        ));
    }

    while lines.len() < height {
        lines.push(String::new());
    }
    lines
}

fn render_tree_line(
    node: &TrieNodeSnapshot,
    focused: bool,
    overlay: &NodeOverlayStats,
    selected: bool,
    width: usize,
) -> String {
    let marker = if selected { '>' } else { ' ' };
    let focus = if focused { '*' } else { ' ' };
    let indent = "  ".repeat(node.depth);
    let label = node
        .address
        .last()
        .map(|branch| format!("{branch:02x}"))
        .unwrap_or_else(|| "root".to_string());
    let flags = format!(
        "{}{}{}",
        if node.has_value { 'V' } else { '-' },
        if node.has_summary { 'S' } else { '-' },
        if node.has_residual { 'R' } else { '-' }
    );
    let glyph = node
        .render
        .as_ref()
        .map(compact_glyph)
        .unwrap_or_else(|| "........".to_string());
    fit_line(
        &format!(
            "{marker}{focus} {indent}{label:<4} {flags} ch:{:<2} sv:{:<3} r:{}/{} w:{} {glyph}",
            node.populated_children,
            node.subtree_values,
            overlay.hit_count,
            overlay.miss_count,
            overlay.write_count,
        ),
        width,
    )
}

fn render_inspector_pane(
    app: &VisualizerApp,
    frame: &TrieFrame,
    width: usize,
    height: usize,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("Inspector".to_string());

    let Some(node) = frame.snapshot.nodes.get(app.selected_index()) else {
        while lines.len() < height {
            lines.push(String::new());
        }
        return lines;
    };

    let selected_writes = matching_writes(frame, &node.address);
    let selected_reads = matching_reads(frame, &node.address);
    let overlay = node_overlay_stats(frame, &node.address);
    let avg_confidence = overlay
        .avg_confidence()
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "n/a".to_string());
    let min_confidence = selected_reads
        .iter()
        .map(|read| read.confidence)
        .reduce(f64::min)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "n/a".to_string());
    let max_confidence = selected_reads
        .iter()
        .map(|read| read.confidence)
        .reduce(f64::max)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "n/a".to_string());

    lines.push(format!("path: {}", format_address(&node.address)));
    lines.push(format!(
        "depth={} density={} children={} subtree_values={}",
        node.depth, node.structural_density, node.populated_children, node.subtree_values
    ));
    lines.push(format!(
        "flags: value={} summary={} residual={}",
        yes_no(node.has_value),
        yes_no(node.has_summary),
        yes_no(node.has_residual)
    ));
    lines.push(format!(
        "overlay: writes={} reads={} hit={} miss={} focus={}",
        selected_writes.len(),
        selected_reads.len(),
        overlay.hit_count,
        overlay.miss_count,
        yes_no(
            frame
                .snapshot
                .focus
                .iter()
                .any(|focus| focus == &node.address)
        )
    ));
    lines.push(format!(
        "selected_conf: avg={} range=[{},{}]",
        avg_confidence, min_confidence, max_confidence
    ));
    lines.push(format!(
        "children: {}",
        if node.children.is_empty() {
            "none".to_string()
        } else {
            node.children
                .iter()
                .map(|child| format!("{child:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        }
    ));
    if let Some(memory) = frame.memory.as_ref() {
        lines.push(format!(
            "memory: hit={}/{} overlap={}/{} conf={:.2}[{:.2},{:.2}]",
            memory.read_hits,
            memory.read_hits + memory.read_misses,
            memory.overlap_pairs,
            memory.total_pairs,
            memory.avg_confidence,
            memory.min_confidence,
            memory.max_confidence,
        ));
        lines.push(format!(
            "density: mean={:.2} final={:.2} corr={:.2}",
            memory.avg_mean_density, memory.avg_final_density, memory.density_confidence_corr
        ));
    }

    lines.push("write heads:".to_string());
    if frame.writes.writes.is_empty() {
        lines.push("  none".to_string());
    } else {
        for write in &frame.writes.writes {
            lines.push(format_write_head(write, &node.address));
        }
    }

    lines.push("read heads:".to_string());
    if frame.reads.reads.is_empty() {
        lines.push("  none".to_string());
    } else {
        for read in &frame.reads.reads {
            lines.push(format_read_head(read, &node.address));
        }
    }

    match node.render.as_ref() {
        Some(render) => {
            lines.push(format!(
                "tensor: {}  mean={:+.3}  min={:+.3}  max={:+.3}  l2={:.3}",
                tensor_source_label(render.source),
                render.stats.mean,
                render.stats.min,
                render.stats.max,
                render.stats.l2_norm,
            ));
            lines.push("heatmap (12x8):".to_string());
            lines.extend(render_heatmap(render));
            lines.push("top-k:".to_string());
            for chunk in render.top_channels.chunks(2) {
                let row = chunk
                    .iter()
                    .map(|entry| format!("#{:02} {:+7.3}", entry.index, entry.value))
                    .collect::<Vec<_>>()
                    .join("  ");
                lines.push(row);
            }
        }
        None => lines.push("tensor: unavailable for this node".to_string()),
    }

    if let Some(frame_time) = frame_time_label(frame.emitted_at_ms) {
        lines.push(frame_time);
    }

    while lines.len() < height {
        lines.push(String::new());
    }
    lines.truncate(height);
    for line in &mut lines {
        *line = fit_line(line, width);
    }
    lines
}

fn format_write_head(write: &TrieWriteRecord, selected: &[u8]) -> String {
    format!(
        "{}w{} b{} {} str={:.2}",
        if write.address == selected { '*' } else { ' ' },
        write.slot,
        write.batch_index,
        format_address(&write.address),
        write.strength,
    )
}

fn format_read_head(read: &TrieReadRecord, selected: &[u8]) -> String {
    let densities = if read.density_chain.is_empty() {
        "-".to_string()
    } else {
        read.density_chain
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join("/")
    };
    format!(
        "{}r{} b{} t{} {} {} conf={:.2} d={}",
        if read.address == selected { '*' } else { ' ' },
        read.slot,
        read.batch_index,
        read.token_index,
        format_address(&read.address),
        if read.hit { "hit " } else { "miss" },
        read.confidence,
        densities,
    )
}

fn compact_glyph(render: &TrieTensorSnapshot) -> String {
    if render.values.is_empty() {
        return "........".to_string();
    }
    let max_abs = render
        .values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let buckets = 8;
    let chunk_size = render.values.len().div_ceil(buckets);
    (0..buckets)
        .map(|bucket| {
            let start = bucket * chunk_size;
            let end = ((bucket + 1) * chunk_size).min(render.values.len());
            if start >= end {
                return ' ';
            }
            let avg = render.values[start..end]
                .iter()
                .map(|value| value.abs())
                .sum::<f32>()
                / (end - start) as f32;
            let index = ((avg / max_abs) * (COMPACT_LEVELS.len() - 1) as f32).round() as usize;
            COMPACT_LEVELS[index.min(COMPACT_LEVELS.len() - 1)]
        })
        .collect()
}

fn render_heatmap(render: &TrieTensorSnapshot) -> Vec<String> {
    let cols = 12;
    let rows = 8;
    let max_abs = render
        .values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let mut lines = Vec::new();
    for row in 0..rows {
        let mut line = String::with_capacity(cols);
        for col in 0..cols {
            let index = row * cols + col;
            let value = render.values.get(index).copied().unwrap_or_default().abs();
            let level = ((value / max_abs) * (HEATMAP_LEVELS.len() - 1) as f32).round() as usize;
            line.push(HEATMAP_LEVELS[level.min(HEATMAP_LEVELS.len() - 1)]);
        }
        lines.push(line);
    }
    lines
}

fn fit_line(text: &str, width: usize) -> String {
    let mut fitted = text.chars().take(width).collect::<String>();
    let len = fitted.chars().count();
    if len < width {
        fitted.push_str(&" ".repeat(width - len));
    }
    fitted
}

fn format_address(address: &[u8]) -> String {
    if address.is_empty() {
        "root".to_string()
    } else {
        address
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join("/")
    }
}

fn tensor_source_label(source: TrieTensorSource) -> &'static str {
    match source {
        TrieTensorSource::Value => "value",
        TrieTensorSource::Summary => "summary",
        TrieTensorSource::Residual => "residual",
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn frame_time_label(emitted_at_ms: u128) -> Option<String> {
    if emitted_at_ms == 0 {
        None
    } else {
        Some(format!("emitted_at_ms: {emitted_at_ms}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trie::{TrieEdgeSnapshot, TrieTensorStats, TrieTopChannel};

    fn render_payload() -> TrieTensorSnapshot {
        TrieTensorSnapshot {
            source: TrieTensorSource::Summary,
            shape: [12, 8],
            values: (0..96).map(|index| index as f32).collect(),
            stats: TrieTensorStats {
                min: 0.0,
                max: 95.0,
                mean: 47.5,
                l2_norm: 1.0,
            },
            top_channels: vec![
                TrieTopChannel {
                    index: 95,
                    value: 95.0,
                },
                TrieTopChannel {
                    index: 94,
                    value: 94.0,
                },
            ],
        }
    }

    fn frame() -> TrieFrame {
        TrieFrame {
            seq: 9,
            emitted_at_ms: 42,
            phase: "phase4".to_string(),
            step: Some(7),
            trie_entries: 11,
            writes: TrieWriteBatch {
                writes: vec![
                    TrieWriteRecord {
                        slot: 0,
                        batch_index: 0,
                        address: vec![1, 2],
                        strength: 0.8,
                    },
                    TrieWriteRecord {
                        slot: 1,
                        batch_index: 0,
                        address: vec![1],
                        strength: 0.5,
                    },
                ],
            },
            reads: TrieReadBatch {
                reads: vec![
                    TrieReadRecord {
                        slot: 0,
                        batch_index: 0,
                        token_index: 7,
                        address: vec![1, 2],
                        density_chain: vec![1, 2],
                        mean_density: 1.5,
                        final_density: 2,
                        confidence: 0.9,
                        hit: true,
                    },
                    TrieReadRecord {
                        slot: 1,
                        batch_index: 0,
                        token_index: 7,
                        address: vec![1, 2],
                        density_chain: vec![1, 0],
                        mean_density: 0.5,
                        final_density: 0,
                        confidence: 0.2,
                        hit: false,
                    },
                ],
            },
            memory: Some(TrieMemoryDiagnostics {
                read_hits: 5,
                read_misses: 1,
                overlap_pairs: 1,
                total_pairs: 3,
                avg_mean_density: 1.2,
                avg_final_density: 0.8,
                avg_confidence: 0.7,
                min_confidence: 0.2,
                max_confidence: 0.9,
                avg_hit_confidence: 0.8,
                avg_miss_confidence: 0.2,
                density_confidence_corr: 0.3,
            }),
            snapshot: SparseTrieSnapshot {
                focus: vec![vec![1, 2]],
                nodes: vec![
                    TrieNodeSnapshot {
                        address: vec![],
                        depth: 0,
                        has_value: false,
                        has_summary: true,
                        has_residual: false,
                        populated_children: 1,
                        structural_density: 1,
                        subtree_values: 2,
                        children: vec![1],
                        render: Some(render_payload()),
                    },
                    TrieNodeSnapshot {
                        address: vec![1],
                        depth: 1,
                        has_value: false,
                        has_summary: true,
                        has_residual: true,
                        populated_children: 1,
                        structural_density: 1,
                        subtree_values: 2,
                        children: vec![2],
                        render: Some(render_payload()),
                    },
                    TrieNodeSnapshot {
                        address: vec![1, 2],
                        depth: 2,
                        has_value: false,
                        has_summary: true,
                        has_residual: true,
                        populated_children: 2,
                        structural_density: 2,
                        subtree_values: 2,
                        children: vec![3, 4],
                        render: Some(render_payload()),
                    },
                ],
                edges: vec![
                    TrieEdgeSnapshot {
                        parent: vec![],
                        child: vec![1],
                        branch: 1,
                    },
                    TrieEdgeSnapshot {
                        parent: vec![1],
                        child: vec![1, 2],
                        branch: 2,
                    },
                ],
            },
        }
    }

    #[test]
    fn compact_glyph_uses_eight_buckets() {
        let glyph = compact_glyph(&render_payload());
        assert_eq!(glyph.chars().count(), 8);
        assert!(glyph.contains('█'));
    }

    #[test]
    fn screen_render_includes_tree_and_inspector() {
        let trie_frame = frame();
        let app = VisualizerApp {
            input: "events.jsonl".to_string(),
            follow: false,
            current_frame: 0,
            selected_address: Some(vec![1, 2]),
            frames: vec![trie_frame.clone()],
            source: JsonlFrameSource {
                reader: BufReader::new(File::open("Cargo.toml").unwrap()),
            },
        };

        let screen = render_screen(&app, 100, 36);
        assert!(screen.contains("Trie nodes (3)"));
        assert!(screen.contains("path: 01/02"));
        assert!(screen.contains("overlay: writes=1 reads=2 hit=1 miss=1"));
        assert!(screen.contains("write heads:"));
        assert!(screen.contains("read heads:"));
        assert!(screen.contains("heatmap (12x8):"));
        assert!(screen.contains("top-k:"));
        assert!(screen.contains("phase=phase4"));
    }
}
