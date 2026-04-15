use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::config::ZenoConfig;
use crate::trie::SparseTrieSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrieWriteRecord {
    pub slot: usize,
    pub batch_index: usize,
    pub address: Vec<u8>,
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrieWriteBatch {
    pub writes: Vec<TrieWriteRecord>,
}

impl TrieWriteBatch {
    pub fn len(&self) -> usize {
        self.writes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }

    pub fn focus_addresses(&self) -> Vec<Vec<u8>> {
        self.writes
            .iter()
            .map(|write| write.address.clone())
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrieReadRecord {
    pub slot: usize,
    pub batch_index: usize,
    pub token_index: usize,
    pub address: Vec<u8>,
    pub density_chain: Vec<usize>,
    pub mean_density: f64,
    pub final_density: usize,
    pub confidence: f64,
    pub hit: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TrieReadBatch {
    pub reads: Vec<TrieReadRecord>,
}

impl TrieReadBatch {
    pub fn len(&self) -> usize {
        self.reads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
    }

    pub fn focus_addresses(&self) -> Vec<Vec<u8>> {
        self.reads.iter().map(|read| read.address.clone()).collect()
    }

    pub fn hit_count(&self) -> usize {
        self.reads.iter().filter(|read| read.hit).count()
    }

    pub fn miss_count(&self) -> usize {
        self.reads.len().saturating_sub(self.hit_count())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrieMemoryDiagnostics {
    pub read_hits: usize,
    pub read_misses: usize,
    pub overlap_pairs: usize,
    pub total_pairs: usize,
    pub avg_mean_density: f64,
    pub avg_final_density: f64,
    pub avg_confidence: f64,
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub avg_hit_confidence: f64,
    pub avg_miss_confidence: f64,
    pub density_confidence_corr: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeSession {
    pub command: String,
    pub phase: Option<u32>,
    pub use_memory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RuntimeEvent {
    SessionStarted {
        session: RuntimeSession,
        trie_depth: usize,
        d_model: usize,
    },
    PhaseStarted {
        phase: String,
    },
    StepMetrics {
        phase: String,
        step: usize,
        total_steps: usize,
        loss: f64,
        perplexity: Option<f64>,
        trie_entries: Option<usize>,
    },
    TrieUpdated {
        phase: String,
        step: Option<usize>,
        trie_entries: usize,
        writes: TrieWriteBatch,
        reads: TrieReadBatch,
        memory: Option<TrieMemoryDiagnostics>,
        snapshot: SparseTrieSnapshot,
    },
    ReencounterSummary {
        phase: String,
        step: Option<usize>,
        pass1_loss: f64,
        pass2_loss: f64,
        improvement: f64,
    },
    EvaluationSummary {
        use_memory: bool,
        loss: f64,
        perplexity: f64,
    },
    PhaseCompleted {
        phase: String,
        best_metric: f64,
        metric_name: String,
    },
    SessionCompleted {
        command: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeEnvelope {
    pub seq: u64,
    pub emitted_at_ms: u128,
    #[serde(flatten)]
    pub event: RuntimeEvent,
}

#[derive(Clone)]
pub struct TrieVisualizerRuntime {
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    sequence: Arc<AtomicU64>,
}

impl TrieVisualizerRuntime {
    pub fn create(target: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating visualizer output dir {}", parent.display()))?;
        }
        let file = File::create(target)
            .with_context(|| format!("creating visualizer output {}", target.display()))?;
        Ok(Self::from_writer(Box::new(BufWriter::new(file))))
    }

    pub fn stdout() -> Self {
        Self::from_writer(Box::new(BufWriter::new(io::stdout())))
    }

    pub fn from_writer(writer: Box<dyn Write + Send>) -> Self {
        Self {
            writer: Arc::new(Mutex::new(writer)),
            sequence: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn emit_session_started(
        &self,
        command: &str,
        phase: Option<u32>,
        use_memory: bool,
        cfg: &ZenoConfig,
    ) -> anyhow::Result<()> {
        self.emit(RuntimeEvent::SessionStarted {
            session: RuntimeSession {
                command: command.to_string(),
                phase,
                use_memory,
            },
            trie_depth: cfg.trie_depth,
            d_model: cfg.d_model,
        })
    }

    pub fn emit_phase_started(&self, phase: impl Into<String>) -> anyhow::Result<()> {
        self.emit(RuntimeEvent::PhaseStarted {
            phase: phase.into(),
        })
    }

    pub fn emit_step_metrics(
        &self,
        phase: impl Into<String>,
        step: usize,
        total_steps: usize,
        loss: f64,
        perplexity: Option<f64>,
        trie_entries: Option<usize>,
    ) -> anyhow::Result<()> {
        self.emit(RuntimeEvent::StepMetrics {
            phase: phase.into(),
            step,
            total_steps,
            loss,
            perplexity,
            trie_entries,
        })
    }

    pub fn emit_trie_updated(
        &self,
        phase: impl Into<String>,
        step: Option<usize>,
        trie_entries: usize,
        writes: TrieWriteBatch,
        reads: TrieReadBatch,
        memory: Option<TrieMemoryDiagnostics>,
        snapshot: SparseTrieSnapshot,
    ) -> anyhow::Result<()> {
        if writes.is_empty() {
            return Ok(());
        }
        self.emit(RuntimeEvent::TrieUpdated {
            phase: phase.into(),
            step,
            trie_entries,
            writes,
            reads,
            memory,
            snapshot,
        })
    }

    pub fn emit_reencounter_summary(
        &self,
        phase: impl Into<String>,
        step: Option<usize>,
        pass1_loss: f64,
        pass2_loss: f64,
        improvement: f64,
    ) -> anyhow::Result<()> {
        self.emit(RuntimeEvent::ReencounterSummary {
            phase: phase.into(),
            step,
            pass1_loss,
            pass2_loss,
            improvement,
        })
    }

    pub fn emit_evaluation_summary(
        &self,
        use_memory: bool,
        loss: f64,
        perplexity: f64,
    ) -> anyhow::Result<()> {
        self.emit(RuntimeEvent::EvaluationSummary {
            use_memory,
            loss,
            perplexity,
        })
    }

    pub fn emit_phase_completed(
        &self,
        phase: impl Into<String>,
        metric_name: impl Into<String>,
        best_metric: f64,
    ) -> anyhow::Result<()> {
        self.emit(RuntimeEvent::PhaseCompleted {
            phase: phase.into(),
            metric_name: metric_name.into(),
            best_metric,
        })
    }

    pub fn emit_session_completed(&self, command: &str) -> anyhow::Result<()> {
        self.emit(RuntimeEvent::SessionCompleted {
            command: command.to_string(),
        })
    }

    pub fn emit(&self, event: RuntimeEvent) -> anyhow::Result<()> {
        let envelope = RuntimeEnvelope {
            seq: self.sequence.fetch_add(1, Ordering::Relaxed) + 1,
            emitted_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            event,
        };
        let mut writer = self.writer.lock().unwrap();
        serde_json::to_writer(&mut *writer, &envelope)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trie::{SparseTrieSnapshot, TrieEdgeSnapshot, TrieNodeSnapshot};

    #[derive(Clone)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn runtime_emits_jsonl_events() {
        let shared = Arc::new(Mutex::new(Vec::new()));
        let runtime = TrieVisualizerRuntime::from_writer(Box::new(SharedBuffer(shared.clone())));
        let snapshot = SparseTrieSnapshot {
            focus: vec![vec![1, 2]],
            nodes: vec![TrieNodeSnapshot {
                address: vec![1, 2],
                depth: 2,
                has_value: true,
                has_summary: true,
                has_residual: false,
                populated_children: 0,
                structural_density: 1,
                subtree_values: 1,
                children: Vec::new(),
                render: None,
            }],
            edges: vec![TrieEdgeSnapshot {
                parent: vec![1],
                child: vec![1, 2],
                branch: 2,
            }],
        };

        runtime
            .emit_trie_updated(
                "phase4",
                Some(7),
                11,
                TrieWriteBatch {
                    writes: vec![TrieWriteRecord {
                        slot: 0,
                        batch_index: 0,
                        address: vec![1, 2],
                        strength: 0.75,
                    }],
                },
                TrieReadBatch {
                    reads: vec![TrieReadRecord {
                        slot: 1,
                        batch_index: 0,
                        token_index: 15,
                        address: vec![1, 2],
                        density_chain: vec![1, 1],
                        mean_density: 1.0,
                        final_density: 1,
                        confidence: 0.9,
                        hit: true,
                    }],
                },
                Some(TrieMemoryDiagnostics {
                    read_hits: 1,
                    read_misses: 0,
                    overlap_pairs: 0,
                    total_pairs: 0,
                    avg_mean_density: 1.0,
                    avg_final_density: 1.0,
                    avg_confidence: 0.9,
                    min_confidence: 0.9,
                    max_confidence: 0.9,
                    avg_hit_confidence: 0.9,
                    avg_miss_confidence: 0.0,
                    density_confidence_corr: 0.0,
                }),
                snapshot,
            )
            .unwrap();

        let text = String::from_utf8(shared.lock().unwrap().clone()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1);

        let parsed: RuntimeEnvelope = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.seq, 1);
        match parsed.event {
            RuntimeEvent::TrieUpdated {
                phase,
                step,
                trie_entries,
                writes,
                reads,
                memory,
                ..
            } => {
                assert_eq!(phase, "phase4");
                assert_eq!(step, Some(7));
                assert_eq!(trie_entries, 11);
                assert_eq!(writes.len(), 1);
                assert_eq!(reads.len(), 1);
                assert_eq!(memory.unwrap().read_hits, 1);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
