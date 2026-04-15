use candle_core::{backprop::GradStore, DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::config::ZenoConfig;
use crate::data::ByteDataset;
use crate::model::{
    ForwardOutput, MemoryDiagnostics, MemoryTrainingSignals, RingBuffer, ZenoAgent,
};
use crate::trie::ProtoTrie;
use crate::visualizer::{
    TrieMemoryDiagnostics, TrieReadBatch, TrieReadRecord, TrieVisualizerRuntime, TrieWriteBatch,
    TrieWriteRecord,
};

// ---------------------------------------------------------------------------
// Shared training infrastructure
// ---------------------------------------------------------------------------

/// Checkpoint configuration shared across all phases.
pub struct CheckpointConfig {
    /// Directory to save checkpoints in (None = no checkpoints)
    pub checkpoint_dir: Option<PathBuf>,
    /// Save checkpoint every N steps (0 = only at phase end)
    pub checkpoint_every: usize,
    /// Interrupt flag — set by Ctrl+C handler
    pub interrupted: Arc<AtomicBool>,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: None,
            checkpoint_every: 0,
            interrupted: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Save a checkpoint to disk.
fn save_checkpoint(varmap: &VarMap, dir: &Path, label: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{label}.safetensors"));
    varmap.save(&path)?;
    println!("  💾 Saved checkpoint: {}", path.display());
    Ok(())
}

/// Check if training should stop (interrupted flag set).
fn should_stop(ckpt: &CheckpointConfig) -> bool {
    ckpt.interrupted.load(Ordering::Relaxed)
}

/// Gradient-clipped optimizer step.
///
/// Computes backward pass, measures global gradient norm, and if it exceeds
/// `max_norm`, scales gradients down proportionally before applying the update.
///
/// Returns the gradient norm (before clipping).
#[derive(Debug, Clone, Default)]
struct GradientModuleDiagnostics {
    module: String,
    active_vars: usize,
    total_vars: usize,
    grad_norm: f64,
}

#[derive(Debug, Clone, Default)]
struct BackwardStepDiagnostics {
    total_norm: f64,
    active_vars: usize,
    total_vars: usize,
    active_modules: Vec<GradientModuleDiagnostics>,
}

#[derive(Debug, Default)]
struct MemoryDiagnosticsAccumulator {
    read_hits: usize,
    read_misses: usize,
    overlap_pairs: usize,
    total_pairs: usize,
    total_slots: usize,
    total_hit_slots: usize,
    total_miss_slots: usize,
    sum_mean_density: f64,
    sum_final_density: f64,
    sum_confidence: f64,
    sum_hit_confidence: f64,
    sum_miss_confidence: f64,
    sum_corr: f64,
    corr_count: usize,
    min_confidence: Option<f64>,
    max_confidence: Option<f64>,
}

impl MemoryDiagnosticsAccumulator {
    fn add(&mut self, diag: &MemoryDiagnostics) {
        let slots = diag.read_hits + diag.read_misses;
        self.read_hits += diag.read_hits;
        self.read_misses += diag.read_misses;
        self.overlap_pairs += diag.overlap_pairs;
        self.total_pairs += diag.total_pairs;
        self.total_slots += slots;
        self.total_hit_slots += diag.read_hits;
        self.total_miss_slots += diag.read_misses;
        self.sum_mean_density += diag.avg_mean_density * slots as f64;
        self.sum_final_density += diag.avg_final_density * slots as f64;
        self.sum_confidence += diag.avg_confidence * slots as f64;
        self.sum_hit_confidence += diag.avg_hit_confidence * diag.read_hits as f64;
        self.sum_miss_confidence += diag.avg_miss_confidence * diag.read_misses as f64;
        self.sum_corr += diag.density_confidence_corr;
        self.corr_count += 1;
        self.min_confidence = Some(match self.min_confidence {
            Some(current) => current.min(diag.min_confidence),
            None => diag.min_confidence,
        });
        self.max_confidence = Some(match self.max_confidence {
            Some(current) => current.max(diag.max_confidence),
            None => diag.max_confidence,
        });
    }

    fn finish(&self) -> Option<MemoryDiagnostics> {
        if self.total_slots == 0 {
            return None;
        }
        Some(MemoryDiagnostics {
            read_hits: self.read_hits,
            read_misses: self.read_misses,
            overlap_pairs: self.overlap_pairs,
            total_pairs: self.total_pairs,
            avg_mean_density: self.sum_mean_density / self.total_slots as f64,
            avg_final_density: self.sum_final_density / self.total_slots as f64,
            avg_confidence: self.sum_confidence / self.total_slots as f64,
            min_confidence: self.min_confidence.unwrap_or(0.0),
            max_confidence: self.max_confidence.unwrap_or(0.0),
            avg_hit_confidence: if self.total_hit_slots == 0 {
                0.0
            } else {
                self.sum_hit_confidence / self.total_hit_slots as f64
            },
            avg_miss_confidence: if self.total_miss_slots == 0 {
                0.0
            } else {
                self.sum_miss_confidence / self.total_miss_slots as f64
            },
            density_confidence_corr: if self.corr_count == 0 {
                0.0
            } else {
                self.sum_corr / self.corr_count as f64
            },
        })
    }
}

#[derive(Debug, Clone, Default)]
struct ReencounterEvalMetrics {
    pass1_loss: f64,
    pass2_loss: f64,
    improvement: f64,
    memory: Option<MemoryDiagnostics>,
    retrieval: RetrievalMetrics,
}

#[derive(Debug, Clone, Default)]
struct Phase3ObjectiveMetrics {
    entropy: f64,
    locality_prefix: f64,
    depth_usage: f64,
    head_overlap: f64,
    read_write_prefix: f64,
    read_write_exact: f64,
}

#[derive(Debug, Clone, Default)]
struct RetrievalMetrics {
    exact_match_rate: f64,
    prefix_overlap_rate: f64,
    density_target_mean: f64,
    confidence_mean: f64,
    density_loss: f64,
    retrieval_loss: f64,
}

fn clipped_backward_step(
    loss: &Tensor,
    opt: &mut AdamW,
    named_vars: &[(String, Var)],
    module_prefixes: &[String],
    max_norm: f64,
) -> candle_core::Result<BackwardStepDiagnostics> {
    let grads = loss.backward()?;
    let diagnostics = collect_gradient_diagnostics(&grads, named_vars, module_prefixes)?;

    if diagnostics.total_norm > max_norm && max_norm > 0.0 {
        // Scale loss so gradients are proportionally reduced
        let scale = max_norm / (diagnostics.total_norm + 1e-8);
        let scaled_loss = (loss * scale)?;
        let clipped_grads = scaled_loss.backward()?;
        opt.step(&clipped_grads)?;
    } else {
        opt.step(&grads)?;
    }

    Ok(diagnostics)
}

fn filter_named_vars(varmap: &VarMap, prefixes: &[String]) -> Vec<(String, Var)> {
    let data = varmap.data().lock().unwrap();
    let mut vars: Vec<(String, Var)> = data
        .iter()
        .filter(|(name, _)| prefixes.iter().any(|p| name.starts_with(p)))
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect();
    vars.sort_by(|a, b| a.0.cmp(&b.0));
    vars.dedup_by(|a, b| a.0 == b.0);
    vars
}

fn collect_gradient_diagnostics(
    grads: &GradStore,
    named_vars: &[(String, Var)],
    module_prefixes: &[String],
) -> candle_core::Result<BackwardStepDiagnostics> {
    let mut modules: BTreeMap<String, GradientModuleDiagnostics> = BTreeMap::new();
    let mut total_norm_sq = 0.0;
    let mut active_vars = 0usize;

    for (name, var) in named_vars {
        let module = module_name_for_var(name, module_prefixes);
        let entry = modules
            .entry(module.clone())
            .or_insert_with(|| GradientModuleDiagnostics {
                module,
                ..GradientModuleDiagnostics::default()
            });
        entry.total_vars += 1;

        if let Some(grad) = grads.get(var.as_tensor()) {
            let norm_sq: f32 = grad.sqr()?.sum_all()?.to_scalar()?;
            let norm_sq = norm_sq as f64;
            total_norm_sq += norm_sq;
            if norm_sq > 0.0 {
                active_vars += 1;
                entry.active_vars += 1;
                entry.grad_norm += norm_sq.sqrt();
            }
        }
    }

    Ok(BackwardStepDiagnostics {
        total_norm: total_norm_sq.sqrt(),
        active_vars,
        total_vars: named_vars.len(),
        active_modules: modules
            .into_values()
            .filter(|module| module.active_vars > 0)
            .collect(),
    })
}

fn module_name_for_var(name: &str, module_prefixes: &[String]) -> String {
    module_prefixes
        .iter()
        .filter(|prefix| name.starts_with(prefix.as_str()))
        .max_by_key(|prefix| prefix.len())
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

fn format_gradient_modules(diag: &BackwardStepDiagnostics) -> String {
    if diag.active_modules.is_empty() {
        return "none".to_string();
    }
    diag.active_modules
        .iter()
        .map(|module| {
            format!(
                "{}({}/{},|g|={:.3})",
                module.module, module.active_vars, module.total_vars, module.grad_norm
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_memory_diagnostics(diag: &MemoryDiagnostics) -> String {
    format!(
        "hit={:.1}% overlap={:.1}% density={:.2}/{:.2} conf={:.2}[{:.2},{:.2}] hit_conf={:.2} miss_conf={:.2} corr={:.2}",
        diag.hit_rate() * 100.0,
        diag.overlap_rate() * 100.0,
        diag.avg_mean_density,
        diag.avg_final_density,
        diag.avg_confidence,
        diag.min_confidence,
        diag.max_confidence,
        diag.avg_hit_confidence,
        diag.avg_miss_confidence,
        diag.density_confidence_corr,
    )
}

fn format_phase3_metrics(metrics: &Phase3ObjectiveMetrics) -> String {
    format!(
        "entropy={:.2} locality={:.2} depth={:.2} head_overlap={:.2} query→write={:.2} exact={:.2}",
        metrics.entropy,
        metrics.locality_prefix,
        metrics.depth_usage,
        metrics.head_overlap,
        metrics.read_write_prefix,
        metrics.read_write_exact,
    )
}

fn format_retrieval_metrics(metrics: &RetrievalMetrics) -> String {
    format!(
        "retrieval_exact={:.1}% prefix={:.1}% density_target={:.2} conf={:.2} density_loss={:.4} retrieval_loss={:.4}",
        metrics.exact_match_rate * 100.0,
        metrics.prefix_overlap_rate * 100.0,
        metrics.density_target_mean,
        metrics.confidence_mean,
        metrics.density_loss,
        metrics.retrieval_loss,
    )
}

fn emit_trie_update(
    visualizer: Option<&TrieVisualizerRuntime>,
    phase: &str,
    step: Option<usize>,
    trie: &ProtoTrie,
    writes: TrieWriteBatch,
    reads: TrieReadBatch,
    memory: Option<TrieMemoryDiagnostics>,
) -> anyhow::Result<()> {
    if let Some(visualizer) = visualizer {
        let mut focus_addresses = writes.focus_addresses();
        focus_addresses.extend(reads.focus_addresses());
        focus_addresses.sort();
        focus_addresses.dedup();
        let snapshot = trie.sparse_snapshot(&focus_addresses, 1);
        visualizer.emit_trie_updated(phase, step, trie.len(), writes, reads, memory, snapshot)?;
    }
    Ok(())
}

fn read_batch_from_training(
    training: Option<&MemoryTrainingSignals>,
) -> candle_core::Result<TrieReadBatch> {
    let Some(training) = training else {
        return Ok(TrieReadBatch::default());
    };

    let confidence_rows = training.confidence.to_vec3::<f32>()?;
    let mut reads = Vec::new();

    for (slot, slot_addresses) in training.read_addresses.iter().enumerate() {
        let seq_len = if confidence_rows.is_empty() {
            0
        } else {
            confidence_rows[0].len()
        };
        if seq_len == 0 {
            continue;
        }
        let last_token = seq_len - 1;
        for (batch_index, token_addresses) in slot_addresses.chunks(seq_len).enumerate() {
            let address = token_addresses.get(last_token).cloned().unwrap_or_default();
            let density_chain: Vec<usize> = training
                .density
                .get(batch_index)?
                .get(last_token)?
                .get(slot)?
                .to_vec1::<u32>()?
                .into_iter()
                .map(|value| value as usize)
                .collect();
            let trimmed_chain = density_chain
                .into_iter()
                .take(address.len())
                .collect::<Vec<_>>();
            let mean_density = if trimmed_chain.is_empty() {
                0.0
            } else {
                trimmed_chain.iter().sum::<usize>() as f64 / trimmed_chain.len() as f64
            };
            let final_density = trimmed_chain.last().copied().unwrap_or(0);
            reads.push(TrieReadRecord {
                slot,
                batch_index,
                token_index: last_token,
                address,
                density_chain: trimmed_chain,
                mean_density,
                final_density,
                confidence: confidence_rows[batch_index][last_token][slot] as f64,
                hit: final_density > 0,
            });
        }
    }

    Ok(TrieReadBatch { reads })
}

fn visualizer_memory_diagnostics(
    memory: Option<&MemoryDiagnostics>,
) -> Option<TrieMemoryDiagnostics> {
    memory.map(|diag| TrieMemoryDiagnostics {
        read_hits: diag.read_hits,
        read_misses: diag.read_misses,
        overlap_pairs: diag.overlap_pairs,
        total_pairs: diag.total_pairs,
        avg_mean_density: diag.avg_mean_density,
        avg_final_density: diag.avg_final_density,
        avg_confidence: diag.avg_confidence,
        min_confidence: diag.min_confidence,
        max_confidence: diag.max_confidence,
        avg_hit_confidence: diag.avg_hit_confidence,
        avg_miss_confidence: diag.avg_miss_confidence,
        density_confidence_corr: diag.density_confidence_corr,
    })
}

/// Compute cross-entropy loss from logits and targets.
///
/// Reshapes [batch, seq_len, vocab] → [batch*seq_len, vocab] for candle's CE.
fn compute_lm_loss(logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
    let (b, s, v) = logits.dims3()?;
    let logits_2d = logits.reshape((b * s, v))?;
    let targets_1d = targets.reshape(b * s)?;
    candle_nn::loss::cross_entropy(&logits_2d, &targets_1d)
}

// ---------------------------------------------------------------------------
// Phase 2 — Base Language Model (no memory)
// ---------------------------------------------------------------------------

pub struct Phase2Config {
    pub lr: f64,
    pub steps: usize,
    pub batch_size: usize,
    pub log_every: usize,
    pub gate_loss: f64, // target: loss < 3.0
    pub max_grad_norm: f64,
}

impl Default for Phase2Config {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            steps: 10_000,
            batch_size: 4,
            log_every: 100,
            gate_loss: 3.0,
            max_grad_norm: 1.0,
        }
    }
}

pub fn train_phase2(
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    pcfg: &Phase2Config,
    ckpt: &CheckpointConfig,
    device: &Device,
    visualizer: Option<&TrieVisualizerRuntime>,
) -> anyhow::Result<f64> {
    println!("=== Phase 2: Base Language Model ===");
    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_started("phase2")?;
    }
    println!(
        "  lr={}, steps={}, batch={}, grad_clip={}",
        pcfg.lr, pcfg.steps, pcfg.batch_size, pcfg.max_grad_norm
    );
    println!("  Gate: loss < {:.1}", pcfg.gate_loss);

    let prefixes = agent.phase2_params();
    let named_vars = filter_named_vars(varmap, &prefixes);
    let vars: Vec<Var> = named_vars.iter().map(|(_, var)| var.clone()).collect();
    println!(
        "  Training {} vars (of {} total)",
        vars.len(),
        varmap.all_vars().len()
    );

    let params = ParamsAdamW {
        lr: pcfg.lr,
        weight_decay: 0.01,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    let trie = ProtoTrie::new(cfg);
    let mut ring = RingBuffer::new(cfg.n_register);
    let mut running_loss = 0.0;
    let mut running_count = 0usize;
    let mut best_loss = f64::MAX;

    for step in 1..=pcfg.steps {
        if should_stop(ckpt) {
            println!("  ⚠ Interrupted at step {}", step);
            save_checkpoint(
                varmap,
                ckpt.checkpoint_dir
                    .as_deref()
                    .unwrap_or(Path::new("checkpoints")),
                "phase2_interrupt",
            )?;
            break;
        }

        let (input, target) = dataset.get_random_batch(pcfg.batch_size, device)?;

        let out: ForwardOutput = agent.forward(&input, &trie, &mut ring, 1.0, false, device)?;
        let loss = compute_lm_loss(&out.logits, &target)?;

        let _grad_diag =
            clipped_backward_step(&loss, &mut opt, &named_vars, &prefixes, pcfg.max_grad_norm)?;

        let loss_val: f64 = loss.to_scalar::<f32>()? as f64;
        running_loss += loss_val;
        running_count += 1;
        if loss_val < best_loss {
            best_loss = loss_val;
        }

        if step % pcfg.log_every == 0 || step == pcfg.steps {
            let avg = running_loss / running_count as f64;
            let ppl = avg.exp();
            println!(
                "  [step {}/{}] loss={:.4} ppl={:.1}{}",
                step,
                pcfg.steps,
                avg,
                ppl,
                if avg < pcfg.gate_loss {
                    " ✓ GATE"
                } else {
                    ""
                }
            );
            if let Some(visualizer) = visualizer {
                visualizer.emit_step_metrics("phase2", step, pcfg.steps, avg, Some(ppl), None)?;
            }
            running_loss = 0.0;
            running_count = 0;
        }

        // Periodic checkpoint
        if ckpt.checkpoint_every > 0 && step % ckpt.checkpoint_every == 0 {
            if let Some(ref dir) = ckpt.checkpoint_dir {
                save_checkpoint(varmap, dir, &format!("phase2_step{step}"))?;
            }
        }
    }

    // End-of-phase checkpoint
    if let Some(ref dir) = ckpt.checkpoint_dir {
        save_checkpoint(varmap, dir, "phase2_final")?;
    }

    println!("  Phase 2 complete. Best loss={:.4}", best_loss);
    if best_loss < pcfg.gate_loss {
        println!(
            "  ✓ Gate PASSED (loss {:.4} < {:.1})",
            best_loss, pcfg.gate_loss
        );
    } else {
        println!(
            "  ✗ Gate FAILED (loss {:.4} >= {:.1})",
            best_loss, pcfg.gate_loss
        );
    }

    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_completed("phase2", "best_loss", best_loss)?;
    }

    Ok(best_loss)
}

// ---------------------------------------------------------------------------
// Phase 3 — AddrNet Training (freeze base)
// ---------------------------------------------------------------------------

pub struct Phase3Config {
    pub lr: f64,
    pub steps: usize,
    pub batch_size: usize,
    pub log_every: usize,
    pub tau_start: f64,
    pub tau_end: f64,
    pub gate_entropy: f64, // target: entropy > 7.0
    pub locality_weight: f64,
    pub depth_usage_weight: f64,
    pub diversity_weight: f64,
    pub query_write_weight: f64,
    pub max_grad_norm: f64,
}

impl Default for Phase3Config {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            steps: 5_000,
            batch_size: 4,
            log_every: 100,
            tau_start: 5.0,
            tau_end: 0.5,
            gate_entropy: 7.0,
            locality_weight: 1.0,
            depth_usage_weight: 0.25,
            diversity_weight: 0.5,
            query_write_weight: 1.0,
            max_grad_norm: 1.0,
        }
    }
}

/// Compute address entropy loss (want high entropy = diverse addresses).
///
/// Takes softmax logits [batch, seq_len, trie_depth, trie_arity], computes
/// per-level entropy, returns negative mean entropy (minimize → maximize entropy).
fn address_entropy_loss(logits_list: &[Tensor]) -> candle_core::Result<(Tensor, f64)> {
    let mut total_entropy = 0.0;
    let mut count = 0usize;
    let mut loss_terms = Vec::new();

    for logits in logits_list {
        // logits: [batch, seq_len, depth, arity]
        let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
        let log_probs = probs.log()?;
        // entropy = -sum(p * log(p)) per level
        let entropy = (probs * log_probs)?.neg()?.sum(candle_core::D::Minus1)?;
        // mean over batch, seq, depth
        let mean_ent = entropy.mean_all()?;
        total_entropy += mean_ent.to_scalar::<f32>()? as f64;
        count += 1;
        // Negative entropy as loss (we want to maximize entropy)
        loss_terms.push(mean_ent.neg()?);
    }

    let loss = if loss_terms.len() == 1 {
        loss_terms.into_iter().next().unwrap()
    } else {
        let stacked = Tensor::stack(&loss_terms, 0)?;
        stacked.mean_all()?
    };

    let avg_entropy = if count > 0 {
        total_entropy / count as f64
    } else {
        0.0
    };
    Ok((loss, avg_entropy))
}

fn last_token_logits(logits: &Tensor) -> candle_core::Result<Tensor> {
    let (batch, seq_len, _depth, _arity) = logits.dims4()?;
    let mut rows = Vec::with_capacity(batch);
    for batch_index in 0..batch {
        rows.push(logits.get(batch_index)?.get(seq_len - 1)?);
    }
    Tensor::stack(&rows, 0)
}

fn shared_prefix_ratio(left: &[u32], right: &[u32]) -> f64 {
    let denom = left.len().min(right.len()).max(1);
    let shared = left
        .iter()
        .zip(right.iter())
        .take_while(|(a, b)| a == b)
        .count();
    shared as f64 / denom as f64
}

fn prefix_match_scores(
    probs_a: &Tensor,
    probs_b: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    let depth = probs_a.dim(0)?;
    let device = probs_a.device();
    let mut prefix_product = Tensor::ones((), DType::F32, device)?;
    let mut prefix_sum = Tensor::zeros((), DType::F32, device)?;

    for depth_index in 0..depth {
        let level_a = probs_a.get(depth_index)?;
        let level_b = probs_b.get(depth_index)?;
        let match_prob = (&level_a * &level_b)?.sum_all()?;
        prefix_product = (&prefix_product * &match_prob)?;
        prefix_sum = (prefix_sum + &prefix_product)?;
    }

    Ok(((prefix_sum / depth as f64)?, prefix_product))
}

fn address_locality_loss(
    input: &Tensor,
    write_logits: &[Tensor],
) -> candle_core::Result<(Tensor, f64)> {
    if write_logits.is_empty() {
        return Ok((Tensor::zeros((), DType::F32, input.device())?, 0.0));
    }
    let rows = input.to_vec2::<u32>()?;
    if rows.len() < 2 {
        return Ok((Tensor::zeros((), DType::F32, input.device())?, 0.0));
    }

    let device = input.device();
    let mut loss_sum = Tensor::zeros((), DType::F32, device)?;
    let mut prefix_total = 0.0;
    let mut pair_count = 0usize;

    for logits in write_logits {
        let final_probs =
            candle_nn::ops::softmax(&last_token_logits(logits)?, candle_core::D::Minus1)?;
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                let target = shared_prefix_ratio(&rows[i], &rows[j]) as f32;
                let score = prefix_match_scores(&final_probs.get(i)?, &final_probs.get(j)?)?.0;
                let diff = (&score - Tensor::new(target, device)?)?;
                loss_sum = (loss_sum + diff.sqr()?)?;
                prefix_total += score.to_scalar::<f32>()? as f64;
                pair_count += 1;
            }
        }
    }

    if pair_count == 0 {
        Ok((Tensor::zeros((), DType::F32, device)?, 0.0))
    } else {
        Ok((
            (loss_sum / pair_count as f64)?,
            prefix_total / pair_count as f64,
        ))
    }
}

fn address_depth_usage_loss(write_logits: &[Tensor]) -> candle_core::Result<(Tensor, f64)> {
    if write_logits.is_empty() {
        return Ok((Tensor::zeros((), DType::F32, &Device::Cpu)?, 0.0));
    }

    let device = write_logits[0].device();
    let one = Tensor::ones((), DType::F32, device)?;
    let mut loss_sum = Tensor::zeros((), DType::F32, device)?;
    let mut novelty_total = 0.0;
    let mut count = 0usize;

    for logits in write_logits {
        let final_probs =
            candle_nn::ops::softmax(&last_token_logits(logits)?, candle_core::D::Minus1)?;
        let (batch, depth, _arity) = final_probs.dims3()?;
        for batch_index in 0..batch {
            let sample = final_probs.get(batch_index)?;
            for depth_index in 1..depth {
                let novelty = (&one
                    - (&sample.get(depth_index - 1)? * &sample.get(depth_index)?)?.sum_all()?)?;
                loss_sum = (loss_sum - &novelty)?;
                novelty_total += novelty.to_scalar::<f32>()? as f64;
                count += 1;
            }
        }
    }

    if count == 0 {
        Ok((Tensor::zeros((), DType::F32, device)?, 0.0))
    } else {
        Ok(((loss_sum / count as f64)?, novelty_total / count as f64))
    }
}

/// Diversity loss: penalize if different write heads collapse onto the same prefix.
fn address_diversity_loss(addr_logits: &[Tensor]) -> candle_core::Result<(Tensor, f64)> {
    if addr_logits.len() < 2 {
        return Ok((Tensor::zeros((), DType::F32, addr_logits[0].device())?, 0.0));
    }

    let device = addr_logits[0].device();
    let final_probs: Vec<Tensor> = addr_logits
        .iter()
        .map(|logits| last_token_logits(logits))
        .collect::<candle_core::Result<Vec<_>>>()?
        .into_iter()
        .map(|logits| candle_nn::ops::softmax(&logits, candle_core::D::Minus1))
        .collect::<candle_core::Result<Vec<_>>>()?;

    let batch = final_probs[0].dim(0)?;
    let mut loss_sum = Tensor::zeros((), DType::F32, device)?;
    let mut overlap_total = 0.0;
    let mut pair_count = 0usize;

    for batch_index in 0..batch {
        for i in 0..final_probs.len() {
            for j in (i + 1)..final_probs.len() {
                let score = prefix_match_scores(
                    &final_probs[i].get(batch_index)?,
                    &final_probs[j].get(batch_index)?,
                )?
                .0;
                loss_sum = (loss_sum + &score)?;
                overlap_total += score.to_scalar::<f32>()? as f64;
                pair_count += 1;
            }
        }
    }

    if pair_count == 0 {
        Ok((Tensor::zeros((), DType::F32, device)?, 0.0))
    } else {
        Ok((
            (loss_sum / pair_count as f64)?,
            overlap_total / pair_count as f64,
        ))
    }
}

fn read_write_alignment_loss(
    read_logits: &[Tensor],
    write_logits: &[Tensor],
) -> candle_core::Result<(Tensor, f64, f64)> {
    if read_logits.is_empty() || write_logits.is_empty() {
        return Ok((Tensor::zeros((), DType::F32, &Device::Cpu)?, 0.0, 0.0));
    }

    let slots = read_logits.len().min(write_logits.len());
    let device = read_logits[0].device();
    let one = Tensor::ones((), DType::F32, device)?;
    let mut loss_sum = Tensor::zeros((), DType::F32, device)?;
    let mut prefix_total = 0.0;
    let mut exact_total = 0.0;
    let mut count = 0usize;

    for slot in 0..slots {
        let read_probs = candle_nn::ops::softmax(&read_logits[slot], candle_core::D::Minus1)?;
        let write_probs = candle_nn::ops::softmax(
            &last_token_logits(&write_logits[slot])?,
            candle_core::D::Minus1,
        )?;
        let (batch, seq_len, _depth, _arity) = read_probs.dims4()?;

        for batch_index in 0..batch {
            let write_target = write_probs.get(batch_index)?;
            let read_rows = read_probs.get(batch_index)?;
            for token_index in 0..seq_len {
                let (prefix_score, exact_score) =
                    prefix_match_scores(&read_rows.get(token_index)?, &write_target)?;
                loss_sum = (loss_sum + (&one - &prefix_score)?)?;
                prefix_total += prefix_score.to_scalar::<f32>()? as f64;
                exact_total += exact_score.to_scalar::<f32>()? as f64;
                count += 1;
            }
        }
    }

    if count == 0 {
        Ok((Tensor::zeros((), DType::F32, device)?, 0.0, 0.0))
    } else {
        Ok((
            (loss_sum / count as f64)?,
            prefix_total / count as f64,
            exact_total / count as f64,
        ))
    }
}

pub fn train_phase3(
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    pcfg: &Phase3Config,
    ckpt: &CheckpointConfig,
    device: &Device,
    visualizer: Option<&TrieVisualizerRuntime>,
) -> anyhow::Result<f64> {
    println!("=== Phase 3: AddrNet Training ===");
    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_started("phase3")?;
    }
    println!(
        "  lr={}, steps={}, τ: {}→{}, grad_clip={}",
        pcfg.lr, pcfg.steps, pcfg.tau_start, pcfg.tau_end, pcfg.max_grad_norm
    );
    println!("  Gate: address entropy > {:.1}", pcfg.gate_entropy);
    println!(
        "  Objective weights: locality={} depth={} diversity={} query→write={}",
        pcfg.locality_weight,
        pcfg.depth_usage_weight,
        pcfg.diversity_weight,
        pcfg.query_write_weight
    );

    let prefixes = agent.phase3_params();
    let named_vars = filter_named_vars(varmap, &prefixes);
    let vars: Vec<Var> = named_vars.iter().map(|(_, var)| var.clone()).collect();
    println!("  Training {} AddrNet vars", vars.len());

    let params = ParamsAdamW {
        lr: pcfg.lr,
        weight_decay: 0.0,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    let trie = ProtoTrie::new(cfg);
    let mut ring = RingBuffer::new(cfg.n_register);
    let mut best_entropy = 0.0;
    let mut best_query_write = 0.0;

    for step in 1..=pcfg.steps {
        if should_stop(ckpt) {
            println!("  ⚠ Interrupted at step {}", step);
            save_checkpoint(
                varmap,
                ckpt.checkpoint_dir
                    .as_deref()
                    .unwrap_or(Path::new("checkpoints")),
                "phase3_interrupt",
            )?;
            break;
        }

        let progress = step as f64 / pcfg.steps as f64;
        let tau = pcfg.tau_start + (pcfg.tau_end - pcfg.tau_start) * progress;

        let (input, _target) = dataset.get_random_batch(pcfg.batch_size, device)?;

        // Forward with memory enabled to get read_addr_logits
        let out = agent.forward(&input, &trie, &mut ring, tau, true, device)?;

        // Entropy + locality objectives on read/write addresses.
        let (entropy_loss, avg_entropy) = address_entropy_loss(&out.read_addr_logits)?;
        let write_logits: Vec<Tensor> =
            out.write_addresses.iter().map(|(_, l)| l.clone()).collect();
        let (locality_loss, locality_prefix) = address_locality_loss(&input, &write_logits)?;
        let (depth_loss, depth_usage) = address_depth_usage_loss(&write_logits)?;
        let (div_loss, head_overlap) = address_diversity_loss(&write_logits)?;
        let (query_write_loss, read_write_prefix, read_write_exact) =
            read_write_alignment_loss(&out.read_addr_logits, &write_logits)?;

        let total_loss = ((((entropy_loss + (locality_loss * pcfg.locality_weight)?)?
            + (depth_loss * pcfg.depth_usage_weight)?)?
            + (div_loss * pcfg.diversity_weight)?)?
            + (query_write_loss * pcfg.query_write_weight)?)?;

        let _grad_diag = clipped_backward_step(
            &total_loss,
            &mut opt,
            &named_vars,
            &prefixes,
            pcfg.max_grad_norm,
        )?;

        if avg_entropy > best_entropy {
            best_entropy = avg_entropy;
        }
        if read_write_prefix > best_query_write {
            best_query_write = read_write_prefix;
        }

        if step % pcfg.log_every == 0 || step == pcfg.steps {
            let loss_val: f32 = total_loss.to_scalar()?;
            let metrics = Phase3ObjectiveMetrics {
                entropy: avg_entropy,
                locality_prefix,
                depth_usage,
                head_overlap,
                read_write_prefix,
                read_write_exact,
            };
            println!(
                "  [step {}/{}] loss={:.4} τ={:.2}{}",
                step,
                pcfg.steps,
                loss_val,
                tau,
                if avg_entropy > pcfg.gate_entropy {
                    " ✓ GATE"
                } else {
                    ""
                }
            );
            println!("    {}", format_phase3_metrics(&metrics));
            if let Some(visualizer) = visualizer {
                visualizer.emit_step_metrics(
                    "phase3",
                    step,
                    pcfg.steps,
                    loss_val as f64,
                    None,
                    None,
                )?;
            }
        }

        if ckpt.checkpoint_every > 0 && step % ckpt.checkpoint_every == 0 {
            if let Some(ref dir) = ckpt.checkpoint_dir {
                save_checkpoint(varmap, dir, &format!("phase3_step{step}"))?;
            }
        }
    }

    if let Some(ref dir) = ckpt.checkpoint_dir {
        save_checkpoint(varmap, dir, "phase3_final")?;
    }

    println!(
        "  Phase 3 complete. Best entropy={:.2}, best query→write prefix={:.2}",
        best_entropy, best_query_write
    );
    if best_entropy > pcfg.gate_entropy {
        println!(
            "  ✓ Gate PASSED (entropy {:.2} > {:.1})",
            best_entropy, pcfg.gate_entropy
        );
    } else {
        println!(
            "  ✗ Gate FAILED (entropy {:.2} <= {:.1})",
            best_entropy, pcfg.gate_entropy
        );
    }

    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_completed("phase3", "best_query_write_prefix", best_query_write)?;
    }

    Ok(best_query_write)
}

// ---------------------------------------------------------------------------
// Phase 4 — Memory Integration (trie active, re-encounter test)
// ---------------------------------------------------------------------------

pub struct Phase4Config {
    pub lr: f64,
    pub steps: usize,
    pub batch_size: usize,
    pub log_every: usize,
    pub tau: f64,
    pub gate_retrieval: f64, // target: > 0.9
    pub eval_every: usize,   // how often to run re-encounter eval
    pub retrieval_supervision_weight: f64,
    pub density_supervision_weight: f64,
    pub max_grad_norm: f64,
}

impl Default for Phase4Config {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            steps: 15_000,
            batch_size: 4,
            log_every: 100,
            tau: 0.5,
            eval_every: 1000,
            gate_retrieval: 0.9,
            retrieval_supervision_weight: 1.0,
            density_supervision_weight: 0.5,
            max_grad_norm: 1.0,
        }
    }
}

/// Perform trie writes from a forward pass output.
fn perform_trie_writes(
    out: &ForwardOutput,
    trie: &mut ProtoTrie,
    cfg: &ZenoConfig,
) -> candle_core::Result<TrieWriteBatch> {
    let mut writes = Vec::new();

    for i in 0..cfg.n_addr_nets {
        let (addresses, _) = &out.write_addresses[i];
        let values = &out.write_values[i];
        let strength = &out.write_strength;

        // addresses: [batch, seq_len, depth, arity] → argmax → byte indices
        let (batch, seq_len, _depth, _arity) = addresses.dims4()?;
        let byte_indices = addresses.argmax(candle_core::D::Minus1)?; // [batch, seq, depth]

        // Mean-pool values and strength across seq_len for trie write
        let mean_val = values.mean(1)?; // [batch, d_model]
        let mean_str = strength.mean(1)?.squeeze(1)?; // [batch]

        for b in 0..batch {
            // Use last token's address (most context)
            let addr_row = byte_indices.get(b)?.get(seq_len - 1)?;
            let addr_bytes: Vec<u8> = addr_row
                .to_vec1::<u32>()?
                .iter()
                .map(|&v| v as u8)
                .collect();

            let val = mean_val.get(b)?;
            let str_val: f64 = mean_str.get(b)?.to_scalar::<f32>()? as f64;

            trie.write(&addr_bytes, &val, str_val)?;
            writes.push(TrieWriteRecord {
                slot: i,
                batch_index: b,
                address: addr_bytes,
                strength: str_val,
            });
        }
    }

    Ok(TrieWriteBatch { writes })
}

fn shared_prefix_fraction_u8(left: &[u8], right: &[u8], depth: usize) -> f32 {
    let shared = left
        .iter()
        .zip(right.iter())
        .take(depth)
        .take_while(|(a, b)| a == b)
        .count();
    shared as f32 / depth.max(1) as f32
}

fn structural_density_target(chain: &[u32], trie_arity: usize) -> f32 {
    if chain.is_empty() {
        return 0.0;
    }
    let denom = (trie_arity.max(1) as f32 + 1.0).ln();
    let mut weighted = 0.0;
    let mut weight_sum = 0.0;
    for (depth, &density) in chain.iter().enumerate() {
        let weight = (depth + 1) as f32;
        weighted += weight * ((density as f32 + 1.0).ln() / denom);
        weight_sum += weight;
    }
    (weighted / weight_sum.max(1.0)).clamp(0.0, 1.0)
}

fn retrieval_supervision(
    training: &MemoryTrainingSignals,
    writes: &TrieWriteBatch,
    cfg: &ZenoConfig,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor, RetrievalMetrics)> {
    let (batch, seq_len, n_slots) = training.confidence.dims3()?;
    let depth = cfg.trie_depth;
    let mut writes_by_batch = vec![Vec::<Vec<u8>>::new(); batch];
    for write in &writes.writes {
        if write.batch_index < batch {
            writes_by_batch[write.batch_index].push(write.address.clone());
        }
    }

    let confidence_rows = training.confidence.to_vec3::<f32>()?;
    let mut density_targets = Vec::with_capacity(batch * seq_len * n_slots);
    let mut retrieval_targets = Vec::with_capacity(batch * seq_len * n_slots);
    let mut exact_hits = 0usize;
    let mut prefix_sum = 0.0;
    let mut density_sum = 0.0;
    let mut count = 0usize;

    for batch_index in 0..batch {
        let sample_writes = &writes_by_batch[batch_index];
        for token_index in 0..seq_len {
            for slot in 0..n_slots {
                let address = &training.read_addresses[slot][batch_index * seq_len + token_index];
                let mut best_prefix = 0.0f32;
                let mut exact = false;
                for target in sample_writes {
                    let prefix = shared_prefix_fraction_u8(address, target, depth);
                    if prefix > best_prefix {
                        best_prefix = prefix;
                    }
                    if address == target {
                        exact = true;
                    }
                }
                let density_chain = training
                    .density
                    .get(batch_index)?
                    .get(token_index)?
                    .get(slot)?
                    .to_vec1::<u32>()?;
                let density_target = structural_density_target(&density_chain, cfg.trie_arity);
                retrieval_targets.push(best_prefix);
                density_targets.push(density_target);
                prefix_sum += best_prefix as f64;
                density_sum += density_target as f64;
                if exact {
                    exact_hits += 1;
                }
                count += 1;
            }
        }
    }

    let density_target = Tensor::from_vec(density_targets, (batch, seq_len, n_slots), device)?;
    let retrieval_target = Tensor::from_vec(retrieval_targets, (batch, seq_len, n_slots), device)?;
    let confidence = &training.confidence;
    let density_loss = ((confidence - &density_target)?.sqr()?.mean_all())?;
    let retrieval_loss = ((confidence - &retrieval_target)?.sqr()?.mean_all())?;

    Ok((
        density_loss.clone(),
        retrieval_loss.clone(),
        RetrievalMetrics {
            exact_match_rate: exact_hits as f64 / count.max(1) as f64,
            prefix_overlap_rate: prefix_sum / count.max(1) as f64,
            density_target_mean: density_sum / count.max(1) as f64,
            confidence_mean: confidence_rows
                .iter()
                .flat_map(|batch| batch.iter().flat_map(|token| token.iter()))
                .map(|&value| value as f64)
                .sum::<f64>()
                / count.max(1) as f64,
            density_loss: density_loss.to_scalar::<f32>()? as f64,
            retrieval_loss: retrieval_loss.to_scalar::<f32>()? as f64,
        },
    ))
}

/// Re-encounter evaluation: process same text twice, check if perplexity drops.
fn reencounter_eval(
    agent: &ZenoAgent,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    device: &Device,
    tau: f64,
    n_samples: usize,
    visualizer: Option<&TrieVisualizerRuntime>,
    phase: &str,
    step: Option<usize>,
) -> anyhow::Result<ReencounterEvalMetrics> {
    let batches = dataset.create_reencounter_batches(n_samples, device)?;
    if batches.is_empty() {
        return Ok(ReencounterEvalMetrics::default());
    }

    let mut total_loss_pass1 = 0.0;
    let mut total_loss_pass2 = 0.0;
    let mut count = 0;
    let mut memory_acc = MemoryDiagnosticsAccumulator::default();
    let mut retrieval_acc = RetrievalMetrics::default();

    for (first_in, first_tgt, second_in, second_tgt) in &batches {
        let mut trie = ProtoTrie::new(cfg);
        let mut ring = RingBuffer::new(cfg.n_register);

        // Pass 1: process and write to trie
        let out1 = agent.forward(first_in, &trie, &mut ring, tau, true, device)?;
        let loss1 = compute_lm_loss(&out1.logits, first_tgt)?;
        total_loss_pass1 += loss1.to_scalar::<f32>()? as f64;

        // Write to trie from pass 1
        let writes = perform_trie_writes(&out1, &mut trie, cfg)?;
        let retrieval_writes = writes.clone();
        let reads = read_batch_from_training(out1.memory_training.as_ref())?;
        let memory = visualizer_memory_diagnostics(out1.memory_diagnostics.as_ref());
        emit_trie_update(visualizer, phase, step, &trie, writes, reads, memory)?;

        // Pass 2: process same content with trie populated
        let out2 = agent.forward(second_in, &trie, &mut ring, tau, true, device)?;
        let loss2 = compute_lm_loss(&out2.logits, second_tgt)?;
        total_loss_pass2 += loss2.to_scalar::<f32>()? as f64;
        if let Some(diag) = out2.memory_diagnostics.as_ref() {
            memory_acc.add(diag);
        }
        if let Some(training) = out2.memory_training.as_ref() {
            let (_, _, retrieval) =
                retrieval_supervision(training, &retrieval_writes, cfg, device)?;
            retrieval_acc.exact_match_rate += retrieval.exact_match_rate;
            retrieval_acc.prefix_overlap_rate += retrieval.prefix_overlap_rate;
            retrieval_acc.density_target_mean += retrieval.density_target_mean;
            retrieval_acc.confidence_mean += retrieval.confidence_mean;
            retrieval_acc.density_loss += retrieval.density_loss;
            retrieval_acc.retrieval_loss += retrieval.retrieval_loss;
        }

        count += 1;
    }

    let avg_loss1 = total_loss_pass1 / count as f64;
    let avg_loss2 = total_loss_pass2 / count as f64;
    let improvement = (avg_loss1 - avg_loss2) / avg_loss1;

    if let Some(visualizer) = visualizer {
        visualizer.emit_reencounter_summary(phase, step, avg_loss1, avg_loss2, improvement)?;
    }

    Ok(ReencounterEvalMetrics {
        pass1_loss: avg_loss1,
        pass2_loss: avg_loss2,
        improvement,
        memory: memory_acc.finish(),
        retrieval: RetrievalMetrics {
            exact_match_rate: retrieval_acc.exact_match_rate / count.max(1) as f64,
            prefix_overlap_rate: retrieval_acc.prefix_overlap_rate / count.max(1) as f64,
            density_target_mean: retrieval_acc.density_target_mean / count.max(1) as f64,
            confidence_mean: retrieval_acc.confidence_mean / count.max(1) as f64,
            density_loss: retrieval_acc.density_loss / count.max(1) as f64,
            retrieval_loss: retrieval_acc.retrieval_loss / count.max(1) as f64,
        },
    })
}

pub fn train_phase4(
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    pcfg: &Phase4Config,
    ckpt: &CheckpointConfig,
    device: &Device,
    visualizer: Option<&TrieVisualizerRuntime>,
) -> anyhow::Result<f64> {
    println!("=== Phase 4: Memory Integration ===");
    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_started("phase4")?;
    }
    println!(
        "  lr={}, steps={}, τ={}, grad_clip={}",
        pcfg.lr, pcfg.steps, pcfg.tau, pcfg.max_grad_norm
    );
    println!(
        "  Gate: re-encounter improvement > {:.0}%",
        pcfg.gate_retrieval * 100.0
    );
    println!(
        "  Supervision weights: retrieval={} density={}",
        pcfg.retrieval_supervision_weight, pcfg.density_supervision_weight
    );

    let prefixes = agent.phase4_params();
    let named_vars = filter_named_vars(varmap, &prefixes);
    let vars: Vec<Var> = named_vars.iter().map(|(_, var)| var.clone()).collect();
    println!(
        "  Training {} memory vars across {:?}",
        vars.len(),
        prefixes
    );

    let params = ParamsAdamW {
        lr: pcfg.lr,
        weight_decay: 0.01,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    let mut trie = ProtoTrie::new(cfg);
    let mut ring = RingBuffer::new(cfg.n_register);
    let mut running_loss = 0.0;
    let mut running_count = 0usize;
    let mut best_improvement = 0.0;
    let mut best_retrieval_exact = 0.0_f64;

    for step in 1..=pcfg.steps {
        if should_stop(ckpt) {
            println!("  ⚠ Interrupted at step {}", step);
            save_checkpoint(
                varmap,
                ckpt.checkpoint_dir
                    .as_deref()
                    .unwrap_or(Path::new("checkpoints")),
                "phase4_interrupt",
            )?;
            break;
        }

        let (first_in, first_tgt, second_in, second_tgt) =
            dataset.get_random_reencounter_batch(pcfg.batch_size, device)?;

        let out1 = agent.forward(&first_in, &trie, &mut ring, pcfg.tau, true, device)?;
        let lm_loss_pass1 = compute_lm_loss(&out1.logits, &first_tgt)?;
        let writes = perform_trie_writes(&out1, &mut trie, cfg)?;

        let out2 = agent.forward(&second_in, &trie, &mut ring, pcfg.tau, true, device)?;
        let lm_loss_pass2 = compute_lm_loss(&out2.logits, &second_tgt)?;
        let (density_loss, retrieval_loss, retrieval_metrics) =
            if let Some(training) = out2.memory_training.as_ref() {
                retrieval_supervision(training, &writes, cfg, device)?
            } else {
                (
                    Tensor::zeros((), DType::F32, device)?,
                    Tensor::zeros((), DType::F32, device)?,
                    RetrievalMetrics::default(),
                )
            };

        let loss = (((lm_loss_pass1 + &lm_loss_pass2)?
            + (retrieval_loss * pcfg.retrieval_supervision_weight)?)?
            + (density_loss * pcfg.density_supervision_weight)?)?;

        let grad_diag =
            clipped_backward_step(&loss, &mut opt, &named_vars, &prefixes, pcfg.max_grad_norm)?;

        let reads = read_batch_from_training(out2.memory_training.as_ref())?;
        let memory = visualizer_memory_diagnostics(out2.memory_diagnostics.as_ref());
        emit_trie_update(
            visualizer,
            "phase4",
            Some(step),
            &trie,
            writes,
            reads,
            memory,
        )?;

        let loss_val: f64 = loss.to_scalar::<f32>()? as f64;
        running_loss += loss_val;
        running_count += 1;
        best_retrieval_exact = best_retrieval_exact.max(retrieval_metrics.exact_match_rate);

        if step % pcfg.log_every == 0 || step == pcfg.steps {
            let avg = running_loss / running_count as f64;
            if let Some(diag) = out2.memory_diagnostics.as_ref() {
                println!(
                    "  [step {}/{}] loss={:.4} trie_entries={} grad={:.4} active_grads={}/{} {}",
                    step,
                    pcfg.steps,
                    avg,
                    trie.len(),
                    grad_diag.total_norm,
                    grad_diag.active_vars,
                    grad_diag.total_vars,
                    format_memory_diagnostics(diag),
                );
                println!("    {}", format_retrieval_metrics(&retrieval_metrics));
            } else {
                println!(
                    "  [step {}/{}] loss={:.4} trie_entries={} grad={:.4} active_grads={}/{}",
                    step,
                    pcfg.steps,
                    avg,
                    trie.len(),
                    grad_diag.total_norm,
                    grad_diag.active_vars,
                    grad_diag.total_vars,
                );
            }
            println!("    grad_modules: {}", format_gradient_modules(&grad_diag));
            if let Some(visualizer) = visualizer {
                visualizer.emit_step_metrics(
                    "phase4",
                    step,
                    pcfg.steps,
                    avg,
                    None,
                    Some(trie.len()),
                )?;
            }
            running_loss = 0.0;
            running_count = 0;
        }

        // Periodic re-encounter evaluation
        if step % pcfg.eval_every == 0 {
            let eval = reencounter_eval(
                agent,
                dataset,
                cfg,
                device,
                pcfg.tau,
                8,
                visualizer,
                "phase4",
                Some(step),
            )?;
            println!(
                "  [EVAL step {}] pass1_loss={:.4} pass2_loss={:.4} improvement={:.1}%{}",
                step,
                eval.pass1_loss,
                eval.pass2_loss,
                eval.improvement * 100.0,
                if eval.improvement > pcfg.gate_retrieval {
                    " ✓ GATE"
                } else {
                    ""
                }
            );
            if let Some(diag) = eval.memory.as_ref() {
                println!("    eval_memory: {}", format_memory_diagnostics(diag));
            }
            println!(
                "    eval_retrieval: {}",
                format_retrieval_metrics(&eval.retrieval)
            );
            if eval.improvement > best_improvement {
                best_improvement = eval.improvement;
            }
            best_retrieval_exact = best_retrieval_exact.max(eval.retrieval.exact_match_rate);
        }

        // Periodic checkpoint
        if ckpt.checkpoint_every > 0 && step % ckpt.checkpoint_every == 0 {
            if let Some(ref dir) = ckpt.checkpoint_dir {
                save_checkpoint(varmap, dir, &format!("phase4_step{step}"))?;
            }
        }

        // Reset trie periodically to prevent unbounded growth
        if trie.len() > 50_000 {
            trie.reset();
            ring.reset();
        }
    }

    if let Some(ref dir) = ckpt.checkpoint_dir {
        save_checkpoint(varmap, dir, "phase4_final")?;
    }

    println!(
        "  Phase 4 complete. Best improvement={:.1}%, best exact retrieval={:.1}%",
        best_improvement * 100.0,
        best_retrieval_exact * 100.0
    );
    if best_improvement > pcfg.gate_retrieval {
        println!("  ✓ Gate PASSED");
    } else {
        println!(
            "  ✗ Gate FAILED (best {:.1}% < {:.0}%)",
            best_improvement * 100.0,
            pcfg.gate_retrieval * 100.0
        );
    }

    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_completed("phase4", "best_retrieval_exact", best_retrieval_exact)?;
    }

    Ok(best_retrieval_exact)
}

// ---------------------------------------------------------------------------
// Phase 5 — Coherence Unfreeze (progressive)
// ---------------------------------------------------------------------------

pub struct Phase5Config {
    pub lr_base: f64,
    pub lr_addr_multiplier: f64, // 0.01× for AddrNet
    pub steps_per_sub: usize,
    pub batch_size: usize,
    pub log_every: usize,
    pub tau: f64,
    pub max_grad_norm: f64,
}

impl Default for Phase5Config {
    fn default() -> Self {
        Self {
            lr_base: 1e-5,
            lr_addr_multiplier: 0.01,
            steps_per_sub: 5_000,
            batch_size: 4,
            log_every: 100,
            tau: 0.5,
            max_grad_norm: 1.0,
        }
    }
}

fn train_phase5_sub(
    label: &str,
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    prefixes: &[String],
    lr: f64,
    steps: usize,
    batch_size: usize,
    log_every: usize,
    tau: f64,
    max_grad_norm: f64,
    ckpt: &CheckpointConfig,
    device: &Device,
    visualizer: Option<&TrieVisualizerRuntime>,
) -> anyhow::Result<f64> {
    println!("  --- Phase 5{} ---", label);
    let phase_name = format!("phase5{}", label);
    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_started(phase_name.clone())?;
    }
    let named_vars = filter_named_vars(varmap, prefixes);
    let vars: Vec<Var> = named_vars.iter().map(|(_, var)| var.clone()).collect();
    println!("    Training {} vars at lr={:.1e}", vars.len(), lr);

    let params = ParamsAdamW {
        lr,
        weight_decay: 0.01,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    let mut trie = ProtoTrie::new(cfg);
    let mut ring = RingBuffer::new(cfg.n_register);
    let mut running_loss = 0.0;
    let mut running_count = 0usize;
    let mut best_loss = f64::MAX;

    for step in 1..=steps {
        if should_stop(ckpt) {
            println!("    ⚠ Interrupted at step {}", step);
            save_checkpoint(
                varmap,
                ckpt.checkpoint_dir
                    .as_deref()
                    .unwrap_or(Path::new("checkpoints")),
                &format!("phase5{label}_interrupt"),
            )?;
            break;
        }

        let (input, target) = dataset.get_random_batch(batch_size, device)?;
        let out = agent.forward(&input, &trie, &mut ring, tau, true, device)?;
        let loss = compute_lm_loss(&out.logits, &target)?;

        let _grad_diag =
            clipped_backward_step(&loss, &mut opt, &named_vars, prefixes, max_grad_norm)?;
        let writes = perform_trie_writes(&out, &mut trie, cfg)?;
        let reads = read_batch_from_training(out.memory_training.as_ref())?;
        let memory = visualizer_memory_diagnostics(out.memory_diagnostics.as_ref());
        emit_trie_update(
            visualizer,
            &phase_name,
            Some(step),
            &trie,
            writes,
            reads,
            memory,
        )?;

        let loss_val: f64 = loss.to_scalar::<f32>()? as f64;
        running_loss += loss_val;
        running_count += 1;
        if loss_val < best_loss {
            best_loss = loss_val;
        }

        if step % log_every == 0 || step == steps {
            let avg = running_loss / running_count as f64;
            println!("    [step {}/{}] loss={:.4}", step, steps, avg);
            if let Some(visualizer) = visualizer {
                visualizer.emit_step_metrics(
                    phase_name.clone(),
                    step,
                    steps,
                    avg,
                    None,
                    Some(trie.len()),
                )?;
            }
            running_loss = 0.0;
            running_count = 0;
        }

        if trie.len() > 50_000 {
            trie.reset();
            ring.reset();
        }
    }

    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_completed(phase_name, "best_loss", best_loss)?;
    }

    Ok(best_loss)
}

pub fn train_phase5(
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    pcfg: &Phase5Config,
    ckpt: &CheckpointConfig,
    device: &Device,
    visualizer: Option<&TrieVisualizerRuntime>,
) -> anyhow::Result<f64> {
    println!("=== Phase 5: Coherence Unfreeze ===");
    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_started("phase5")?;
    }

    // 5a: unfreeze mem_attn + write heads
    let prefixes_5a: Vec<String> = vec!["output_heads".to_string(), "confidence_gate".to_string()];
    // Also include block memory cross-attention weights
    let mut p5a = prefixes_5a;
    for i in 0..cfg.n_layers {
        p5a.push(format!("block_{i}.mem_attn"));
        p5a.push(format!("block_{i}.norm3"));
    }
    let loss_5a = train_phase5_sub(
        "a (mem_attn + write heads)",
        agent,
        varmap,
        dataset,
        cfg,
        &p5a,
        pcfg.lr_base,
        pcfg.steps_per_sub,
        pcfg.batch_size,
        pcfg.log_every,
        pcfg.tau,
        pcfg.max_grad_norm,
        ckpt,
        device,
        visualizer,
    )?;

    if should_stop(ckpt) {
        return Ok(loss_5a);
    }

    // 5b: unfreeze context cross-attention
    let mut p5b = p5a.clone();
    for i in 0..cfg.n_layers {
        p5b.push(format!("block_{i}.ctx_attn"));
        p5b.push(format!("block_{i}.norm2"));
    }
    let loss_5b = train_phase5_sub(
        "b (+ context cross-attn)",
        agent,
        varmap,
        dataset,
        cfg,
        &p5b,
        pcfg.lr_base,
        pcfg.steps_per_sub,
        pcfg.batch_size,
        pcfg.log_every,
        pcfg.tau,
        pcfg.max_grad_norm,
        ckpt,
        device,
        visualizer,
    )?;

    if should_stop(ckpt) {
        return Ok(loss_5b);
    }

    // 5c: unfreeze self-attn + FFN at 0.1× LR, AddrNet at 0.01×
    // First: self-attn + FFN at reduced LR
    let mut p5c: Vec<String> = Vec::new();
    for i in 0..cfg.n_layers {
        p5c.push(format!("block_{i}.self_attn"));
        p5c.push(format!("block_{i}.norm1"));
        p5c.push(format!("block_{i}.ffn"));
        p5c.push(format!("block_{i}.norm4"));
    }
    p5c.push("embedding".to_string());
    p5c.push("context_vectors".to_string());
    p5c.push("final_norm".to_string());

    let loss_5c_base = train_phase5_sub(
        "c (self-attn + FFN at 0.1×)",
        agent,
        varmap,
        dataset,
        cfg,
        &p5c,
        pcfg.lr_base * 0.1,
        pcfg.steps_per_sub,
        pcfg.batch_size,
        pcfg.log_every,
        pcfg.tau,
        pcfg.max_grad_norm,
        ckpt,
        device,
        visualizer,
    )?;

    if should_stop(ckpt) {
        return Ok(loss_5c_base);
    }

    // Then: AddrNet at 0.01× LR
    let p5c_addr = vec!["addr_nets".to_string()];
    let loss_5c_addr = train_phase5_sub(
        "c (AddrNet at 0.01×)",
        agent,
        varmap,
        dataset,
        cfg,
        &p5c_addr,
        pcfg.lr_base * pcfg.lr_addr_multiplier,
        pcfg.steps_per_sub / 2,
        pcfg.batch_size,
        pcfg.log_every,
        pcfg.tau,
        pcfg.max_grad_norm,
        ckpt,
        device,
        visualizer,
    )?;

    if let Some(ref dir) = ckpt.checkpoint_dir {
        save_checkpoint(varmap, dir, "phase5_final")?;
    }

    println!(
        "  Phase 5 complete. Losses: 5a={:.4} 5b={:.4} 5c_base={:.4} 5c_addr={:.4}",
        loss_5a, loss_5b, loss_5c_base, loss_5c_addr
    );

    // Final re-encounter evaluation
    let eval = reencounter_eval(
        agent, dataset, cfg, device, pcfg.tau, 16, visualizer, "phase5", None,
    )?;
    println!(
        "  Final eval: pass1={:.4} pass2={:.4} improvement={:.1}%",
        eval.pass1_loss,
        eval.pass2_loss,
        eval.improvement * 100.0
    );

    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_completed("phase5", "best_loss", loss_5b.min(loss_5c_base))?;
    }

    Ok(loss_5b.min(loss_5c_base))
}

// ---------------------------------------------------------------------------
// Quick evaluation
// ---------------------------------------------------------------------------

pub fn evaluate(
    agent: &ZenoAgent,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    device: &Device,
    use_memory: bool,
    visualizer: Option<&TrieVisualizerRuntime>,
) -> anyhow::Result<()> {
    println!("=== Evaluation (memory={}) ===", use_memory);
    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_started("eval")?;
    }

    let n_batches = (dataset.len() / 4).min(50).max(1);
    let mut total_loss = 0.0;

    let trie = ProtoTrie::new(cfg);
    let mut ring = RingBuffer::new(cfg.n_register);

    for _ in 0..n_batches {
        let (input, target) = dataset.get_random_batch(4, device)?;
        let out = agent.forward(&input, &trie, &mut ring, 0.5, use_memory, device)?;
        let loss = compute_lm_loss(&out.logits, &target)?;
        total_loss += loss.to_scalar::<f32>()? as f64;
    }

    let avg_loss = total_loss / n_batches as f64;
    let ppl = avg_loss.exp();
    println!("  Loss: {:.4}", avg_loss);
    println!("  Perplexity: {:.1}", ppl);
    if let Some(visualizer) = visualizer {
        visualizer.emit_evaluation_summary(use_memory, avg_loss, ppl)?;
    }

    if use_memory {
        println!("\n  Re-encounter test:");
        let eval = reencounter_eval(
            agent, dataset, cfg, device, 0.5, 16, visualizer, "eval", None,
        )?;
        println!(
            "    Pass 1 loss: {:.4} (ppl {:.1})",
            eval.pass1_loss,
            eval.pass1_loss.exp()
        );
        println!(
            "    Pass 2 loss: {:.4} (ppl {:.1})",
            eval.pass2_loss,
            eval.pass2_loss.exp()
        );
        println!("    Improvement: {:.1}%", eval.improvement * 100.0);
        if let Some(diag) = eval.memory.as_ref() {
            println!("    Memory: {}", format_memory_diagnostics(diag));
        }
        println!(
            "    Retrieval: {}",
            format_retrieval_metrics(&eval.retrieval)
        );
    }

    if let Some(visualizer) = visualizer {
        visualizer.emit_phase_completed("eval", "loss", avg_loss)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use candle_nn::VarBuilder;

    #[test]
    fn test_filter_vars() -> candle_core::Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
        let _agent = ZenoAgent::new(vb, &cfg)?;

        let all = vm.all_vars().len();
        assert!(all > 0);

        let embed_vars = filter_named_vars(&vm, &["embedding".to_string()]);
        assert!(!embed_vars.is_empty());
        assert!(embed_vars.len() < all);

        let addr_vars = filter_named_vars(&vm, &["addr_nets".to_string()]);
        assert!(!addr_vars.is_empty());

        Ok(())
    }

    #[test]
    fn test_phase4_filter_matches_memory_read_path() -> candle_core::Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
        let agent = ZenoAgent::new(vb, &cfg)?;

        let prefixes = agent.phase4_params();
        let vars = filter_named_vars(&vm, &prefixes);
        assert!(!vars.is_empty());
        assert!(
            vars.iter().all(|(name, _)| {
                name.starts_with("confidence_gate")
                    || name.contains(".norm3.")
                    || name.contains(".mem_attn.")
            }),
            "phase4 should only select confidence gate + memory read path vars"
        );
        assert!(
            vars.iter().all(
                |(name, _)| !name.starts_with("addr_nets") && !name.starts_with("output_heads")
            ),
            "phase4 should exclude addr_nets and output_heads"
        );

        Ok(())
    }

    #[test]
    fn test_compute_lm_loss() -> candle_core::Result<()> {
        let dev = &Device::Cpu;
        let logits = Tensor::randn(0f32, 1.0, (2, 8, 256), dev)?;
        let targets = Tensor::zeros((2, 8), DType::U32, dev)?;
        let loss = compute_lm_loss(&logits, &targets)?;
        assert_eq!(loss.dims(), &[] as &[usize]);
        let val: f32 = loss.to_scalar()?;
        assert!(val > 0.0, "CE loss should be positive");
        Ok(())
    }

    #[test]
    fn test_entropy_loss() -> candle_core::Result<()> {
        let dev = &Device::Cpu;
        // Uniform logits should give high entropy
        let logits = Tensor::zeros((1, 1, 8, 256), DType::F32, dev)?;
        let (loss, entropy) = address_entropy_loss(&[logits])?;
        // log(256) ≈ 5.545 for uniform distribution
        assert!(
            entropy > 5.0,
            "uniform should give high entropy, got {entropy}"
        );
        let loss_val: f32 = loss.to_scalar()?;
        assert!(loss_val < 0.0, "neg entropy loss should be negative");
        Ok(())
    }

    #[test]
    fn test_prefix_match_scores_for_identical_paths() -> candle_core::Result<()> {
        let dev = &Device::Cpu;
        let probs = Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0], (2, 2), dev)?;
        let (prefix, exact) = prefix_match_scores(&probs, &probs)?;
        assert!((prefix.to_scalar::<f32>()? - 1.0).abs() < 1e-6);
        assert!((exact.to_scalar::<f32>()? - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_retrieval_supervision_uses_written_addresses() -> candle_core::Result<()> {
        let dev = &Device::Cpu;
        let cfg = ZenoConfig::default_proto();
        let address = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let training = MemoryTrainingSignals {
            density: Tensor::from_vec(vec![2u32; cfg.trie_depth], (1, 1, 1, cfg.trie_depth), dev)?,
            confidence: Tensor::from_vec(vec![0.5f32], (1, 1, 1), dev)?,
            read_addresses: vec![vec![address.clone()]],
        };
        let writes = TrieWriteBatch {
            writes: vec![TrieWriteRecord {
                slot: 0,
                batch_index: 0,
                address,
                strength: 1.0,
            }],
        };

        let (_density_loss, _retrieval_loss, metrics) =
            retrieval_supervision(&training, &writes, &cfg, dev)?;
        assert_eq!(metrics.exact_match_rate, 1.0);
        assert_eq!(metrics.prefix_overlap_rate, 1.0);
        assert!(metrics.density_target_mean > 0.0);
        Ok(())
    }

    #[test]
    fn test_read_batch_from_training_uses_last_token_per_batch() -> candle_core::Result<()> {
        let dev = &Device::Cpu;
        let training = MemoryTrainingSignals {
            density: Tensor::from_vec(
                vec![
                    1u32, 1, 0, 0, 0, 0, 0, 0, // batch 0 token 0 slot 0/1
                    2, 1, 0, 0, 1, 0, 0, 0, // batch 0 token 1 slot 0/1
                    3, 2, 1, 0, 5, 4, 3, 0, // batch 0 token 2 slot 0/1
                    0, 0, 0, 0, 0, 0, 0, 0, // batch 1 token 0 slot 0/1
                    1, 0, 0, 0, 2, 1, 0, 0, // batch 1 token 1 slot 0/1
                    4, 2, 0, 0, 6, 3, 2, 0, // batch 1 token 2 slot 0/1
                ],
                (2, 3, 2, 4),
                dev,
            )?,
            confidence: Tensor::from_vec(
                vec![
                    0.1f32, 0.2, 0.3, 0.4, 0.9, 0.8, //
                    0.5, 0.6, 0.7, 0.8, 0.4, 0.3,
                ],
                (2, 3, 2),
                dev,
            )?,
            read_addresses: vec![
                vec![
                    vec![0x10],
                    vec![0x11],
                    vec![0xaa, 0xbb],
                    vec![0x20],
                    vec![0x21],
                    vec![0xcc],
                ],
                vec![
                    vec![0x30],
                    vec![0x31],
                    vec![0xdd, 0xee, 0xff],
                    vec![0x40],
                    vec![0x41],
                    vec![0x99, 0x88],
                ],
            ],
        };

        let reads = read_batch_from_training(Some(&training))?;
        assert_eq!(reads.len(), 4);
        assert_eq!(reads.reads[0].address, vec![0xaa, 0xbb]);
        assert_eq!(reads.reads[0].token_index, 2);
        assert_eq!(reads.reads[0].density_chain, vec![3, 2]);
        assert!(reads.reads[0].hit);
        assert_eq!(reads.reads[1].address, vec![0xcc]);
        assert_eq!(reads.reads[1].density_chain, vec![4]);
        assert!(reads.reads[1].hit);
        assert_eq!(reads.reads[2].density_chain, vec![5, 4, 3]);
        assert_eq!(reads.reads[3].density_chain, vec![6, 3]);
        Ok(())
    }
}
