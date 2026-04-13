use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::config::ZenoConfig;
use crate::data::ByteDataset;
use crate::model::{ForwardOutput, RingBuffer, ZenoAgent};
use crate::trie::ProtoTrie;

/// Filter VarMap variables by name prefixes, returning only matching Vars.
fn filter_vars(varmap: &VarMap, prefixes: &[String]) -> Vec<candle_core::Var> {
    let data = varmap.data().lock().unwrap();
    data.iter()
        .filter(|(name, _)| prefixes.iter().any(|p| name.starts_with(p)))
        .map(|(_, var)| var.clone())
        .collect()
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
}

impl Default for Phase2Config {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            steps: 10_000,
            batch_size: 4,
            log_every: 100,
            gate_loss: 3.0,
        }
    }
}

pub fn train_phase2(
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    pcfg: &Phase2Config,
    device: &Device,
) -> anyhow::Result<f64> {
    println!("=== Phase 2: Base Language Model ===");
    println!("  lr={}, steps={}, batch={}", pcfg.lr, pcfg.steps, pcfg.batch_size);
    println!("  Gate: loss < {:.1}", pcfg.gate_loss);

    let prefixes = agent.phase2_params();
    let vars = filter_vars(varmap, &prefixes);
    println!("  Training {} vars (of {} total)", vars.len(), varmap.all_vars().len());

    let params = ParamsAdamW {
        lr: pcfg.lr,
        weight_decay: 0.01,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    let trie = ProtoTrie::new(cfg);
    let ring = RingBuffer::new(cfg.n_register);
    let mut running_loss = 0.0;
    let mut running_count = 0usize;
    let mut best_loss = f64::MAX;

    for step in 1..=pcfg.steps {
        let (input, target) = dataset.get_random_batch(pcfg.batch_size, device)?;

        let out: ForwardOutput = agent.forward(&input, &trie, &ring, 1.0, false, device)?;
        let loss = compute_lm_loss(&out.logits, &target)?;

        opt.backward_step(&loss)?;

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
                if avg < pcfg.gate_loss { " ✓ GATE" } else { "" }
            );
            running_loss = 0.0;
            running_count = 0;
        }
    }

    println!("  Phase 2 complete. Best loss={:.4}", best_loss);
    if best_loss < pcfg.gate_loss {
        println!("  ✓ Gate PASSED (loss {:.4} < {:.1})", best_loss, pcfg.gate_loss);
    } else {
        println!("  ✗ Gate FAILED (loss {:.4} >= {:.1})", best_loss, pcfg.gate_loss);
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
        }
    }
}

/// Compute address entropy loss (want high entropy = diverse addresses).
///
/// Takes softmax logits [batch, 1, trie_depth, trie_arity], computes
/// per-level entropy, returns negative mean entropy (minimize → maximize entropy).
fn address_entropy_loss(logits_list: &[Tensor]) -> candle_core::Result<(Tensor, f64)> {
    let mut total_entropy = 0.0;
    let mut count = 0usize;
    let mut loss_terms = Vec::new();

    for logits in logits_list {
        // logits: [batch, 1, depth, arity]
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

    let avg_entropy = if count > 0 { total_entropy / count as f64 } else { 0.0 };
    Ok((loss, avg_entropy))
}

/// Diversity loss: penalize if different AddrNets produce the same addresses.
fn address_diversity_loss(addr_logits: &[Tensor]) -> candle_core::Result<Tensor> {
    if addr_logits.len() < 2 {
        return Tensor::zeros((), DType::F32, addr_logits[0].device());
    }

    let mut loss_sum = Tensor::zeros((), DType::F32, addr_logits[0].device())?;
    let mut pair_count = 0;

    for i in 0..addr_logits.len() {
        for j in (i + 1)..addr_logits.len() {
            // Cosine similarity between address distributions
            let a = candle_nn::ops::softmax(&addr_logits[i], candle_core::D::Minus1)?;
            let b = candle_nn::ops::softmax(&addr_logits[j], candle_core::D::Minus1)?;
            let sim = (&a * &b)?.sum_all()?;
            loss_sum = (loss_sum + sim)?;
            pair_count += 1;
        }
    }

    if pair_count > 0 {
        loss_sum / pair_count as f64
    } else {
        Ok(loss_sum)
    }
}

pub fn train_phase3(
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    pcfg: &Phase3Config,
    device: &Device,
) -> anyhow::Result<f64> {
    println!("=== Phase 3: AddrNet Training ===");
    println!("  lr={}, steps={}, τ: {}→{}", pcfg.lr, pcfg.steps, pcfg.tau_start, pcfg.tau_end);
    println!("  Gate: address entropy > {:.1}", pcfg.gate_entropy);

    let prefixes = agent.phase3_params();
    let vars = filter_vars(varmap, &prefixes);
    println!("  Training {} AddrNet vars", vars.len());

    let params = ParamsAdamW {
        lr: pcfg.lr,
        weight_decay: 0.0,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    let trie = ProtoTrie::new(cfg);
    let ring = RingBuffer::new(cfg.n_register);
    let mut best_entropy = 0.0;

    for step in 1..=pcfg.steps {
        let progress = step as f64 / pcfg.steps as f64;
        let tau = pcfg.tau_start + (pcfg.tau_end - pcfg.tau_start) * progress;

        let (input, _target) = dataset.get_random_batch(pcfg.batch_size, device)?;

        // Forward with memory enabled to get read_addr_logits
        let out = agent.forward(&input, &trie, &ring, tau, true, device)?;

        // Entropy loss on read addresses
        let (entropy_loss, avg_entropy) = address_entropy_loss(&out.read_addr_logits)?;

        // Diversity loss across the 3 AddrNets (use write addresses)
        let write_logits: Vec<Tensor> = out.write_addresses.iter().map(|(_, l)| l.clone()).collect();
        let div_loss = address_diversity_loss(&write_logits)?;

        // Combined loss: entropy (neg, minimize) + diversity penalty
        let total_loss = (entropy_loss + div_loss)?;

        opt.backward_step(&total_loss)?;

        if avg_entropy > best_entropy {
            best_entropy = avg_entropy;
        }

        if step % pcfg.log_every == 0 || step == pcfg.steps {
            let loss_val: f32 = total_loss.to_scalar()?;
            println!(
                "  [step {}/{}] loss={:.4} entropy={:.2} τ={:.2}{}",
                step,
                pcfg.steps,
                loss_val,
                avg_entropy,
                tau,
                if avg_entropy > pcfg.gate_entropy { " ✓ GATE" } else { "" }
            );
        }
    }

    println!("  Phase 3 complete. Best entropy={:.2}", best_entropy);
    if best_entropy > pcfg.gate_entropy {
        println!("  ✓ Gate PASSED (entropy {:.2} > {:.1})", best_entropy, pcfg.gate_entropy);
    } else {
        println!("  ✗ Gate FAILED (entropy {:.2} <= {:.1})", best_entropy, pcfg.gate_entropy);
    }

    Ok(best_entropy)
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
    pub gate_retrieval: f64,   // target: > 0.9
    pub eval_every: usize,     // how often to run re-encounter eval
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
        }
    }
}

/// Perform trie writes from a forward pass output.
fn perform_trie_writes(
    out: &ForwardOutput,
    trie: &mut ProtoTrie,
    cfg: &ZenoConfig,
) -> candle_core::Result<usize> {
    let mut write_count = 0;

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
            write_count += 1;
        }
    }

    Ok(write_count)
}

/// Re-encounter evaluation: process same text twice, check if perplexity drops.
fn reencounter_eval(
    agent: &ZenoAgent,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    device: &Device,
    tau: f64,
    n_samples: usize,
) -> anyhow::Result<(f64, f64, f64)> {
    let batches = dataset.create_reencounter_batches(n_samples, device)?;
    if batches.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }

    let mut total_loss_pass1 = 0.0;
    let mut total_loss_pass2 = 0.0;
    let mut count = 0;

    for (first_in, first_tgt, second_in, second_tgt) in &batches {
        let mut trie = ProtoTrie::new(cfg);
        let ring = RingBuffer::new(cfg.n_register);

        // Pass 1: process and write to trie
        let out1 = agent.forward(first_in, &trie, &ring, tau, true, device)?;
        let loss1 = compute_lm_loss(&out1.logits, first_tgt)?;
        total_loss_pass1 += loss1.to_scalar::<f32>()? as f64;

        // Write to trie from pass 1
        perform_trie_writes(&out1, &mut trie, cfg)?;

        // Pass 2: process same content with trie populated
        let out2 = agent.forward(second_in, &trie, &ring, tau, true, device)?;
        let loss2 = compute_lm_loss(&out2.logits, second_tgt)?;
        total_loss_pass2 += loss2.to_scalar::<f32>()? as f64;

        count += 1;
    }

    let avg_loss1 = total_loss_pass1 / count as f64;
    let avg_loss2 = total_loss_pass2 / count as f64;
    let improvement = (avg_loss1 - avg_loss2) / avg_loss1;

    Ok((avg_loss1, avg_loss2, improvement))
}

pub fn train_phase4(
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    pcfg: &Phase4Config,
    device: &Device,
) -> anyhow::Result<f64> {
    println!("=== Phase 4: Memory Integration ===");
    println!("  lr={}, steps={}, τ={}", pcfg.lr, pcfg.steps, pcfg.tau);
    println!("  Gate: re-encounter improvement > {:.0}%", pcfg.gate_retrieval * 100.0);

    let prefixes = agent.phase4_params();
    let vars = filter_vars(varmap, &prefixes);
    println!("  Training {} memory vars", vars.len());

    let params = ParamsAdamW {
        lr: pcfg.lr,
        weight_decay: 0.01,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    let mut trie = ProtoTrie::new(cfg);
    let ring = RingBuffer::new(cfg.n_register);
    let mut running_loss = 0.0;
    let mut running_count = 0usize;
    let mut best_improvement = 0.0;

    for step in 1..=pcfg.steps {
        let (input, target) = dataset.get_random_batch(pcfg.batch_size, device)?;

        // Forward with memory
        let out = agent.forward(&input, &trie, &ring, pcfg.tau, true, device)?;
        let loss = compute_lm_loss(&out.logits, &target)?;

        opt.backward_step(&loss)?;

        // Perform trie writes (after gradient step, detached)
        perform_trie_writes(&out, &mut trie, cfg)?;

        let loss_val: f64 = loss.to_scalar::<f32>()? as f64;
        running_loss += loss_val;
        running_count += 1;

        if step % pcfg.log_every == 0 || step == pcfg.steps {
            let avg = running_loss / running_count as f64;
            println!(
                "  [step {}/{}] loss={:.4} trie_entries={}",
                step,
                pcfg.steps,
                avg,
                trie.len()
            );
            running_loss = 0.0;
            running_count = 0;
        }

        // Periodic re-encounter evaluation
        if step % pcfg.eval_every == 0 {
            let (loss1, loss2, improvement) =
                reencounter_eval(agent, dataset, cfg, device, pcfg.tau, 8)?;
            println!(
                "  [EVAL step {}] pass1_loss={:.4} pass2_loss={:.4} improvement={:.1}%{}",
                step,
                loss1,
                loss2,
                improvement * 100.0,
                if improvement > pcfg.gate_retrieval { " ✓ GATE" } else { "" }
            );
            if improvement > best_improvement {
                best_improvement = improvement;
            }
        }

        // Reset trie periodically to prevent unbounded growth
        if trie.len() > 50_000 {
            trie.reset();
        }
    }

    println!("  Phase 4 complete. Best improvement={:.1}%", best_improvement * 100.0);
    if best_improvement > pcfg.gate_retrieval {
        println!("  ✓ Gate PASSED");
    } else {
        println!("  ✗ Gate FAILED (best {:.1}% < {:.0}%)", best_improvement * 100.0, pcfg.gate_retrieval * 100.0);
    }

    Ok(best_improvement)
}

// ---------------------------------------------------------------------------
// Phase 5 — Coherence Unfreeze (progressive)
// ---------------------------------------------------------------------------

pub struct Phase5Config {
    pub lr_base: f64,
    pub lr_addr_multiplier: f64,  // 0.01× for AddrNet
    pub steps_per_sub: usize,
    pub batch_size: usize,
    pub log_every: usize,
    pub tau: f64,
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
    device: &Device,
) -> anyhow::Result<f64> {
    println!("  --- Phase 5{} ---", label);
    let vars = filter_vars(varmap, prefixes);
    println!("    Training {} vars at lr={:.1e}", vars.len(), lr);

    let params = ParamsAdamW {
        lr,
        weight_decay: 0.01,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    let mut trie = ProtoTrie::new(cfg);
    let ring = RingBuffer::new(cfg.n_register);
    let mut running_loss = 0.0;
    let mut running_count = 0usize;
    let mut best_loss = f64::MAX;

    for step in 1..=steps {
        let (input, target) = dataset.get_random_batch(batch_size, device)?;
        let out = agent.forward(&input, &trie, &ring, tau, true, device)?;
        let loss = compute_lm_loss(&out.logits, &target)?;

        opt.backward_step(&loss)?;
        perform_trie_writes(&out, &mut trie, cfg)?;

        let loss_val: f64 = loss.to_scalar::<f32>()? as f64;
        running_loss += loss_val;
        running_count += 1;
        if loss_val < best_loss {
            best_loss = loss_val;
        }

        if step % log_every == 0 || step == steps {
            let avg = running_loss / running_count as f64;
            println!("    [step {}/{}] loss={:.4}", step, steps, avg);
            running_loss = 0.0;
            running_count = 0;
        }

        if trie.len() > 50_000 {
            trie.reset();
        }
    }

    Ok(best_loss)
}

pub fn train_phase5(
    agent: &ZenoAgent,
    varmap: &VarMap,
    dataset: &ByteDataset,
    cfg: &ZenoConfig,
    pcfg: &Phase5Config,
    device: &Device,
) -> anyhow::Result<f64> {
    println!("=== Phase 5: Coherence Unfreeze ===");

    // 5a: unfreeze mem_attn + write heads
    let prefixes_5a: Vec<String> = vec![
        "output_heads".to_string(),
        "confidence_gate".to_string(),
    ];
    // Also include block memory cross-attention weights
    let mut p5a = prefixes_5a;
    for i in 0..cfg.n_layers {
        p5a.push(format!("block_{i}.mem_attn"));
        p5a.push(format!("block_{i}.norm3"));
    }
    let loss_5a = train_phase5_sub(
        "a (mem_attn + write heads)",
        agent, varmap, dataset, cfg,
        &p5a, pcfg.lr_base, pcfg.steps_per_sub,
        pcfg.batch_size, pcfg.log_every, pcfg.tau, device,
    )?;

    // 5b: unfreeze context cross-attention
    let mut p5b = p5a.clone();
    for i in 0..cfg.n_layers {
        p5b.push(format!("block_{i}.ctx_attn"));
        p5b.push(format!("block_{i}.norm2"));
    }
    let loss_5b = train_phase5_sub(
        "b (+ context cross-attn)",
        agent, varmap, dataset, cfg,
        &p5b, pcfg.lr_base, pcfg.steps_per_sub,
        pcfg.batch_size, pcfg.log_every, pcfg.tau, device,
    )?;

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
        agent, varmap, dataset, cfg,
        &p5c, pcfg.lr_base * 0.1, pcfg.steps_per_sub,
        pcfg.batch_size, pcfg.log_every, pcfg.tau, device,
    )?;

    // Then: AddrNet at 0.01× LR
    let p5c_addr = vec!["addr_nets".to_string()];
    let loss_5c_addr = train_phase5_sub(
        "c (AddrNet at 0.01×)",
        agent, varmap, dataset, cfg,
        &p5c_addr, pcfg.lr_base * pcfg.lr_addr_multiplier, pcfg.steps_per_sub / 2,
        pcfg.batch_size, pcfg.log_every, pcfg.tau, device,
    )?;

    println!("  Phase 5 complete. Losses: 5a={:.4} 5b={:.4} 5c_base={:.4} 5c_addr={:.4}",
        loss_5a, loss_5b, loss_5c_base, loss_5c_addr);

    // Final re-encounter evaluation
    let (loss1, loss2, improvement) =
        reencounter_eval(agent, dataset, cfg, device, pcfg.tau, 16)?;
    println!(
        "  Final eval: pass1={:.4} pass2={:.4} improvement={:.1}%",
        loss1, loss2, improvement * 100.0
    );

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
) -> anyhow::Result<()> {
    println!("=== Evaluation (memory={}) ===", use_memory);

    let n_batches = (dataset.len() / 4).min(50).max(1);
    let mut total_loss = 0.0;

    let trie = ProtoTrie::new(cfg);
    let ring = RingBuffer::new(cfg.n_register);

    for _ in 0..n_batches {
        let (input, target) = dataset.get_random_batch(4, device)?;
        let out = agent.forward(&input, &trie, &ring, 0.5, use_memory, device)?;
        let loss = compute_lm_loss(&out.logits, &target)?;
        total_loss += loss.to_scalar::<f32>()? as f64;
    }

    let avg_loss = total_loss / n_batches as f64;
    let ppl = avg_loss.exp();
    println!("  Loss: {:.4}", avg_loss);
    println!("  Perplexity: {:.1}", ppl);

    if use_memory {
        println!("\n  Re-encounter test:");
        let (loss1, loss2, improvement) =
            reencounter_eval(agent, dataset, cfg, device, 0.5, 16)?;
        println!("    Pass 1 loss: {:.4} (ppl {:.1})", loss1, loss1.exp());
        println!("    Pass 2 loss: {:.4} (ppl {:.1})", loss2, loss2.exp());
        println!("    Improvement: {:.1}%", improvement * 100.0);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let embed_vars = filter_vars(&vm, &["embedding".to_string()]);
        assert!(!embed_vars.is_empty());
        assert!(embed_vars.len() < all);

        let addr_vars = filter_vars(&vm, &["addr_nets".to_string()]);
        assert!(!addr_vars.is_empty());

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
        assert!(entropy > 5.0, "uniform should give high entropy, got {entropy}");
        let loss_val: f32 = loss.to_scalar()?;
        assert!(loss_val < 0.0, "neg entropy loss should be negative");
        Ok(())
    }
}