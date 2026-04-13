use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::addrnet::AddrNetBank;
use crate::attention::{CrossAttention, MultiHeadSelfAttention};
use crate::confidence::ConfidenceGate;
use crate::config::ZenoConfig;
use crate::heads::{lm_head, OutputHeadResults, OutputHeads};
use crate::trie::ProtoTrie;

// ---------------------------------------------------------------------------
// TransformerBlock — one block with 4 sublayers
// ---------------------------------------------------------------------------

pub struct TransformerBlock {
    norm1: RmsNorm,
    self_attn: MultiHeadSelfAttention,
    norm2: RmsNorm,
    ctx_attn: CrossAttention,
    norm3: RmsNorm,
    mem_attn: CrossAttention,
    norm4: RmsNorm,
    ffn_up: Linear,
    ffn_down: Linear,
}

impl TransformerBlock {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let eps = 1e-6;
        Ok(Self {
            norm1: rms_norm(cfg.d_model, eps, vb.pp("norm1"))?,
            self_attn: MultiHeadSelfAttention::new(vb.pp("self_attn"), cfg)?,
            norm2: rms_norm(cfg.d_model, eps, vb.pp("norm2"))?,
            ctx_attn: CrossAttention::new(vb.pp("ctx_attn"), cfg)?,
            norm3: rms_norm(cfg.d_model, eps, vb.pp("norm3"))?,
            mem_attn: CrossAttention::new(vb.pp("mem_attn"), cfg)?,
            norm4: rms_norm(cfg.d_model, eps, vb.pp("norm4"))?,
            ffn_up: linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("ffn_up"))?,
            ffn_down: linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("ffn_down"))?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        pos_offset: usize,
        context_pool: &Tensor,
        memory_pool: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention with residual (self_attn already adds residual internally)
        let x = self.self_attn.forward(&self.norm1.forward(x)?, pos_offset)?;

        // Context cross-attention with residual
        let x = (&x + self.ctx_attn.forward(&self.norm2.forward(&x)?, context_pool)?)?;

        // Memory cross-attention with residual
        let x = (&x + self.mem_attn.forward(&self.norm3.forward(&x)?, memory_pool)?)?;

        // SiLU FFN with residual
        let ffn_out = self.ffn_up.forward(&self.norm4.forward(&x)?)?.silu()?;
        let ffn_out = self.ffn_down.forward(&ffn_out)?;
        &x + ffn_out
    }
}

// ---------------------------------------------------------------------------
// RingBuffer — FIFO ring for register slots (NOT an nn module)
// ---------------------------------------------------------------------------

pub struct RingBuffer {
    slots: Vec<Tensor>,
    write_idx: usize,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::new(),
            write_idx: 0,
            capacity,
        }
    }

    /// Push a new hidden state (detached), dropping oldest if full.
    pub fn push(&mut self, hidden: &Tensor) -> Result<()> {
        let detached = hidden.detach();
        if self.slots.len() < self.capacity {
            self.slots.push(detached);
        } else {
            self.slots[self.write_idx] = detached;
        }
        self.write_idx = (self.write_idx + 1) % self.capacity;
        Ok(())
    }

    /// Return `[capacity, d_model]` tensor, zero-padding empty slots.
    pub fn to_tensor(&self, device: &Device, d_model: usize) -> Result<Tensor> {
        if self.slots.is_empty() {
            return Tensor::zeros((self.capacity, d_model), DType::F32, device);
        }
        let mut tensors = Vec::with_capacity(self.capacity);
        for slot in &self.slots {
            tensors.push(slot.clone());
        }
        let n_pad = self.capacity - self.slots.len();
        for _ in 0..n_pad {
            tensors.push(Tensor::zeros(d_model, DType::F32, device)?);
        }
        Tensor::stack(&tensors, 0)
    }

    pub fn reset(&mut self) {
        self.slots.clear();
        self.write_idx = 0;
    }
}

// ---------------------------------------------------------------------------
// ForwardOutput
// ---------------------------------------------------------------------------

pub struct ForwardOutput {
    /// LM logits: `[batch, seq_len, vocab_size]`
    pub logits: Tensor,
    /// 3 × (addresses, logits) from write AddrNets
    pub write_addresses: Vec<(Tensor, Tensor)>,
    /// 3 × `[batch, seq_len, d_model]` write values
    pub write_values: Vec<Tensor>,
    /// `[batch, seq_len, 1]` write strength
    pub write_strength: Tensor,
    /// 3 × `[batch, 1, trie_depth, trie_arity]` raw read-address logits (Phase 3 loss)
    pub read_addr_logits: Vec<Tensor>,
}

// ---------------------------------------------------------------------------
// ZenoAgent — the full model
// ---------------------------------------------------------------------------

pub struct ZenoAgent {
    embedding: Embedding,
    /// 10 learnable context vectors: `[n_context_vectors, d_model]`
    context_vectors: Tensor,
    blocks: Vec<TransformerBlock>,
    addr_nets: AddrNetBank,
    output_heads: OutputHeads,
    confidence_gate: ConfidenceGate,
    final_norm: RmsNorm,
    config: ZenoConfig,
}

impl ZenoAgent {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let embedding = embedding(cfg.vocab_size, cfg.d_model, vb.pp("embedding"))?;

        let context_vectors = vb.get_with_hints(
            (cfg.n_context_vectors, cfg.d_model),
            "context_vectors",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(TransformerBlock::new(vb.pp(format!("block_{i}")), cfg)?);
        }

        let addr_nets = AddrNetBank::new(vb.pp("addr_nets"), cfg)?;
        let output_heads = OutputHeads::new(vb.pp("output_heads"), cfg)?;
        let confidence_gate = ConfidenceGate::new(vb.pp("confidence_gate"), cfg)?;
        let final_norm = rms_norm(cfg.d_model, 1e-6, vb.pp("final_norm"))?;

        Ok(Self {
            embedding,
            context_vectors,
            blocks,
            addr_nets,
            output_heads,
            confidence_gate,
            final_norm,
            config: cfg.clone(),
        })
    }

    /// Core forward pass.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        trie: &ProtoTrie,
        ring_buffer: &RingBuffer,
        gumbel_temp: f64,
        use_memory: bool,
        device: &Device,
    ) -> Result<ForwardOutput> {
        let (batch, _seq_len) = input_ids.dims2()?;
        let d = self.config.d_model;
        let n_ctx = self.config.n_context_vectors;
        let n_reg = self.config.n_register;
        let n_mem = self.config.n_mem_slots;

        // 1. Embed input_ids → [batch, seq_len, d_model]
        let embedded = self.embedding.forward(input_ids)?;

        // 2. Build context pool [batch, n_ctx + n_reg, d_model]
        let ctx_learned = self
            .context_vectors
            .unsqueeze(0)?
            .broadcast_as((batch, n_ctx, d))?
            .contiguous()?;

        let reg_tensor = ring_buffer
            .to_tensor(device, d)?
            .unsqueeze(0)?
            .broadcast_as((batch, n_reg, d))?
            .contiguous()?;

        let context_pool = Tensor::cat(&[&ctx_learned, &reg_tensor], 1)?;

        // 3. Memory pool via trie reads (or zeros if memory disabled)
        let (memory_pool, read_addr_logits) = if use_memory {
            self.build_memory_pool(&embedded, trie, gumbel_temp, batch, d, n_mem, device)?
        } else {
            let zeros = Tensor::zeros((batch, n_mem, d), DType::F32, device)?;
            (zeros, Vec::new())
        };

        // 4. Run through transformer blocks
        let mut hidden = embedded;
        for block in &self.blocks {
            hidden = block.forward(&hidden, 0, &context_pool, &memory_pool)?;
        }

        // 5. Final norm
        hidden = self.final_norm.forward(&hidden)?;

        // 6. LM logits via weight-tied head (flatten to 2D for matmul)
        let (b, s, _d) = hidden.dims3()?;
        let logits = lm_head(
            &hidden.reshape((b * s, self.config.d_model))?,
            self.embedding.embeddings(),
        )?
        .reshape((b, s, self.config.vocab_size))?;

        // 7. Write addresses from final hidden
        let write_addresses = self.addr_nets.forward(&hidden, gumbel_temp, true)?;

        // 8. Write values + strength from output heads
        let OutputHeadResults {
            write_values,
            write_strength,
        } = self.output_heads.forward(&hidden)?;

        Ok(ForwardOutput {
            logits,
            write_addresses,
            write_values,
            write_strength,
            read_addr_logits,
        })
    }

    /// Build memory pool from trie reads using mean-pooled address generation.
    ///
    /// Computes one set of 3 read addresses from the mean of all embedded tokens,
    /// reads 3 vectors from the trie, applies confidence gating, and returns
    /// `[batch, n_mem_slots, d_model]`.
    fn build_memory_pool(
        &self,
        embedded: &Tensor,
        trie: &ProtoTrie,
        gumbel_temp: f64,
        batch: usize,
        d: usize,
        n_mem: usize,
        device: &Device,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let trie_depth = self.config.trie_depth;

        // Mean-pool across seq_len → [batch, 1, d_model]
        let mean_emb = embedded.mean(1)?.unsqueeze(1)?;

        // Run AddrNets on the mean embedding: 3 × (addresses, logits)
        let addr_results = self.addr_nets.forward(&mean_emb, gumbel_temp, true)?;

        let read_addr_logits: Vec<Tensor> = addr_results.iter().map(|(_, l)| l.clone()).collect();

        // For each AddrNet output, convert hard one-hot to byte indices and read from trie
        let mut mem_vecs = Vec::with_capacity(n_mem);
        let mut all_density_chains = Vec::with_capacity(batch * n_mem);

        for (addresses, _logits) in &addr_results {
            // addresses: [batch, 1, trie_depth, trie_arity] → argmax → [batch, 1, trie_depth]
            let byte_indices = addresses.argmax(candle_core::D::Minus1)?.squeeze(1)?;

            let mut batch_vecs = Vec::with_capacity(batch);
            for b in 0..batch {
                let addr_row = byte_indices.get(b)?;
                let addr_bytes: Vec<u8> = addr_row
                    .to_vec1::<u32>()?
                    .iter()
                    .map(|&v| v as u8)
                    .collect();

                batch_vecs.push(trie.read(&addr_bytes, device)?);
                all_density_chains.push(trie.density_chain(&addr_bytes));
            }
            mem_vecs.push(Tensor::stack(&batch_vecs, 0)?); // [batch, d_model]
        }

        // Stack → [batch, n_mem, d_model]
        let mem_stacked = Tensor::stack(&mem_vecs, 1)?;

        // Build density chain tensor [batch, n_mem, trie_depth]
        let density_flat: Vec<u32> = all_density_chains
            .iter()
            .flat_map(|chain| {
                let mut padded: Vec<u32> = chain.iter().map(|&v| v as u32).collect();
                padded.resize(trie_depth, 0);
                padded
            })
            .collect();

        // density_chains is ordered [addr0_batch0, addr0_batch1, ..., addr1_batch0, ...]
        // but we need [batch, n_mem, trie_depth] — reshape handles row-major ordering
        // which matches: for each addr_net (n_mem), for each batch item, we have trie_depth values
        // so shape is [n_mem, batch, trie_depth] → transpose to [batch, n_mem, trie_depth]
        let density_tensor =
            Tensor::from_vec(density_flat, (n_mem, batch, trie_depth), device)?
                .transpose(0, 1)?
                .contiguous()?;

        // Confidence gate → [batch, n_mem]
        let confidence = self.confidence_gate.forward(&density_tensor)?;

        // Modulate memory by confidence: broadcast [batch, n_mem] → [batch, n_mem, d]
        let confidence_expanded = confidence.unsqueeze(2)?.broadcast_as((batch, n_mem, d))?;
        let memory_pool = (mem_stacked * confidence_expanded)?;

        Ok((memory_pool, read_addr_logits))
    }

    /// Param name prefixes to train in Phase 2 (base LM, no memory).
    pub fn phase2_params(&self) -> Vec<String> {
        vec![
            "embedding".to_string(),
            "context_vectors".to_string(),
            "block_".to_string(),
            "final_norm".to_string(),
        ]
    }

    /// Param name prefixes to train in Phase 3 (AddrNet only).
    pub fn phase3_params(&self) -> Vec<String> {
        vec!["addr_nets".to_string()]
    }

    /// Param name prefixes to train in Phase 4 (memory integration).
    pub fn phase4_params(&self) -> Vec<String> {
        vec![
            "addr_nets".to_string(),
            "confidence_gate".to_string(),
            "output_heads".to_string(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn test_cfg() -> ZenoConfig {
        ZenoConfig::default_proto()
    }

    #[test]
    fn test_ring_buffer_empty() -> Result<()> {
        let rb = RingBuffer::new(4);
        let t = rb.to_tensor(&Device::Cpu, 96)?;
        assert_eq!(t.dims(), &[4, 96]);
        let sum: f32 = t.sum_all()?.to_scalar()?;
        assert_eq!(sum, 0.0);
        Ok(())
    }

    #[test]
    fn test_ring_buffer_push_and_wrap() -> Result<()> {
        let mut rb = RingBuffer::new(2);
        let v1 = Tensor::ones(96, DType::F32, &Device::Cpu)?;
        let v2 = (Tensor::ones(96, DType::F32, &Device::Cpu)? * 2.0)?;
        let v3 = (Tensor::ones(96, DType::F32, &Device::Cpu)? * 3.0)?;

        rb.push(&v1)?;
        rb.push(&v2)?;
        assert_eq!(rb.slots.len(), 2);

        rb.push(&v3)?;
        // Wrapped: slot[0]=v3, slot[1]=v2
        let t = rb.to_tensor(&Device::Cpu, 96)?;
        assert_eq!(t.dims(), &[2, 96]);
        Ok(())
    }

    #[test]
    fn test_ring_buffer_reset() -> Result<()> {
        let mut rb = RingBuffer::new(4);
        let v = Tensor::ones(96, DType::F32, &Device::Cpu)?;
        rb.push(&v)?;
        assert_eq!(rb.slots.len(), 1);
        rb.reset();
        assert!(rb.slots.is_empty());
        Ok(())
    }

    #[test]
    fn test_transformer_block_shape() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let block = TransformerBlock::new(vb, &cfg)?;

        let batch = 2;
        let seq = 8;
        let x = Tensor::randn(0f32, 1.0, (batch, seq, cfg.d_model), dev)?;
        let ctx = Tensor::randn(0f32, 1.0, (batch, cfg.context_pool_size(), cfg.d_model), dev)?;
        let mem = Tensor::randn(0f32, 1.0, (batch, cfg.n_mem_slots, cfg.d_model), dev)?;

        let out = block.forward(&x, 0, &ctx, &mem)?;
        assert_eq!(out.dims(), &[batch, seq, cfg.d_model]);
        Ok(())
    }

    #[test]
    fn test_zeno_agent_creation() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let _agent = ZenoAgent::new(vb, &cfg)?;

        let total: usize = vm.all_vars().iter().map(|v| v.elem_count()).sum();
        assert!(total > 0, "model should have parameters");
        Ok(())
    }

    #[test]
    fn test_zeno_agent_forward_no_memory() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let agent = ZenoAgent::new(vb, &cfg)?;
        let rb = RingBuffer::new(cfg.n_register);
        let trie = ProtoTrie::new(&cfg);

        let batch = 1;
        let seq = 8;
        let input_ids = Tensor::zeros((batch, seq), DType::U32, dev)?;

        let out = agent.forward(&input_ids, &trie, &rb, 1.0, false, dev)?;

        assert_eq!(out.logits.dims(), &[batch, seq, cfg.vocab_size]);
        assert_eq!(out.write_addresses.len(), cfg.n_addr_nets);
        assert_eq!(out.write_values.len(), cfg.n_addr_nets);
        assert_eq!(out.write_strength.dims(), &[batch, seq, 1]);
        assert!(out.read_addr_logits.is_empty());
        Ok(())
    }

    #[test]
    fn test_zeno_agent_forward_with_memory() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let agent = ZenoAgent::new(vb, &cfg)?;
        let rb = RingBuffer::new(cfg.n_register);
        let trie = ProtoTrie::new(&cfg);

        let batch = 1;
        let seq = 8;
        let input_ids = Tensor::zeros((batch, seq), DType::U32, dev)?;

        let out = agent.forward(&input_ids, &trie, &rb, 1.0, true, dev)?;

        assert_eq!(out.logits.dims(), &[batch, seq, cfg.vocab_size]);
        assert_eq!(out.read_addr_logits.len(), cfg.n_addr_nets);
        for logit in &out.read_addr_logits {
            assert_eq!(logit.dims(), &[batch, 1, cfg.trie_depth, cfg.trie_arity]);
        }
        Ok(())
    }

    #[test]
    fn test_phase_param_prefixes() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let agent = ZenoAgent::new(vb, &cfg)?;

        let p2 = agent.phase2_params();
        assert!(p2.contains(&"embedding".to_string()));
        assert!(p2.contains(&"block_".to_string()));

        let p3 = agent.phase3_params();
        assert!(p3.contains(&"addr_nets".to_string()));

        let p4 = agent.phase4_params();
        assert!(p4.contains(&"confidence_gate".to_string()));
        assert!(p4.contains(&"output_heads".to_string()));
        Ok(())
    }
}
