use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use crate::config::ZenoConfig;

// ---------------------------------------------------------------------------
// RoPE helpers
// ---------------------------------------------------------------------------

/// Build interleaved cos/sin tables for Rotary Position Embeddings.
///
/// Returns `(cos, sin)` each of shape `[max_len, head_dim/2]`.
fn build_rope_tables(max_len: usize, head_dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;

    // θ_i = 1 / 10000^(2i / head_dim)  for i in 0..half
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000_f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?; // [1, half]

    let positions: Vec<f32> = (0..max_len).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (max_len, 1), device)?; // [max_len, 1]

    // angles: [max_len, half]
    let angles = positions.matmul(&inv_freq)?;
    let cos = angles.cos()?;
    let sin = angles.sin()?;
    Ok((cos, sin))
}

/// Apply interleaved RoPE to a tensor of shape `[batch, n_heads, seq_len, head_dim]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, pos_offset: usize) -> Result<Tensor> {
    let seq_len = x.dim(2)?;
    let cos = cos.narrow(0, pos_offset, seq_len)?;
    let sin = sin.narrow(0, pos_offset, seq_len)?;

    let x = x.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    candle_nn::rotary_emb::rope_i(&x, &cos, &sin)
}

// ---------------------------------------------------------------------------
// MultiHeadSelfAttention
// ---------------------------------------------------------------------------

/// Causal multi-head self-attention with Rotary Position Embeddings.
///
/// Q, K, V projections are bias-free; the output projection has a bias.
pub struct MultiHeadSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope_cos: Tensor,
    rope_sin: Tensor,
    n_heads: usize,
    head_dim: usize,
}

impl MultiHeadSelfAttention {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let d = cfg.d_model;
        let q_proj = linear_no_bias(d, d, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(d, d, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(d, d, vb.pp("v_proj"))?;
        let o_proj = linear(d, d, vb.pp("o_proj"))?;

        // Pre-compute RoPE cos/sin tables up to 2× context window for generation.
        let max_len = cfg.context_window * 2;
        let (rope_cos, rope_sin) = build_rope_tables(max_len, cfg.head_dim(), vb.device())?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope_cos,
            rope_sin,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` — `[batch, seq_len, d_model]`
    /// * `pos_offset` — starting position index (for generation beyond training)
    ///
    /// # Returns
    /// `[batch, seq_len, d_model]`  (residual added internally)
    pub fn forward(&self, x: &Tensor, pos_offset: usize) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let h = self.n_heads;
        let hd = self.head_dim;

        // Project: [b, t, d] → [b, t, d]
        let q = x.apply(&self.q_proj)?;
        let k = x.apply(&self.k_proj)?;
        let v = x.apply(&self.v_proj)?;

        // Reshape to [b, t, h, hd] then transpose to [b, h, t, hd]
        let q = q.reshape((b, t, h, hd))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, t, h, hd))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, t, h, hd))?.transpose(1, 2)?;

        // Apply RoPE to Q and K
        let q = apply_rope(&q, &self.rope_cos, &self.rope_sin, pos_offset)?;
        let k = apply_rope(&k, &self.rope_cos, &self.rope_sin, pos_offset)?;

        // Scaled dot-product attention: [b, h, t, t]
        let scale = (hd as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

        // Causal mask: fill upper triangle with -inf
        let mask = build_causal_mask(t, x.dtype(), x.device())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        // Softmax over last dim
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Weighted sum: [b, h, t, hd]
        let out = attn_weights.matmul(&v.contiguous()?)?;

        // Reshape back: [b, h, t, hd] → [b, t, h, hd] → [b, t, d]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, h * hd))?;

        // Output projection + residual
        let out = out.apply(&self.o_proj)?;
        out + x
    }
}

/// Build a causal (upper-triangular) mask of shape `[1, 1, t, t]`.
///
/// Allowed positions are 0.0; masked positions are -inf.
fn build_causal_mask(t: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..t)
        .flat_map(|row| {
            (0..t).map(move |col| {
                if col <= row {
                    0.0_f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, t, t), device)?.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// CrossAttention
// ---------------------------------------------------------------------------

/// Multi-head cross-attention (no causal mask, no RoPE).
///
/// Used for both context cross-attention (14 slots) and memory cross-attention
/// (3+ slots).  Queries come from the hidden state; keys and values come from
/// an external pool.
pub struct CrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let d = cfg.d_model;
        let q_proj = linear_no_bias(d, d, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(d, d, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(d, d, vb.pp("v_proj"))?;
        let o_proj = linear(d, d, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `q_input` — `[batch, seq_len, d_model]` (hidden states)
    /// * `kv_input` — `[batch, n_slots, d_model]` (external pool)
    ///
    /// # Returns
    /// `[batch, seq_len, d_model]`
    pub fn forward(&self, q_input: &Tensor, kv_input: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = q_input.dims3()?;
        let s = kv_input.dim(1)?;
        let h = self.n_heads;
        let hd = self.head_dim;

        let q = q_input.apply(&self.q_proj)?;
        let k = kv_input.apply(&self.k_proj)?;
        let v = kv_input.apply(&self.v_proj)?;

        // [b, *, h, hd] → [b, h, *, hd]
        let q = q.reshape((b, t, h, hd))?.transpose(1, 2)?;
        let k = k.reshape((b, s, h, hd))?.transpose(1, 2)?;
        let v = v.reshape((b, s, h, hd))?.transpose(1, 2)?;

        let scale = (hd as f64).sqrt();
        let attn_weights = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let out = attn_weights.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, h * hd))?;
        out.apply(&self.o_proj)
    }

    /// Forward pass with token-local key/value slots.
    ///
    /// Flattens `[batch, seq_len]` into an expanded batch dimension so each token
    /// attends only over its own memory slots.
    pub fn forward_per_token(&self, q_input: &Tensor, kv_input: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, d_model) = q_input.dims3()?;
        let (kv_batch, kv_seq_len, n_slots, kv_d_model) = kv_input.dims4()?;
        debug_assert_eq!(batch, kv_batch);
        debug_assert_eq!(seq_len, kv_seq_len);
        debug_assert_eq!(d_model, kv_d_model);

        let q_flat = q_input.reshape((batch * seq_len, 1, d_model))?;
        let kv_flat = kv_input.reshape((batch * seq_len, n_slots, d_model))?;
        self.forward(&q_flat, &kv_flat)?
            .reshape((batch, seq_len, d_model))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_self_attention_shape() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let attn = MultiHeadSelfAttention::new(vb, &cfg)?;

        let batch = 2;
        let seq = 16;
        let x = Tensor::randn(0f32, 1.0, (batch, seq, cfg.d_model), dev)?;
        let out = attn.forward(&x, 0)?;
        assert_eq!(out.dims(), &[batch, seq, cfg.d_model]);
        Ok(())
    }

    #[test]
    fn test_self_attention_pos_offset() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let attn = MultiHeadSelfAttention::new(vb, &cfg)?;

        let x = Tensor::randn(0f32, 1.0, (1, 8, cfg.d_model), dev)?;
        let out = attn.forward(&x, 128)?;
        assert_eq!(out.dims(), &[1, 8, cfg.d_model]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_shape() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let xattn = CrossAttention::new(vb, &cfg)?;

        let batch = 2;
        let seq = 16;
        let n_slots = cfg.context_pool_size(); // 14
        let q = Tensor::randn(0f32, 1.0, (batch, seq, cfg.d_model), dev)?;
        let kv = Tensor::randn(0f32, 1.0, (batch, n_slots, cfg.d_model), dev)?;
        let out = xattn.forward(&q, &kv)?;
        assert_eq!(out.dims(), &[batch, seq, cfg.d_model]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_mem_slots() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let xattn = CrossAttention::new(vb, &cfg)?;

        let q = Tensor::randn(0f32, 1.0, (1, 8, cfg.d_model), dev)?;
        let kv = Tensor::randn(0f32, 1.0, (1, cfg.n_mem_slots, cfg.d_model), dev)?;
        let out = xattn.forward(&q, &kv)?;
        assert_eq!(out.dims(), &[1, 8, cfg.d_model]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_per_token_slots() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let xattn = CrossAttention::new(vb, &cfg)?;

        let q = Tensor::randn(0f32, 1.0, (2, 8, cfg.d_model), dev)?;
        let kv = Tensor::randn(0f32, 1.0, (2, 8, cfg.n_mem_slots, cfg.d_model), dev)?;
        let out = xattn.forward_per_token(&q, &kv)?;
        assert_eq!(out.dims(), &[2, 8, cfg.d_model]);
        Ok(())
    }

    #[test]
    fn test_self_attention_is_causal() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let attn = MultiHeadSelfAttention::new(vb, &cfg)?;

        // Changing a later token should not affect earlier positions.
        let x1 = Tensor::randn(0f32, 1.0, (1, 4, cfg.d_model), dev)?;
        let mut x2_data = x1.to_vec3::<f32>()?;
        x2_data[0][3] = vec![99.0; cfg.d_model]; // mutate last token
        let x2 = Tensor::new(x2_data, dev)?;

        let o1 = attn.forward(&x1, 0)?;
        let o2 = attn.forward(&x2, 0)?;

        // First 3 positions should be identical.
        let diff = (o1.narrow(1, 0, 3)? - o2.narrow(1, 0, 3))?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "causal violation: diff = {diff}");
        Ok(())
    }

    #[test]
    fn test_self_attention_param_count() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let _attn = MultiHeadSelfAttention::new(vb, &cfg)?;

        // Q,K,V: 96*96=9216 each (no bias) = 27648
        // O: 96*96 + 96 = 9312
        // Total = 36960
        let total: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
        assert_eq!(total, 36_960);
        Ok(())
    }

    #[test]
    fn test_cross_attention_param_count() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let _xattn = CrossAttention::new(vb, &cfg)?;

        // Same structure as self-attention projections
        let total: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
        assert_eq!(total, 36_960);
        Ok(())
    }
}
