use candle_core::{DType, Result, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

use crate::config::ZenoConfig;

/// Converts trie structural density chains into scalar confidence values.
///
/// Dense paths → high confidence (trust memory).
/// Sparse paths → low confidence (suppress memory).
///
/// ~2,129 parameters total.
pub struct ConfidenceGate {
    density_embedding: Embedding,
    depth_embedding: Embedding,
    w_gate: Linear,
    /// Pre-built depth indices [0, 1, …, trie_depth-1].
    depth_indices: Tensor,
    trie_depth: usize,
}

impl ConfidenceGate {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let e = cfg.confidence_embed_dim; // 8

        let density_embedding = embedding(cfg.trie_arity, e, vb.pp("density_embedding"))?;
        let depth_embedding = embedding(cfg.trie_depth, e, vb.pp("depth_embedding"))?;
        let w_gate = linear(e * 2, 1, vb.pp("w_gate"))?;

        let depth_indices = Tensor::arange(0u32, cfg.trie_depth as u32, vb.device())?;

        Ok(Self {
            density_embedding,
            depth_embedding,
            w_gate,
            depth_indices,
            trie_depth: cfg.trie_depth,
        })
    }

    /// Compute per-slot confidence scalars from density chains.
    ///
    /// # Arguments
    /// * `density_chain` — `[batch, n_mem_slots, trie_depth]` integer density counts
    ///
    /// # Returns
    /// `[batch, n_mem_slots]` confidence scalars in (0, 1)
    pub fn forward(&self, density_chain: &Tensor) -> Result<Tensor> {
        let (_batch, _n_slots, _depth) = density_chain.dims3()?;

        // Clamp density counts to [0, 255] for safe embedding lookup.
        let clamped = density_chain.clamp(0u32, 255u32)?.to_dtype(DType::U32)?;

        // Density embeddings: [batch, n_slots, trie_depth] → [batch, n_slots, trie_depth, e]
        let d_embed = self.density_embedding.forward(&clamped)?;

        // Depth embeddings: [trie_depth] → [trie_depth, e], broadcast to [1, 1, trie_depth, e]
        let l_embed = self
            .depth_embedding
            .forward(&self.depth_indices)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(d_embed.shape())?;

        // Concatenate along last dim: [batch, n_slots, trie_depth, 2*e]
        let combined = Tensor::cat(&[&d_embed, &l_embed], 3)?;

        // Mean over depth levels: [batch, n_slots, 2*e]
        let mean_combined = (combined.sum(2)? / self.trie_depth as f64)?;

        // Project to scalar and sigmoid: [batch, n_slots, 1] → [batch, n_slots]
        let logits = self.w_gate.forward(&mean_combined)?.squeeze(2)?;
        candle_nn::ops::sigmoid(&logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_confidence_gate_shape() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let gate = ConfidenceGate::new(vb, &cfg)?;

        let batch = 2;
        let density = Tensor::zeros((batch, cfg.n_mem_slots, cfg.trie_depth), DType::U32, dev)?;

        let conf = gate.forward(&density)?;
        assert_eq!(conf.dims(), &[batch, cfg.n_mem_slots]);

        // All-zero density → should still produce valid (0,1) values.
        let vals = conf.to_vec2::<f32>()?;
        for row in &vals {
            for &v in row {
                assert!(v > 0.0 && v < 1.0, "confidence {v} not in (0,1)");
            }
        }
        Ok(())
    }

    #[test]
    fn test_param_count() -> Result<()> {
        let cfg = ZenoConfig::default_proto();
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let _gate = ConfidenceGate::new(vb, &cfg)?;

        let total: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
        assert_eq!(total, 2_129);
        Ok(())
    }
}
