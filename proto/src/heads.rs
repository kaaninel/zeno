use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::ZenoConfig;

// ---------------------------------------------------------------------------
// LM Head — weight-tied with embedding (free function, not a module)
// ---------------------------------------------------------------------------

/// Computes logits by projecting hidden states through the transposed embedding
/// weight matrix: `logits = hidden @ embed_weight^T`.
pub fn lm_head(hidden: &Tensor, embed_weight: &Tensor) -> Result<Tensor> {
    // hidden:       [batch, seq_len, d_model]
    // embed_weight: [vocab_size, d_model]
    // result:       [batch, seq_len, vocab_size]
    hidden.matmul(&embed_weight.t()?)
}

// ---------------------------------------------------------------------------
// VProj — base value projection
// ---------------------------------------------------------------------------

pub struct VProj {
    proj: Linear,
}

impl VProj {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let proj = linear(cfg.d_model, cfg.d_model, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        self.proj.forward(hidden)
    }
}

// ---------------------------------------------------------------------------
// AspectHead — two-layer MLP producing a channel-specific residual
// ---------------------------------------------------------------------------

pub struct AspectHead {
    lin1: Linear,
    lin2: Linear,
}

impl AspectHead {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let lin1 = linear(cfg.d_model, cfg.aspect_bottleneck, vb.pp("lin1"))?;
        let lin2 = linear(cfg.aspect_bottleneck, cfg.d_model, vb.pp("lin2"))?;
        Ok(Self { lin1, lin2 })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let x = self.lin1.forward(hidden)?.relu()?;
        self.lin2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// AspectHeadBank — holds n_addr_nets (3) AspectHead instances
// ---------------------------------------------------------------------------

pub struct AspectHeadBank {
    heads: Vec<AspectHead>,
}

impl AspectHeadBank {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let mut heads = Vec::with_capacity(cfg.n_addr_nets);
        for i in 0..cfg.n_addr_nets {
            heads.push(AspectHead::new(vb.pp(format!("head_{i}")), cfg)?);
        }
        Ok(Self { heads })
    }

    /// Returns one residual tensor per write channel.
    pub fn forward(&self, hidden: &Tensor) -> Result<Vec<Tensor>> {
        self.heads.iter().map(|h| h.forward(hidden)).collect()
    }
}

// ---------------------------------------------------------------------------
// WriteStrengthHead — scalar α ∈ (0,1) per token
// ---------------------------------------------------------------------------

pub struct WriteStrengthHead {
    proj: Linear,
}

impl WriteStrengthHead {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let proj = linear(cfg.d_model, 1, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let logit = self.proj.forward(hidden)?;
        candle_nn::ops::sigmoid(&logit)
    }
}

// ---------------------------------------------------------------------------
// OutputHeads — bundles VProj + AspectHeadBank + WriteStrengthHead
// ---------------------------------------------------------------------------

pub struct OutputHeadResults {
    /// 3 tensors, each [batch, seq_len, d_model]
    pub write_values: Vec<Tensor>,
    /// [batch, seq_len, 1]
    pub write_strength: Tensor,
}

pub struct OutputHeads {
    pub v_proj: VProj,
    pub aspects: AspectHeadBank,
    pub write_strength: WriteStrengthHead,
}

impl OutputHeads {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let v_proj = VProj::new(vb.pp("v_proj"), cfg)?;
        let aspects = AspectHeadBank::new(vb.pp("aspects"), cfg)?;
        let write_strength = WriteStrengthHead::new(vb.pp("write_strength"), cfg)?;
        Ok(Self {
            v_proj,
            aspects,
            write_strength,
        })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<OutputHeadResults> {
        let base_value = self.v_proj.forward(hidden)?;
        let aspect_residuals = self.aspects.forward(hidden)?;

        // write_values[i] = base_value + aspect_residual[i]
        let write_values = aspect_residuals
            .iter()
            .map(|r| base_value.add(r))
            .collect::<Result<Vec<_>>>()?;

        let write_strength = self.write_strength.forward(hidden)?;

        Ok(OutputHeadResults {
            write_values,
            write_strength,
        })
    }
}
