use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

use crate::config::ZenoConfig;

// ---------------------------------------------------------------------------
// Gumbel-Softmax
// ---------------------------------------------------------------------------

/// Gumbel-Softmax with optional straight-through hard mode.
///
/// `logits` shape: [..., vocab]
/// Returns soft probabilities (or hard one-hot via straight-through estimator).
pub fn gumbel_softmax(logits: &Tensor, temperature: f64, hard: bool) -> Result<Tensor> {
    // g ~ Gumbel(0,1) = -log(-log(U)),  U ~ Uniform(0,1)
    let u = logits.rand_like(1e-7, 1.0 - 1e-7)?;
    let g = u.log()?.neg()?.log()?.neg()?;
    let perturbed = ((logits + g)? / temperature)?;
    let y_soft = candle_nn::ops::softmax(&perturbed, candle_core::D::Minus1)?;

    if !hard {
        return Ok(y_soft);
    }

    // Straight-through: y_hard - y_soft.detach() + y_soft
    let indices = y_soft.argmax(candle_core::D::Minus1)?; // [...,]
    let num_classes = logits.dim(candle_core::D::Minus1)?;
    let y_hard = one_hot(&indices, num_classes, logits.device())?;
    // straight-through gradient: forward uses hard, backward flows through soft
    let y = ((&y_hard - &y_soft.detach())? + &y_soft)?;
    Ok(y)
}

/// Create a one-hot tensor from integer indices.
/// `indices` shape: [...], output shape: [..., num_classes]
fn one_hot(indices: &Tensor, num_classes: usize, device: &candle_core::Device) -> Result<Tensor> {
    let shape = indices.dims();
    let flat = indices.flatten_all()?;
    let n = flat.dim(0)?;
    let zeros = Tensor::zeros((n, num_classes), DType::F32, device)?;
    let ones = Tensor::ones((n, 1), DType::F32, device)?;
    let idx = flat.reshape((n, 1))?.to_dtype(DType::U32)?;
    let result = zeros.scatter_add(&idx, &ones, 1)?;
    let mut out_shape = shape.to_vec();
    out_shape.push(num_classes);
    result.reshape(out_shape)
}

// ---------------------------------------------------------------------------
// AddrNet  (~4,396 params)
// ---------------------------------------------------------------------------

/// Single address-generation network.
///
/// Pipeline: Linear(d_model→32) → reshape(4,8) → DepthwiseConv1D(k=3,g=4)
///           → Pointwise(4→256) per level → Gumbel-Softmax
#[derive(Debug)]
pub struct AddrNet {
    fc: Linear,
    dw_conv: Conv1d,
    pointwise: Linear,
    trie_depth: usize,
    trie_arity: usize,
    n_groups: usize,
}

impl AddrNet {
    /// `n_groups` = number of depthwise channels (fixed at 4 so fc output = n_groups * trie_depth).
    const N_GROUPS: usize = 4;

    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let n_groups = Self::N_GROUPS;
        let fc_out = n_groups * cfg.trie_depth; // 4 * 8 = 32

        let fc = candle_nn::linear(cfg.d_model, fc_out, vb.pp("fc"))?;
        let dw_conv = candle_nn::conv1d_no_bias(
            n_groups,
            n_groups,
            3,
            Conv1dConfig {
                padding: 1,
                groups: n_groups,
                ..Default::default()
            },
            vb.pp("dw_conv"),
        )?;
        let pointwise = candle_nn::linear(n_groups, cfg.trie_arity, vb.pp("pointwise"))?;

        Ok(Self {
            fc,
            dw_conv,
            pointwise,
            trie_depth: cfg.trie_depth,
            trie_arity: cfg.trie_arity,
            n_groups,
        })
    }

    /// Forward pass.
    ///
    /// * `hidden`      — `[batch, seq_len, d_model]`
    /// * `temperature`  — Gumbel-Softmax temperature τ
    /// * `hard`         — if true, use straight-through hard samples
    ///
    /// Returns `(addresses, logits)`:
    /// * `addresses` — `[batch, seq_len, trie_depth, trie_arity]` (soft probs or hard one-hot)
    /// * `logits`    — `[batch, seq_len, trie_depth, trie_arity]` (raw, for loss)
    pub fn forward(
        &self,
        hidden: &Tensor,
        temperature: f64,
        hard: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _d_model) = hidden.dims3()?;

        // 1) FC projection → [batch, seq_len, n_groups * trie_depth]
        let h = self.fc.forward(hidden)?;

        // 2) Reshape to [batch * seq_len, n_groups, trie_depth] for Conv1D
        //    Conv1d expects [N, C, L]
        let h = h.reshape((batch * seq_len, self.n_groups, self.trie_depth))?;

        // 3) Depthwise Conv1D over the levels dimension
        let h = self.dw_conv.forward(&h)?;
        //    h: [batch * seq_len, n_groups, trie_depth]

        // 4) Transpose to [batch * seq_len, trie_depth, n_groups] for pointwise
        let h = h.transpose(1, 2)?.contiguous()?;

        // 5) Shared pointwise linear: [batch * seq_len * trie_depth, n_groups] → [..., trie_arity]
        let h = h.reshape((batch * seq_len * self.trie_depth, self.n_groups))?;
        let logits = self.pointwise.forward(&h)?;
        //    logits: [batch * seq_len * trie_depth, trie_arity]

        // 6) Reshape logits to [batch, seq_len, trie_depth, trie_arity]
        let logits = logits.reshape((batch, seq_len, self.trie_depth, self.trie_arity))?;

        // 7) Gumbel-Softmax per level (last dim is trie_arity)
        let addresses = gumbel_softmax(&logits, temperature, hard)?;

        Ok((addresses, logits))
    }
}

// ---------------------------------------------------------------------------
// AddrNetBank — holds n_addr_nets (3) AddrNet instances
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct AddrNetBank {
    pub nets: Vec<AddrNet>,
}

impl AddrNetBank {
    pub fn new(vb: VarBuilder, cfg: &ZenoConfig) -> Result<Self> {
        let nets = (0..cfg.n_addr_nets)
            .map(|i| AddrNet::new(vb.pp(format!("addr_net_{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { nets })
    }

    /// Run all AddrNets in the bank, returning parallel vectors of (addresses, logits).
    pub fn forward(
        &self,
        hidden: &Tensor,
        temperature: f64,
        hard: bool,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        self.nets
            .iter()
            .map(|net| net.forward(hidden, temperature, hard))
            .collect()
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

    fn test_cfg() -> ZenoConfig {
        ZenoConfig::default_proto()
    }

    #[test]
    fn test_addrnet_shapes() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let net = AddrNet::new(vb.pp("test"), &cfg)?;

        let batch = 2;
        let seq = 4;
        let hidden = Tensor::randn(0f32, 1.0, (batch, seq, cfg.d_model), dev)?;
        let (addresses, logits) = net.forward(&hidden, 1.0, false)?;

        assert_eq!(logits.dims(), &[batch, seq, cfg.trie_depth, cfg.trie_arity]);
        assert_eq!(
            addresses.dims(),
            &[batch, seq, cfg.trie_depth, cfg.trie_arity]
        );
        Ok(())
    }

    #[test]
    fn test_addrnet_hard_mode() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let net = AddrNet::new(vb.pp("test"), &cfg)?;
        let hidden = Tensor::randn(0f32, 1.0, (1, 2, cfg.d_model), dev)?;
        let (addresses, _logits) = net.forward(&hidden, 0.5, true)?;

        // Hard mode: each level should be a one-hot vector (sums to 1.0)
        let sums = addresses.sum(candle_core::D::Minus1)?;
        let expected = Tensor::ones_like(&sums)?;
        let diff = (sums - expected)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 1e-4,
            "hard one-hot rows should sum to 1.0, diff={diff}"
        );
        Ok(())
    }

    #[test]
    fn test_addrnet_bank() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let bank = AddrNetBank::new(vb.pp("bank"), &cfg)?;
        assert_eq!(bank.nets.len(), cfg.n_addr_nets);

        let hidden = Tensor::randn(0f32, 1.0, (1, 3, cfg.d_model), dev)?;
        let results = bank.forward(&hidden, 1.0, false)?;
        assert_eq!(results.len(), cfg.n_addr_nets);
        Ok(())
    }

    #[test]
    fn test_param_count() -> Result<()> {
        let cfg = test_cfg();
        let dev = &Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);

        let _net = AddrNet::new(vb.pp("count"), &cfg)?;
        let total: usize = vm.all_vars().iter().map(|v| v.elem_count()).sum();
        // fc: 96*32+32=3104, dw_conv: 4*1*3=12, pointwise: 4*256+256=1280 → 4396
        assert_eq!(total, 4396, "expected ~4396 params, got {total}");
        Ok(())
    }
}
