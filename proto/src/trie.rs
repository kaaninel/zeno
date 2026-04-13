use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};

use crate::config::ZenoConfig;

/// HashMap-based L0 trie memory for the prototype.
///
/// Keys are byte addresses (up to `trie_depth` bytes from AddrNet output).
/// Values are `d_model`-dimensional vectors stored as detached tensors.
///
/// Gradient flow: writes detach the value (fire-and-forget); reads return
/// stored tensors as-is so gradients flow through the read path back to
/// memory-cross-attention weights, not to the original writer.
pub struct ProtoTrie {
    store: HashMap<Vec<u8>, Tensor>,
    d_model: usize,
}

impl ProtoTrie {
    pub fn new(config: &ZenoConfig) -> Self {
        Self {
            store: HashMap::new(),
            d_model: config.d_model,
        }
    }

    /// Look up a single address. Returns the stored d_model vector or zeros.
    pub fn read(&self, address: &[u8], device: &Device) -> Result<Tensor> {
        match self.store.get(address) {
            Some(v) => Ok(v.clone()),
            None => Tensor::zeros(self.d_model, candle_core::DType::F32, device),
        }
    }

    /// Blended write: `stored = (1-α) × old + α × new`.
    /// The incoming value is detached before storage to break the compute graph.
    pub fn write(&mut self, address: &[u8], value: &Tensor, strength: f64) -> Result<()> {
        let detached = value.detach();
        let key = address.to_vec();

        let to_store = if let Some(old) = self.store.get(&key) {
            ((old * (1.0 - strength))? + (detached * strength)?)?
        } else {
            detached
        };

        self.store.insert(key, to_store);
        Ok(())
    }

    /// Count entries whose key starts with `prefix`.
    pub fn density_at_prefix(&self, prefix: &[u8]) -> usize {
        self.store
            .keys()
            .filter(|k| k.starts_with(prefix))
            .count()
    }

    /// For address `[a0, a1, …, aN]` return
    /// `[density([a0]), density([a0,a1]), …, density([a0,…,aN])]`.
    pub fn density_chain(&self, address: &[u8]) -> Vec<usize> {
        (1..=address.len())
            .map(|i| self.density_at_prefix(&address[..i]))
            .collect()
    }

    /// Read multiple addresses and stack into `[n_addresses, d_model]`.
    pub fn batch_read(&self, addresses: &[Vec<u8>], device: &Device) -> Result<Tensor> {
        let vecs: Vec<Tensor> = addresses
            .iter()
            .map(|a| self.read(a, device))
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&vecs, 0)
    }

    /// Clear all entries.
    pub fn reset(&mut self) {
        self.store.clear();
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the trie is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ZenoConfig {
        ZenoConfig::default_proto()
    }

    #[test]
    fn read_missing_returns_zeros() {
        let trie = ProtoTrie::new(&cfg());
        let v = trie.read(&[0, 1, 2], &Device::Cpu).unwrap();
        assert_eq!(v.dims(), &[96]);
        let sum: f32 = v.sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn write_then_read() {
        let mut trie = ProtoTrie::new(&cfg());
        let val = Tensor::ones(96, candle_core::DType::F32, &Device::Cpu).unwrap();
        trie.write(&[1, 2, 3], &val, 1.0).unwrap();

        let out = trie.read(&[1, 2, 3], &Device::Cpu).unwrap();
        let sum: f32 = out.sum_all().unwrap().to_scalar().unwrap();
        assert!((sum - 96.0).abs() < 1e-5);
    }

    #[test]
    fn blended_write() {
        let mut trie = ProtoTrie::new(&cfg());
        let ones = Tensor::ones(96, candle_core::DType::F32, &Device::Cpu).unwrap();
        trie.write(&[0], &ones, 1.0).unwrap();

        let twos = (Tensor::ones(96, candle_core::DType::F32, &Device::Cpu).unwrap() * 2.0)
            .unwrap();
        trie.write(&[0], &twos, 0.5).unwrap();

        // Expected: 0.5 * 1.0 + 0.5 * 2.0 = 1.5 per element
        let out = trie.read(&[0], &Device::Cpu).unwrap();
        let mean: f32 = out.mean_all().unwrap().to_scalar().unwrap();
        assert!((mean - 1.5).abs() < 1e-5);
    }

    #[test]
    fn density_chain_counts() {
        let mut trie = ProtoTrie::new(&cfg());
        let v = Tensor::zeros(96, candle_core::DType::F32, &Device::Cpu).unwrap();

        // Three entries sharing prefix [1]
        trie.write(&[1, 0], &v, 1.0).unwrap();
        trie.write(&[1, 1], &v, 1.0).unwrap();
        trie.write(&[1, 1, 5], &v, 1.0).unwrap();

        let chain = trie.density_chain(&[1, 1, 5]);
        assert_eq!(chain, vec![3, 2, 1]);
    }

    #[test]
    fn batch_read_shape() {
        let mut trie = ProtoTrie::new(&cfg());
        let v = Tensor::ones(96, candle_core::DType::F32, &Device::Cpu).unwrap();
        trie.write(&[0], &v, 1.0).unwrap();

        let out = trie
            .batch_read(&[vec![0], vec![1], vec![2]], &Device::Cpu)
            .unwrap();
        assert_eq!(out.dims(), &[3, 96]);
    }

    #[test]
    fn reset_clears() {
        let mut trie = ProtoTrie::new(&cfg());
        let v = Tensor::zeros(96, candle_core::DType::F32, &Device::Cpu).unwrap();
        trie.write(&[0], &v, 1.0).unwrap();
        assert_eq!(trie.len(), 1);
        trie.reset();
        assert!(trie.is_empty());
    }
}