/// Hyperparameter configuration for the Zeno prototype agent.
#[derive(Debug, Clone)]
pub struct ZenoConfig {
    /// Embedding / hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer blocks
    pub n_layers: usize,
    /// FFN intermediate dimension (2 × d_model)
    pub d_ff: usize,
    /// Vocabulary size (byte-level: 256)
    pub vocab_size: usize,
    /// Context window (tokens per chunk)
    pub context_window: usize,
    /// Maximum trie address depth
    pub trie_depth: usize,
    /// Trie branching factor (256 = byte-level)
    pub trie_arity: usize,
    /// Number of register ring buffer slots
    pub n_register: usize,
    /// Number of learnable context vectors
    pub n_context_vectors: usize,
    /// Number of memory cross-attention slots (trie reads)
    pub n_mem_slots: usize,
    /// Number of AddrNet instances (= write channels)
    pub n_addr_nets: usize,
    /// Aspect head bottleneck dimension
    pub aspect_bottleneck: usize,
    /// Confidence gate density embedding dimension
    pub confidence_embed_dim: usize,
}

impl ZenoConfig {
    pub fn default_proto() -> Self {
        Self {
            d_model: 96,
            n_heads: 4,
            n_layers: 4,
            d_ff: 192,
            vocab_size: 256,
            context_window: 256,
            trie_depth: 8,
            trie_arity: 256,
            n_register: 4,
            n_context_vectors: 10,
            n_mem_slots: 3,
            n_addr_nets: 3,
            aspect_bottleneck: 16,
            confidence_embed_dim: 8,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Total context cross-attention slots: learnable vectors + register ring buffer
    pub fn context_pool_size(&self) -> usize {
        self.n_context_vectors + self.n_register
    }
}
