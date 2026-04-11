# Zeno

> A tiled swarm of sub-1M byte-level transformer agents with external hierarchical trie memory.

## Overview

Zeno is a high-performance, chat-focused multimodal AI system built as a swarm
of tiny (~666K parameter) transformer agents. Each agent processes 256-token
windows of VQ-encoded content, with **all knowledge stored in a persistent
hierarchical trie** — completely separate from model weights.

The weights learn *how* to read and write. The trie stores *what* was learned.

### Core Principles

- **Memory IS the architecture** — 666K params store zero knowledge. All knowledge
  lives in the trie. Every forward pass reads from the trie, every cycle writes
  to the trie. Memory is not optional.
- **Unified VQ-256 pipeline** — All modalities (text, image, audio) go through
  VQ-256 codecs. Core agents are truly modality-blind.
- **Async non-blocking memory** — Tiered: GPU (0ns) → CPU (μs) → Disk (ms) →
  Internet (10-100ms). Agents never block.
- **Pure Rust** — candle ML framework. Single language, no FFI overhead per token.
- **Hardware priority** — Metal (Apple Silicon) > CUDA > CPU.

```
 ┌──────────────────────────────── ZENO ─────────────────────────────────┐
 │                                                                       │
 │  ┌─────────────── Dynamic Tile Pool (N core agents) ───────────────┐  │
 │  │                                                                  │  │
 │  │   ┌────────┐  ┌────────┐  ┌────────┐       ┌────────┐          │  │
 │  │   │ Agent0 │  │ Agent1 │  │ Agent2 │  ...  │ AgentN │          │  │
 │  │   │ ~666K  │  │ ~666K  │  │ ~666K  │       │ ~666K  │          │  │
 │  │   │Unit[0] │  │Unit[1] │  │Unit[2] │       │Unit[N] │          │  │
 │  │   │ 256tok │  │ 256tok │  │ 256tok │       │ 256tok │          │  │
 │  │   └───┬────┘  └───┬────┘  └───┬────┘       └───┬────┘          │  │
 │  │       │            │            │                │               │  │
 │  │       ▼            ▼            ▼                ▼               │  │
 │  │  ┌──────────────────────────────────────────────────────┐       │  │
 │  │  │        Unified Cross-Attention Pool (per agent)      │       │  │
 │  │  │                                                      │       │  │
 │  │  │  [L0 trie cache] + [scratchpad] + [causal peers]    │       │  │
 │  │  │  + [async L1/L2/L3 results as they arrive]          │       │  │
 │  │  │                                                      │       │  │
 │  │  │  Single attention mechanism, model learns routing    │       │  │
 │  │  │  Loss penalizes access cost + inaccurate data        │       │  │
 │  │  └──────────────────────────────────────────────────────┘       │  │
 │  └──────────────────────────────────────────────────────────────────┘  │
 │                            │                                          │
 │                            ▼                                          │
 │  ┌──────────── Shared Scratchpad (GPU) ──────────────┐                │
 │  │  Fixed 16 × d_model tensor                        │                │
 │  │  Agents read at start, write at end (async)       │                │
 │  │  Additive/EMA blending for concurrent writes      │                │
 │  └───────────────────────┬───────────────────────────┘                │
 │                          │                                            │
 │                          ▼                                            │
 │  ┌────────────── Hierarchical Trie Memory ───────────────────────┐    │
 │  │                                                               │    │
 │  │  L0 (GPU): Depth 0..K       ◄── configurable depth level     │    │
 │  │    Direct flat-array indexing, lazy allocation, instant reads  │    │
 │  │                                                               │    │
 │  │  L1 (CPU RAM): Depth K+1..cap                                │    │
 │  │    Full Rust trie, RwLock, async read, batched writes         │    │
 │  │                                                               │    │
 │  │  L2 (Disk): Overflow / cold storage                          │    │
 │  │    mmap'd pages, OS page cache, page-aligned buckets          │    │
 │  │                                                               │    │
 │  │  L3 (Internet): Remote trie servers (read-only at inference)  │    │
 │  │    Subscribed trusted providers, address-based lookup          │    │
 │  │    Triggered by attention signal, cached locally               │    │
 │  │                                                               │    │
 │  │  Properties:                                                  │    │
 │  │    256-ary, up to 8 levels, d_model vectors at every node    │    │
 │  │    EMA blending, density-driven confidence gate               │    │
 │  │    3 AddrNet read/write paths                                 │    │
 │  └───────────────────────────────────────────────────────────────┘    │
 │                                                                       │
 │  ┌──── VQ Codecs (separate models, ~300K each) ───────────────────┐   │
 │  │                                                                │   │
 │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │   │
 │  │  │ Text Codec │  │Image Codec │  │Audio Codec │               │   │
 │  │  │ VQ-256 RVQ │  │ VQ-256 RVQ │  │ VQ-256 RVQ │               │   │
 │  │  └────────────┘  └────────────┘  └────────────┘               │   │
 │  │                                                                │   │
 │  │  Dedicated Generators (5-20M, singleton per swarm):            │   │
 │  │  ┌────────────┐  ┌────────────┐                                │   │
 │  │  │ Image Gen  │  │ Audio Gen  │                                │   │
 │  │  └────────────┘  └────────────┘                                │   │
 │  └────────────────────────────────────────────────────────────────┘   │
 │                                                                       │
 │  ┌──── Rust Runtime / Orchestrator ──────────────────────────────┐    │
 │  │  • VQ codec routing (content-type → encoder/decoder dispatch) │    │
 │  │  • Work-queue tiling (spatiotemporal boundary splitting)      │    │
 │  │  • Tool event-queue dispatch (parse at chunk boundary)        │    │
 │  │  • Async memory transport (L0↔L1↔L2↔L3)                      │    │
 │  │  • Dual-channel tag injection (inline bytes + side-channel)   │    │
 │  │  • Agent lifecycle management                                  │    │
 │  │  • Scratchpad synchronization                                  │    │
 │  └───────────────────────────────────────────────────────────────┘    │
 └───────────────────────────────────────────────────────────────────────┘
```

## Key Features

1. **Memory-separated architecture** — Weights store computation, trie stores
   knowledge. Unlimited context without parameter growth.

2. **Hierarchical trie** — Multi-scale representation (root=global,
   leaves=specific) with free ancestor collection during traversal.

3. **AddrNet co-processors** — 3 learned address generators via Gumbel-softmax.
   3 paths = 3 perspectives. Fully differentiable.

4. **Spatiotemporal tags** — HTTP-inspired dual-channel headers define tile
   boundaries and provide temporal/spatial identity.

5. **Dynamic tiling** — Work decomposition along 3 axes: chunking (temporal),
   tiling (spatial/spectral), refinement (quality hierarchy).

6. **Unified attention pool** — One mechanism handles all context sources
   (trie, scratchpad, peers, async). Model learns routing implicitly.

7. **Async non-blocking memory** — L0 gives instant coarse context, deeper
   results arrive progressively. Agents never stall.

8. **Full Rust + candle** — No Python, no FFI overhead. Metal > CUDA > CPU.
   Pure Rust from training to inference for maximum throughput.

9. **Unified VQ-256** — All modalities use the same codec pipeline. Core
   agents are modality-blind — code 0x41 means "codebook entry 65", not "A".
   All content knowledge lives in codecs + trie, weights are pure computation.

10. **Symmetric encoder/decoder per modality** — Each codec is a standalone
    VQ-VAE (Perceiver compress → RVQ quantize → Perceiver decompress).

11. **Learned adaptive text compression** — Perceiver cross-attention discovers
    word/character boundaries per language automatically. No language-specific
    rules. Multi-byte scripts (Russian, Hindi) get 4× context window improvement.

12. **Tool token protocol** — Byte-level tool dispatch at chunk boundaries.
    Tool results are standard tagged units, same format as any input.

13. **Emoji enrichment** — RVQ Layer 3 captures tone/emotion in continuous VAD
    space. Emojis are the training signal, cross-modal bridge for audio tone.

14. **Partiality & imagination** — Trie density encodes what's known vs
    imagined. Confidence gate prevents hallucination. Imagination → weak write,
    no halo → observation → strong write + diffusion halo → solid knowledge.

## Repository Structure

```
zeno/
  crates/
    zeno-core/        # Core agent model (transformer, AddrNet, output heads)
    zeno-trie/        # Hierarchical trie memory (L0-L3, arena-allocated)
    zeno-codec/       # VQ codecs (text, image, audio — Perceiver + RVQ)
    zeno-swarm/       # Multi-agent orchestrator (tiling, scheduling, scratchpad)
    zeno-runtime/     # Inference runtime (async transport, tool dispatch, headers)
    zeno-train/       # Training pipeline (8-phase curriculum, candle)
  docs/
    ARCHITECTURE.md   # Core architecture (agent, codecs, headers, design decisions)
    MEMORY.md         # Trie memory system (L0-L3, read/write, diffusion, confidence)
    TRAINING.md       # 8-phase training curriculum
    SWARM.md          # Multi-agent coordination, tiling, generation, tools
    PARTIALITY.md     # Partiality, imagination, confidence system
  configs/            # Model and deployment configurations
  data/               # Training data scripts and HuggingFace dataset loaders
```

## Getting Started

**First milestone:** Single agent + trie memory generating text.

```bash
# Build (requires Rust nightly + candle)
cargo build --release

# Run single-agent text generation
cargo run --release --bin zeno -- chat

# Train text VQ codec (Phase 1)
cargo run --release --bin zeno-train -- --phase 1

# Train base language model (Phase 2)
cargo run --release --bin zeno-train -- --phase 2
```

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Core agent, VQ codecs, headers, parameter budget, design decisions |
| [Memory System](docs/MEMORY.md) | Hierarchical trie, L0-L3 tiers, read/write paths, diffusion, confidence |
| [Training](docs/TRAINING.md) | 8-phase progressive freeze/unfreeze curriculum |
| [Swarm](docs/SWARM.md) | Multi-agent coordination, tiling, generation, tool dispatch |
| [Partiality](docs/PARTIALITY.md) | Partiality awareness, imagination, confidence gate |

## Roadmap

### Phase 1: Core Agent + Trie (Text Only)

- [ ] Project scaffolding (Cargo workspace, candle deps)
- [ ] Config system (ModelConfig, MemoryConfig, SwarmConfig)
- [ ] Text VQ codec (Perceiver encoder/decoder, RVQ K=3, semantic refinement)
- [ ] Spatiotemporal header system (tag format, parsing)
- [ ] Core transformer (RMSNorm, RoPE, Attention, MemoryAttention, FFN)
- [ ] AddrNet (Gumbel-softmax, 3 co-processors)
- [ ] Hierarchical trie (flat-array, lazy alloc, EMA, density)
- [ ] Tiered memory (L0 GPU / L1 CPU / L2 disk)
- [ ] Async memory transport
- [ ] Engine (per-token READ→PROCESS→WRITE cycle)
- [ ] Single-agent inference (chat mode)
- [ ] Training: Phase 1→2→3→4→5→6

### Phase 2: Swarm + Tools + L3

- [ ] Dynamic tiling + shared scratchpad
- [ ] Swarm orchestrator
- [ ] Tool event-queue dispatch
- [ ] L3 Internet memory
- [ ] Training: Phase 7 (swarm) + Phase 8 (tools + chat)

### Phase 3: Multimodal

- [ ] Image VQ codec (spatial tiling + RVQ K=4)
- [ ] Audio VQ codec (frequency-band tiling + RVQ K=4)
- [ ] Shared coarse codebook for multimodal sync
- [ ] Video pipeline (spatial + spectral + temporal)

## License

MIT — see [LICENSE](LICENSE)

---

*Repository: [github.com/kaaninel/zeno](https://github.com/kaaninel/zeno)*
