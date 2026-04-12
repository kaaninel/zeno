# Copilot Instructions for Zeno

## Build & Run

```bash
# Build (requires Rust nightly + candle)
cargo build --release

# Run single-agent text generation
cargo run --release --bin zeno -- chat

# Train a specific phase (1-8)
cargo run --release --bin zeno-train -- --phase 1

# Run a single test
cargo test -p zeno-core -- test_name

# Run all tests in a crate
cargo test -p zeno-trie
```

Hardware backends: Metal (Apple Silicon) > CUDA > CPU. Use `--release` for
anything touching model inference or training.

## Architecture

Zeno is a swarm of tiny (~666K parameter) byte-level transformer agents. Each
agent processes 256-token windows of VQ-encoded content. **All knowledge lives
in a persistent hierarchical trie, completely separate from model weights.** The
weights learn *how* to read and write; the trie stores *what* was learned.

### Workspace Crates

| Crate | Purpose |
|---|---|
| `zeno-core` | Core agent transformer (AddrNet, attention, output heads) |
| `zeno-trie` | Hierarchical trie memory (L0 GPU, L1 CPU, L2 disk, L3 internet) |
| `zeno-codec` | VQ codecs for text/image/audio (Perceiver + RVQ) |
| `zeno-swarm` | Multi-agent orchestrator (tiling, scheduling, scratchpad) |
| `zeno-runtime` | Inference runtime (async transport, tool dispatch, headers) |
| `zeno-train` | Training pipeline (8-phase curriculum) |

### Core Agent Forward Pass

Every agent cycle follows: **READ trie → READ scratchpad → Forward transformer
→ WRITE trie (async) → WRITE scratchpad → Output**. Agents never block on
memory; L0 returns instantly, deeper tier results arrive asynchronously into the
unified attention pool.

### Transformer Block Structure (4 layers per block)

1. **Self-Attention** — causal, RoPE, 4 heads
2. **Context Cross-Attention** — 14 slots (10 tag-encoder + 4 internal registers)
3. **Memory Cross-Attention** — 22-slot unified pool (3 L0 trie + 16 scratchpad + 3 async queue)
4. **SiLU FFN** — d_model → 2×d_model → d_model

### Output Heads

- LM head → 256 logits (weight-tied with code embedding)
- 3× AddrNet → write addresses (8 levels each, Conv1D, Gumbel-softmax)
- V_proj + 3 aspect heads → write values with channel-specific residuals
- write_strength head → α modifier
- Confidence gate → modulates reads by trie density (READ-ONLY)

## Key Conventions

### VQ-256 Pipeline

All modalities (text, image, audio) pass through VQ-256 codecs before reaching
core agents. Core agents are modality-blind — they only see codes 0-255. Content
knowledge lives in codecs + trie, never in agent weights.

RVQ uses 3 layers with progressive residual refinement:
Layer 1 = coarse self-sufficient output, Layer 2 = language refinement residual,
Layer 3 = final polish residual (style/emoji/nuance). All layers get reconstruction loss.
Byte ordering: `[all coarse] + [all mid] + [all fine]`.

### Trie Memory

- 256-ary, up to 8 levels deep, `d_model` vectors at every node
- L0 (GPU): flat-array direct indexing, instant reads
- L1 (CPU): full Rust trie with RwLock, async reads
- L2 (Disk): mmap'd pages for overflow/cold storage
- L3 (Internet): remote trie servers, read-only at inference
- Confidence = structural density (populated_children count), not counters
- Write diffusion creates probabilistic halos around primary writes
- Writes are fire-and-forget via lock-free queue → background thread

### Spatiotemporal Headers

Every content unit carries HTTP-inspired headers (author, origin, location,
timestamp, content-type, etc.). Headers define work-unit boundaries — changes in
author/location/timestamp trigger new units. Headers are injected via dual
channels: inline bytes in the token stream + structured side-channel via
tag encoder (mini cross-attention producing 10 context vectors).

### AddrNet Constraints

AddrNet (address generation) is the most fragile component. It must:
- Only train AFTER base model produces meaningful hidden states (Phase 3+)
- Stay frozen or at very low LR (0.01×) after Phase 3
- Be monitored for entropy collapse (target > 7.0 bits)
- Use Gumbel-softmax for differentiable discrete address selection

### Training Phases

The 8-phase progressive curriculum freezes learned components before training
harder ones. Never advance phases without meeting gate criteria. Phase order:
VQ codec → base LM → AddrNet → memory integration → coherence unfreeze →
partiality → swarm → tools/chat. See `docs/TRAINING.md` for gate criteria.

### Partiality & Confidence

The confidence gate (~2,129 params) converts trie density chains into a scalar
that modulates memory attention values (READ-ONLY). Sparse paths →
weak reads (imagination). Dense paths → strong reads (knowledge).
This is not binary — it's a continuous gradient derived from structural density.

### Scratchpad

Fixed 16 × d_model tensor on GPU shared by all agents in a work group.
Attention-based write: agent hidden cross-attends to scratchpad slots (W_q+W_k, 1,536 params).
Content-aware slot targeting, self-organizing. Fully parallel, fully differentiable.
Defines work group boundaries (GPU SMEM analogy).

### Constants

| Name | Value | Notes |
|---|---|---|
| `d_model` | 96 | Core embedding dimension |
| `d_codec` | 64 | VQ codec embedding dimension |
| `vocab_size` | 256 | VQ codebook size (byte-level) |
| `context_window` | 256 | Tokens per chunk |
| `n_heads` | 4 | Attention heads |
| `agent_params` | ~655K | Total per agent |
| `RVQ layers (K)` | 3 | Progressive residual refinement |
| `trie_depth` | 8 | Maximum trie levels |
| `trie_arity` | 256 | Children per node |
| `scratchpad_slots` | 16 | Shared coordination slots |
| `attention_pool_slots` | 14 | Context cross-attn: 10 tag + 4 register |
| `memory_pool_slots` | 22 | Memory cross-attn: 3 trie + 16 scratchpad + 3 async |
| `register_bank` | 4 | FIFO ring buffer (last 4 hiddens) |

## MCP Servers

| Server | Purpose | Auth needed |
|---|---|---|
| `rust-analyzer` | Semantic code analysis — diagnostics, hover, go-to-def. Binary: `rust-analyzer-mcp` | — |
| `colab` | Create/run notebooks on Colab CUDA GPUs for training phases | Google OAuth (browser, first use) |
| `huggingface` | Search models, datasets, papers; browse Hub | `HF_TOKEN` env var (optional, raises rate limits) |
| `github` | Manage issues/PRs/Actions for this repo | `GITHUB_TOKEN` env var |
| `fetch` | Fetch ML papers, docs, remote configs | — |

Config lives in `.mcp.json` at the repo root. `rust-analyzer-mcp` is installed at `~/.cargo/bin/`. All `uvx`-based servers auto-install on first run.

## Documentation

Detailed design docs live in `docs/`:
- `ARCHITECTURE.md` — agent model, VQ codecs, headers, parameter budget
- `MEMORY.md` — trie tiers, read/write paths, diffusion, confidence gate
- `TRAINING.md` — 8-phase curriculum with gate criteria
- `SWARM.md` — tiling (spatial/temporal/refinement), tool dispatch, scratchpad
- `PARTIALITY.md` — imagination, confidence, cross-modal alignment
- `ISSUES.md` — resolved design decisions log
- `EXPECTATIONS.md` — hardware performance analysis and training estimates

## Decision Authority

**The human decides all architectural and design choices.** Copilot's role is to
code, research, document, and present options. Never make design decisions
unilaterally — always present options with trade-offs and wait for confirmation
before writing them into docs. If uncertain, ask. When documenting a confirmed
decision, show the exact text before editing and get approval.
