# Zeno — Architecture Decisions Log

Resolved design issues and decision rationale. Each entry records what was ambiguous, the
decision made, and where it is documented.

---

## Issue #1 — Write-before-read in Phase 4 training (CRITICAL → RESOLVED)

**Problem:** Phase 4 training computes retrieval accuracy loss which requires reading back a
just-written value. Fire-and-forget trie writes are not visible in the same forward pass.

**Decision:** L0 writes ARE synchronous (in-process DRAM, zero async). The "fire-and-forget"
phrasing in earlier drafts referred only to L0→L3 propagation, not the initial L0 commit.
Training data must be constructed so write-step strictly precedes read-step by at least 1 step
(see Phase 4 single-pass order: Read → Forward → Loss → Generate addresses → Write).

**Documented in:** `docs/TRAINING.md` (Phase 4 section, "Training data constraint" block)

---

## Issue #2 — 250K TPS ambiguity (CRITICAL → RESOLVED)

**Problem:** ARCHITECTURE.md risk table asked: "Is 250K TPS total swarm throughput or
per-request latency?" — explicitly flagged as needing clarification.

**Decision:** 250K TPS = **total swarm tokens per second** across all concurrent requests.
Not per-request latency. A single agent on M4 DRAM-bound asymptote ≈ 90K tok/s;
ANE SRAM-cached regime achieves 250K TPS with ~13 concurrent agents on one M4 base.

**Documented in:** `docs/ARCHITECTURE.md` (risk table, §Throughput), `docs/EXPECTATIONS.md`

---

## Issue #3 — Memory attention pool size undefined (SIGNIFICANT → RESOLVED)

**Problem:** Unified memory cross-attention pools multiple source types but the spec never
fixed a maximum. ARCHITECTURE.md itself flagged "~100+ slots" as a dilution risk.

**Decision:** Pool is fixed at **22 slots**:
- L0 trie: **3 vectors** (one leaf value per read address — path nodes NOT included in pool)
- Shared scratchpad: **16 slots** (see Issue #5)
- Async queue slots: **3** (FIFO queue drained up to 3 per token; empty slots zero-masked)

**"Causal peer hiddens" dropped** from the pool. The scratchpad IS the only inter-agent
channel (see Issue #5). Peer communication happens through scratchpad, not a separate
peer-hidden-state broadcast.

The 4 per-agent register slots live in the **context cross-attention** (not memory cross-attention)
and are not double-counted here.

With 4 heads at d_model=96: ~5–6 keys per head. No dilution.

**Documented in:** `docs/ARCHITECTURE.md` (memory cross-attention section)

---

## Issue #4 — 256^8 address space near-zero density (SIGNIFICANT → RESOLVED)

**Problem:** Training writes ~24M values into a 256^8 ≈ 10^19 address space.
Coverage ≈ 0%. The confidence gate never sees a non-empty address during training,
so it has no positive examples for "skip redundant write".

**Decision:** Resolved by **training data design, not address space reduction**.
The confidence gate trains on VALUE similarity, not address collision. Training sequences
must explicitly include re-encounter patterns: the same content is written at step N and
seen again at step M > N. When the real trie already contains a matching value at that address,
the confidence gate learns to suppress write_strength (redundant write). The 256^8 address
space is kept for inference.

**Documented in:** `docs/TRAINING.md` (Phase 4 training data requirements)

---

## Issue #5 — Scratchpad conflict resolution (SIGNIFICANT → RESOLVED)

**Problem:** SWARM.md presented two options for concurrent scratchpad writes:
(a) GPU atomic f32 adds, (b) per-agent owned ring-buffer slots.

**Decision:** Scratchpad is **shared global state** — the only inter-agent communication
channel, analogous to GPU shared memory (SMEM). Just as SMEM defines a GPU workgroup,
scratchpad defines a Zeno work group: only agents in the same work group share a scratchpad.
Agents in different work groups cannot communicate.

- **16 slots** (fixed, regardless of work group size)
- **Global shared bus**: all agents in the work group blend into the same 16 slots
- **Conflict resolution:** strength-weighted mean across concurrent writes (fully parallel, no atomics):
  ```
  scratchpad[i] = Σ_agents(strength[a][i] × value[a][i]) / Σ_agents(strength[a][i])
  ```
- **Automatic update**: each agent applies a small FFN to its hidden state to produce write
  values and strengths per slot. No explicit write decision — all agents update every step.
- **Work group size**: dynamic, 1–N_max agents. N_max configurable at launch (supports up to
  128+ agents for large tasks like tiling a high-resolution image). Inactive agent slots are
  zero-masked (strength=0 → no contribution).
- **Per-agent register bank** (4 slots) handles private per-agent bookkeeping and is separate
  from the shared scratchpad (lives in context cross-attention, not memory cross-attention).

The `write_strength` sigmoid head used for trie writes doubles as scratchpad write strength
per slot. No new parameters needed beyond a small FFN (~16 × d_model params) to project
hidden state to per-slot write values.

**Documented in:** `docs/SWARM.md` (scratchpad section), `docs/ARCHITECTURE.md` (work group)

---

## Issue #6 — Tag encoder identity ambiguous (MODERATE → RESOLVED)

**Problem:** `pool(text_encoder("author_name"))` — which encoder? Separate tiny embedding or
shared VQ codec?

**Decision:** `text_encoder` = **VQ codec byte encoder, shared, frozen after Phase 1**.
The `~2K` params in the parameter budget are for the cross-attention layer attending to tag
vectors, NOT for a separate encoder. Tags are encoded via the same byte-level VQ pipeline,
then pooled, then attended to. No new encoder params.

**Documented in:** `docs/ARCHITECTURE.md` (tag encoder section)

---

## Issue #7 — RVQ Layer 3 dual-role conflict (MODERATE → RESOLVED, REVISED)

**Problem:** Layer 3 must both close reconstruction to 99.5% accuracy AND specialize for
emoji/emotional enrichment. The two losses pull the codebook in opposite directions.

**Decision (revised):** **No partition needed.** Layers 1-2 alone provide 256×256 = 65K code
combinations — massive overkill for 256 byte values. Text reconstruction does not need Layer 3.
Layer 3 is 100% dedicated to enrichment/emoji/emotion. No reconstruction loss on Layer 3.
No partition, no dual-objective conflict. Simpler training, cleaner design.

**Documented in:** `docs/TRAINING.md` (Phase 1 section)

---

## Issue #8 — candle vs PyTorch contradiction (MODERATE → RESOLVED)

**Problem:** `docs/TRAINING.md` states "candle-only". ARCHITECTURE.md risk table (line ~691)
said "Start with candle for inference, keep PyTorch for training initially." Contradictory.

**Decision:** **Candle-only for all training and inference.** The risk table language was a
hedge written before the final framework decision. We accept candle's current limitations
(no FlashAttention, young training ecosystem). Benefit: single codebase, no Python dependency,
pure Rust from day one.

**Documented in:** `docs/ARCHITECTURE.md` (risk table updated to remove PyTorch fallback)

---

## Issue #9 — NOOP design and training (MODERATE → RESOLVED, REVISED)

**Problem:** If NOOP is never present in training data before Phase 7, the model may learn to
suppress the NOOP logit entirely.

**Decision (revised):** **NOOP is a filler token, like "ehm" in speech.** The model emits it
when it needs more processing time before committing to real content. Maps to UTF-8 ACK at
the Rust runtime boundary. All training data across ALL phases includes NOOP at natural pause
points (~1-2% of positions). Dataset construction places NOOP where complicated concepts or
low-confidence reads would naturally cause a pause. NOOP is always safe — better to stall
than emit garbage.

**Documented in:** `docs/TRAINING.md` (data requirements section + per-phase data notes)

---

## Issue #10 — Phase 5 base LR not specified (MINOR → RESOLVED)

**Problem:** Phase 5c specifies multipliers (0.1×, 0.5×, 0.01×) but never names the "1.0×"
reference LR.

**Decision:** **Phase 5 base LR = Phase 4 final learning rate**, taken from the Phase 4
checkpoint metadata (`checkpoint['lr']`). All Phase 5 multipliers apply to this base.
This is now explicit in the training spec.

**Documented in:** `docs/TRAINING.md` (Phase 5 section)

---

## Issue #11 — L2 page size M4-specific (MINOR → RESOLVED, REVISED)

**Problem:** `docs/MEMORY.md` specifies "Buckets sized to M4 16KB pages". Linux/Windows
default is 4KB; only Apple Silicon uses 16KB. Hardcoding breaks cross-platform correctness.

**Decision (revised):** **4KB-aligned buckets** — works on all platforms (4KB is a common
divider of all OS page sizes: 4KB, 16KB, 64KB). 8 nodes per bucket (4096 / 384 bytes per
node = 10, rounded down to power-of-2). No runtime detection needed.

**Documented in:** `docs/MEMORY.md` (L2 section, TrieBucket struct)

---

## Issue #12 — Phase 2 mem_attn trains on empty slots (MINOR → RESOLVED)

**Problem:** Phase 2 trains mem_attn with memory disabled (empty/random slots). The model
may learn near-zero memory attention weights, requiring un-learning in Phase 4.

**Decision:** **Not a bug — expected behavior.** Near-zero Phase 2 mem_attn is easy to shift
in Phase 4 when real memory content provides gradient signal. Monitoring required: log mean
absolute mem_attn weight values at Phase 3→4 boundary. If still < 0.01 after 500 Phase 4
steps, add explicit mem_attn warmup (increase retrieval loss weight for 1K warmup steps).

**Documented in:** `docs/TRAINING.md` (Phase 2 and Phase 4 sections, monitoring notes)

---

## Issue #13 — Phase 4 trie strategy (design decision, REVISED)

**Problem:** How should Phase 4 interact with trie memory during training?

**Decision (revised):** **Phase 4 trains on the real L0 trie, not a mock.**
- Full 256-ary trie, K=8 levels, stored as GPU tensors (same as production L0)
- AddrNet is frozen — hard addressing (argmax) is fine, no gradient needed through addresses
- Gradient flows through continuous values: trie stored value → mem_attn → LM loss
- Confidence gate trains on real structural density (populated_children counts)
- Training data volume kept small so trie stays GPU-resident
- Periodic trie resets (determined by device memory budget + dataset size) bound growth
  and teach the model to rebuild context from scratch

**Why not a mock:** The recall mechanism is not isolated — the model may look up arbitrary
content. A mock (e.g., DNC-style dense tensor with soft attention) trains on an abstraction
that doesn't match real trie behavior (sparsity, density patterns, address misses).
Real trie = zero transfer gap to inference.

**Documented in:** `docs/TRAINING.md` (Phase 4 section)
