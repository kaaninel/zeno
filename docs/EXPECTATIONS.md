# Zeno Performance Expectations

This document provides hardware-specific performance estimates for Zeno inference and training,
with explicit reasoning chains so deviations can be traced to their root cause.

**Key model parameters:**
- Core agent: d_model=96, 4 transformer blocks, 4 heads, d_ff=192 (2×d_model), vocab=256 → **~655K params ≈ 2.62 MB FP32**
- VQ codec: ~156K params (d_codec=64, M=256 latents, 1 encoder/decoder layer, commitment β=0.25)
- KV cache per request at seq_len=256: 4 layers × 2 (K,V) × 256 × 96 × 4 bytes = **786 KB**

---

## Part 1: Inference Expectations

### 1.1 Methodology

The model is always **memory-bandwidth-bound** during inference, never compute-bound.

**Why:** FLOP/byte ratio needed = 2 FLOPs per parameter × 655K params / (2.62 MB weights + 0.786 MB KV)
= 1.31 GFLOP / 3.41 MB = **0.38 FLOP/byte**. All hardware listed delivers 10–100 FLOP/byte of
arithmetic intensity capability. The computation is 25–250× faster than memory bandwidth allows,
so bandwidth is the bottleneck 100% of the time.

**Two inference regimes:**

**Regime A: Cache-hot** — model weights fit in GPU L2 (A100: ~40MB, T4: ~4MB) or CPU LLC
(Ryzen L3: 32MB; Pi 5: 2MB). With weights in cache, bandwidth is 3–10 TB/s instead of DRAM speed.
This applies to single-user or low-concurrency scenarios.

**Regime B: DRAM-bound** — large batches evict weights from cache. Throughput asymptotes to
`DRAM_BW / bytes_per_token` regardless of batch size N. This is the sustainable high-throughput ceiling.

**Cache spill analysis (M4 base, GPU L2 ≈ 8MB, SLC = 12MB):**
- Model weights (FP16): 1.33 MB
- KV per concurrent request: 0.786 MB
- GPU L2 spill at: N = (8 − 1.33) / 0.786 ≈ **8 concurrent requests**
- SLC spill at: N = (12 − 1.33) / 0.786 ≈ **14 concurrent requests**
- Below spill: L2/SLC bandwidth-bound (~1–5 TB/s) → effectively instantaneous generation

**VQ codec overhead** (one-shot decode per response, NOT autoregressive):
~5–20 ms across all platforms (300K params, single forward over 256 codes). Negligible.

### 1.2 Hardware Inference Table

All numbers are **per-agent, autoregressive token generation**, FP32 weights unless noted.

| Hardware | DRAM BW | Cache-hot tok/s | DRAM ceiling tok/s | Cache-hot ms/1K words | Confidence |
|---|---|---|---|---|---|
| A100 80GB | 2,000 GB/s | ~1.7M | ~960K | <1 ms | HIGH |
| T4 16GB | 320 GB/s | ~200K¹ | ~150K | ~5 ms | HIGH |
| M4 base 24GB (GPU) | 120 GB/s | ~800K² | ~90K | ~1 ms cache / ~11 ms DRAM | HIGH |
| M4 base 24GB (ANE) | ~SRAM-bound | ~150K–2M³ | ~150K | ~5 ms–<1 ms | MEDIUM |
| Ryzen 5 7600X | 50 GB/s | ~65K⁴ | ~38K | ~15 ms | HIGH |
| Snapdragon 8 Gen 3 | 77 GB/s | ~85K⁵ | ~49K | ~12 ms | MEDIUM |
| Raspberry Pi 5 | 34 GB/s | ~5K⁶ | ~22K | ~200 ms | HIGH |

¹ **T4:** 4MB L2 → model (2.66 MB FP32 = 1.33 MB FP16) fits in L2. KV evicts after ~3 concurrent requests.
  Cache-hot: 320 GB/s × 10 TB/s ratio ≈ ~200K tok/s estimate.

² **M4 GPU cache-hot:** 10-core GPU with ~8MB L2 + 12MB SLC. Both weights and KV fit in SLC (1.33 + 0.786 = 2.1 MB FP16).
  With SLC at ~4–5 TB/s effective bandwidth → ~800K tok/s. Drops to 90K tok/s when DRAM-bound.
  Unified memory architecture: no CPU→GPU staging cost. This is the clearest advantage over discrete GPUs for this workload.

³ **M4 ANE (candle-coreml, target deployment):** ANE has ~16–24 MB on-chip SRAM (estimated from die analysis).
  FP16 weights (1.33 MB) + 1 agent KV (0.786 MB) = 2.1 MB → likely fits in ANE SRAM.
  If in SRAM at 4–8 TB/s internal bandwidth: **1.9M–3.8M tok/s** (upper bound).
  If only weights in SRAM, KV from unified DRAM: **~150K tok/s** (KV-bandwidth-bound).
  ANE SRAM fits ~18–28 concurrent FP16 agents (shared weights); ~39–59 at INT8.
  Access path: `candle-coreml` crate (see ARCHITECTURE.md). **Status: targeted, not yet implemented.**

⁴ **Ryzen 5 7600X:** 32MB L3 fits model (2.66 MB FP32) easily. KV evicts after ~37 concurrent requests.
  L3 BW ≈ 400 GB/s → ~65K tok/s. DRAM ceiling: 50 GB/s / 3.45 MB = ~14K samples/s = ~38K tok/s.

⁵ **Snapdragon 8 Gen 3:** 12MB LLC. Model fits. KV evicts after ~17 concurrent requests.
  LLC BW ≈ 350 GB/s → ~85K tok/s. Good mobile inference target.

⁶ **Raspberry Pi 5:** L2 = 2MB (per Cortex-A76 cluster). Model (2.66 MB) does NOT fit in L2.
  Always DRAM-bound: 34 GB/s / 2.66 MB weights = ~12.8K weight loads/s → ~5K tok/s after overhead.
  Pi 5 is the only hardware where the model doesn't fit in cache. Confirms Pi 5 = functional but slow.

### 1.3 Swarm Throughput (250K TPS Target)

250K TPS = **total swarm tokens/second**, not per-agent latency.
See ARCHITECTURE.md §Throughput for full analysis.

At M4 DRAM ceiling (90K tok/s per agent group), 250K TPS requires:
- Pure DRAM regime: 250K / 90K ≈ **3 parallel M4 machines** (unrealistic for single-device target)
- ANE SRAM-cached regime: 250K / 1.9M = **0.13 of one ANE** → single M4 achieves 250K TPS with 13 concurrent SRAM-cached agents

**ANE is the feasibility path for the 250K TPS goal on a single M4.**

---

## Part 2: Training Expectations

### 2.1 Training FLOPs Per Step

Training processes a full sequence at once (not autoregressive), so attention is O(seq²):

- Attention (4 layers, seq=256, d=96): `2 × 256² × 96 × 4` = 50.3 MFLOP
- FFN (4 layers, seq=256, d_ff=192): `2 × 256 × 96 × 192 × 2 × 4` = 75.5 MFLOP
- Output head: `256 × 256 × 2` = 0.13 MFLOP
- **Forward per sample: 126 MFLOP**
- **Backward: ~2× forward = 252 MFLOP**
- **Full step (batch=32): 12.1 GFLOP**

**Attention matrix memory per step:** `256² × 32 × 4 bytes × 4 layers` = **33.6 MB** (must be materialized;
candle has no FlashAttention). This is under 0.15% of M4's 24GB RAM — memory is not a constraint.

**AdamW optimizer state** (666K params): grad (2.66 MB) + m1 (2.66 MB) + m2 (2.66 MB) = **8 MB**. Trivial.

### 2.2 Candle Training Efficiency

candle's efficiency for training is significantly lower than inference due to:
1. **No FlashAttention**: full O(seq²) attention matrix → poor memory access patterns
2. **d_model=96 too narrow for tensor cores**: NVIDIA Tensor Cores and Apple AMX require at least 128-dim for maximum utilization; 96-dim falls back to regular GEMM
3. **Kernel launch overhead**: tiny matrices (96×96, 96×384) → many small CUDA/Metal kernels with high launch latency relative to compute time
4. **Autograd graph traversal**: gradient tape traversal adds 30–80% wall-clock beyond pure FLOPs
5. **Optimizer memory bandwidth**: AdamW requires 5 reads + 2 writes per parameter per step

**Candle training efficiency estimates vs FP32 theoretical peak:**
- A100 / T4 CUDA: **6–12%** (kernel launch overhead dominates for small ops)
- M4 Metal: **8–15%** (Metal backend less mature for training than inference)
- Ryzen CPU (BLAS): **45–65%** (OpenBLAS/MKL well-tuned for small matrices; no kernel launch overhead)
- Snapdragon CPU: **35–55%** (NEON, decent for small GEMM)
- Pi 5 CPU: **55–70%** (NEON, similar to Snapdragon but slower clock)

**Surprising result:** Ryzen CPU competes with or beats M4 Metal for training at this model size.
This is because CPU BLAS (OpenBLAS) is extremely well-optimized for 96-dim matrices, while Metal's
training pipeline involves more overhead per kernel for these tiny operations. This will flip once
model size grows (d_model > ~256) or candle's Metal backend matures.

### 2.3 Hardware Training Speed Table

Full curriculum = 95K nominal steps. Gate criteria can add 50–200% wall-clock on gated phases.

| Hardware | Candle steps/sec | Steps/min | 95K steps | Gate-adjusted total |
|---|---|---|---|---|
| A100 80GB | 60–120 | 3,600–7,200 | 13–26 min | 0.5–1.5 hr |
| T4 16GB | 25–50 | 1,500–3,000 | 32–63 min | 2–5 hr |
| M4 base (Metal) | 5–10 | 300–600 | 2.6–5.3 hr | 8–24 hr |
| Ryzen 5 7600X | 12–18 | 720–1,100 | 1.4–2.2 hr | 5–12 hr |
| Snapdragon 8 Gen 3 | 4–7 | 240–420 | 3.8–6.6 hr | 10–20 hr |
| Raspberry Pi 5 | 1.5–2 | 90–120 | 13–17 hr | 40–72 hr |

**Gate-adjusted:** Phases 1, 4, 5, and 6 have convergence gates (diversity, retrieval accuracy,
hallucination rate). Each gated phase may need 1.5–3× nominal steps to reliably meet criteria.
Gate-adjusted total assumes 3 of the 8 phases need 2× steps, adding 30–50% wall-clock.

**Practical training workflow recommendation:**
- **Development and phase validation:** Ryzen or A100 (Colab). Full curriculum in 1 Colab session or overnight Ryzen.
- **Iteration (hyperparameter search):** A100 or T4 (Colab). Phase-isolated, 15–60 min per phase.
- **Production training run:** A100 on Colab (1–2 hrs wall-clock for full curriculum), export to M4 for inference.
- **Pi 5 / Snapdragon:** Inference only. Not recommended for training unless no other option.

### 2.4 Phase-by-Phase Training Expectations

#### Phase 1: VQ Codec (10K steps + 5-stage curriculum)

**Model:** VQ codec (~300K params, lighter than core agent). Faster per step.
**Primary bottleneck:** Codebook diversity requirements.
- Stage 3 (codebook diversity > 95%): codebook collapse is the most common failure mode in VQ training.
  Expect 1–3 restarts with different temperature settings before this gate passes.
- Stage 5 (emoji enrichment): Layer 3 gets both reconstruction loss (residual) and emoji
  classification loss. The residual naturally captures style/emotion, emoji loss sharpens it.

**Expected convergence on M4:** 45–90 min nominal + 1–2× overhead for restarts = **2–4 hrs total**.
**Success criterion:** Reconstruction loss < 0.01 AND codebook utilization > 95%.
**Risk:** MEDIUM. Codebook collapse is common; recovery requires learning rate warmup + temperature annealing.

#### Phase 2: Base Language Model (10K steps)

**Primary bottleneck:** None significant. Straightforward cross-entropy on VQ token sequences.
**Expected convergence on M4:** 25–50 min. This is the "sanity check" phase.
**Success criterion:** Perplexity < 10 (rough). Watch for gradient explosion early (clip at 1.0).
**Risk:** LOW. Most standard LM training behavior applies here.

#### Phase 3: AddrNet (5K steps)

**Primary bottleneck:** Address entropy is invisible to loss — you can minimize reconstruction loss while
AddrNet produces degenerate addresses (all zeros, or identity-hash of input). Must monitor address
entropy bits and locality ratio metrics explicitly.
**Expected convergence on M4:** 12–25 min nominal. May be longer if address entropy fails to grow.
**Success criterion:** Address entropy > 7 bits (out of 8 max for 8-byte addresses). Locality ratio < 0.3 (addresses disperse across trie rather than clustering at root).
**Risk:** MEDIUM. Cannot rely on loss curve alone; needs explicit entropy monitoring in observability.

#### Phase 4: Memory Integration (15K steps) ⚠️ CRITICAL PHASE

**Primary bottleneck:** Write-before-read constraint (see TRAINING.md and ISSUES.md #1).
Training data must be constructed such that writes precede reads by ≥1 step.
If sync-write path fails, retrieval accuracy will be 0% and the phase will not converge.
**Expected convergence on M4:** 40–80 min (optimistic) if sync writes work correctly.
**Success criterion:** Retrieval accuracy > 90%. Gate is binary — passes or fails.
**Risk:** HIGH. This phase has the only confirmed architectural constraint that can cause total failure.
First benchmark: run Phase 4 with 100-step smoke test and check `write_before_read_success_pct` metric is 100%.

#### Phase 5: Coherence Under Load (15K steps + gate)

**Primary bottleneck:** LM loss must not regress vs Phase 2 while maintaining retrieval accuracy.
Two competing objectives. Phase 5c LR schedule reference point is implicit (see ISSUES.md #10).
**Expected convergence on M4:** 40–80 min + possible extra steps if LM loss regresses.
**Risk:** MEDIUM. The LR ambiguity (ISSUES.md #10) could cause instability.

#### Phase 6: Partiality / Controlled Imagination (10K steps) ⚠️ HARDEST PHASE

**Primary bottleneck:** No prior art for this architecture. Imagination accuracy (predicting future writes)
vs hallucination rate (inventing paths that don't exist) is a precision/recall trade-off with no known
optimal operating point. Initial runs will likely produce extremes: either never imagines (safe but useless)
or always hallucinates (creative but unreliable).
**Expected convergence on M4:** 25–50 min nominal + high probability of needing >1 run.
**Risk:** HIGH. Budget 3–5× nominal time. This phase defines the project's central claim.

#### Phase 7: Swarm Integration (20K steps)

**Primary bottleneck:** Compute scales with N_agents. At N=8 swarm agents:
- Memory: 8× core agent KV + swarm coordination overhead
- FLOPs: approximately N×linear in agents (scratchpad attention over 16 shared slots)
- Scratchpad: strength-weighted mean, fully parallel, no atomics

**Expected convergence on M4:** 55–110 min nominal at N=4–8 swarm agents.
**NOOP calibration:** NOOP (filler token) is present in training data for ALL phases at ~1-2% of positions.
Phase 7 NOOP rate should already be reasonable; swarm coordination may initially under-use it.
**Risk:** HIGH for swarm coordination quality. MEDIUM for convergence speed.

#### Phase 8: Tool Integration (10K steps)

**Primary bottleneck:** Training data quality. Tool-call datasets (function calls, structured outputs)
are the primary dependency.
**Expected convergence on M4:** 25–50 min. Likely fastest phase to converge if data quality is good.
**Risk:** LOW for convergence; HIGH for real-world tool-call quality (hard to measure offline).

---

## Part 3: Confidence Levels Summary

| Estimate | Confidence | Main uncertainty |
|---|---|---|
| Inference tok/s (cache-hot, GPU) | HIGH | Real candle Metal backend efficiency |
| Inference tok/s (DRAM ceiling) | HIGH | Arithmetic, confirmed by bandwidth analysis |
| ANE inference tok/s | MEDIUM | ANE SRAM size is estimated, not confirmed |
| Training steps/sec (GPU) | MEDIUM | candle kernel launch overhead for 96-dim ops |
| Training steps/sec (CPU) | MEDIUM-HIGH | OpenBLAS 96-dim GEMM is well-characterized |
| Phase gate timing | LOW | Curriculum gates are empirical, may need restarts |
| Phase 6 partiality convergence | VERY LOW | No prior art; exploratory research |

**First calibration point:** Phase 1 VQ training on the target hardware will provide the first
real candle training throughput number. All training estimates above should be updated after Phase 1.

---

## Part 4: Deployment Targets Summary

| Platform | Primary use | Notes |
|---|---|---|
| A100 80GB (Colab) | Training | Full curriculum in 1–2 Colab sessions |
| T4 16GB (Colab) | Training iteration | 1–4 hrs per phase |
| M4 base 24GB (GPU Metal) | Training + inference | ~8–24 hr full curriculum; inference with candle Metal |
| M4 base 24GB (ANE) | Production inference | Target: candle-coreml; 250K TPS goal achievable here |
| Ryzen 5 7600X | Development + training | Surprisingly competitive for training at this model size |
| Snapdragon 8 Gen 3 | Mobile inference | On-device via candle (CPU or GPU backend) |
| Raspberry Pi 5 | Edge inference | Functional but slow; inference only |
