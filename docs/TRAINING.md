# Zeno — Training Curriculum

8-phase progressive freeze/unfreeze curriculum. Each phase builds on stable
foundations from the previous phase. The core principle: **train easiest
components first, freeze what's learned, train harder parts on top, gradually
unfreeze for coherence.**

AddrNet trains AFTER the base model produces meaningful hidden states — never
on random noise. AddrNet stays frozen or very-low-LR after Phase 3 to prevent
address space collapse.

---

## Phase 1 — Text VQ Codec (separate model, ~300K params)

```
  Goal:    Text VQ codec with semantic refinement layers
  Train:   Text VQ codec ONLY (separate model, ~300K params)
           Core agent is NOT involved yet.
  Data:    Diverse multilingual text (wiki, code, chat, social media)
           Include text WITH emojis for enrichment bootstrapping.

  Semantic refinement layer design:
    Layer 1 (K=1, coarse): Semantic intent / pseudocode
      "return boolean true", "greet user positively", "define function"
      → Language-agnostic concepts. 256 entries = 256 semantic atoms.
    Layer 2 (K=2, mid): Target language / actual words
      "return True" (Python), "return true;" (Rust), "Hey! How are you?"
      → Language-specific syntax and wording.
    Layer 3 (K=3, fine): Format / style / detail / emotion
      "    return True  # early exit\n", "Hey! 😊 How are you?"
      → Exact formatting, comments, emoji, whitespace, type annotations.

  Step 1 — Reconstruction (autoencoder):
    text bytes → Perceiver encoder → RVQ → Perceiver decoder → text bytes
    Loss: cross-entropy per reconstructed byte (exact match required)
    K=3 RVQ layers, codebook size = 256 each, d_codec=64, M=256 latents
    Standard progressive RVQ: each layer encodes cumulative residual.
    All 3 layers get reconstruction loss (Layer N loss = residual after Layers 1..N-1).
    Gate: reconstruction accuracy > 99.5% across all scripts (all 3 layers)

  Step 2 — Codebook diversity:
    Loss: maximize entropy of code usage per RVQ layer
    Random restarts for underused codes
    EMA codebook updates
    Gate: all 256 codes used > 0.1% frequency per layer

  Step 3 — Emoji enrichment (Layer 3 fine-tuning):
    Data: text + emoji pairs (social media, chat)
    Loss: predict emoji from Layer 3 codes (classification)
    Layer 3 residual naturally captures style/emotion (reconstruction handled by L1+L2).
    Additional emoji classification loss sharpens L3's emotional signal.
    Gate: emoji prediction accuracy > 70% (top-5)

  Step 4 — Continuous emotional space:
    Expand from categorical emoji prediction to continuous VAD vectors
    VAD = Valence/Arousal/Dominance
    Each emoji maps to VAD: 😊=(0.8,0.5,0.5), 😤=(-0.5,0.8,0.9)
    Loss: Layer 3 latent → predict VAD vector (MSE)

  Step 5 (optional, Phase 3+ data required) — Audio-text pairing:
    Paired speech transcripts → align text latents with audio features
    Text codec learns prosodic cues from speech
    "How are you?" spoken happily vs sarcastically → different Layer 3

  Duration: ~10K steps
  Outcome: Frozen text VQ codec. Core agents train on its output codes.

  Logging:
    recon_loss: X.XXX          (reconstruction cross-entropy, target: low enough for >99.5% accuracy)
    recon_accuracy: X.XX%      target > 99.5%
    codebook_usage[layer]: min_freq    target > 0.1% per code per layer
    emoji_top5_acc: X.XX%      target > 70%
```

---

## Phase 2 — Base Language Model (no memory)

```
  Goal:    Learn VQ code patterns, code embeddings, attention
  Train:   Full core agent (embedding, self-attn, context cross-attn,
           mem_attn, FFN, norms, tag system, register bank, LM head)
           EXCEPT: AddrNet (random init, frozen — addresses meaningless
           without trained base), V_proj, aspect heads, write_strength,
           confidence_gate (all memory-write components stay frozen)
  Memory:  OFF (no trie access)
  Data:    Wiki + shell + code → TEXT VQ ENCODED (using frozen Phase 1 codec)
           All training data passes through text VQ codec first.
           Core agent trains on VQ codes, never raw bytes.
  Loss:    Causal LM (next VQ code prediction)

  Gate criterion: LM loss < 3.0 (code-level perplexity < 20)
  Duration: ~10K steps
  Outcome: Meaningful hidden states. Stable embeddings + attention.

  Logging:
    lm_loss: X.XXX             target < 3.0
    mem_attn_mean_abs: X.XXXX  (expected near-zero, memory is OFF — baseline for Phase 4)
```

---

## Phase 3 — Address Formation (freeze base)

```
  Goal:    Content-aware address space from meaningful hidden states
  Freeze:  Everything from Phase 2 (embedding, self-attn, context,
           mem_attn, FFN, norms, tags, register, LM head)
  Train:   3× AddrNet ONLY
  Data:    Diverse VQ-encoded text (wiki, code, chat)
  Method:
    1. Run FROZEN base model on VQ-encoded text → meaningful hidden states
    2. AddrNet generates addresses from these real representations
    3. Losses:
       a. Diversity loss: maximize entropy of address distribution
          (prevent address collapse to few bins)
       b. Locality loss: similar hidden states → nearby addresses
          (cosine similarity of hidden → address prefix overlap)
       c. Depth utilization: penalize always-shallow or always-deep
          (encourage full use of 8 levels)
    4. Gumbel-softmax temperature annealing: start high (τ=5), decay to τ=0.5

  Why after base (not before):
    Training AddrNet on random hidden states produces meaningless addresses.
    Locality loss on noise is meaningless. Now AddrNet learns content-
    aware addressing from real semantic representations.

  Gate criterion: Address entropy > 7.0 bits (out of 8.0 max)
                  AND depth distribution is approximately uniform
                  AND locality: similar inputs → overlapping address prefixes
  Duration: ~5K steps
  Outcome: Stable, well-distributed, content-aware addresses.

  Logging:
    addr_entropy: X.XX         target > 7.0 (bits, out of 8.0 max)
    depth_distribution_std: X.XX   (target: approximately uniform across 8 levels)
    locality_correlation: X.XX     (similar inputs → overlapping address prefixes)
    gumbel_temperature: X.XX       (annealing from 5.0 → 0.5)
```

---

## Phase 4 — Memory Integration + Confidence Gate (freeze base + AddrNet)

```
  Goal:    Learn to read/write trie effectively + confidence gating
  Freeze:  Base model (from Phase 2) + AddrNet (from Phase 3)
  Train:   V_proj, aspect heads, write_strength, mem_attn weights,
           confidence_gate (density chain → confidence)
  Memory:  ON (trie active, L0+L1)
  Data:    Same as Phase 2 + QA pairs (write fact → read back → answer)
  Losses:
    a. LM loss (through memory path — trains mem_attn to use trie)
    b. Contrastive address loss (same passage → similar addresses)
    c. Retrieval accuracy (write value, read back, compare)
    d. Access cost (penalize attention to expensive L1/L2 slots)
    e. Data quality (penalize attending to low-quality/stale vectors)
    f. Confidence calibration:
       - Write facts to trie (creates dense neighborhoods via diffusion)
         → read back → confidence gate should output HIGH
       - Read from unpopulated addresses (sparse neighborhoods)
         → confidence gate should output LOW
       - Train confidence gate to correlate with density chain

  Single-pass forward (no two-pass bootstrap):
    Read trie → forward with memory → loss → generate write addresses → write
    Confidence gate + density signals provide gradient flow through memory path.

  Trie during training:
    Full L0 trie on GPU — same 256-ary structure, K=8 levels, as production.
    AddrNet (frozen) generates hard addresses. Reads and writes hit the real trie.
    Gradient flows through the continuous values stored in the trie:
      trie stored value → mem_attn weights → LM loss
    Confidence gate sees real structural density (populated_children counts).
    
    Training data volume is kept small so the trie stays GPU-resident.
    Periodic trie resets bound memory growth and teach the model to rebuild
    context from scratch — useful for inference robustness. Reset frequency is
    determined by device memory budget and dataset size (reset when trie
    approaches GPU memory limit).

  Re-encounter training data (write_strength + confidence calibration):
    Training sequences must include re-encounter patterns:
    - Write fact X at step N → encounter same content again at step M > N
    - At step M: trie node at that address is already populated (from halo or direct write)
    - write_strength should be LOW → gentle nudge, accumulate without overwriting:
        trie[addr] = (1 - α) × trie[addr] + α × new_value,  α = write_strength
    - confidence gate should be HIGH → dense neighborhood = reliable data, attend strongly
    write_strength and confidence gate are separate mechanisms:
      write_strength = blend ratio for writes (low = nudge, high = overwrite)
      confidence gate = read attention strength (dense = trust, sparse = uncertain)
    Training data construction must include deliberate re-encounter sequences.

  Training data constraint (critical):
    The step order above means writes land in L0 at END of step N and are
    readable at the START of step N+1. QA-pair sequences MUST be constructed
    so that the write step precedes the read-back step by at least one step:
      step N:   agent outputs write instruction → fact written to L0 at step end
      step N+1: query requires reading that fact → found in L0 → retrieval loss computed
    Constructing write and read-back in the same step will yield 0% retrieval
    accuracy because L0 is read at the start of the same step (before the write).
    This is a training data construction rule, not a code constraint.

  Gate criterion: Retrieval accuracy > 90%
                  AND LM loss with memory < LM loss without memory
                  AND confidence gate correlates with density (r > 0.7)
  Duration: ~15K steps
  Outcome: Working memory read/write. Calibrated confidence gate.

  Logging:
    retrieval_accuracy: X.XX%      target > 90%
    lm_loss_with_memory: X.XXX     target < lm_loss_without_memory
    confidence_density_corr: X.XX  target > 0.7
    mem_attn_mean_abs: X.XXXX      target > 0.01 (if < 0.01 after 500 steps → warmup)
    write_strength_mean: X.XX      (monitor for collapse to 0 or 1)
    trie_node_count: XXXXX         (monitor GPU memory, reset when approaching limit)
```

---

## Phase 5 — Coherence Unfreeze (gradual)

```
  Goal:    All components work together coherently
  Method:  Progressive unfreezing in sub-phases:

  Phase 5a — Unfreeze memory cross-attention (keep base + AddrNet frozen)
    Train: mem_attn weights + V_proj + aspect heads + write_strength
    LR: standard
    Duration: ~5K steps

  Phase 5b — Unfreeze context cross-attention + register bank
    Train: context cross-attn + register access + everything from 5a
    Keep frozen: self-attn, FFN, embedding, AddrNet
    LR: standard for newly unfrozen, 0.5× for 5a components
    Duration: ~5K steps

  Phase 5c — Unfreeze self-attention + FFN
    Train: all base components + all memory components
    Keep frozen: AddrNet (ALWAYS low LR after Phase 3)
    LR base: Phase 4 final learning rate (from checkpoint['lr'])
    LR: 0.1× base for self-attn/FFN (prevent catastrophic forgetting)
         0.5× base for context/memory components
         0.01× base for AddrNet (minimal drift, or fully frozen)
    Duration: ~5K steps

  Gate criterion: LM loss not degraded vs Phase 4
                  AND retrieval accuracy maintained > 90%
                  AND no address distribution collapse
  Duration: ~15K steps total
  Outcome: Coherent model where all components cooperate.

  Logging:
    lm_loss: X.XXX             target <= phase4_final_lm_loss
    retrieval_accuracy: X.XX%  target > 90%
    addr_entropy: X.XX         target > 7.0 (no collapse during unfreeze)
```

---

## Phase 6 — Partiality Training

```
  Goal:    Teach model that content is partial, build imagination skills
  Train:   Full model, AddrNet at very low LR (0.01×)
  Memory:  ON (full tiered)
  Data:    PARTIAL descriptions paired with full content:
           - Image + multiple partial captions (each describes ~15%)
           - Full text + summary (summary describes ~10%)
           - Audio + partial text description
           - QA where answer requires imagining beyond given facts

  Method:
    1. Write FULL content to trie (e.g., all details of an image)
       → creates dense neighborhoods via write diffusion
    2. Present PARTIAL description to agent (e.g., "a dog in a park")
    3. Agent reads trie — gets mix of dense paths (written areas,
       high density) and sparse paths (unwritten, low density)
    4. Agent must generate response acknowledging what's known vs imagined

  Losses:
    a. LM loss (still predicting next VQ codes)
    b. Asymmetric matching:
       loss = -log P(partial description | full content)
       NOT: -log P(full content | partial description)
       Partial descriptions should be CONSISTENT WITH full content,
       not required to CAPTURE ALL of it.
    c. Confidence prediction:
       For each memory read, model predicts its own confidence level.
       Ground truth: dense neighborhood (from full write + diffusion)
       vs sparse neighborhood (never written). Density-based signal.
    d. Imagination quality:
       Compare imagined fine-level representations with actual
       fine-level data (when available from full content write).
       Reward accurate imaginations, don't penalize creative ones
       that are consistent but different.
    e. Hallucination penalty:
       Penalize generating SPECIFIC claims from imagined data.
       "There's probably greenery" (hedged, OK from imagination)
       vs "There are exactly 3 oak trees" (specific, BAD from imagination)

  Multi-view training protocol:
    Same content, 3-5 different partial descriptions:
      Image: caption1 "a dog", caption2 "sunny park", caption3 "wooden bench"
    All get paired with the same full-content trie state.
    Model learns: each is one valid partial view of many possible views.

  RVQ layer dropout:
    During training, randomly drop RVQ layers 2 and/or 3.
    Model sees Layer 1 only (gist) → must work with partial VQ info.
    Teaches: "I might only have coarse codes, handle it."

  Gate criterion: Imagination accuracy > 60% (imagined vectors close to actual)
                  AND confidence correlates with neighborhood density (r > 0.8)
                  AND no hallucination increase in eval
  Duration: ~10K steps
  Outcome: Model understands partiality, calibrated imagination.

  Logging:
    partiality_loss: X.XXX         (asymmetric matching loss)
    imagination_accuracy: X.XX%    target > 60%
    confidence_density_corr: X.XX  target > 0.8
    hallucination_rate: X.XX%      (monitor, no increase from Phase 5 baseline)
```

---

## Phase 7 — Swarm Training (multi-agent coordination)

```
  Goal:    Multi-agent coordination, scratchpad-based communication
  Train:   Full model, AddrNet at very low LR (0.01×)
  Memory:  ON (full tiered, async)
  Data:    Long documents (multi-tile), chat, QA
  Method:
    1. Tile documents into N chunks, process as batch of N agents
    2. All inter-agent communication flows via shared scratchpad (16 slots)
    3. All agents share same weights, different tiles
  Losses:
    a. LM loss per tile (each agent predicts next codes in its tile)
    b. Scratchpad utilization (reward agents that use scratchpad productively)
    c. Scratchpad efficiency (minimize redundant writes)
    d. NOOP supervision (ingestion agents should output NOOP)
    e. Tool token accuracy (correct tool invocation format)

  Duration: ~20K steps
  Outcome: Working swarm with tile coordination.

  Logging:
    lm_loss_per_tile: X.XXX        (per-agent loss, should be comparable across tiles)
    noop_rate: X.XX%               (should emerge naturally, ~1-5% in collaborative tasks)
    scratchpad_utilization: X.XX   (fraction of 16 slots actively written per step)
    multi_agent_coherence: X.XX    (cross-tile consistency metric)
```

---

## Phase 8 — Tool & Chat Fine-tuning

```
  Goal:    Practical chat + tool use
  Train:   Full model + optional LoRA adapters for tool specialization
  Memory:  ON (full)
  Data:    Multi-turn chat, tool-use examples, QA with trie recall
  Losses:  LM + tool format accuracy + response quality

  Duration: ~10K steps
  Outcome: Deployment-ready chat agent.

  Logging:
    tool_dispatch_accuracy: X.XX%  (correct tool invocation format + selection)
    chat_coherence: X.XX           (multi-turn consistency metric)
    lm_loss: X.XXX                 (should not degrade from Phase 7)
```

---

## Dataset Requirements

Training data sources (HuggingFace + custom curation):

| Phase | Data Needed | Sources |
|---|---|---|
| 1 (VQ Codec) | Diverse multilingual text + emoji pairs | Wikipedia multilingual, Common Crawl, social media corpora |
| 2 (Base LM) | Wiki + code + shell transcripts (VQ-encoded) | Wikipedia, The Stack, shell command datasets |
| 3 (AddrNet) | Same as Phase 2 (diverse VQ-encoded text) | Reuse Phase 2 data |
| 4 (Memory) | QA pairs + same as Phase 2 | SQuAD, Natural Questions, TriviaQA + Phase 2 data |
| 5 (Coherence) | Same as Phase 4 | Reuse Phase 4 data |
| 6 (Partiality) | Image+caption pairs, summaries, audio+text | COCO Captions, Flickr30k, CNN/DailyMail, LibriSpeech |
| 7 (Swarm) | Long documents, multi-turn chat | BookCorpus, Reddit conversations, long-form QA |
| 8 (Tools+Chat) | Tool-use examples, chat | ShareGPT, tool-use datasets, custom chat data |

All text data passes through the frozen Phase 1 VQ codec before use in Phases 2+.
Core agents never see raw UTF-8 bytes.

All training data across all phases includes NOOP filler tokens at natural pause points
(~1-2% of positions). NOOP is a verbal filler ("ehm") — the model emits it when it needs
more processing time. Maps to UTF-8 ACK at the Rust runtime boundary. Dataset construction
places NOOP where complicated concepts or low-confidence reads would naturally cause a pause.

Phase 4+ training data must include re-encounter sequences (same content written at step N,
encountered again at step M > N) to calibrate write_strength and confidence gate.

---

## Training Principles

1. **Easiest first** — VQ codec → base LM → addresses → memory → coherence →
   partiality → swarm → tools. Each phase builds on stable foundations.

2. **Progressive freeze/unfreeze** — Freeze what's learned, train harder parts
   on top, gradually unfreeze for coherence. Prevents catastrophic forgetting.

3. **AddrNet is fragile** — Always frozen or very-low-LR after Phase 3. Address
   space collapse is the #1 training risk. Monitor entropy continuously.

4. **Gate criteria** — Every phase has explicit gate criteria. Do not advance
   until gates are met. Premature advancement compounds errors.

5. **Single-pass forward** — No two-pass bootstrap. Confidence gate + density
   signals provide gradient flow through memory path.

6. **candle-only** — All training runs in pure Rust via candle. No PyTorch
   dependency. Accept the development cost for ecosystem consistency.
