# Zeno — Memory System

The hierarchical trie is Zeno's knowledge store. Model weights (~666K params)
contain zero knowledge — they only learn how to read and write. The trie stores
what was learned. Every forward pass reads from the trie. Every cycle writes
to the trie. Memory is not optional.

---

## Memory Tiers

```
  L0 (GPU):      Depth 0..K, configurable
    Direct flat-array indexing (byte value = array index)
    Lazy allocation (only populated parents get child arrays)
    Instant reads (pure tensor index), zero latency
    Static: loaded at startup, held until shutdown

  L1 (CPU RAM):  Depth K+1..cap
    Full Rust trie, RwLock for concurrent readers
    Lazy allocation (only populated parents get child arrays)
    Async read API (non-blocking, results arrive via channel)
    Batched writes (fire-and-forget from agents)
    Static: loaded at startup, held until shutdown

  L2 (Disk):     Overflow / cold storage
    mmap'd pages, OS page cache handles hot/cold
    Page-aligned buckets (4KB-aligned, works on all platforms)
    On-demand, high latency reads (~ms)

  L3 (Internet): Remote trie servers (read-only at inference)
    Subscribed trusted providers (simple config list)
    Address-based lookup (same trie protocol as local)
    Triggered by attention signal (low/uniform mem-attn)
    Results cached locally (promoted to L2/L1)
    Non-blocking async (~10-100ms latency)
    Contribution is SEPARATE offline process (privacy)

  Latency tiers:
    L0 (GPU):      ~0 ns    (tensor index)
    L1 (CPU RAM):  ~1 μs    (async, lock-free read)
    L2 (Disk):     ~1 ms    (mmap, OS page cache)
    L3 (Internet): ~10-100ms (network, async)

  Properties:
    256-ary, up to 8 levels, d_model vectors at every node
    EMA blending (write strength modulated by density-driven confidence gate)
    Ancestor propagation with depth-dependent decay
    3 AddrNet paths: read from input embed, write from output hidden
```

---

## Level-Configurable GPU Residency

```
  User configures: gpu_depth = 2  (keep levels 0, 1, 2 on GPU)

  Size estimation per depth configuration:
  ┌───────┬──────────────┬───────────────────┬──────────────────┐
  │ Depth │ Max nodes    │ At d_model=96     │ Notes            │
  │       │ (dense)      │ (bytes per node   │                  │
  │       │              │  = 96×4 = 384B)   │                  │
  ├───────┼──────────────┼───────────────────┼──────────────────┤
  │  0    │           1  │ 384 B             │ Just root        │
  │  0-1  │         257  │ 99 KB             │ Root + level 1   │
  │  0-2  │      65,793  │ 25 MB             │ Comfortable      │
  │  0-3  │  16,843,009  │ 6.4 GB (dense)    │ Dense=too much   │
  │       │              │ ~50 MB (sparse*)   │ *typical pop.    │
  └───────┴──────────────┴───────────────────┴──────────────────┘

  * Depth 0-3 sparse: depends on actual population. With lazy alloc,
    only ~130K nodes typical after moderate use → ~50MB.

  L0 and L1 are static: loaded at startup, held in memory until shutdown.
  No eviction, no dynamic resizing at runtime.
```

---

## Flat-Array Trie Node Structure

```
  Direct byte→index addressing (no hash maps):
    [Option<u32>; 256] per node — byte value = array index

  Full [Option<u32>; 256] = 1KB per node just for children.
  Solution: lazy child arrays. Only allocate when first child created.

  struct TrieLevel {
      values: Vec<[f32; D_MODEL]>,    // contiguous value storage
      children: Vec<Option<Box<[Option<u32>; 256]>>>,  // lazy child arrays
      // No write_count — confidence derived from structural density
      // (populated children count) + write diffusion halo
  }

  Or better — page-aligned bucket storage:

  struct TrieBucket {
      // One bucket = one group of nodes at same depth
      // 4KB-aligned: 4096 / 384 bytes per node (d=96) = 10 → use 8 nodes per bucket (power of 2)
      values: [[f32; D_MODEL]; BUCKET_SIZE],  // BUCKET_SIZE = 8
      children: [Option<u32>; BUCKET_SIZE * 256],
      // populated_children derived on read: count non-None in children slice
      // Zero extra storage — the trie structure IS the confidence signal
  }
```

---

## Write Diffusion — Gravitational Density Halo

```
  Each write to a leaf PROBABILISTICALLY creates weak sibling and cousin
  nodes nearby, gradually building a density halo around frequently-written
  addresses. The halo IS the confidence signal — no metadata counters needed.

  Mechanism:
  ──────────
  On each primary write to leaf address [A,B,C,D,E,F,G,H]:

    Level 0 — Sibling diffusion (same parent, different last byte):
      Probability: p₁ ≈ 0.05 per write
      Target:      [A,B,C,D,E,F,G,X]  where X ≠ H, chosen randomly
      Strength:    α_primary × 0.02
      Creates:     1 new leaf node under same parent (if not exists)

    Level 1 — Cousin diffusion (grandparent's other child):
      Probability: p₂ ≈ 0.01 per write
      Target:      [A,B,C,D,E,F,Y,Z]  where Y ≠ G, Z random
      Strength:    α_primary × 0.005
      Creates:     1 intermediate node + 1 leaf (2 new nodes)

    Level 2 — Second cousin diffusion (great-grandparent):
      Probability: p₃ ≈ 0.002 per write
      Target:      [A,B,C,D,E,X,Y,Z]  where X ≠ F, Y,Z random
      Strength:    α_primary × 0.001
      Creates:     up to 3 new nodes along the path

  Diffused values: weakened COPY of the primary write value.
  Not empty markers — the sibling carries a faint version of the concept.


  Halo growth over time:
  ──────────────────────
  Writes    Siblings    Cousins    Parent density   GP density
  ────────  ──────────  ─────────  ──────────────   ──────────
     1         0           0         1/256            unchanged
    10        ~0-1         0         ~1-2/256         unchanged
    50        ~2-3        ~0-1       ~3-4/256         ~1-2 more
   100        ~5          ~1         ~6/256           ~2 more
   500        ~25         ~5         ~26/256          ~6 more
  1000        ~50         ~10        ~51/256          ~11 more


  Why this works:
  ───────────────
  1. Imagination (single write) → no halo → sparse → model knows it's uncertain
  2. Repeated observation → growing halo → density increases → confidence rises
  3. Well-established concept → dense gravity well → model trusts this data
  4. The halo carries real information (weakened concept copies)
     If model reads a nearby address, it gets faint related data
  5. Density is STRUCTURAL — derived from trie shape at read time
     No counters, no metadata, no session tracking

  Implementation:
  ───────────────
  Write diffusion runs in the SAME fire-and-forget write path.
  Primary write + diffusion writes all go to the write buffer.
  Random byte selection: use fast PRNG (xoshiro256) seeded per write.
  Check node existence before creating (avoid redundant allocation).
  Diffusion probability can be tuned per trie tier:
    L0 (GPU):  higher p (fast to allocate)
    L1 (CPU):  moderate p
    L2 (disk): lower p (node creation is I/O)

  Write diffusion is LOCAL ONLY — diffusion never propagates to L3.
  All diffused writes go to the same local write buffer.

  Storage cost:
  ─────────────
  Per 1M primary writes → ~50K diffusion nodes → ~20MB at d_model=96
  (Each node = 96×4 = 384 bytes for value + children index)
  Negligible compared to primary node storage.
```

---

## Async Read/Write Paths

```
  Agent issues non-blocking read request:
    ┌─────────┐                    ┌──────────┐
    │  Agent  │──read_request───►  │  Memory  │
    │         │                    │  Manager  │
    │         │◄─L0_result (0ns)── │  (async)  │
    │         │                    │           │
    │ ...continues processing...   │           │
    │         │                    │           │
    │         │◄─L1_result (μs)─── │           │
    │         │                    │           │
    │ ...next tile uses result...  │           │
    │         │                    │           │
    │         │◄─L2_result (ms)─── │           │
    │         │                    │           │
    │ ...if mem-attn signal low... │           │
    │         │                    │           │
    │         │◄─L3_result (10ms+) │ ◄── Remote trie server(s)
    └─────────┘                    └──────────┘

  Agent NEVER blocks. L0 is instant (GPU tensor index).
  L1/L2 results arrive asynchronously and get merged into the
  unified attention pool's available slots. More time = more detail.
  L3 triggers only when attention signal indicates insufficient local
  knowledge — results cached locally (promoted to L2/L1) for future reads.

  Read path returns STRUCTURAL PROVENANCE alongside values:
    For each memory vector in the ancestor chain:
      value:              d_model float vector (the stored representation)
      populated_children: u8 (how many of 256 child slots are populated)
      depth:              u8 (node depth in trie, 0=root, 7=leaf)

    The density chain [populated_children at each depth] IS the confidence
    signal. Dense ancestors = well-explored territory. Sparse ancestors =
    uncharted territory (possibly imagined). The confidence gate
    (see PARTIALITY.md) converts the density chain into a scalar that
    modulates attention weights.

    Write diffusion ensures that frequently-written addresses develop
    density halos, making the structural signal grow with repeated
    observation.

  Write path: fire-and-forget (LOCAL trie only — never to L3 during inference).
    WRITE addresses ≠ READ addresses:
      Read addrs  = AddrNet(input_embedding)     — "where is relevant knowledge?"
      Write addrs = AddrNet(processed_hidden)     — "where should I store this?"
      Same 3 AddrNets, different inputs. Content-to-address mapping is one
      learned function — similar content → nearby addresses.
    Write values: base + aspect residuals (3 different per-channel values):
      base  = V_proj(hidden)              — shared core representation
      val_i = base + aspect_i(hidden)     — channel-specific emphasis (96→16→96)
    Write strength: α_effective = sigmoid(strength_head(hidden)) × confidence_gate × base_α
      confidence_gate comes from the learned density gate (see PARTIALITY.md):
      Sparse source → low confidence → weak write (tentative).
      Dense source → high confidence → stronger write (confident).
      Always writes (trail of thoughts), but model AND density control intensity.
    Write diffusion: primary write + probabilistic sibling/cousin writes
      (see § Write Diffusion above). LOCAL ONLY — diffusion never propagates to L3.
      All diffused writes go to the same local write buffer.
    Agent → write_buffer (lock-free queue) → background thread → L1/L2
    L0 cache updated inline (GPU tensor write, fast).
```

---

## Memory Roles

```
  System       Purpose                              Speed       Scope
  ─────────   ──────────────────────────────────   ─────────  ──────────
  Trie (L0-3) Knowledge RETRIEVAL                  0ns-100ms  Persistent
              "What does speech sound like?"
              "What pattern matches these bytes?"
              Rich semantic context from training + past inference
              NOT for tile-to-tile state passing
              NOT for adjacent-chunk coordination

  Scratchpad  Agent COORDINATION + state sharing    ~0ns       Ephemeral
              Inter-tile coherence (neighbor hidden states)
              Inter-chunk context (previous chunk's final state)
              Environmental context for new work units
              Coarse codes shared for refinement passes
              Resets between unrelated work groups

  Headers     Temporal/spatial POSITIONING          N/A        Per-unit
              "This tile covers t=3.5s to t=4.78s"
              "This is the top-left quadrant"
              "content-type: audio/vq256-k4"
              Absolute positioning — no dependency on other tiles
```

---

## L3 Internet Memory — Remote Trie Tier

```
  L3 is a read-only (during inference) remote trie tier that extends
  local memory to networked knowledge sources.

  ┌─────────── L3 Architecture ─────────────────────────────────────┐
  │                                                                  │
  │  INFERENCE (read-only):                                          │
  │                                                                  │
  │    Trigger: Memory cross-attention weights are low/uniform       │
  │             → agent isn't finding useful local context            │
  │             → L3 lookup scheduled for next cycle                 │
  │                                                                  │
  │    Flow:                                                         │
  │      1. Agent's AddrNet generates 3 addresses (normal path)      │
  │      2. L0→L1→L2 traversal (local trie)                         │
  │      3. Attention signal check: are mem-attn weights useful?     │
  │      4. If NOT → forward addresses to subscribed remote servers  │
  │      5. Remote server traverses its own trie at same addresses   │
  │      6. Returns ancestor vectors (same format as local read)     │
  │      7. Results arrive async, merge into attention pool           │
  │      8. Results cached locally: written to L2/L1 for future use  │
  │                                                                  │
  │    Protocol:                                                     │
  │      READ request:  [address: [u8; 8]]                           │
  │      READ response: [vectors: Vec<(depth, Vec<f32>)>] or MISS   │
  │      Transport:     gRPC or TCP + protobuf                       │
  │      Payload size:  request ~8 bytes, response ~2.4KB typical    │
  │                     (25 ancestors × 96 floats × 4 bytes)         │
  │                                                                  │
  │    Subscription: simple config list of trusted trie endpoints    │
  │      servers:                                                    │
  │        - url: "trie://knowledge.local:9090"                      │
  │          name: "local-wiki"                                      │
  │        - url: "trie://provider.example.com:9090"                 │
  │          name: "shared-knowledge"                                │
  │    All subscribed servers queried on L3 trigger (broadcast).     │
  │    First valid response wins. Others cached if they arrive.      │
  │                                                                  │
  │  CONTRIBUTION (separate offline process — NEVER during inference)│
  │                                                                  │
  │    Why separate: inference writes carry personal/conversation     │
  │    context. Contributing during inference would leak private data │
  │    to remote servers.                                            │
  │                                                                  │
  │    Flow:                                                         │
  │      1. Standalone CLI/daemon connects to remote trie server     │
  │      2. Loads content (documents, web crawl, structured data)    │
  │      3. Runs Zeno model using REMOTE trie as memory source       │
  │         (reads from remote, writes to remote — maintains          │
  │          coherency within the remote trie)                       │
  │      4. Server applies writes with contributor's trust_weight    │
  │      5. Automizable: cron jobs, crawlers, CI pipelines           │
  │                                                                  │
  │    Trust & anti-override:                                        │
  │      effective_α = base_α × trust_weight × novelty               │
  │                                                                  │
  │      trust_weight: [0.0, 1.0] per contributor, set by server     │
  │        1.0 = full trust (admin/owner)                            │
  │        0.5 = established contributor                              │
  │        0.1 = new contributor (minimal impact per write)           │
  │        0.0 = read-only (writes are no-ops)                       │
  │                                                                  │
  │      novelty: 1.0 - cosine_sim(new_value, existing_value)        │
  │        Writing what's already there → near-zero impact            │
  │        Writing genuinely new content → full impact                │
  │        Zero extra storage (compares against current trie state)   │
  │                                                                  │
  │      Note: spatiotemporal addressing provides natural protection  │
  │      — different timestamps → different addresses → past writes   │
  │      are not easily overridable.                                  │
  │                                                                  │
  │    V2 security (deferred):                                       │
  │      Per-contributor per-address write limits                     │
  │      Count-Min Sketch for bounded memory tracking                 │
  │      Influence budget caps                                        │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```
