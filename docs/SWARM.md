# Zeno — Swarm Coordination

Multi-agent coordination, tiling, work decomposition, ingestion, generation,
and tool dispatch.

---

## Three Dimensions of Work Decomposition

```
  Zeno decomposes work along three orthogonal axes:

  1. CHUNKING (temporal, sequential)
     ─────────────────────────────
     Byte stream → 256-token chunks, processed sequentially.
     Each chunk depends on the previous (context flows via scratchpad).
     One agent handles all chunks within a stream.

     Text:  "The quick brown fox..." → Chunk 0 [0:256] → Chunk 1 [256:512]
     Audio: 10 sec audio → Chunk 0 [0:1.28s] → Chunk 1 [1.28:2.56s] → ...
     Video: frame 0 → frame 1 → frame 2 → ... (sequential in time)

  2. TILING (spatial/spectral, parallel)
     ────────────────────────────────────
     One task decomposed into sub-tasks processed by different agents
     simultaneously. Each tile works on a DIFFERENT PART of the same
     content. Scratchpad provides coherence between tiles.

     Image: 512×512 → 4 spatial quadrants, one agent per quadrant
     Audio: full-band → 3 frequency tiles (bass/mid/treble), one agent each
     Text:  long document → section tiles, one agent per section
     Video: each frame → spatial quadrants (parallel within frame)

  3. REFINEMENT (quality hierarchy, iterative)
     ───────────────────────────────────────
     Coarse pass produces rough structure, fine passes add detail.
     Maps naturally to Residual VQ for audio/image.
     Maps to draft→refine for text generation.

     Audio: RVQ layer 1 (coarse) → layer 2 (mid) → layer 3 (fine)
     Image: low-res sketch → mid detail → high detail
     Text:  outline → draft → polished

  Combined: a video generation task uses ALL THREE:
    Chunking:    frame-by-frame temporal sequence
    Tiling:      each frame split into spatial quadrants (parallel)
    Refinement:  coarse scene codes → fine visual/audio detail
```

---

## Tiling by Modality

```
  IMAGE — Spatial Tiling
  ──────────────────────
  512×512 image → 4 tiles of 256×256 (or 16 tiles of 128×128)

  Each tile gets a spatial header:
    location: image://photo.jpg/tile/0/0     (row 0, col 0 = top-left)
    content-type: image/vq256-k4

  ┌─────────┬─────────┐
  │ Agent 0 │ Agent 1 │  4 agents run in parallel
  │ (0,0)   │ (0,1)   │  Scratchpad shares edge pixels/codes
  ├─────────┼─────────┤  between adjacent tiles for coherence
  │ Agent 2 │ Agent 3 │
  │ (1,0)   │ (1,1)   │  Trie provides: "what does sky look like?"
  └─────────┴─────────┘

  Each agent's byte stream:
    VQ-256 codes for its spatial region
    Refinement: coarse codes first, then fine (progressive within tile)


  AUDIO — Frequency-Band Tiling
  ──────────────────────────────
  Full-band audio → 3 frequency tiles via filterbank decomposition

  ┌──────────────────────────────────┐
  │ Agent 0: Low  (0-300Hz)    bass  │  Codec agent splits audio into
  │ Agent 1: Mid  (300-4kHz)   voice │  frequency bands before tiling
  │ Agent 2: High (4kHz+)     treble │
  └──────────────────────────────────┘

  Each tile gets a spectral header:
    location: audio://clip.wav/band/low
    content-type: audio/vq256-k4-band

  Scratchpad shares: cross-band energy for coherence
    (bass drum hit should correlate with high-freq transient)
  Trie provides: "what does a bass drum pattern sound like?"


  TEXT — Section/Aspect Tiling
  ────────────────────────────
  Long document → section tiles OR parallel generation aspects

  Section tiling (ingestion):
    Chapter 1 → Agent 0
    Chapter 2 → Agent 1
    Chapter 3 → Agent 2

  Aspect tiling (generation):
    Agent 0: generate narrative structure
    Agent 1: generate dialogue
    Agent 2: generate descriptions
    → Merge and refine via additional pass


  VIDEO — Spatial + Spectral + Temporal
  ──────────────────────────────────────
  Video combines all three axes:

  Chunking (temporal):
    frame 0 → frame 1 → ... → frame 23 (1 second at 24fps)
    Sequential: each frame depends on previous via scratchpad

  Per-frame tiling (spatial parallel):
    ┌─────────┬─────────┐
    │ Agent 0 │ Agent 1 │  4 image tile agents
    ├─────────┼─────────┤
    │ Agent 2 │ Agent 3 │
    └─────────┴─────────┘

  Audio tiling (spectral parallel, synchronized):
    ┌──────────────────────────┐
    │ Agent 4: bass             │  3 audio tile agents
    │ Agent 5: mids             │  process same time range
    │ Agent 6: treble           │  as image tiles above
    └──────────────────────────┘

  Refinement (quality hierarchy):
    Shared coarse codes → image fine → audio fine
    Audio fine codes conditioned on image coarse → lip-sync for free

  Scratchpad coordinates ALL of this:
    - Spatial coherence: edge codes shared between image quadrants
    - Spectral coherence: cross-band energy between audio tiles
    - Temporal coherence: previous frame summary for next chunk
    - Cross-modal coherence: shared coarse codes link image + audio
```

---

## Chunking and Work Queue

```
  Chunking handles sequential/temporal progression within a stream.

  Spatiotemporal boundaries define work units:
  Input is NOT split into fixed 256-byte chunks. The Rust runtime
  splits at SPATIOTEMPORAL BOUNDARIES — when author, source, time,
  or position changes, that's a new work unit.

  Examples:
    PDF (3 pages):     page1 → Unit0, page2 → Unit1, page3 → Unit2
    Chat:              alice@10:00 → Unit0, bob@10:01 → Unit1
    Code file:         func_a → Unit0, func_b → Unit1
    Audio:             0-1.28s → Unit0, 1.28-2.56s → Unit1, ...

  Work Queue:
  ┌──────────────────────────────────────────────────────────────┐
  │ [alice@10:00 (50B)] [bob@10:01 (300B)] [readme.md (800B)]   │
  └────────┬───────────────────┬───────────────────┬─────────────┘
           ▼                   ▼                   ▼
       Agent 0             Agent 1             Agent 2
    (done in 50 cyc,    (256 cyc, then      (256→256→256→32,
     picks up next)      44 more)            sequential)

  Rules:
    - Each unit has ONE coherent spatiotemporal context
    - Units up to 256 tokens processed in one pass (one chunk)
    - Units > 256 tokens: SAME agent continues sequentially
      (keeps scratchpad context across chunk continuations)
    - When agent finishes, it picks up next unit from queue
    - Tiled units (image quadrants, audio bands) are PARALLEL
      work units in the same queue — assigned to different agents
```

---

## Within a Chunk

```
  Within a chunk, each agent:
    1. Read TRIE (knowledge retrieval — "what's relevant to this?")
    2. Read SCRATCHPAD (coordination — neighbor states, prev chunk context)
    3. Forward through transformer with unified attention
    4. Write to TRIE (fire-and-forget, async — store learned representations)
    5. Write to SCRATCHPAD (share hidden states with neighbors)
    6. Output NOOP tokens during ingestion (nothing to generate yet)
```

---

## Ingestion (Processing Input)

```
  Input: Chat conversation with multiple speakers

  Rust runtime parses spatiotemporal boundaries:
    alice@10:00: "Hello, how are you today?"    → Unit 0 (50 bytes raw)
    bob@10:01:   "I'm good! Working on the..."  → Unit 1 (300 bytes raw)
    alice@10:02: "Nice!"                         → Unit 2 (6 bytes raw)

  Text VQ codec encodes each unit (headers + content):
    Unit 0: 50 raw bytes → text VQ encoder → ~30 VQ codes (K=3 RVQ)
    Unit 1: 300 raw bytes → text VQ encoder → ~180 VQ codes
    Unit 2: 6 raw bytes → text VQ encoder → ~6 VQ codes

  Agent pool processes VQ codes:
    Agent 0 ← Unit 0 (30 code cycles → done → picks up Unit 2)
    Agent 1 ← Unit 1 (180 code cycles → done)
    Agent 0 ← Unit 2 (6 code cycles → done → idle)

  Dual-channel tag injection still applies:
    Inline: VQ codes carry header+content (compressed together)
    Side-channel: tag_encoder([author_vec, location_vec, time_vec, ...])
    The side-channel provides STRUCTURED tag info that supplements
    what the VQ codes carry in compressed form.

  Image ingestion (tiled):
    Raw image → image VQ codec → VQ codes per spatial tile
    Rust splits into spatial tiles → parallel work units:
      Unit 0: tile(0,0) → Agent 0
      Unit 1: tile(0,1) → Agent 1   (all 4 run in parallel)
      Unit 2: tile(1,0) → Agent 2
      Unit 3: tile(1,1) → Agent 3
    Scratchpad: agents share edge hidden states for coherence
    Trie: each agent writes its tile's semantic representation

  Audio ingestion (tiled):
    Raw audio → filterbank → 3 frequency bands → audio VQ codec each
    Rust splits into spectral tiles → parallel work units:
      Unit 0: bass band → Agent 0
      Unit 1: mids band → Agent 1   (all 3 run in parallel)
      Unit 2: treble band → Agent 2
    Within each band: chunking handles temporal progression
    Scratchpad: cross-band energy sharing for coherence
```

---

## Generation (Producing Output)

```
  Generation uses all three dimensions. Core agents generate VQ CODES.
  Rust routes codes through the appropriate VQ decoder (per content-type).

  TEXT — Chunked sequential generation
  ─────────────────────────────────────
  Agent 0: generates chunk 0 [0:256 VQ codes]
    → writes final hidden state to scratchpad
  Agent 0: generates chunk 1 [256:512 codes]
    → reads previous chunk context from scratchpad
    → continues generating
  ...

  Rust accumulates output VQ codes → text VQ decoder → UTF-8 bytes
  Text decoder produces text WITH emojis when Layer 3 codes indicate them.
  "I'm doing great! 😊" — the emoji emerges from enrichment codes.

  One agent per text stream (sequential).
  Scratchpad carries context between chunks.
  Trie provides semantic retrieval throughout.


  TEXT — RVQ Refinement
  ─────────────────────
  Layer 1 generation (coarse): generate meaning/structure codes
    → text VQ decoder shows rough preview: "Hlo hw r yu?"
  Layer 2 generation (exact): generate reconstruction codes
    → text VQ decoder shows clean text: "Hello, how are you?"
  Layer 3 generation (polish): generate final refinement codes
    → text VQ decoder shows polished: "Hello, how are you? 😊"

  Progressive rendering: user sees text appearing at Layer 1 speed,
  refinement layers improve quality as they arrive.


  IMAGE — Tiled parallel generation with refinement
  ─────────────────────────────────────────────────
  Phase 1 (coarse, per tile):
    4 agents generate coarse VQ codes for their spatial quadrant
    Scratchpad: share edge codes for spatial coherence
    All 4 run in parallel

  Phase 2 (fine, per tile):
    Same 4 agents generate fine VQ codes
    Read their own coarse codes from scratchpad
    Read neighbors' coarse codes for context
    All 4 run in parallel

  Assembly: Rust collects all tiles → image VQ decoder → pixels


  AUDIO — Tiled spectral generation with refinement
  ──────────────────────────────────────────────────
  Phase 1 (coarse, per band):
    3 agents generate coarse VQ codes for their frequency band
    Scratchpad: share cross-band energy
    All 3 run in parallel

  Phase 2 (fine, per band):
    3 agents generate fine codes conditioned on coarse
    All 3 run in parallel

  Chunking: repeat for each temporal segment (1.28s per chunk)
  Scratchpad carries previous-chunk context for temporal continuity


  VIDEO — Combined tiling + chunking + refinement
  ────────────────────────────────────────────────
  Per frame (spatial + spectral tiling):
    4 image agents (quadrants) + 3 audio agents (bands) = 7 parallel agents
    Shared coarse codes via scratchpad

  Across frames (chunking):
    Previous frame's summary in scratchpad → next frame reads it
    Sequential: frame N+1 starts after frame N tiles finish

  Refinement (quality hierarchy):
    Coarse pass: all 7 agents generate coarse codes (parallel)
    Fine pass: all 7 agents refine (parallel, conditioned on coarse)
    Audio fine conditioned on image coarse → lip-sync
    Text enrichment conditioned on shared coarse → emoji matches scene

  Bandwidth (1 second at 24fps):
    Per frame: 4 image tiles + 3 audio tiles = 7 parallel agents
    24 frames × sequential chunks
    Throughput depends on agent speed, not data volume
```

---

## Tool Dispatch (Event Queue at Chunk Boundaries)

```
  Tool calls are processed at 256-token chunk boundaries, NOT mid-stream.
  Rust runtime NEVER pauses the agent mid-generation.

  Agent generates 256-token chunk:
    "The weather in San Francisco is <tool:weather>SF</tool> so I think..."

  Chunk completes → Rust scans output for tool patterns:
    Found: <tool:weather>SF</tool>

  Event queue processing:
    1. Dispatch weather API (async, can dispatch multiple tools in parallel)
    2. Result arrives with full headers:
       author:tool:weather
       origin:kaans-mac.local
       location:https://api.weather.com/forecast/sf
       timestamp:1712765720
       content-type:text/plain
       status:200
       method:RESPONSE
       length:24
       expires:1712769320

       San Francisco: 68°F, sunny
    3. Tool result becomes next input unit for the agent
       (properly tagged, enters via work queue)
    4. Agent continues generating with tool result in memory + attention

  Agent's next chunk output:
    "Based on the 68°F sunny weather, I'd recommend..."

  Properties:
    - Clean chunk boundaries — no mid-stream interruptions
    - Multiple tool calls in one chunk → all dispatched in parallel
    - Tool results are standard tagged units (same format as any input)
    - Agent writes to trie, reads tool result from trie — normal cycle
```

---

## Shared Scratchpad

```
  Fixed 16 × d_model tensor on GPU, shared by all agents in a work group.
  Scratchpad DEFINES the work group: agents sharing a scratchpad are in the same
  work group (analogous to GPU SMEM defining a thread block). Agents in different
  work groups cannot communicate.

  Work group size: dynamic, 1–N_max agents. N_max is configurable at launch
  (supports up to 128+ agents for large tasks). Only active agents participate
  in the scratchpad update.

  Update mechanism (attention-based write):
    - Each agent's hidden state cross-attends to scratchpad contents:
        Q = hidden · W_q (96→8), K = scratchpad · W_k (96→8)
        weights = softmax(Q · K^T / √8)   per slot
    - Blend: scratchpad[i] = weights[i] × hidden + (1-weights[i]) × scratch[i]
    - Content-aware: write targets depend on what's already in each slot
    - Self-organizing: slots specialize by content similarity, not assignment
    - Cross-agent: strength-weighted mean resolves concurrent writes:
        scratchpad[i] = Σ_a(weights[a][i] × hidden[a]) / Σ_a(weights[a][i])
    - Fully parallel, no atomics, fully differentiable
    - Automatic: every agent contributes every step (attention weights gate influence)
    - Params: W_q(96×8) + W_k(96×8) = 1,536 per agent

  Uses:
    - Inter-tile coherence (edge hidden states between image quadrants)
    - Inter-chunk context (previous chunk's final state for continuation)
    - Environmental context (shared coarse codes for refinement passes)
    - Cross-modal coordination (image coarse codes shared with audio)

  Lifecycle:
    - Resets between unrelated work groups
    - Persists within a work group (e.g., one image's spatial tiles)
```
