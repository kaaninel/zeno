# Zeno — Architecture

Core agent design, spatiotemporal headers, VQ codec strategy, parameter
budget, risk assessment, and design decisions.

---

## Spatiotemporal Header System

Every piece of content in Zeno carries structured headers — inspired by HTTP
but designed for agent context. Headers are the spatiotemporal identity of
content and define tile/work-unit boundaries.

### Header Format

```
  Required headers (always present):
  ┌──────────────┬────────────┬────────────────────────────────────────────┐
  │ Header       │ Type       │ Example                                    │
  ├──────────────┼────────────┼────────────────────────────────────────────┤
  │ author       │ text       │ "kaan", "system", "agent-0", "tool:shell" │
  │ origin       │ text       │ "kaans-mac.local", "server-prod-01"       │
  │ location     │ URI        │ "chat://host/conv/work/msg/42"            │
  │ timestamp    │ numeric    │ 1712765715 (unix seconds)                 │
  │ content-type │ text       │ "text/plain", "text/code:rust", "image/png│
  │ status       │ numeric    │ 200, 401, 404, 500 (HTTP status codes)    │
  │ method       │ enum       │ READ, WRITE, QUERY, RESPONSE, OBSERVE     │
  │ length       │ numeric    │ 1024 (payload bytes)                      │
  └──────────────┴────────────┴────────────────────────────────────────────┘

  Optional headers (when applicable):
  ┌──────────────┬────────────┬────────────────────────────────────────────┐
  │ parent       │ URI        │ "chat://host/conv/work/msg/41" (reply-to) │
  │ topic        │ text       │ "code/rust/config" (hierarchical subject) │
  │ priority     │ 0-9        │ Importance level (affects trie write str.) │
  │ expires      │ timestamp  │ Unix seconds, or "indefinite"/"ephemeral" │
  │ chunk        │ text       │ "3/7" (for multi-window units)            │
  │ position     │ fragment   │ "#L120", "#byte:4096", "#page:3"          │
  └──────────────┴────────────┴────────────────────────────────────────────┘
```

### Location URI — Everything is a File

All content has a globally unique URI. Protocol determines access method:

```
  chat://kaans-mac.local/conversations/work/msg/42      (chat message)
  file://kaans-mac.local/Users/kaan/project/main.rs      (local file)
  https://github.com/user/repo/blob/main/src/lib.rs      (web resource)
  tool://kaans-mac.local/shell/exec-42                   (tool output)
  stdin://kaans-mac.local/terminal-0                     (direct input)
  memory://kaans-mac.local/trie/L1/addr[42,128,...]      (memory read)
```

`origin` header identifies WHOSE perspective the location is from.
`file://kaans-mac.local/etc/hosts` ≠ `file://server-01/etc/hosts`.

### Full Tag Example

```
  author:kaan
  origin:kaans-mac.local
  location:chat://kaans-mac.local/conversations/work/msg/42
  timestamp:1712765715
  content-type:text/plain
  status:200
  method:WRITE
  length:35
  parent:chat://kaans-mac.local/conversations/work/msg/41
  expires:indefinite

  How do I fix this config parsing bug?
```

### Dual-Channel Injection (Input)

Headers are injected into the model via TWO simultaneous channels:

```
  Channel 1 — Inline bytes (in token stream):
    Agent reads header text as bytes, building internal representation.
    Cost: ~80-120 bytes at unit start only (not repeated per chunk).
    Model LEARNS the header format from data.

  Channel 2 — Structured side-channel (tag cross-attention):
    Rust computes rich tag vectors using the model's own text encoder:

    Tag Encoder Pipeline (10 fields → 10 d_model vectors):

      Text-encoded fields (pool byte embeddings):
        author_vec    = pool(text_encoder("kaan"))            # d_model
        origin_vec    = pool(text_encoder("kaans-mac.local")) # d_model
        location_vec  = pool(text_encoder("chat://..."))      # d_model
        type_vec      = pool(text_encoder("text/plain"))      # d_model
        parent_vec    = pool(text_encoder("chat://..."))      # d_model
                        (or learned null_vec when no parent)

      Numeric-encoded fields:
        time_vec      = sinusoidal_encode(timestamp)          # d_model
        expires_vec   = sinusoidal_encode(expires_timestamp)  # d_model
                        (or learned ephemeral_vec / indefinite_vec)
        length_vec    = sinusoidal_encode(length)             # d_model

      Categorical-encoded fields (small learned embedding):
        status_vec    = status_embed[status_class]            # d_model
                        (8 classes: 200,301,400,401,403,404,429,500)
        method_vec    = method_embed[method_class]            # d_model
                        (5 classes: READ,WRITE,QUERY,RESPONSE,OBSERVE)

      Sinusoidal encoding (16 frequency bands, reused for time/expires/length):
        Band 0:  period = 1 second     (sub-second resolution)
        Band 1:  period = 10 seconds
        Band 2:  period = 1 minute
        ...
        Band 15: period = 1 year       (seasonal patterns)
        vec[2i]   = sin(value / period_i)
        vec[2i+1] = cos(value / period_i)

      Mini cross-attention (1 layer, 2 heads, ~2K params):
        Stack all 10 vectors → (10, d_model)
        Self-attention → each field becomes aware of the others
        "kaan" + "error 404" → different from "kaan" + "200 OK"
        Output → 10 tag_context vectors for context cross-attention

    These 10 tag vectors join the 4 internal register slots in a
    UNIFIED context cross-attention layer (14 slots total per block).

    The text encoder used here is the SAME encoder the model uses for
    content. Tags and content share the same representation space.
```

### Internal Register Bank

Each agent has a 4-slot register bank (4 × d_model = 384 floats) that serves
as general-purpose internal scratchpad. NOT tag-specific — model learns what
to store (reasoning state, partial results, working memory).

```
  Register bank: [R0] [R1] [R2] [R3]    (each d_model=96 floats)

  Lifecycle:
    - ZERO-RESET when agent picks up a new work unit
    - PERSIST across 256-token chunk continuations (same unit)
    - Updated via GRU gate each token (model controls what to store)

  Access: unified context cross-attention alongside tag vectors
    Attention pool = [T0..T9, R0, R1, R2, R3]   (14 slots total)
                      \_ tag encoder _/  \_ registers _/

  Why 4 slots (not 1):
    Model can compartmentalize — e.g. one slot for speaker identity,
    one for topic tracking, one for reasoning chain, one for misc.
    Or use all 4 for one purpose. Model decides.
```

### Output Headers (Inline Bytes Only)

The model generates response headers as inline bytes in its output stream.
Rust parses the header text before the blank-line separator. No structured
output heads — the side-channel is INPUT-ONLY (Rust→Model).

```
  Model output:
    author:agent-0
    location:chat://host/conv/work/msg/43
    timestamp:1712765720
    content-type:text/plain
    status:200
    method:RESPONSE
    length:42
    parent:chat://host/conv/work/msg/42
                                              ← blank line = header/content boundary
    Based on the weather, I'd recommend...

  Rust parses headers from output bytes:
    - Known fields: strict key:value parsing, one per line
    - Unknown fields: ignored (forward-compatible)
    - Malformed/missing: Rust applies sensible defaults
      (status→200, method→RESPONSE, expires→indefinite, priority→5)
    - Header section ends at first blank line

  Why no structured output heads:
    - Response headers are CONSTANT per output unit (like input)
    - Per-token classification heads for constant values = wasteful
    - Model sees header format constantly on input → learns to produce it
    - Inline bytes are flexible (any field, any value, forward-compatible)
    - ~1K params saved, simpler architecture
```

### Headers Define Tile Boundaries

Spatiotemporal headers are not just metadata — they DEFINE where to split
input into work units. A change in any core header field (author, location,
timestamp beyond threshold) signals a new work unit boundary.

---

## Per-Agent Architecture (Core Agent ~666K params)

```
  Input: 256 VQ code tokens (from text/image/audio VQ codec)
         Core agent NEVER sees raw UTF-8 or raw media.
         Code 0x41 ≠ "A" — it means VQ codebook entry 65.
         │
         ▼
  ┌─── Code Embedding ────────────────────────────────────────────┐
  │  VQ code → d_model embedding (256 × d_model table)           │
  │  Same table for ALL modalities (codes are modality-blind)    │
  │  Output: (256, d_model) tensor                               │
  └──────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
  ┌─── 4× TransformerBlock ──────────────────────────────────────┐
  │                                                              │
  │  1. RMSNorm → Self-Attention (causal, RoPE) → + residual    │
  │     4 heads × (d_model/4) dim                                │
  │     KV cache for generation                                  │
  │                                                              │
  │  2. RMSNorm → Context Cross-Attention → + residual            │
  │     Unified pool of 14 slots:                                │
  │       [10 tag vectors from Tag Encoder (side-channel)]       │
  │       + [4 register slots (general-purpose internal state)]  │
  │     Register bank: 4×d_model, GRU-gated, reset per unit      │
  │       Read-then-update: cross-attend to CURRENT register,    │
  │       THEN GRU updates register from enriched hidden state.  │
  │       Next token reads updated register.                     │
  │       Model learns what to store (reasoning, partial state)  │
  │                                                              │
  │  3. RMSNorm → Unified Memory Cross-Attention → + residual    │
  │     Attends to ALL sources in one pool (22 slots fixed):     │
  │       [3 L0 trie leaf vectors]                               │
  │       + [16 shared scratchpad slots]                         │
  │       + [3 async queue slots (FIFO, zero-masked if empty)]   │
  │     Per-head inv_temp for sharpness                          │
  │     Access cost loss (penalize expensive reads)              │
  │     Data quality loss (penalize attending to bad data)       │
  │                                                              │
  │  4. RMSNorm → SiLU FFN (d→2d→d) → + residual                │
  │                                                              │
  └──────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
  ┌─── Output Heads ─────────────────────────────────────────────┐
  │                                                              │
  │  Hidden → LM head → next token logits (weight-tied w/ embed)│
  │  Hidden → 3× AddrNet → 3 WRITE addresses (8 levels each)    │
  │           (same AddrNets as read, called on processed hidden │
  │            — read addrs from input embed, write from output) │
  │  Hidden → V_proj → base value (shared, 96→96)                │
  │  Hidden → 3× aspect head → 3 residuals (96→16→96 each)      │
  │           val_i = base + aspect_i(hidden) for channel i      │
  │  Hidden → write_strength → α modifier (96→1, sigmoid)        │
  │                                                              │
  │  No structured output heads — model generates response       │
  │  headers as inline bytes, Rust parses text.                  │
  │  Side-channel is INPUT-ONLY (Rust→Model).                    │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

### Code Decoder (LM Head)

```
  The code decoder is the symmetric counterpart to the code embedding:

  Code Embedding (input):  VQ code (0-255) → embedding table row → d_model vector
  Code Decoder (output):   d_model hidden → LM head → 256 logits → sample → VQ code

  Weight-tied: logits = hidden @ embed.weight.T   (no extra params)
    The same embedding that best represents code 0x4A as input
    is also the one most distinguishable from other codes as output.

  Decoding pipeline:
    1. Final transformer layer outputs hidden state (d_model=96)
    2. LM head: hidden @ embed.weight.T → 256 logits (one per VQ code)
    3. Temperature scaling + top-k/top-p sampling → code_id (0-255)
    4. code_id IS a VQ code — meaning depends on which codec decodes it
    5. Accumulate codes → Rust routes to appropriate VQ decoder based
       on content-type header → UTF-8 text, pixels, audio waveform

  Symmetry across ALL modalities (fully unified):
    Text:  text → [text VQ enc] → codes → core → codes → [text VQ dec] → text + 😊
    Image: pixels → [image VQ enc] → codes → core → codes → [image VQ dec] → pixels
    Audio: wave → [audio VQ enc] → codes → core → codes → [audio VQ dec] → wave

  Core agents are modality-blind — they process VQ codes, not content.
  The 256×96 embedding matrix encodes VQ CODE semantics, not byte values.
  All content knowledge lives in the VQ codecs + trie memory.
```

### Parameter Budget (d_model=96, 4 layers, 4 heads)

```
  Component                        Params      %
  ─────────────────────────────    ────────   ─────
  Code Embedding (256 × 96)        24,576     3.7%
  4× TransformerBlock:
    Self-Attention (Q,K,V,O)       36,864
    Context Cross-Attn (Q,K,V,O)   36,868
      (attends to 14 slots: 10 tag + 4 register)
    Register GRU gate (d→4d + d→4) ~1,000
    Memory Cross-Attn (Q,K,V,O)    36,868
    FFN (up + down)                36,864
    RMSNorm × 4                       384
    Subtotal per layer:           ~148,848
    × 4 layers =                             ~595,400   89.4%
  3× AddrNet (~7K each)                       21,000     3.2%
  V_proj (96 → 96)                  9,312     1.4%  (shared base value)
  3× Aspect Head (96→16→96 each)    9,552     1.4%  (channel residuals)
  Write Strength Head (96 → 1)          97     0.0%
  Confidence Gate:
    density_embedding (256 × 8)      2,048     0.3%
    depth_embedding (8 × 8)             64     0.0%
    W_gate (16 → 1) + b_gate           17     0.0%
    Subtotal:                        2,129     0.3%
  Tag Encoder (~2K params):
    Mini cross-attn (1 layer, 2 heads) 2,048     0.3%
    Note: text_encoder() in the tag pipeline = the VQ codec's byte encoder
    (shared, frozen after Phase 1). The ~2K params here are ONLY the
    cross-attention layer, not a separate encoder.
  Tag categorical embeds:
    status_embed(8×96)+method_embed(5×96) 1,248   0.2%
    null_vec + ephemeral_vec + indef_vec    288   0.0%
  Register Bank (4 × 96)              384     —     (state, not params)
  Final RMSNorm                        96     0.0%
  LM Head                     (tied with code embed)
  ─────────────────────────────    ────────   ─────
  TOTAL (estimated)               ~666,000   ~100%
```

---

## VQ-256 Codec Strategy (Unified — All Modalities Including Text)

```
  ALL modalities — text, audio, image, video — use VQ-256 codecs.
  Core agents NEVER see raw bytes (UTF-8) or raw media. They only
  see VQ codes (0-255). This is a fully unified pipeline.

  ┌──────────────────── UNIFIED DATA FLOW ──────────────────────────┐
  │                                                                  │
  │  User input    → [text VQ encoder]  → VQ codes ─┐               │
  │  Image file    → [image VQ encoder] → VQ codes ─┤               │
  │  Audio stream  → [audio VQ encoder] → VQ codes ─┤               │
  │                                                  ▼               │
  │                                          ┌──────────────┐       │
  │                                          │  Core Agent   │       │
  │                                          │ (only sees    │       │
  │                                          │  VQ codes     │       │
  │                                          │  0-255)       │       │
  │                                          └──────┬───────┘       │
  │                                                 │               │
  │                                          VQ codes out           │
  │                                                 │               │
  │               ┌─────────────────────────────────┼──────────┐    │
  │               ▼                                 ▼          ▼    │
  │     [text VQ decoder]             [image VQ dec]  [audio dec]   │
  │               ▼                                 ▼          ▼    │
  │       UTF-8 text + 😊                       pixels    waveform  │
  │                                                                  │
  │  Content-type header tells Rust which decoder to route through.  │
  │  Core agents are TRULY modality-blind — pure code-to-code.       │
  └──────────────────────────────────────────────────────────────────┘

  Why text too (not just audio/image):
    - Text is lossy-quantized speech — tone, emphasis, emotion all lost
    - VQ continuous latent captures enrichment NATIVELY (no extra layer)
    - Context window expands ~2-3× via compression
    - All modalities share identical pipeline — no special cases
    - Emoji enrichment lives naturally in the continuous latent space
    - Core agent weights become pure computation, zero content knowledge
```

### Residual VQ (RVQ) — Shared Mechanics

```
  Each quantization layer captures progressively finer detail:

    Input: continuous vector x (from codec encoder)

    Layer 1: c₁ = nearest(x, codebook₁)        → byte (coarse/meaning)
             r₁ = x - codebook₁[c₁]
    Layer 2: c₂ = nearest(r₁, codebook₂)       → byte (mid/exactness)
             r₂ = r₁ - codebook₂[c₂]
    Layer 3: c₃ = nearest(r₂, codebook₃)       → byte (fine/nuance)

    Reconstruction: codebook₁[c₁] + codebook₂[c₂] + codebook₃[c₃]

  Quality scales linearly with bytes-per-position (K = RVQ layers):
    K=1: rough meaning   (gist of text, phone-quality audio, blocky image)
    K=2: good quality    (readable text, clear speech, recognizable image)
    K=3: exact + nuance  (exact text + enrichment, near-lossless media)
    K=4+: high fidelity  (studio audio, detailed image)

  Layer roles for TEXT specifically:
    Layer 1: semantic gist ("greeting about the weather")
    Layer 2: exact character reconstruction ("Hello, how's the weather?")
    Layer 3: enrichment/nuance (tone, emphasis, emoji → 😊)


  Progressive byte ordering
  ─────────────────────────
  All coarse codes first, then refinement layers:

    [c₁₁ c₁₂ ... c₁ₙ | c₂₁ c₂₂ ... c₂ₙ | c₃₁ ... c₃ₙ]
     └─── coarse ────┘   └──── mid ──────┘   └── fine ──┘

  Why progressive:
    - Coarse codes render immediately (low-latency preview)
    - Fine codes improve quality progressively
    - Autoregressive: fine codes are conditioned on all coarse codes
    - Graceful degradation: drop fine layers for lower bandwidth
```

### Text VQ Codec (Perceiver-Style, Learned Adaptive Compression)

```
  ┌─── Text VQ Codec (~300K params, separate from core agent) ──────┐
  │                                                                   │
  │  ENCODER: UTF-8 bytes → VQ codes                                  │
  │  ─────────────────────────────────                                │
  │  1. Byte embed: 256 × d_codec → (N, d_codec)                     │
  │     d_codec = 64-128 (codec has its own embedding dimension)      │
  │  2. Positional encoding (sinusoidal, byte positions)              │
  │  3. Small transformer (1-2 layers): captures local UTF-8 structure│
  │     → (N, d_codec) encoded byte sequence                          │
  │                                                                   │
  │  4. Perceiver cross-attention (THE KEY STEP):                     │
  │     Q = M learned latent queries (M << N, compression target)     │
  │     K, V = encoded byte sequence (N, d_codec)                     │
  │     → (M, d_codec) compressed latent sequence                     │
  │                                                                   │
  │     Each latent position LEARNS which bytes to attend to.         │
  │     Attention pattern = learned adaptive tokenization:            │
  │       English: latent[i] attends to ~5 bytes (one word)           │
  │       Chinese: latent[i] attends to ~3 bytes (one character=word) │
  │       Russian: latent[i] attends to ~8 bytes (one word)           │
  │       Mixed:   adapts per-position based on content               │
  │                                                                   │
  │     No explicit boundary prediction needed.                       │
  │     Attention IS the segmentation. Fully differentiable.          │
  │                                                                   │
  │  5. RVQ quantization: each latent → K byte codes (K=2-3)         │
  │     Layer 1: meaning/gist                                         │
  │     Layer 2: exact character reconstruction                       │
  │     Layer 3: enrichment/nuance (emotional, tonal, emoji)          │
  │                                                                   │
  │  Output: M × K byte codes (progressive ordering)                  │
  │                                                                   │
  │                                                                   │
  │  DECODER (symmetric): VQ codes → UTF-8 bytes                      │
  │  ──────────────────────────────────────                           │
  │  1. Code lookup: VQ codes → codebook vectors → sum RVQ layers     │
  │     → (M, d_codec) latent sequence                                │
  │                                                                   │
  │  2. Reverse Perceiver cross-attention:                             │
  │     Q = N positional queries (byte positions to reconstruct)      │
  │     K, V = M latent vectors                                       │
  │     → (N, d_codec) reconstructed byte embeddings                  │
  │                                                                   │
  │  3. Small transformer (1-2 layers): local refinement              │
  │  4. Per-position: d_codec → 256 logits → byte                     │
  │  5. Target: EXACT match with original UTF-8 bytes                 │
  │                                                                   │
  │                                                                   │
  │  WHY PERCEIVER (vs fixed-stride convolution):                     │
  │  - Attention pattern adapts to language/script automatically       │
  │  - No UTF-8 character splitting (conv stride can split 3-byte CJK)│
  │  - Same architecture handles ALL languages including mixed scripts │
  │  - Attention visualization = interpretable "tokenization"          │
  │  - Fully differentiable, no Gumbel tricks for boundary prediction │
  └───────────────────────────────────────────────────────────────────┘


  Learned Adaptive Tokenization — How It Works
  ─────────────────────────────────────────────
  Input: "Hello 你好 мир!"  (mixed English + Chinese + Russian)
  Bytes: [48 65 6C 6C 6F 20 | E4 BD A0 E5 A5 BD | 20 D0 BC D0 B8 D1 80 21]
          H  e  l  l  o  sp   你       好        sp м     и     р     !

  With M = 5 latent positions (perceiver):

  Latent 0: attends to bytes 0-5  "Hello " (6 bytes, English word + space)
  Latent 1: attends to bytes 6-8  "你"     (3 bytes, one CJK character)
  Latent 2: attends to bytes 9-11 "好"     (3 bytes, one CJK character)
  Latent 3: attends to bytes 12-18 " мир"  (7 bytes, Russian word)
  Latent 4: attends to byte 19    "!"      (1 byte, punctuation)

  The codec discovers word/character boundaries for EACH language.
  No explicit language detection. Pure learned attention.
```

### Multilingual Compression Analysis

```
  UTF-8 byte costs vary dramatically by script:

  Script              Bytes/char  Chars/word  Bytes/word
  ────────────────   ──────────  ──────────  ──────────
  English (ASCII)     1           ~5          ~5
  French/German       ~1.2        ~5          ~6
  Cyrillic (Russian)  2           ~6          ~12
  Arabic/Hebrew       2           ~4          ~8
  Hindi/Thai          3           ~4          ~12
  CJK (Chinese)       3           ~1 (!)      ~3
  Japanese mixed      2-3         varies      ~3-6
  Korean Hangul       3           ~2          ~6
  Emoji               4           ~1          ~4

  Effective context window (256 VQ codes per chunk, one RVQ layer per chunk):
  Each chunk = 256 content positions. K=3 RVQ layers → 3 sequential chunks
  in a pipeline: Agent→L1 codes → Agent→L2 codes → Agent→L3 codes.
  Same base agent, each pass gets previous layer's output as input.

  Language   Bytes/position  Chars/pos  Words/window  vs raw 256 bytes
  ────────  ──────────────  ─────────  ────────────  ────────────────
  English    ~5              ~5         ~256 words    5.1× (was ~50)
  Chinese    ~3              ~1         ~256 words    3.0× (was ~85)
  Russian    ~8              ~4         ~256 words    12.2× (was ~21)
  Arabic     ~6              ~3         ~256 words    7.3× (was ~35)
  Hindi      ~9              ~3         ~256 words    13× (was ~20)

  Key insight: multi-byte scripts (Russian, Hindi, Arabic) benefit MOST
  from VQ compression because raw byte processing wastes context window
  on UTF-8 continuation bytes. Chinese benefits least because each 3-byte
  character already carries one word of meaning.

  With K=1 only (single chunk, no refinement):
  English: ~256 words (semantic gist, immediate output)
  Chinese: ~256 words (same — uniform positions)
  Russian: ~256 words (massive improvement over raw bytes)

  UTF-8 vs UTF-16 is IRRELEVANT in VQ world:
    The text codec handles raw bytes internally.
    Core agents only see VQ codes (0-255).
    Encoding choice is a codec implementation detail.
```

### Emoji Enrichment (Text VQ Layer 3)

```
  Text is lossy-quantized speech — tone, emphasis, emotion are all
  lost in symbolic representation. Emojis are humanity's attempt to
  add that information back. Zeno formalizes this.

  In the text VQ codec:
    Layer 1 codes: semantic content ("greeting about weather")
    Layer 2 codes: exact characters ("Hello, how's the weather?")
    Layer 3 codes: enrichment ("warm, genuine enthusiasm → 😊")

  The continuous latent space (before VQ quantization) captures ALL of
  this — content, form, and emotion — in one representation. The VQ
  codebook for Layer 3 learns to carve up the emotional/tonal space:

    Code 0x4A: positive-excited (😊🎉😄)
    Code 0x1C: negative-sad (😢😞💔)
    Code 0xB2: sarcastic-ironic (🙃😏🤨)
    Code 0x7F: neutral-factual (no emoji)

  Emojis are the training signal:
    Billions of text+emoji pairs exist (chat, social media, messages).
    "I just got the job 🎉" → codec learns what 🎉-like enrichment is.
    "My cat ran away 😢"   → codec learns what 😢-like enrichment is.
    Emojis = largest naturally-labeled emotional dataset in existence.

  Cross-modal bridge:
    Text Layer 3 code 0x4A (happy) ←→ Audio warm/upbeat tone
    Text Layer 3 code 0x1C (sad)   ←→ Audio soft/slow tone
    Same enrichment code drives text output AND audio generation.
    Emoji becomes the LINGUA FRANCA between modalities.

  External input (no enrichment info):
    Incoming flat text from outside → Layer 3 = UNKNOWN_CODE
    Not inferred (unreliable), not neutral (that's a claim).
    Explicitly "unknown" — the model knows what it doesn't know.

  Emoji in generated output:
    Core agent generates VQ codes → text codec decoder → UTF-8 bytes
    The decoder produces text WITH emojis when Layer 3 indicates them.
    "I'm doing great! 😊" — the emoji emerges naturally from the code.


  Training curriculum for enrichment:
  ─────────────────────────────────
  Step 1: Emoji prediction (bootstrap)
    text+emoji → encode → latent → predict emoji from Layer 3 codes
    Massive training data (every chat message with emojis)

  Step 2: Continuous emotional space
    Move from categorical (predict emoji) to continuous (predict VAD vector)
    VAD = Valence (positive↔negative), Arousal (calm↔excited),
          Dominance (submissive↔assertive)
    Each emoji maps to a VAD point: 😊=(0.8, 0.5, 0.5), 😤=(-0.5, 0.8, 0.9)

  Step 3: Audio-text pairing
    Paired speech + transcripts → both encode → align latents
    Text codec learns prosodic features from speech
    "How are you?" spoken happily vs. sarcastically → different Layer 3

  Step 4: Video/TV show training
    Visual (facial expressions) + audio (tone) + text (subtitles)
    Multi-signal supervision → robust enrichment understanding
```

### Shared Coarse Codebook (Multimodal Sync)

```
  Video + audio share the SAME coarse RVQ layer (scene-level semantics).
  Fine layers are modality-specific.

  Per temporal position:
    shared coarse: 1 byte  ("scene: speaker talking, room reverb")
    text fine:     2 bytes (exact text + enrichment for this moment)
    image fine:    3 bytes (visual detail for this frame)
    audio fine:    3 bytes (per frequency band detail)

  → Temporal sync is inherent (same coarse = same scene)
  → Audio fine conditioned on image coarse + fine → lip-sync emerges
  → Text enrichment conditioned on shared coarse → emoji matches scene
  → Graceful degradation: drop fine layers for lower bandwidth
```

### Codec Agent Architecture & Training

```
  Each modality has a separate VQ codec (trained independently):

  ┌──────────────────────────────────────────────────────────┐
  │  Codec         Params   Encoder            Decoder       │
  │  ─────────    ───────  ─────────────────  ────────────── │
  │  Text VQ      ~300K    Perceiver compress  Perceiver up  │
  │  Image VQ     ~300K    Conv2d + Perceiver  TransConv2d   │
  │  Audio VQ     ~300K    Conv1d + Perceiver  TransConv1d   │
  └──────────────────────────────────────────────────────────┘

  Training: VQ-VAE (reconstruction + commitment loss)
    Reconstruction: exact byte match (text), MSE (audio/image)
    Commitment loss: ||z - sg(e)|| (latent close to codebook entry)
    EMA codebook updates, random restarts to prevent collapse
    End-to-end fine-tuning via straight-through estimator

  Core agents never see raw media OR raw text — only VQ codes.
  The 256-entry byte embedding in the core agent embeds VQ CODES,
  not UTF-8 bytes. Code 0x41 no longer means "A" — it means
  "codebook entry 65." All semantics live in codecs + trie.
```

---

## Risk Assessment

### HIGH RISK

1. **AddrNet training stability (P0)**
   256 bins × 8 levels = 256^8 ≈ 10^19 possible addresses. Gumbel-softmax
   must converge to meaningful clustering in this vast space. With multiple
   agents writing simultaneously, address collisions and chaotic address
   spaces are likely.
   
   Mitigation: Phase 3 dedicated AddrNet training on meaningful hidden
   states (trained AFTER base model, not before). Diversity loss.
   Temperature annealing. Address entropy monitoring. If addresses collapse,
   reduce bins (e.g., 64 bins × 8 levels = still 10^14 addresses).

2. **Unified attention dilution (RESOLVED)**
   Pool fixed at 22 slots (3 trie + 16 scratchpad + 3 async queue). 4 heads × ~5 keys
   each. No dilution. "Causal peer hiddens" dropped — scratchpad is the only inter-agent
   channel. Fixed slot ordering lets attention heads specialize by position.

3. **250K TPS generation target (P1)**
   250K TPS = **total swarm throughput** across concurrent requests, not per-request latency.
   Single agent DRAM ceiling on M4: ~90K tok/s. ANE SRAM-cached: ~1.9M tok/s for 18+ agents.
   250K TPS achievable with ~13 concurrent ANE-cached agents on one M4 base. See EXPECTATIONS.md.

4. **Training in pure Rust / candle (RESOLVED)**
   **Candle-only for all training and inference.** We accept the development cost. Missing
   pieces (DataLoader, Gumbel-softmax, custom schedules) will be built in Rust. See TRAINING.md.
   The model is small enough (666K params) that candle's training limitations are manageable.

5. **Text VQ reconstruction fidelity (P0)**
   Exact round-trip UTF-8 reconstruction from VQ codes is the hardest
   challenge in the unified VQ pipeline. Unlike audio/image where lossy
   is acceptable, text requires EXACT byte-level reconstruction — a single
   wrong byte can change meaning ("not" → "hot") or break UTF-8 encoding.

   Sub-risks:
   - Multi-byte UTF-8 characters (CJK 3 bytes, emoji 4 bytes) must
     reconstruct perfectly — one wrong continuation byte = mojibake
   - Code/markup is unforgiving (missing ; or > breaks programs)
   - Rare scripts with complex combining marks may be underrepresented
   - Compression ratio vs fidelity tradeoff: higher compression = harder
   
   Mitigation: Start with low compression (2:1) and increase gradually.
   Perceiver-style encoder should handle variable-width naturally.
   Per-position byte cross-entropy with heavy penalty on wrong bytes.
   Separate eval sets per script family. Accept lower compression for
   scripts where fidelity suffers. K=3 RVQ gives Layer 2 dedicated to
   exact reconstruction.

### MEDIUM RISK

6. **Memory coherency under concurrent writes (P1)**
   Multiple agents writing to same trie regions simultaneously. EMA
   blending handles value conflicts (deterministic outcome), but the
   ORDER of writes affects the final value. With async fire-and-forget
   writes, ordering is non-deterministic.
   
   Mitigation: For L0 (GPU), use atomic adds. For L1/L2, buffer writes
   and apply in deterministic order per flush cycle. Accept that within
   a cycle, write order is non-deterministic — EMA smooths this.

7. **Imagination vs hallucination boundary (P1)**
   Teaching the model to IMAGINE (fill in partial data creatively) while
   preventing HALLUCINATION (presenting imagination as fact) is a delicate
   balance. Too much imagination encouragement → model fabricates.
   Too little → model is overly conservative and unhelpful.

   Sub-risks:
   - Confidence gate may not calibrate well in early training
     (trie is sparse initially → density uniformly low everywhere)
   - Diffusion halos need writes to accumulate before density diverges
   - Asymmetric loss is harder to tune than symmetric loss
   - Multi-view training requires curated datasets (multiple
     partial descriptions per content item)
   - The model might learn to ALWAYS hedge instead of being direct
   
   Mitigation: Phase 6 starts AFTER Phase 5 (coherent model first).
   Confidence gate is pre-trained on simple write/read density patterns
   before tackling partiality. Write diffusion creates density differences
   quickly (even a few writes create observable halos). Hallucination
   penalty in loss keeps specificity in check. Monitor imagination/factual
   ratio during training.

8. **Flat-array memory overhead (P2)**
   Each trie node with [Option<u32>; 256] children = 1KB overhead just
   for child pointers, even if mostly empty. Sparse tries waste memory.
   
   Mitigation: Lazy allocation (only create child array on first child
   write). Or use compressed sparse format (bitmap + packed indices).
   At depth 0-2 the overhead is negligible (65K nodes × 1KB = 65MB).

9. **candle Metal custom ops (P2)**
   AddrNet's Gumbel-softmax, trie L0 reads (custom indexing), and
   scratchpad atomics may need custom Metal shaders. candle's Metal
   backend covers standard ops but not custom memory access patterns.
   
   Mitigation: Implement custom candle ops via Metal Performance Shaders
   or raw Metal compute shaders. candle supports custom kernel registration.

10. **Spatiotemporal tag overhead in 256-token window (P2)**
    Tags like "localhost/alice/chat@2026-04-10T10:19:22Z: " are ~45 bytes.
    In a 256-token window, that's 17% overhead per tile.
    
    Mitigation: The learned text encoder can compress tags efficiently.
    Or: use compact tag encoding (hash-based tag IDs instead of full text).
    Or: separate tag channel (tags go directly to tag_register, not tokens).

### LOW RISK

11. **LoRA in candle (P3)**
    candle has basic LoRA support but it's less mature than PyTorch's PEFT.
    May need custom implementation for tool-specialized LoRA.

12. **Scratchpad write conflicts (P3) — RESOLVED**
    Strength-weighted mean across all agents: `slot[i] = Σ(strength × value) / Σ(strength)`.
    Fully parallel, no atomics, fully differentiable. See SWARM.md scratchpad section.

13. **Image/audio generators at 5-20M (Phase 2+ risk)**
    Generating high-quality images from 5-20M params is very difficult.
    Even tiny diffusion models need ~50M+ for decent quality.
    Mitigation: Start with 64×64 resolution. Use progressive training.
    Or fall back to tool-dispatch for high-quality generation.

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Language | Full Rust (candle) | 250K TPS, no FFI overhead |
| Agent type | Homogeneous base + LoRA | Simple training, flexible specialization |
| Agent size | ~666K params (d_model=96) | Sweet spot: expressive + fast |
| Context window | 256 VQ codes per chunk, one RVQ layer per chunk | 3-agent pipeline: L1→L2→L3. Each agent gets previous layer's output as input. K=1 alone gives semantic preview. Same base agent. |
| Tiling | Work-queue (spatiotemporal boundaries) | Natural splitting, no wasted context, variable-size units |
| Headers | HTTP-inspired dual-channel (VQ-encoded inline + side-channel) | Headers go through VQ like everything else; side-channel provides structured enrichment |
| Tag injection | VQ-encoded inline + side-channel (input-only) | Same VQ codebook for tags and content; 10 encoded vectors via cross-attention |
| Context cross-attn | Unified 14 slots (10 tag + 4 register) | Simpler, fewer params than separate layers |
| Internal register | 4×d_model bank, GRU-gated (read-then-update), reset per unit | General-purpose scratchpad replaces tag-specific GRU |
| Output headers | Inline bytes only, Rust parses text | Side-channel is input-only; per-token heads wasteful for constant values |
| Tag encoder | Shared VQ codebook embeddings + sinusoidal time | Same embedding space for tags and content |
| Origin header | Separate from author (who-provided) | Tracks provenance: author wrote it, origin served it |
| Memory structure | Hierarchical trie | Proven, elegant multi-scale |
| Memory tiers | L0 GPU / L1 CPU / L2 disk / L3 Internet | Level-configurable depth, L3 read-only at inference |
| Memory access | Non-blocking async | Agent never waits, progressive detail |
| Read/write addrs | Separate (input→read, output→write) | Same AddrNets, different inputs; writes can target novel locations |
| Write values | Shared V_proj + 3 aspect heads (96→16→96) | Each channel stores base + specialized residual |
| Write strength | Learned continuous α modifier | Always writes (trail of thoughts), model controls intensity |
| L3 trigger | Attention weight signal | Low/uniform mem-attn → query remote on next cycle |
| L3 contribution | Separate offline process | Privacy: never leak inference data to remote |
| L3 trust model | Trust weight × novelty gating | EMA strength per contributor, diminishing returns |
| Agent count | Dynamic N, configurable per deployment | Tools to recommend swarm configuration per hardware |
| Adaptive computation | NOOP token (no halt head) | NOOP = "nothing to say yet", replaces ACT. Simpler, same effect |
| Peer comm | Scratchpad + cross-attn hybrid | Fixed cost + rich per-tile context |
| Attention pool | Unified (all sources in one) | Simplest, model learns routing |
| Attention loss | Access cost + quality penalty | Prevents dilution and stale reads |
| Tool dispatch | Event-queue at chunk boundaries | No mid-stream pausing, batch parallel dispatch |
| Multimodal | VQ-256 codec agents + shared coarse RVQ | Bytes everywhere, core agents modality-blind |
| Multimodal sync | Shared coarse codebook | Same coarse codes → inherent temporal sync |
| Code decoder | Weight-tied LM head (embed.weight.T) | Symmetric with Code Embedding, saves 24K params |
| Work decomposition | Chunking + Tiling + Refinement | 3 orthogonal axes: temporal, spatial/spectral, quality |
| Audio tiling | Frequency-band decomposition | Bass/mid/treble as parallel tiles |
| Image tiling | Spatial quadrant decomposition | 4+ agents process regions in parallel |
| Video generation | Spatial + spectral tiling + temporal chunking | All three axes combined |
| Trie role | Knowledge retrieval ONLY | Not for tile coordination or chunk state |
| Scratchpad role | Agent coordination + state sharing | Tiles, chunks, refinement passes all use scratchpad |
| Tokenization | VQ-256 codebooks per modality, byte vocab (0-255) | Everything maps to bytes, uniform interface |
| Training | 8-phase curriculum (1→2→3→4→5→6→7→8) | Easiest first, progressive freeze/unfreeze, AddrNet after base |
| Training principle | Progressive freeze/unfreeze | Freeze what's learned, train harder parts on stable foundations, gradual coherence |
| Text VQ refinement | Semantic layers: K=1=concepts, K=2=language, K=3=formatting | Pseudocode → target language → formatted code. Same pattern for chat: intent → words → emoji/style |
| Text VQ codec | Perceiver-style learned adaptive compression | Discovers word/character boundaries per language, no explicit tokenizer |
| Text compression | Learned adaptive (not fixed-stride) | Fixed stride splits multi-byte UTF-8 chars; Perceiver handles variable-width |
| Emoji enrichment | RVQ Layer 3 + continuous VAD space | Emojis = largest emotional dataset; cross-modal bridge for tone |
| Enrichment output | Actual Unicode emojis inline | 4 bytes per emoji, negligible at 250K TPS |
| Unknown enrichment | UNKNOWN_CODE (not neutral) | External input with no emotional info → explicit unknown, not assumed neutral |
| Core agent input | VQ codes (0-255), not raw bytes | Code 0x41 = "codebook entry 65", not "A" — modality-blind |
| Partiality model | Imagination via hierarchical trie | Sparse fine levels → model infers from coarse. See PARTIALITY.md |
| Confidence signal | Duplex density (structural) | populated_children per ancestor → density chain → confidence gate |
| Write diffusion | Gravitational density halo | Writes probabilistically create siblings, cousins. See MEMORY.md |
| Confidence compounding | Parent density flows to children | Dense parent → confident context for child reads. Sparse parent → uncertain child. Density propagates naturally |
| Confidence gate | Learned gate from density+depth chain | ~2,129 params. density_embedding + depth_embedding → sigmoid |
| Write strength × confidence | Sparse → weak write, dense → strong | Feedback loop: weak data + later observation → diffusion halo grows → density rises → solid knowledge |
| Partiality training | Phase 6 with asymmetric losses | After coherence unfreeze. Multi-view, imagination quality, hallucination penalty |
| UTF-8 vs UTF-16 | Irrelevant (codec handles internally) | Core agents see only VQ codes, never raw bytes |

---

## Prior Art

Zeno's architecture builds on ideas explored in the
[ANT project](https://github.com/kaaninel/ANT) — a 937K parameter byte-level
transformer with a Rust/PyO3 hierarchical trie. Key lessons carried forward:

- **External memory works.** Separating knowledge (trie) from computation
  (weights) enables unbounded context without parameter growth.
- **AddrNet co-processors** are viable for learned address generation via
  Gumbel-softmax, though training stability requires careful curriculum.
- **Spatiotemporal tags** as byte prefixes provide source/time attribution
  with zero architecture changes.
- **EMA blending** at trie nodes gives stable, gradual knowledge accumulation.

Zeno extends these ideas with a multi-agent swarm, unified VQ-256 codecs,
async non-blocking memory, and a pure Rust implementation targeting Metal.
