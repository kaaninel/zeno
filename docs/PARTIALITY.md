# Zeno — Partiality & Imagination

## Core Principle

```
  Nearly all content is PARTIAL.

  A text tag describes ~15% of an image.
  A meeting summary captures ~10% of what was said.
  "Upbeat pop song" describes ~5% of the audio.
  Even an image is partial: the photographer's framing selected a view.

  Every piece of content is someone's attention-applied summary of
  a richer reality. The model MUST understand this — not as a bug
  to work around, but as the fundamental nature of information.
```

---

## Trie Hierarchy as Natural Partiality Encoding

```
  The hierarchical trie already encodes partiality structurally:

  Text tag "a golden retriever in a park":
    → writes to coarse trie levels (depth 1-3)
    → fine levels (depth 4-8) remain sparse/unpopulated

  Full image of that same park:
    → writes to ALL trie levels (depth 1-8), dense
    → coarse: "park scene", fine: "golden retriever, bench, fountain, trees"

  Reading from the text-written path:
    Depth 1-3: solid data (directly written by text processing)
    Depth 4-8: empty or EMA-propagated from ancestors (never directly stored)

  The PATTERN of dense vs sparse levels = a COVERAGE MAP.
  The model can see what it knows and what it's missing.
```

---

## Imagination = Resolving Sparse Levels from Coarse

```
  When the model reads a sparse trie path, it encounters gaps where
  fine levels have no direct data. It can INFER what those levels
  should contain based on coarse context — just like an image upscaler
  reconstructs detail from low-resolution input.

  "a dog in a park"
    Coarse (known): park scene, dog present
    Fine (imagined): probably grass, trees, maybe sky, possibly people

  The transformer's cross-attention naturally does this:
    - Attends to available (coarse) memory vectors
    - Internal FFN predicts/fills what's missing
    - Generated representations include both known and imagined content

  This IS imagination — pattern-completed inference from partial data.

  Key: the model must KNOW what's imagined vs what's known.
  "I see a dog" (known from tag) vs "there's probably grass" (imagined).
  This distinction prevents hallucination: generating imagined detail
  while being aware it's imagined, not presenting guesses as facts.
```

---

## Confidence as Continuous Gradient (Duplex Density)

```
  Partiality is NOT binary (known vs unknown). It's a continuous
  gradient of confidence derived from STRUCTURAL DENSITY — the
  populated_children count at each ancestor along the read path.

  The density chain IS the confidence signal:

  Density Profile              Interpretation
  ─────────────────────────   ──────────────────────────────────
  All ancestors dense          Well-explored path — HIGH confidence
  (200+ children each)         (well-established concept area)

  Dense root, sparse leaves    Known concept, but specific detail
  (200→150→50→5→1→0)          uncertain — MEDIUM confidence
                               (concept exists, details imagined)

  Sparse throughout            Uncharted territory — LOW confidence
  (10→3→1→1→0→0)              (barely explored, likely imagined)

  Dense root, abrupt drop      Known broad category, specific path
  (200→180→2→1→0→0)           is rare — LOW-MEDIUM confidence
                               (unusual specialization of known topic)


  How density grows with evidence (via write diffusion):
  ─────────────────────────────────────────────────────

  Writes to    Parent    Grandparent
  same leaf    density   density        Confidence reading
  ──────────   ───────   ───────────    ──────────────────
     1          1/256    unchanged       VERY LOW (single trace)
    10          ~2/256   unchanged       LOW (faint halo)
    50          ~4/256   ~2 more         LOW-MEDIUM (visible halo)
   200          ~12/256  ~5 more         MEDIUM (clear gravity well)
   500+         ~26/256  ~10 more        HIGH (dense neighborhood)

  The halo grows organically through write diffusion — no counters
  needed. The trie structure IS the evidence.


  Duplex confidence (both directions):
  ─────────────────────────────────────
  Leaf confidence = f(ancestor density chain from root to leaf)

  Dense ancestors, sparse leaf area:
    "I'm in well-known territory, but this specific spot is fresh"
    → moderate confidence (good neighborhood, new data)

  Sparse ancestors, any leaf:
    "The whole path here is unexplored"
    → low confidence regardless of leaf detail

  Confidence flows THROUGH the hierarchy structurally:
    Imagining from dense (well-explored) ancestors → reasonable guesses
    Imagining from sparse (barely populated) ancestors → very uncertain
```

---

## Learned Confidence Gate

```
  A small learned gate converts the structural density chain into a
  confidence signal that modulates memory attention and write strength.

  ┌─────────────────────────── CONFIDENCE GATE ───────────────────────┐
  │                                                                    │
  │  Inputs (per read path):                                           │
  │    density_chain: [u8; path_depth]  (populated_children at each    │
  │                                      ancestor along the read path) │
  │    depth_chain:   [u8; path_depth]  (depth level of each ancestor) │
  │                                                                    │
  │  Computation:                                                      │
  │    For each ancestor i in the path:                                │
  │      density_embed_i = density_embedding(density_chain[i]) [8-dim] │
  │      depth_embed_i   = depth_embedding(depth_chain[i])     [8-dim] │
  │      feat_i = cat(density_embed_i, depth_embed_i)          [16-dim]│
  │                                                                    │
  │    path_feat = mean(feat_0, feat_1, ..., feat_D)           [16-dim]│
  │    confidence = sigmoid(W_gate @ path_feat + b_gate)       [scalar]│
  │                                                                    │
  │  Outputs (modulates two things):                                   │
  │                                                                    │
  │    1. Memory attention:                                            │
  │       mem_value_effective = confidence × value_vector              │
  │       (low-confidence vectors get attenuated in cross-attention)   │
  │                                                                    │
  │    2. Write strength:                                              │
  │       α_effective = sigmoid(strength_head(hidden)) × confidence    │
  │                     × base_α                                       │
  │       (representations derived from sparse/uncertain sources get    │
  │        written weakly — they stay tentative until diffusion halo   │
  │        builds and density confirms the pattern)                    │
  │                                                                    │
  │  Parameter cost:                                                   │
  │    density_embedding: 256 × 8-dim = 2,048 params                  │
  │    depth_embedding:     8 × 8-dim =    64 params                  │
  │    W_gate:            16 × 1      =    16 params                  │
  │    b_gate:                         =     1 param                   │
  │    TOTAL:                          ≈ 2,129 params                  │
  │    Negligible addition to ~666K agent                              │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘


  Why density embedding (not raw count):
  ───────────────────────────────────────
  Raw density 0-255 is a wide range. A learned 8-dim embedding lets
  the model discover that density=0 and density=1 are both "sparse"
  while density=100 and density=200 are both "dense" — nonlinear
  bucketing learned from data, not hand-designed log-scale buckets.

  The depth embedding captures: "density=5 at depth 0 means something
  different than density=5 at depth 7." Near the root, low density is
  unusual. Near leaves, low density is normal.
```

---

## Feedback Loop: Imagination → Observation → Knowledge

```
  The confidence system creates a natural knowledge consolidation loop
  driven by structural density growth through write diffusion:

  1. IMAGINATION (confidence = LOW, sparse density)
     Model reads coarse trie data for "park"
     Path: dense at depth 0-3, sparse at depth 4-8
     Imagines: "probably has trees, grass, benches"
     Writes imagination with LOW strength (sparse source → low confidence)
     Write diffusion: at p=0.05, maybe 0-1 sibling created
     Neighborhood: still sparse → density signal unchanged

  2. PARTIAL CONFIRMATION (density grows)
     Later, processes text: "the park bench was cold"
     "bench" gets direct write → plus diffusion siblings near bench leaf
     Multiple writes in the "park" subtree → parent density increases
     Confidence gate sees denser ancestors → stronger writes allowed
     But "trees" and "grass" subtrees still sparse — stay tentative

  3. FULL CONFIRMATION (density HIGH)
     Processes image of the park → writes across many addresses
     Trees, grass, fountain all get direct writes + diffusion halos
     Parent/grandparent density jumps (many new children created)
     Ancestor density chain now reads dense at every level
     Confidence gate opens wide → solid writes → knowledge solidifies

  4. CONTRADICTION (EMA corrects)
     Processes: "the park had no trees, just open meadow"
     Tree representation gets overwritten (EMA with new value)
     The previous imagination was WRONG — but the system handled it:
       - Imagined tree had sparse neighborhood (no diffusion halo)
       - New observation writes strongly (dense source neighborhood)
       - EMA naturally corrects: α×new + (1-α)×old → new dominates
       - Old diffusion siblings stay (weak, faint) — no harm

  This mirrors human memory consolidation:
    Vague impressions → repeated exposure → solid memories
    Wrong guesses → corrected by direct experience
    Confident knowledge = dense neighborhoods, resistant to noise
```

---

## Partiality in Cross-Modal Alignment

```
  Cross-modal training must use ASYMMETRIC losses.

  WRONG: bidirectional exact match
    loss = ||text_embed - image_embed||²
    This forces text to capture EVERYTHING in image (impossible)

  RIGHT: directional containment
    loss = -log P(text is consistent WITH image)
    Text describes SOME of what's in image → that's sufficient
    The model learns: "dog in park" is a valid PARTIAL view of the image

  Multi-view training:
    Same image paired with DIFFERENT partial captions:
      "a golden retriever"           (focuses on animal)
      "a park with benches"          (focuses on setting)
      "a sunny day outdoors"         (focuses on atmosphere)
    All are VALID partial descriptions of the same image.
    The model learns: any single description is one view among many.

  With VQ codes:
    Image VQ codes capture FULL visual content (all positions)
    Text VQ codes of caption capture PARTIAL semantic content
    Shared coarse codebook aligns what overlaps
    The GAP between text coverage and image coverage IS partiality

  Confidence gate ensures:
    Cross-modal associations from sparse density paths get LOW confidence
    Only associations from densely-confirmed neighborhoods get HIGH confidence
    The model never over-commits to a single partial view
```

---

## Connection to Other Mechanisms

```
  Partiality awareness extends patterns already in the architecture:

  L3 Internet trigger:
    Low/uniform mem-attn weights → "I don't have enough info"
    → Query remote trie for more data
    This IS a partiality response — detecting gaps and seeking more.

  UNKNOWN_CODE for enrichment:
    External text with no emotional info → UNKNOWN_CODE (not neutral)
    Explicitly "I don't know the emotion" rather than guessing.
    Same principle: represent what's missing, don't pretend completeness.

  NOOP token (adaptive computation):
    NOOP (0x06) = "nothing to say yet" — model outputs NOOP when it
    needs more processing before committing to meaningful output.
    Each NOOP cycle still does full memory read/write.
    With confidence gate: sparse-density reads → more NOOP cycles
    → search different addresses → maybe find denser neighborhoods.
    No separate halt head needed — NOOP achieves the same effect
    with simpler, uniform cycle logic.

  RVQ layers:
    Layer 1 = semantic gist (most partial, most abstract)
    Layer 2 = exact reconstruction (more complete)
    Layer 3 = enrichment (nuance, nearly full)
    Progressive VQ IS a partiality hierarchy.
    Training with RVQ layer dropout = training with varying partiality.

  Write strength head:
    Already modulates write intensity.
    Confidence gate multiplies it: sparse source → weaker writes.
    No new head needed — existing mechanism gains partiality awareness
    through density-driven confidence gating.
```
