# Video Shot Analyzer

Automated shot-by-shot video analysis pipeline. Detects every cut in a video using a custom multi-pass scene splitting algorithm, transcribes dialogue with word-level timestamps, and generates detailed cinematographic descriptions of each shot using Google Gemini — all wrapped in an interactive web viewer.

---

## What it does

Given any video, the pipeline produces a structured JSON breakdown of every individual shot: shot type, camera angle, lens, focus, movement, full composition description, dialogue, music, and sound effects. A second JSON file stores a persistent entity registry — every character, location, and prop seen in the video, described once and referenced consistently across all shots.

The web viewer lets you load the video and JSON side-by-side, scrub through shots on an interactive zoomable timeline, and read the full analysis for each clip as you watch.

<!-- Screenshot: viewer with video loaded and shot list visible -->

---

## Architecture

The pipeline separates concerns across four sequential steps, each using the right tool for the job rather than one monolithic LLM call.

```
Input Video
    │
    ├─→ Step 0: Global Entity Registry
    │          Full video (4fps) → Gemini pre-pass → Characters, Locations, Props
    │
    ├─→ Step 1: Custom Scene Splitting Algorithm
    │          Pass 1: ffmpeg over-split (aggressive threshold)
    │          Pass 2: CLIP window-vote merge (remove false cuts)
    │          Pass 3: Validator loop (find hidden cuts, 3 iterations)
    │
    ├─→ Step 2: Clip Extraction
    │          Each scene → ffmpeg → MP4 (h264 + AAC)
    │
    ├─→ Step 3: Full-Video Transcription
    │          Whisper → word-level timestamps → sliced per scene
    │
    └─→ Step 4: Per-Clip Gemini Analysis
               Video + transcript + registry + rolling context → JSON
```

---

## Custom Scene Splitting Algorithm

This is the core technical contribution. Rather than relying on a pretrained shot boundary model, I built a custom iterative algorithm in `scene_splitter.py` that is fully deterministic, debuggable, and generalizes well across different content types (ads, music videos, documentary, narrative).

### Why not just use TransNet?

The first version used TransNet V2, a neural network trained specifically on shot boundary datasets. It performed reasonably on conventional cuts but consistently missed shots in fast-cut commercial and music video content — particularly sub-second cuts and rapid-fire montage sequences that are underrepresented in its training data. Critically, it provided no mechanism to recover from a miss. If TransNet didn't catch a cut, that was final.

Replacing it with a deterministic algorithm solved both problems. Every decision can be traced back to a concrete numerical comparison, and the architecture is designed to recover from both over-splits and missed cuts.

### Pass 1 — Aggressive ffmpeg over-split

`ffmpeg`'s scene filter runs at a very low threshold (default 0.1) to intentionally over-split the video. The goal is a ceiling of correctness: any real cut that escapes this pass cannot be recovered later. False positives are fine — they're cleaned up in Pass 2.

### Pass 2 — CLIP window-vote merge

For each pair of adjacent scenes, 3 frames are sampled from the tail of scene A and 3 frames from the head of scene B. All 9 cross-product cosine similarities are computed using CLIP (ViT-B/32). The two scenes are merged if **2 or more of the 9 pairs** exceed the similarity threshold (default 0.85). This repeats until a full sweep produces no merges.

**Why a window instead of a single frame?** A single blurry or motion-smeared boundary frame (common in whip-pans and glitch cuts) can produce an anomalously low CLIP similarity and prevent a merge that should happen. With 9 pairs, one outlier frame is outvoted by the other 8. This eliminated the need for blur-aware threshold adjustments that worked on some content but catastrophically failed on others (described further in the design notes below).

### Pass 3 — Suspicious-scene validator loop

After merge, every remaining scene is scored against three heuristics to find hidden cuts — real shot boundaries that survived the merge pass. A scene is only re-split if its **total weighted score exceeds 2**, which requires either H1 alone or any combination of two other signals. This threshold prevents any single heuristic from over-triggering.

| Heuristic | Weight | What it detects |
|-----------|--------|-----------------|
| **H1: CLIP uniform-sample distinctness** | 2 | 12 frames sampled evenly across the scene, full 66-pair similarity matrix computed. Flags if minimum similarity < 0.70. Uniform sampling means a cut between sparse samples can't hide. Strongest signal — double weight. |
| **H2: Color histogram discontinuity** | 1 | 6 frames sampled, consecutive pairs compared via Bhattacharyya distance on HS histograms. Flags if max jump > 0.5. Catches lighting and color-grade transitions that CLIP misses. |
| **H3: Duration prior** | 1 | A single continuous shot rarely exceeds 15 seconds in edited content. Simple but effective. |

Suspicious scenes are re-split with tighter thresholds (`ffmpeg_threshold × 0.5`, `clip_threshold + 0.1`) and marked `immune` if they can't be split further, preventing infinite loops. The loop runs up to 3 iterations.

### Motion hint system

Before each Gemini call, a two-track optical flow computation runs on the extracted clip and injects a plain-English motion intensity description into the prompt.

- **Track 1 (fast motion):** Consecutive frame pairs sampled at 3 positions (20%, 50%, 80%), Farneback optical flow magnitude computed on each pair. Catches rapid pans, whips, handheld shake.
- **Track 2 (slow motion):** Mean absolute pixel difference between the first and last frame of the scene. Catches slow push-ins and creeping zooms that produce near-zero per-frame flow.

The higher of the two scores drives the output tier: static / low-to-moderate / clear / strong. Direction is intentionally omitted — the hint tells Gemini what intensity to expect so it doesn't default to "static," but the final call on movement type is left to the model.

<!-- Screenshot: example output JSON showing motion and composition fields -->

---

## Memory & Context System

A key challenge in per-clip LLM analysis is that each clip is short and lacks context about the broader video. Without context, the same character can be described differently in every shot, and the model has no sense of narrative continuity.

The `VideoMemory` class in `memory.py` addresses this with two mechanisms:

**Global entity registry (Step 0 pre-pass).** Before any splitting, the full video is uploaded to Gemini at 4fps and scanned for all recurring characters, locations, and props. Each entity gets a stable ID (`[CHAR_1]`, `[LOC_1]`, `[PROP_1]`, or a real name if spoken). Every subsequent shot analysis receives this registry and uses the IDs rather than re-describing known entities from scratch.

**Rolling shot cache.** A window of the 10 most recent shot summaries is maintained and injected into every Gemini prompt. This gives the model short-term narrative context — it knows what just happened, where the camera was, and who was in frame — without re-uploading the entire video or consuming unbounded context.

Together these mean that Gemini analyzes each clip as an informed collaborator who has seen the full video and remembers recent events, rather than a blank-slate model seeing a decontextualized fragment.

<!-- Screenshot: example registry JSON with character and location entries -->

---

## Models & Technology

| Component | Model / Tool |
|-----------|-------------|
| Shot analysis | Google Gemini 2.5 Pro (or Gemini 3 Pro via `--gemini3`) |
| Entity registry | Gemini 2.5 Flash Preview (pre-pass) |
| Speech transcription | OpenAI `whisper-1` (API) or local Whisper (any size) |
| Visual similarity | OpenAI CLIP ViT-B/32 (via `transformers`) |
| Scene filter | `ffmpeg` scene detection filter |
| Optical flow | OpenCV Farneback dense flow |
| Color histograms | OpenCV HS histogram + Bhattacharyya distance |

**Gemini configuration:** 32,768 thinking tokens for shot analysis, 8,192 for the registry pre-pass, 4,096 for the optional deep movement pass. Temperature 0.2 throughout. `MEDIA_RESOLUTION_HIGH` for all video uploads.

**Optional deep movement pass (`--deep-movement`).** Sends each clip to Gemini in a second dedicated call with a single focused prompt at 24fps — no other task, just camera movement analysis. Produces significantly more accurate movement classification than the optical flow hint alone, at the cost of one extra Gemini call per scene.

---

## Interactive Viewer

`viewer.html` is a self-contained React app (no build step, no dependencies) for browsing the pipeline output.

- Drag-and-drop upload for the video and analysis JSON
- Zoomable, interactive timeline with shot markers and a scrubbing playhead
- Frame-accurate seeking (1/24s resolution), keyboard navigation
- Shot list panel with full metadata — clicking a shot jumps the video to that point
- Auto-scrolls the current shot into view as the video plays

<!-- Screenshot: timeline zoomed in showing individual shot markers -->

<!-- Screenshot: shot card expanded showing composition, audio, and metadata fields -->

---

## Installation

**System dependencies**

```bash
brew install ffmpeg
```

**Python dependencies** (Python 3.12 recommended)

```bash
pip install -r requirements.txt
```

**Environment variables** — create a `.env` in the project root:

```
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key   # only needed for API transcription (default)
```

---

## Usage

```bash
# Default — Gemini 2.5 Pro, ffmpeg threshold 0.1, CLIP threshold 0.85, whisper-1
python pipeline.py my_video.mp4

# Lower ffmpeg threshold for fast-cut music videos
python pipeline.py music_video.mp4 --threshold 0.05

# Stricter CLIP merge (keep more cuts)
python pipeline.py ad.mp4 --clip-threshold 0.90

# Gemini 3 Pro for stronger visual descriptions
python pipeline.py ad.mp4 --gemini3

# On-device Whisper instead of API
python pipeline.py ad.mp4 --transcriber local --local-model large

# Skip transcription
python pipeline.py ad.mp4 --no-transcribe

# Skip entity registry pre-pass (faster, less cross-shot context)
python pipeline.py ad.mp4 --skip-registry

# Skip the validator loop
python pipeline.py ad.mp4 --skip-validator

# Dedicated per-clip camera movement pass (more accurate, extra cost)
python pipeline.py ad.mp4 --deep-movement

# Keep extracted clips after processing
python pipeline.py ad.mp4 --keep-clips

# Custom output path
python pipeline.py ad.mp4 --output results/ad_shots.json
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold F` | 0.1 | ffmpeg scene filter threshold (0–1). Lower = more initial cuts. |
| `--clip-threshold F` | 0.85 | CLIP cosine similarity threshold for merging adjacent scenes (0–1). Higher = merge less. |
| `--fps N` | 8 | Frame sampling rate for the registry pre-pass. |
| `--gemini3` | off | Use Gemini 3 Pro instead of 2.5 Pro. |
| `--transcriber` | `api` | `api` for OpenAI whisper-1, `local` for on-device Whisper. |
| `--local-model` | `large` | Local Whisper model size: `tiny`, `base`, `small`, `medium`, `large`. |
| `--no-transcribe` | off | Skip transcription entirely. |
| `--skip-registry` | off | Skip the global entity registry pre-pass. |
| `--skip-validator` | off | Skip the suspicious-scene validator loop. |
| `--deep-movement` | off | Run a dedicated Gemini movement analysis pass per clip. |
| `--keep-clips` | off | Keep extracted per-shot clips after processing. |
| `--output / -o` | auto | Output JSON path (defaults to `<video_name>_analysis.json`). |

### Viewing the results

Once the pipeline finishes, open `viewer.html` directly in your browser — no server needed.

```bash
open viewer.html        # macOS
xdg-open viewer.html   # Linux
# or just double-click viewer.html in Finder / Explorer
```

On the upload screen, drop in (or browse for) two files:

1. **Your video** — the same file you passed to `pipeline.py`
2. **The analysis JSON** — `<video_name>_analysis.json` output by the pipeline

Both files need to be loaded before the viewer opens. Once they are, you can scrub the timeline, click any shot in the list to jump to it, and read the full per-shot breakdown alongside the video.

> **Note:** Browsers block local file access by default, so the video and JSON must be loaded through the viewer's file picker rather than referenced by path. Everything runs client-side — no data leaves your machine.

---

## Output Format

The pipeline writes two JSON files.

### `<video_name>_analysis.json`

Shot-by-shot breakdown. Entity references in `composition` use IDs from the registry for consistency across all shots.

```json
{
  "shots": [
    {
      "id": 1,
      "startTime": "00:00.000",
      "endTime": "00:02.417",
      "cut": "hard cut",
      "shotType": "wide",
      "angle": "low angle",
      "lens": "wide angle",
      "focus": "deep focus, full scene sharp",
      "movement": "static",
      "composition": "[Drake] stands at the edge of the ridge overlooking [LOC_ARENA_EXTERIOR] at golden hour...",
      "graphicsOverlays": "None",
      "audio": {
        "dialogue": "What if money moved at the speed of the internet?",
        "music": "Sparse electronic drone, building tension",
        "sfx": "Wind ambience, distant thunder"
      }
    }
  ]
}
```

### `<video_name>_registry.json`

Entity legend built during the pre-pass. Maps every ID to a human-readable description.

```json
{
  "CHARACTERS": {
    "[Drake]": "Male with light-brown skin, a full beard, and black hair styled in cornrows. Outfit 1: Black long-sleeved shirt and black pants.",
    "[J. Cole]": "Male with brown skin, a full beard, and long black hair in dreadlocks."
  },
  "LOCATIONS": {
    "[LOC_ARENA_INTERIOR]": "A massive, dark stadium. The central floor holds a single ping pong table. The stands are filled with a crowd holding up small lights.",
    "[LOC_OFFICE_CUBICLES]": "A brightly lit open-plan office space filled with grey cubicles and mannequins in office attire."
  },
  "PROPS": {
    "[PROP_CLASH_TROPHY]": "A large metallic circular trophy with two paddle-shaped wings bearing 'OVO' and 'Dreamville' logos."
  }
}
```

---

## Design Notes

### The blur-aware merge problem

An early version of Pass 2 included blur-aware threshold adjustment: if either boundary frame had low Laplacian variance (i.e. was motion-blurred), the merge threshold was lowered to bias toward keeping adjacent shots together. The motivation was real — CLIP produces unreliable embeddings from motion-smeared frames.

This worked well on some content. On narrative and ad footage with sparse whip-pans, blurred transition frames were correctly identified and the merge behaved sensibly. But on music video content with dense whip-pan editing throughout, blur-aware merging collapsed large numbers of genuinely distinct shots into a handful of incorrectly long segments. There was no single threshold that worked across both content types.

The fix wasn't to tune the heuristic — it was to make the underlying decision more robust. The 3x3 window voting approach naturally absorbs one outlier frame in the boundary region, eliminating the need for blur-aware logic entirely. The algorithm now performs consistently without content-specific tuning.

### Full-video transcription vs. per-clip

An earlier version transcribed each clip individually. Per-clip transcription meant each ASR call had no audio context from neighboring clips, which hurt quality at scene boundaries and caused more hallucination on silent or music-only clips. Switching to a single full-video Whisper pass with word-level timestamps fixed both: the model has full audio context, and each word is assigned to the right scene by timestamp with no bleed between adjacent scenes.

### Slight over-splitting bias

The algorithm deliberately errs toward splitting. A missed cut is unrecoverable; a spurious cut can be flagged and handled. The practical consequence is that shots with significant in-shot motion or rapid lighting changes may occasionally be split when they should remain one. This is an accepted tradeoff — the outputs are consistently usable across all tested content types.

---

## Known Limitations

- **Effect-heavy transitions** (glitch cuts, flash frames, extreme color grading) can confuse both the CLIP merge pass and the histogram heuristic.
- **English hallucinations** on silent clips: Whisper occasionally produces short filler phrases on clips with no dialogue. The charset filter catches non-Latin hallucinations; English ones require a phrase blocklist or voice activity detection pre-filter.
- **Optical flow in high-action scenes**: Flow magnitude becomes noisy when scene content (object motion, graphics) dominates over camera motion. The `--deep-movement` pass is the workaround.
- **Word-level timestamps**: `gpt-4o-transcribe` produces higher quality ASR but doesn't support `timestamp_granularities=["word"]`, which is required for correct per-scene slicing. The default backend is therefore `whisper-1`.
