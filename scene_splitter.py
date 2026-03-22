"""
Shot boundary detection and clip extraction.

scene_splitter_algorithm() is the main entry point for the pipeline.

Internally runs three passes:
  Pass 1 — ffmpeg scene filter (threshold 0.1).
            Over-splits intentionally to catch every possible cut.
  Pass 2 — CLIP visual similarity merge.
            Merges adjacent scenes that are visually similar (false cuts).
  Pass 3 — Validator loop.
            Two heuristics check every scene for hidden cuts:
              H1: CLIP frame distinctness (first / mid / last frame pairs)
              H2: Optical flow motion peak (mid segment vs start/end segments)
            Suspicious scenes are re-split with tighter thresholds.
            Repeats until clean or max_iterations reached.

Dependencies:
  pip install ffmpeg-python transformers torch pillow opencv-python numpy
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import ffmpeg
import numpy as np
from PIL import Image


# Gemini Files API floor — clips below this are dropped during detection
GEMINI_MIN_CLIP_DURATION = 0.1  # seconds

# Heuristic thresholds (hardcoded — physical properties of a single shot)
_FRAME_DISTINCTNESS_THRESHOLD = 0.70   # CLIP sim below this = suspicious (so higher should be better right?)
_MIN_DURATION_FOR_H1          = 0.5    # scenes shorter than this skip H1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    scene_id: int
    start_time: float       # seconds, absolute in original video
    end_time: float
    start_frame: int
    end_frame: int
    start_timecode: str     # MM:SS.mmm
    end_timecode: str
    clip_path: str          = field(default="")
    suspicious: bool        = field(default=False)
    suspicious_reason: str  = field(default="")
    immune: bool            = field(default=False)  # tried to re-split and failed — skip validator forever


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def seconds_to_timecode(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def get_video_fps(video_path: str) -> float:
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        num, den = video_stream["r_frame_rate"].split("/")
        return float(num) / float(den)
    except Exception as exc:
        print(f"  WARNING: could not read FPS ({exc}), defaulting to 25.0")
        return 25.0


def _extract_frame(video_path: str, timestamp: float) -> Image.Image:
    """Extract a single frame at the given timestamp as a PIL Image."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(max(0.0, timestamp)),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-loglevel", "error",
            tmp_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return Image.open(tmp_path).convert("RGB")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Pass 1 — ffmpeg scene detection
# ---------------------------------------------------------------------------

def detect_scenes_ffmpeg(
    video_path: str,
    threshold: float = 0.1,
    time_offset: float = 0.0,
    duration_limit: Optional[float] = None,
) -> List[Scene]:
    """
    Detect shot boundaries using ffmpeg's scene filter.

    Args:
        video_path:      Path to video file.
        threshold:       Scene change sensitivity (0–1). Lower = more cuts.
        time_offset:     Start scanning from this absolute timestamp (seconds).
                         All returned Scene timestamps are absolute.
        duration_limit:  Only scan this many seconds from time_offset.
                         None = scan to end of video.
    """
    cmd = ["ffmpeg", "-ss", str(time_offset), "-i", video_path]
    if duration_limit is not None:
        cmd += ["-t", str(duration_limit)]
    cmd += [
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # With -ss before -i (input seeking), ffmpeg does NOT reset PTS.
    # pts_time values from showinfo are already absolute timestamps in the
    # original video — do NOT add time_offset (that would double-count it).
    raw_cuts = sorted(
        float(t)
        for t in re.findall(r"pts_time:([\d.]+)", result.stderr)
    )

    try:
        probe = ffmpeg.probe(video_path)
        full_duration = float(probe["format"]["duration"])
    except Exception:
        full_duration = raw_cuts[-1] + 1.0 if raw_cuts else time_offset + (duration_limit or 1.0)

    if duration_limit is not None:
        end_time = min(time_offset + duration_limit, full_duration)
    else:
        end_time = full_duration

    fps = get_video_fps(video_path)

    boundaries = [time_offset] + [c for c in raw_cuts if time_offset < c < end_time] + [end_time]
    scenes: List[Scene] = []
    for i in range(len(boundaries) - 1):
        start_sec = boundaries[i]
        end_sec   = boundaries[i + 1]
        scenes.append(Scene(
            scene_id=len(scenes) + 1,
            start_time=start_sec,
            end_time=end_sec,
            start_frame=int(start_sec * fps),
            end_frame=max(0, int(end_sec * fps) - 1),
            start_timecode=seconds_to_timecode(start_sec),
            end_timecode=seconds_to_timecode(end_sec),
        ))

    if time_offset == 0.0:
        print(f"  FPS: {fps:.3f}  |  Duration: {full_duration:.2f}s  |  Cuts: {len(raw_cuts)}")
    print(f"  → {len(scenes)} scenes after pass 1 (offset={time_offset:.2f}s, threshold={threshold})")
    return scenes


# ---------------------------------------------------------------------------
# Pass 2 — CLIP visual similarity merge
# ---------------------------------------------------------------------------

_clip_cache = None

def _get_clip_model():
    global _clip_cache
    if _clip_cache is None:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
        except ImportError:
            raise ImportError(
                "transformers and torch are required for scene merging.\n"
                "Install with: pip install transformers torch"
            )
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model = HFCLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_cache = (model, processor, device)
        print(f"  CLIP model loaded (device: {device})")
    return _clip_cache


def get_blur_score(image: Image.Image) -> float:
    """Laplacian variance blur score — lower means more blurry (e.g. whip-pan)."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(img_cv, cv2.CV_64F).var()


def _clip_cosine_similarity(model, processor, device, img_a: Image.Image, img_b: Image.Image) -> float:
    import torch
    inputs = processor(images=[img_a, img_b], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        feats = model.vision_model(pixel_values=pixel_values).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] * feats[1]).sum())


#okay i thnk i may have figured outht problem if there is a lot of blur my merger 
#drops down the simialrity threshold for merging since its hard for the clip model to identify
#features in blurred. using the laplacian measure. thats why we alwyas merge
#perhaps to fix this? we might need to create a new function but the problem with that 
#approach is.. is that since its a huerisitic it will break other splitting for most other files.
#since i have accounted for blurred not to split shots but what about blur need to split shots.
#this is where the tug and war is happening


def merge_scenes_clip(
    video_path: str,
    scenes: List[Scene],
    similarity_threshold: float = 0.85,
) -> List[Scene]:
    """
    Pass 2: merge adjacent scenes whose boundary frames are visually similar.

    For each adjacent pair, extracts the last 3 frames of scene A and the
    first 3 frames of scene B, computes all 9 cross-product CLIP cosine
    similarities, and merges if more than 2 pairs exceed similarity_threshold.
    Sampling multiple frames removes the need for blur-aware threshold hacks —
    a single blurry boundary frame no longer drives the decision.
    Repeats until no merges happen in a full pass.
    """
    if len(scenes) <= 1:
        return scenes

    model, preprocess, device = _get_clip_model()
    fps = get_video_fps(video_path)
    frame_step = max(1.0 / fps, 0.04)

    total_before = len(scenes)
    changed = True
    passes = 0

    while changed:
        changed = False
        passes += 1
        merged: List[Scene] = []
        i = 0

        while i < len(scenes):
            if i + 1 >= len(scenes):
                merged.append(scenes[i])
                i += 1
                continue

            scene_a = scenes[i]
            scene_b = scenes[i + 1]

            # Last 3 frames of scene A (stepping back from end)
            dur_a = scene_a.end_time - scene_a.start_time
            tail_times = [
                max(scene_a.start_time, scene_a.end_time - k * frame_step)
                for k in range(1, 4)
            ]
            # First 3 frames of scene B (stepping forward from start)
            dur_b = scene_b.end_time - scene_b.start_time
            head_times = [
                min(scene_b.end_time, scene_b.start_time + k * frame_step)
                for k in range(1, 4)
            ]

            try:
                tail_frames = [_extract_frame(video_path, t) for t in tail_times]
                head_frames = [_extract_frame(video_path, t) for t in head_times]
            except Exception as exc:
                print(f"    WARNING: CLIP frame extract failed at {scene_a.end_timecode}: {exc}")
                merged.append(scene_a)
                i += 1
                continue

            # All 9 cross-product similarities
            similarities = [
                _clip_cosine_similarity(model, preprocess, device, fa, fb)
                for fa in tail_frames
                for fb in head_frames
            ]
            above = sum(1 for s in similarities if s >= similarity_threshold)

            if above >= 2:
                scene_a.end_time      = scene_b.end_time
                scene_a.end_frame     = scene_b.end_frame
                scene_a.end_timecode  = scene_b.end_timecode
                merged.append(scene_a)
                i += 2
                changed = True
            else:
                merged.append(scene_a)
                i += 1

        scenes = merged

    # Absorb scenes below Gemini's minimum into their preceding neighbour.
    # Dropping them outright creates coverage gaps — absorbing keeps full coverage.
    before_filter = len(scenes)
    absorbed: List[Scene] = []
    for s in scenes:
        if s.end_time - s.start_time < GEMINI_MIN_CLIP_DURATION and absorbed:
            # Extend previous scene to swallow this micro-scene
            absorbed[-1].end_time     = s.end_time
            absorbed[-1].end_frame    = s.end_frame
            absorbed[-1].end_timecode = s.end_timecode
        else:
            absorbed.append(s)
    scenes = absorbed
    dropped = before_filter - len(scenes)

    for idx, s in enumerate(scenes):
        s.scene_id = idx + 1

    print(f"  CLIP merge: {total_before} → {len(scenes)} scenes "
          f"({passes} pass(es), threshold={similarity_threshold})" +
          (f"  [{dropped} too-short dropped]" if dropped else ""))
    return scenes


# ---------------------------------------------------------------------------
# Pass 3 — Validator heuristics
# ---------------------------------------------------------------------------


def heuristic_clip_distinctness(
    video_path: str,
    scene: Scene,
    n_frames: int = 12,
) -> Tuple[bool, float]:
    """
    Heuristic 1: sample n_frames evenly across the scene, compute a full
    pairwise CLIP similarity matrix (all N*(N-1)/2 pairs), and return the
    minimum similarity found.

    If that minimum < _FRAME_DISTINCTNESS_THRESHOLD the scene is suspicious —
    at least one pair of frames looks like it belongs to a different shot,
    which a 3-frame first/mid/last check would miss if the cut falls between
    those sparse sample points.

    Returns: (is_suspicious, min_similarity_across_all_pairs)
    """
    duration = scene.end_time - scene.start_time
    if duration < _MIN_DURATION_FOR_H1:
        return False, 1.0

    # Sample evenly, with a small inset from the boundaries
    inset = min(0.05, duration * 0.05)
    timestamps = [
        scene.start_time + inset + (i / (n_frames - 1)) * (duration - 2 * inset)
        for i in range(n_frames)
    ]

    try:
        frames = [_extract_frame(video_path, t) for t in timestamps]
    except Exception as exc:
        print(f"    WARNING: H1 frame extraction failed for scene {scene.scene_id}: {exc}")
        return False, 1.0

    import torch
    model, processor, device = _get_clip_model()

    inputs = processor(images=frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        feats = model.vision_model(pixel_values=pixel_values).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True) 

    # Full similarity matrix via dot product — shape (n_frames, n_frames)
    sim_matrix = (feats @ feats.T).cpu()

    #extract upper triangle (exclude diagonal) to get all unique pairs
    upper = sim_matrix.triu(diagonal=1)
    #the other half of matrix is just same values, (i,j), and (j,i) here direction doesnt maytter in similarity.
    pair_sims = upper[upper != 0]
    min_sim = float(pair_sims.min())

    return min_sim < _FRAME_DISTINCTNESS_THRESHOLD, min_sim



def histogram_distance(img1: Image.Image, img2: Image.Image) -> float:
    """Bhattacharyya distance between HS histograms of two frames. Higher = more different."""
    hsv1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2HSV)

    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def heuristic_histogram_discontinuity(
    video_path: str,
    scene: Scene,
    samples: int = 6,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Heuristic 3: sample frames evenly across the scene and compute the
    Bhattacharyya histogram distance between each consecutive pair.

    A large jump in color distribution between adjacent samples indicates a
    scene change that ffmpeg and CLIP both missed.

    Returns: (is_suspicious, max_consecutive_histogram_jump)
    """
    duration = scene.end_time - scene.start_time
    if duration < 2.0:
        return False, 0.0

    frames = []
    for i in range(samples):
        t = scene.start_time + (i / (samples - 1)) * duration
        try:
            frames.append(_extract_frame(video_path, t))
        except Exception:
            return False, 0.0

    dists = [
        histogram_distance(frames[i], frames[i + 1])
        for i in range(len(frames) - 1)
    ]

    max_jump = max(dists)
    return max_jump > threshold, max_jump




_DURATION_PRIOR_THRESHOLD = 15.0  # seconds — shots longer than this are inherently suspicious


def heuristic_duration_prior(
    scene: Scene,
) -> Tuple[bool, float]:
    """
    Heuristic 4: a single continuous shot rarely exceeds 15 seconds.
    If the scene duration is above the threshold it is a prior for suspicion.

    Returns: (is_suspicious, duration)
    """
    duration = scene.end_time - scene.start_time
    return duration > _DURATION_PRIOR_THRESHOLD, duration


def compute_suspicion_score(
    video_path: str,
    scene: Scene,
) -> Tuple[int, str]:
    """
    Run all three heuristics and return a suspicion score + reason string.

    Scoring:
      +2  heuristic_clip_distinctness       fired  (CLIP — strongest signal)
      +1  heuristic_histogram_discontinuity fired
      +1  heuristic_duration_prior          fired

    Max score: 4. Scene is flagged suspicious if score > 2.

    Returns: (score, reason_string)
    """
    h1_flag, h1_val = heuristic_clip_distinctness(video_path, scene)
    h3_flag, h3_val = heuristic_histogram_discontinuity(video_path, scene)
    h4_flag, h4_val = heuristic_duration_prior(scene)

    score = 0
    reasons = []

    if h1_flag:
        score += 2
        reasons.append(f"clip_distinctness(min_sim={h1_val:.3f})")
    if h3_flag:
        score += 1
        reasons.append(f"histogram_jump(max={h3_val:.3f})")
    if h4_flag:
        score += 1
        reasons.append(f"duration_prior({h4_val:.1f}s)")

    return score, " + ".join(reasons)


def find_suspicious_scenes(
    video_path: str,
    scenes: List[Scene],
) -> List[Scene]:
    """
    Score every scene with compute_suspicion_score. Scenes with score > 2
    are marked suspicious. All checked scenes are marked immune so the
    validator never re-runs them.
    """
    suspicious: List[Scene] = []
    for scene in scenes:
        if scene.immune:
            continue

        scene.suspicious        = False
        scene.suspicious_reason = ""

        score, reason = compute_suspicion_score(video_path, scene)
        scene.immune = True  # checked — don't re-run regardless of result

        if score > 2:
            scene.suspicious        = True
            scene.suspicious_reason = reason
            print(
                f"  Scene {scene.scene_id:03d} "
                f"[{scene.start_timecode} → {scene.end_timecode}] "
                f"SUSPICIOUS (score={score}): {reason}"
            )
            suspicious.append(scene)

    return suspicious


# ---------------------------------------------------------------------------
# Main algorithm — entry point for pipeline
# ---------------------------------------------------------------------------

def scene_splitter_algorithm(
    video_path: str,
    ffmpeg_threshold: float = 0.1,
    clip_threshold: float = 0.85,
    max_iterations: int = 3,
    skip_validator: bool = False,
) -> List[Scene]:
    """
    Full iterative scene splitting algorithm.

    1. Pass 1 + Pass 2 on the full video.
    2. Validator: find suspicious scenes via H1 (frame distinctness) + H2 (motion peak).
    3. Re-split only suspicious scenes with tighter thresholds.
    4. Repeat until no suspicious scenes remain or max_iterations hit.

    Threshold biasing per iteration:
      ffmpeg_threshold → × 0.7  (more splits),  floor  at 0.03
      clip_threshold   → + 0.04 (less merging),  ceiling at 0.97

    Args:
        video_path:        Path to input video.
        ffmpeg_threshold:  Starting ffmpeg scene sensitivity (default 0.1).
        clip_threshold:    Starting CLIP merge threshold (default 0.85).
        max_iterations:    Max validator iterations (default 3).
        skip_validator:    If True, skip the validator loop entirely.

    Returns:
        Final list of Scene objects. Scenes still suspicious after max_iterations
        have suspicious=True and will be skipped during clip extraction.
    """
    # --- Initial full-video pass 1 + pass 2 ---
    print(f"  Pass 1 — ffmpeg scene detection (threshold={ffmpeg_threshold}) …")
    scenes = detect_scenes_ffmpeg(video_path, threshold=ffmpeg_threshold)

    print(f"  Pass 2 — CLIP merge (threshold={clip_threshold}) …")
    scenes = merge_scenes_clip(video_path, scenes, similarity_threshold=clip_threshold)

    if skip_validator:
        print(f"  Validator skipped (--skip-validator).")
        return scenes

    # --- Validator loop ---
    current_ffmpeg_threshold = ffmpeg_threshold
    current_clip_threshold   = clip_threshold
    converged = False

    for iteration in range(max_iterations):
        print(f"\n  [Validator iter {iteration + 1}/{max_iterations}] "
              f"Checking {len(scenes)} scenes …")

        suspicious = find_suspicious_scenes(video_path, scenes) #splits out suspicious ids bascially.

        if not suspicious:
            print(f"  [Validator] All clean — done.")
            converged = True
            break

        # On last iteration just flag and stop — don't re-split
        if iteration == max_iterations - 1:
            print(f"  [Validator] Max iterations reached — "
                  f"{len(suspicious)} scene(s) still suspicious (will be skipped during extraction).")
            break

        # Tighten thresholds for re-splitting
        current_ffmpeg_threshold = max(0.03, current_ffmpeg_threshold * 0.5) #even more aggreisve suspicious splitting. since 0.1 is already so low, 
        #if things cannot be caught by that we need to hevily go deeper.
        current_clip_threshold   = min(0.99, current_clip_threshold   + 0.1) #and more aggreisve merging (like needs to be very similary to merge)

        #on manual ffmpg split testing seems like merhing is the problme it merges too much, effectively nullifying
        #ourt aggreisive split so lets tweak those values a bit.
        print(f"  [Validator] {len(suspicious)} suspicious — re-splitting "
              f"(ffmpeg={current_ffmpeg_threshold:.3f}, CLIP={current_clip_threshold:.2f}) …")

        suspicious_ids = {s.scene_id for s in suspicious}
        new_scenes: List[Scene] = []

        for scene in scenes:
            #go over each scene and analyze the suspicious scenes and ids.
            if scene.scene_id not in suspicious_ids:
                new_scenes.append(scene)
                continue

            scene_duration = scene.end_time - scene.start_time
            sub = detect_scenes_ffmpeg(
                video_path,
                threshold=current_ffmpeg_threshold,
                time_offset=scene.start_time,
                duration_limit=scene_duration,
            )

            #biased merge (harder to merge)
            sub = merge_scenes_clip(
                video_path, sub,
                similarity_threshold=current_clip_threshold,
            ) 

            #okay so the issue here is we have two different herusitics fighting against each other.

            if len(sub) <= 1:
                # Could not split further — mark immune so validator never re-flags it
                scene.suspicious        = False
                scene.suspicious_reason = ""
                scene.immune            = True
                new_scenes.append(scene)
                print(f"    Scene {scene.scene_id:03d} — could not split further, marked immune")
            else:
                print(f"    Scene {scene.scene_id:03d} → {len(sub)} sub-scenes")
                new_scenes.extend(sub)

        scenes = new_scenes

        # Renumber all IDs after replacement
        for idx, s in enumerate(scenes):
            s.scene_id = idx + 1

    n_clean      = len([s for s in scenes if not s.suspicious])
    n_suspicious = len([s for s in scenes if s.suspicious])
    print(f"\n  [Validator] Final: {len(scenes)} scenes "
          f"({n_clean} clean, {n_suspicious} suspicious)")

    return scenes


# ---------------------------------------------------------------------------
# Motion hint — deterministic optical flow for Gemini prompt injection
# ---------------------------------------------------------------------------

def check_for_motion(video_path: str, scene: Scene, threshold: float = 1.5) -> str:
    """
    Two-track motion detection:

    Track 1 — Consecutive frame optical flow (fast motion):
      Grabs actual adjacent frame pairs via VideoCapture at 3 positions in the
      scene. Farneback works correctly here because frames are 1/fps apart.
      Catches fast pans, whips, handheld shake.

    Track 2 — First-vs-last frame pixel difference (slow/gradual motion):
      Computes mean absolute pixel difference between the first and last frame
      of the scene. A slow push-in or creeping zoom barely moves per-frame but
      the overall scene change is clearly visible across the full duration.
      Catches slow dolly, slow zoom, gentle tilt that Track 1 misses entirely.

    Both scores are computed; the higher tier from either track wins.
    Output is intentionally vague — no direction, just motion intensity level.
    """
    duration = scene.end_time - scene.start_time
    if duration < 0.2:
        return "Minimal motion detectable (clip too short to measure)."

    # ------------------------------------------------------------------
    # Track 1: consecutive frame optical flow
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    sample_positions = [
        scene.start_time + duration * frac
        for frac in [0.20, 0.50, 0.80]
    ]
    flow_mags = []
    for t in sample_positions:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret1, f1 = cap.read()
        ret2, f2 = cap.read()
        if not ret1 or not ret2:
            continue
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(float(np.mean(mag)))
    cap.release()

    fast_score = float(np.mean(flow_mags)) if flow_mags else 0.0

    # ------------------------------------------------------------------
    # Track 2: first-vs-last frame absolute pixel difference (slow motion)
    # Normalised to 0–255 grayscale scale.
    # Typical values: static ~2–5, slow zoom ~8–20, pan ~20–50+
    # ------------------------------------------------------------------
    try:
        f_first = _extract_frame(video_path, scene.start_time + 0.05)
        f_last  = _extract_frame(video_path, scene.end_time   - 0.05)
        g_first = cv2.cvtColor(np.array(f_first), cv2.COLOR_RGB2GRAY)
        g_last  = cv2.cvtColor(np.array(f_last),  cv2.COLOR_RGB2GRAY)
        slow_score = float(np.mean(cv2.absdiff(g_first, g_last)))
    except Exception:
        slow_score = 0.0

    # Map slow_score to equivalent motion tiers
    # (calibrated so slow_score ~8 ≈ fast_score ~1.5 "subtle" threshold)
    slow_normalised = slow_score / 5.0  # rough scale to match fast_score units

    # Use whichever track sees more motion
    combined = max(fast_score, slow_normalised)

    if combined < threshold:
        return "Minimal / no camera motion detected. The camera appears static or locked-off."
    elif combined < 4.0:
        return "Low-to-moderate camera motion detected. There is likely a subtle camera move (slow push, gentle zoom, slight pan or tilt)."
    elif combined < 9.0:
        return "Clear camera motion detected. There is a definite camera move in this shot (pan, tilt, dolly, tracking, or zoom)."
    else:
        return "Strong camera motion detected. This shot has significant movement (fast pan, whip, handheld, or heavy action)."



#actual creation of mulliple clip files function.
def extract_scene_clips(
    video_path: str,
    scenes: List[Scene],
    output_dir: str,
) -> List[Scene]:
    """
    Extract each non-suspicious scene as a standalone MP4 clip using ffmpeg.
    Audio is re-encoded to AAC so Gemini and Whisper can read it cleanly.
    Scenes marked suspicious=True are skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for scene in scenes:
        clip_path = os.path.join(output_dir, f"{video_name}_scene_{scene.scene_id:03d}.mp4")
        duration  = scene.end_time - scene.start_time

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(scene.start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            "-loglevel", "error",
            clip_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for scene {scene.scene_id}:\n" + result.stderr.decode()
            )

        scene.clip_path = clip_path
        suspicious_tag = f"  [suspicious: {scene.suspicious_reason}]" if scene.suspicious else ""
        print(
            f"  Scene {scene.scene_id:03d}: "
            f"{scene.start_timecode} → {scene.end_timecode}  "
            f"({duration:.2f}s)  →  {os.path.basename(clip_path)}{suspicious_tag}"
        )

    return scenes
