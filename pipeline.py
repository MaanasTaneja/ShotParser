"""
Main pipeline CLI.

Flow:
  0.  Gemini (full video, 1 fps)       → Global Entity Registry (characters, locations, props)
  1a. ffmpeg scene filter (threshold 0.1) → over-split into maximum cuts
  1b. CLIP visual similarity merge         → join false cuts, repeat until stable
  2.  ffmpeg                               → extract final clips
  3.  Whisper (full-video, one call)       → transcript with word timestamps
  4.  Gemini                               → visual + audio analysis per clip
  5.  Assemble                             → shots[] JSON for the viewer

Usage:
  python pipeline.py my_video.mp4
  python pipeline.py my_video.mp4 --threshold 0.1 --clip-threshold 0.85
  python pipeline.py my_video.mp4 --transcriber local --local-model medium
  python pipeline.py my_video.mp4 --no-transcribe
  python pipeline.py my_video.mp4 --gemini3 --keep-clips

Environment variables (in .env):
  GEMINI_API_KEY   — required
  OPENAI_API_KEY   — required unless --no-transcribe or --transcriber local
"""

import argparse
import json
import mimetypes
import os
import re
import shutil
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from audio_transcriber import format_transcript, transcribe_full_video, segments_for_scene
from memory import VideoMemory
from prompts import GLOBAL_REGISTRY_PROMPT, MOVEMENT_ANALYSIS_PROMPT, NO_TRANSCRIPT_PROMPT, SHOT_ANALYSIS_PROMPT
from scene_splitter import Scene, scene_splitter_algorithm, extract_scene_clips, check_for_motion

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def upload_clip(clip_path: str) -> str:
    """Upload a video clip to the Gemini Files API and wait until ACTIVE."""
    mime_type, _ = mimetypes.guess_type(clip_path)
    if not mime_type or not mime_type.startswith("video/"):
        mime_type = "video/mp4"

    with open(clip_path, "rb") as f:
        uploaded = client.files.upload(file=f, config={"mime_type": mime_type})

    while True:
        info = client.files.get(name=uploaded.name)
        if info.state.name == "ACTIVE":
            return uploaded.uri
        if info.state.name == "FAILED":
            raise RuntimeError(f"Gemini file upload failed: {info.error}")
        time.sleep(2)


def build_global_registry(video_path: str, memory: VideoMemory, use_gemini_3: bool) -> None:
    """
    Pre-pass: upload the full video at 1 fps and ask Gemini to identify all
    recurring characters, locations, and props. Populates memory.global_registry.
    """
    print(f"  Uploading full video for entity scan …")
    try:
        file_uri = upload_clip(video_path)
    except RuntimeError as exc:
        print(f"  WARNING: pre-pass upload failed — {exc}. Skipping registry build.")
        return

    parts = [
        types.Part(
            file_data=types.FileData(file_uri=file_uri),
            video_metadata=types.VideoMetadata(fps=4),
        ),
        types.Part(text=GLOBAL_REGISTRY_PROMPT),
    ]

    try:
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=types.Content(parts=parts),
            config=types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.2,
                media_resolution="MEDIA_RESOLUTION_HIGH",
                thinking_config=types.ThinkingConfig(thinking_budget=16384),
            ),
        )
    except Exception as exc:
        print(f"  WARNING: registry Gemini call failed — {exc}. Skipping registry build.")
        return

    text = response.text or ""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    json_str = match.group(1).strip() if match else text.strip()

    try:
        data = json.loads(json_str)
        memory.update_registry(data)
        n_chars = len(memory.global_registry["CHARACTERS"])
        n_locs  = len(memory.global_registry["LOCATIONS"])
        n_props = len(memory.global_registry["PROPS"])
        print(f"  Registry: {n_chars} characters, {n_locs} locations, {n_props} props")
        print(memory.format_registry())
    except json.JSONDecodeError as exc:
        print(f"  WARNING: registry JSON parse failed — {exc}")
        print(f"  Raw response (first 400 chars): {text[:400]}")


def analyze_scene_clip(
    scene: Scene,
    file_uri: str,
    transcript: str,
    start_id: int,
    fps: int,
    use_gemini_3: bool,
    global_registry: str,
    rolling_context: str,
    motion_hint: str = "",
) -> list[dict]:
    """
    Send one scene clip to Gemini and return the shots list.

    Args:
        scene:           Scene metadata (used for timestamp offsetting in the prompt)
        file_uri:        Gemini Files URI returned by upload_clip()
        transcript:      Formatted transcript string (may be "")
        start_id:        Shot ID to start numbering from
        fps:             Frame sampling rate passed to Gemini
        use_gemini_3:    Use Gemini 3 Pro instead of 2.5 Pro
        global_registry: Formatted PERSISTENT STORY LEGEND string
        rolling_context: Formatted recent shot summaries string

    Returns:
        List of shot dicts, or [] on parse failure
    """
    if transcript.strip():
        prompt = SHOT_ANALYSIS_PROMPT.format(
            transcript=transcript,
            start_id=start_id,
            offset_timecode=scene.start_timecode,
            offset_seconds=scene.start_time,
            global_registry=global_registry,
            rolling_context=rolling_context,
            motion_hint=motion_hint,
        )
    else:
        prompt = NO_TRANSCRIPT_PROMPT.format(
            start_id=start_id,
            offset_timecode=scene.start_timecode,
            offset_seconds=scene.start_time,
            global_registry=global_registry,
            rolling_context=rolling_context,
            motion_hint=motion_hint,
        )

    # Gemini recommends: video part first, then text instruction
    parts = [
        types.Part(
            file_data=types.FileData(file_uri=file_uri),
            video_metadata=types.VideoMetadata(fps=fps),
        ),
        types.Part(text=prompt),
    ]

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            max_output_tokens=32768,
            temperature=0.2,
            media_resolution="MEDIA_RESOLUTION_HIGH",
            thinking_config=types.ThinkingConfig(thinking_budget=32768),
        ),
    )

    text = response.text or ""

    # Strip markdown fences if the model wrapped its output
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    json_str = match.group(1).strip() if match else text.strip()

    try:
        data = json.loads(json_str)
        shots = data.get("shots", [])
        return shots
    except json.JSONDecodeError as exc:
        print(f"    WARNING: JSON parse failed for scene {scene.scene_id}: {exc}")
        print(f"    Raw response (first 400 chars): {text[:400]}")
        return []


def analyze_movement(file_uri: str, fps: int) -> str:
    """
    Dedicated Gemini call to identify camera movement for a single clip.
    Returns a movement string from the taxonomy (e.g. 'pan left', 'static/locked').
    """
    parts = [
        types.Part(
            file_data=types.FileData(file_uri=file_uri),
            video_metadata=types.VideoMetadata(fps=fps),
        ),
        types.Part(text=MOVEMENT_ANALYSIS_PROMPT),
    ]

    try:
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=types.Content(parts=parts),
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.1,
                media_resolution="MEDIA_RESOLUTION_HIGH",
                thinking_config=types.ThinkingConfig(thinking_budget=4096),
            ),
        )
    except Exception as exc:
        print(f"    WARNING: deep-movement Gemini call failed — {exc}")
        return ""

    text = response.text or ""
    match = re.search(r'"movement"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1).strip()

    # Fallback: strip fences and parse JSON
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    json_str = fence_match.group(1).strip() if fence_match else text.strip()
    try:
        data = json.loads(json_str)
        return data.get("movement", "").strip()
    except json.JSONDecodeError:
        print(f"    WARNING: deep-movement JSON parse failed. Raw: {text[:200]}")
        return ""


def run_pipeline(
    video_path: str,
    output_path: str | None = None,
    threshold: float = 0.1,
    clip_threshold: float = 0.85,
    fps: int = 8,
    use_gemini_3: bool = False,
    skip_transcription: bool = False,
    transcriber_backend: str = "api",
    local_model: str = "large",
    keep_clips: bool = False,
    skip_registry: bool = False,
    skip_validator: bool = False,
    deep_movement: bool = False,
) -> dict:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    clips_dir = os.path.join(
        os.path.dirname(os.path.abspath(video_path)), f"{video_name}_clips"
    )

    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_analysis.json"

    transcriber_label = (
        "skipped" if skip_transcription
        else f"OpenAI whisper-1 (full-video)" if transcriber_backend == "api"
        else f"local Whisper/{local_model} (full-video)"
    )

    print("=" * 60)
    print(f"Video        : {video_path}")
    print(f"Output       : {output_path}")
    print(f"Gemini model : {'Gemini 3 Pro' if use_gemini_3 else 'Gemini 2.5 Pro'}  |  clip FPS: 16  |  registry FPS: {fps}")
    print(f"Transcription: {transcriber_label}")
    print(f"Scene detect : ffmpeg threshold={threshold}  CLIP merge threshold={clip_threshold}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 0 — Global Entity Registry (pre-pass, full video at 1 fps)
    # ------------------------------------------------------------------
    memory = VideoMemory()
    registry_path = os.path.splitext(video_path)[0] + "_registry.json"

    if skip_registry:
        print(f"\n[0/4] Global Entity Registry — SKIPPED (--skip-registry)")
    else:
        print(f"\n[0/4] Building Global Entity Registry (full video at 4 fps) …")
        build_global_registry(video_path, memory, use_gemini_3)
        with open(registry_path, "w") as f:
            json.dump(memory.global_registry, f, indent=2)
        print(f"       Registry saved → {registry_path}")

    # ------------------------------------------------------------------
    # Step 1 — Scene splitting algorithm (pass1 + pass2 + validator loop)
    # ------------------------------------------------------------------
    print(f"\n[1/4] Scene splitting algorithm …")
    scenes = scene_splitter_algorithm(
        video_path,
        ffmpeg_threshold=threshold,
        clip_threshold=clip_threshold,
        skip_validator=skip_validator,
    )
    print(f"       → {len(scenes)} scenes")

    # ------------------------------------------------------------------
    # Step 2 — Clip extraction (suspicious scenes are skipped)
    # ------------------------------------------------------------------
    print(f"\n[2/4] Extracting clips with ffmpeg …")
    scenes = extract_scene_clips(video_path, scenes, clips_dir)

    # ------------------------------------------------------------------
    # Step 3 — Transcribe full video once with Whisper
    # ------------------------------------------------------------------
    all_segments = []
    if not skip_transcription:
        print(f"\n[3/4] Transcribing full video with Whisper …")
        try:
            all_segments = transcribe_full_video(
                video_path,
                backend=transcriber_backend,
                local_model=local_model,
            )
            print(f"      → {len(all_segments)} transcript segments")
        except Exception as exc:
            print(f"  WARNING: transcription failed — {exc}")

    # ------------------------------------------------------------------
    # Step 4 — Analyze each scene clip with Gemini
    # ------------------------------------------------------------------
    print(f"\n[4/4] Analyzing {len(scenes)} clips with Gemini …")

    all_shots: list[dict] = []
    shot_id = 1

    for scene in scenes:
        duration = scene.end_time - scene.start_time
        print(
            f"\n  Scene {scene.scene_id:03d}/{len(scenes):03d} "
            f"[{scene.start_timecode} → {scene.end_timecode}  {duration:.2f}s]"
        )

        # --- Slice the global transcript to this scene's time window ---
        transcript = ""
        if not skip_transcription and all_segments:
            scene_segs = segments_for_scene(all_segments, scene.start_time, scene.end_time)
            transcript = format_transcript(scene_segs)
            if transcript:
                preview = transcript[:120].replace("\n", " ")
                print(f"    Transcript: {preview}{'…' if len(transcript) > 120 else ''}")
            else:
                print("    Transcript: (silent / no speech detected)")

        # --- Optical flow motion hint ---
        motion_hint = check_for_motion(video_path, scene)
        print(f"    Motion: {motion_hint}")

        # --- Upload clip to Gemini ---
        try:
            print(f"    Uploading {os.path.basename(scene.clip_path)} …")
            file_uri = upload_clip(scene.clip_path)
        except RuntimeError as exc:
            print(f"    ERROR uploading: {exc} — skipping scene")
            continue

        # --- Analyze with Gemini ---
        shots = analyze_scene_clip(
            scene=scene,
            file_uri=file_uri,
            transcript=transcript,
            start_id=shot_id,
            fps=16,
            use_gemini_3=use_gemini_3,
            global_registry=memory.format_registry(),
            rolling_context=memory.format_rolling_cache(),
            motion_hint=motion_hint,
        )

        # --- Deep movement analysis (optional parallel pass) ---
        if deep_movement and shots:
            print(f"    Running deep-movement analysis …")
            movement = analyze_movement(file_uri, fps=24)
            if movement:
                print(f"    Deep movement: {movement}")
                for shot in shots:
                    shot["movement"] = movement
            else:
                print(f"    Deep movement: (parse failed, keeping original)")

        if shots:
            for shot in shots:
                # Pop the hidden sceneSummary field before saving to output JSON
                summary = shot.pop("sceneSummary", "").strip()
                if summary:
                    memory.add_to_shot_cache(shot_id, summary)

            all_shots.extend(shots)
            shot_id += len(shots)
            print(f"    → {len(shots)} shots extracted (running total: {len(all_shots)})")
        else:
            print(f"    → WARNING: no shots parsed for scene {scene.scene_id}")

        with open(output_path, "w") as f:
            json.dump({"shots": all_shots}, f, indent=2)

    if not keep_clips:
        shutil.rmtree(clips_dir, ignore_errors=True)
        print(f"\nCleaned up clips directory: {clips_dir}")

    print(f"\n{'=' * 60}")
    print(f"Done — {len(all_shots)} shots → {output_path}")
    print("=" * 60)

    return {"shots": all_shots}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accurate shot-by-shot video analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Output JSON path")

    # Scene detection
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="ffmpeg scene filter threshold (0–1). Lower = more cuts. Default 0.1.",
    )
    parser.add_argument(
        "--clip-threshold", type=float, default=0.85, dest="clip_threshold",
        help=(
            "CLIP cosine similarity threshold for merging adjacent scenes (0–1). "
            "Higher = merge less (keep more cuts). Default 0.85."
        ),
    )

    # Gemini
    parser.add_argument(
        "--fps", type=int, default=8,
        help="Frame sampling rate for the registry pre-pass (step 0). Clips are always sent at 16 fps.",
    )
    parser.add_argument(
        "--gemini3", action="store_true",
        help="Use Gemini 3 Pro instead of Gemini 2.5 Pro",
    )

    # Transcription
    parser.add_argument(
        "--no-transcribe", action="store_true",
        help="Skip transcription entirely",
    )
    parser.add_argument(
        "--transcriber", default="api", choices=["api", "local"],
        dest="transcriber_backend",
        help=(
            "Transcription backend: 'api' (OpenAI whisper-1, default) or "
            "'local' (on-device Whisper). The full video is transcribed once "
            "and segments are assigned to scenes by timestamp."
        ),
    )
    parser.add_argument(
        "--local-model", default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Local Whisper model size (only used when --transcriber local)",
    )

    # Misc
    parser.add_argument(
        "--keep-clips", action="store_true",
        help="Keep extracted scene clips after processing",
    )
    parser.add_argument(
        "--skip-registry", action="store_true", dest="skip_registry",
        help="Skip the global entity registry pre-pass (step 0/4)",
    )
    parser.add_argument(
        "--skip-validator", action="store_true", dest="skip_validator",
        help="Skip the suspicious-scene validator loop in the splitting algorithm",
    )
    parser.add_argument(
        "--deep-movement", action="store_true", dest="deep_movement",
        help=(
            "Run a dedicated Gemini pass per clip to determine camera movement. "
            "Replaces the movement field from the main shot analysis with a focused result."
        ),
    )

    args = parser.parse_args()
    run_pipeline(
        video_path=args.video_path,
        output_path=args.output,
        threshold=args.threshold,
        clip_threshold=args.clip_threshold,
        fps=args.fps,
        use_gemini_3=args.gemini3,
        skip_transcription=args.no_transcribe,
        transcriber_backend=args.transcriber_backend,
        local_model=args.local_model,
        keep_clips=args.keep_clips,
        skip_registry=args.skip_registry,
        skip_validator=args.skip_validator,
        deep_movement=args.deep_movement,
    )


if __name__ == "__main__":
    main()
