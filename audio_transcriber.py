"""
Audio transcription module.

Supports two backends, selectable via the `backend` parameter:

  "api"   (DEFAULT) — OpenAI whisper-1 via the Transcriptions API.
                      Returns per-segment timestamps needed for scene assignment.

  "local"           — OpenAI Whisper on-device. Free, slower.
                      Model sizes: tiny / base / small / medium / large

Full-video transcription (used by pipeline):
  The whole video is transcribed once at the start of the pipeline.
  whisper-1 / local Whisper both return segment-level timestamps.
  segments_for_scene() then slices the global transcript to each scene's
  time window and converts to clip-relative timestamps for Gemini.

Both backends return List[TranscriptSegment] so the rest of the pipeline is
backend-agnostic.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List


@dataclass
class TranscriptSegment:
    start: float   # seconds, relative to current clip start
    end: float
    text: str


def transcribe_full_video( video_path: str, backend: str = "api", local_model: str = "large" ) -> List[TranscriptSegment]:
    tmp_mp3 = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_mp3 = tmp.name
        tmp.close()

        _extract_audio_mp3(video_path, tmp_mp3)

        if backend == "local":
            segments = _transcribe_local(tmp_mp3, local_model)
        else:
            segments = _transcribe_api(tmp_mp3, "whisper-1")
    finally:
        if tmp_mp3 and os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)

    if segments:
        full_text = " ".join(s.text for s in segments)
        if not _is_valid_charset(full_text):
            print(f"  [transcription filtered] suspicious character set: {full_text[:80]!r}")
            return []

    return segments


def segments_for_scene(
    all_segments: List[TranscriptSegment],
    scene_start: float,
    scene_end: float,
) -> List[TranscriptSegment]:
    """
    Find all words whose start time falls inside [scene_start, scene_end) and
    join them into a single TranscriptSegment with clip-relative timestamps.

    Word start time is used for assignment (not overlap) so each word belongs
    to exactly one scene.

    Args:
        all_segments:  Word-level segments from transcribe_full_video().
        scene_start:   Scene start time in seconds (absolute).
        scene_end:     Scene end time in seconds (absolute).

    Returns:
        List with one TranscriptSegment (the joined words), or [] if no words
        fall in this scene's window.
    """
    words = [s for s in all_segments if scene_start <= s.start < scene_end]
    if not words:
        return []
    return [TranscriptSegment(
        start=words[0].start - scene_start,
        end=words[-1].end - scene_start,
        text=" ".join(w.text for w in words),
    )]


def format_transcript(segments: List[TranscriptSegment]) -> str:
    """
    Format transcript segments into a readable string for Gemini prompts.
    Timestamps are clip-relative (seconds).
    """
    if not segments:
        return ""
    lines = [f"[{s.start:.2f}s – {s.end:.2f}s]: {s.text}" for s in segments]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hallucination filters (heuristics)
# ---------------------------------------------------------------------------

def _is_valid_charset(text: str) -> bool:
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    has_cjk   = bool(re.search(r'[\u4e00-\u9fff]', text))
    return has_latin and not has_cjk


# ---------------------------------------------------------------------------
# API backend — default
# ---------------------------------------------------------------------------

def _transcribe_api(audio_path: str, model_name: str) -> List[TranscriptSegment]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env or use --transcriber local."
        )
    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as f:
        #Request word-level timestamps — each word gets its own precise start/end.
        #This is the only way to correctly assign speech to short scenes.
        
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
        segments: List[TranscriptSegment] = []

        for word in (response.words or []):
            text = word.word.strip()
            if text:
                segments.append(TranscriptSegment(
                    start=float(word.start),
                    end=float(word.end),
                    text=text,
                ))
        return segments
        #create a list of transcription segments extarct each word from response and its timestamps, so we store aeach 
        #word with its strt and end ts..


def _transcribe_local(audio_path: str, model_name: str) -> List[TranscriptSegment]:
    import whisper

    # local Whisper wants WAV; convert from the MP3 we already have
    wav_path = audio_path.replace(".mp3", "_local.wav")
    _convert_to_wav(audio_path, wav_path)
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(wav_path, word_timestamps=True, verbose=False)
        segments: List[TranscriptSegment] = []
        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if text:
                segments.append(TranscriptSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=text,
                ))
        return segments
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _extract_audio_mp3(video_path: str, output_path: str) -> None:
    #extract audio from ffmpeg get the mp3 only.
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg MP3 extraction failed:\n{result.stderr}")


def _convert_to_wav(mp3_path: str, wav_path: str) -> None:
    """Convert MP3 to mono 16 kHz WAV for local Whisper."""
    cmd = [
        "ffmpeg", "-y",
        "-i", mp3_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg WAV conversion failed:\n{result.stderr}")
