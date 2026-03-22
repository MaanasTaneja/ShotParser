"""
Gemini prompts used by the pipeline.

Three prompts:
  GLOBAL_REGISTRY_PROMPT  — pre-pass over the full video at 1 fps.
                            Returns a JSON entity legend (characters, locations, props).

  SHOT_ANALYSIS_PROMPT    — per-clip analysis when a transcript is available.
  NO_TRANSCRIPT_PROMPT    — per-clip analysis fallback (no speech).

Both per-clip prompts inject:
  {global_registry}  — the PERSISTENT STORY LEGEND built in the pre-pass
  {rolling_context}  — rolling window of recent shot summaries
"""

# ---------------------------------------------------------------------------
# Pre-pass prompt — builds the global entity registry from the full video
# ---------------------------------------------------------------------------
MOVEMENT_ANALYSIS_PROMPT = """\
You are an expert camera operator and cinematographer. Your ONLY job is to identify the camera movement in this video clip.

Analyze EVERY frame of this clip carefully. You must identify exactly what the camera (not the subject) is doing.

CAMERA MOVEMENT TAXONOMY — pick exactly ONE:
• static/locked      — camera is completely still. Zero movement throughout.
• pan left           — camera rotates horizontally to the left on a fixed axis
• pan right          — camera rotates horizontally to the right on a fixed axis
• tilt up            — camera rotates vertically upward on a fixed axis
• tilt down          — camera rotates vertically downward on a fixed axis
• dolly in           — camera physically moves toward the subject (background gets smaller)
• dolly out          — camera physically moves away from subject (background gets larger)
• push-in            — slow, subtle dolly toward subject
• pull-out           — slow, subtle dolly away from subject
• zoom in            — focal length increases (background compresses, subject gets larger WITHOUT parallax)
• zoom out           — focal length decreases
• tracking           — camera moves laterally following a subject
• crane up           — camera moves upward through space
• crane down         — camera moves downward through space
• handheld           — irregular, organic shake/drift — no coherent directional movement
• drone/aerial       — smooth floating motion from above
• whip pan           — extremely fast pan causing motion blur

HOW TO DISTINGUISH KEY PAIRS:
- Dolly vs Zoom: In a dolly, background elements shift in parallax relative to the subject. In a zoom, the entire frame scales uniformly with no parallax.
- Pan vs Tracking: A pan rotates (foreground and background move at different rates). A tracking shot translates (camera physically moves sideways).
- Static vs Subtle push: Check if the subject very slowly grows in frame over the duration. Even 5% size increase = push-in.
- Handheld vs Static: Look for small, irregular drifts. If the frame never perfectly settles = handheld.

Output ONLY a valid JSON object — no markdown fences, no explanation:
{{"movement": "your single choice from the taxonomy above"}}"""


GLOBAL_REGISTRY_PROMPT = """\
SYSTEM: You are a Lead Script Supervisor and Visual Continuity Engineer.
TASK: Analyze the provided video and generate a "Global Entity Registry."

OBJECTIVE: Identify every recurring Character, Location, and Prop that has narrative or visual significance. Focus ONLY on permanent, immutable physical traits.

STRICT INSTRUCTIONS:
1. NO ACTIONS: Do not describe what someone is doing (e.g., "sitting").
2. NO EMOTIONS: Do not describe how someone feels (e.g., "angry").
3. ATTRIBUTES ONLY: Focus on clothing, physical features, architectural layout, and distinct objects.
4. NAMING: Assign each a unique ID (e.g., [CHAR_1], [LOC_1], [PROP_1]). Use real names if they are explicitly mentioned in the video (e.g., "[Tom Cruise]").

Output ONLY a valid JSON object — no markdown fences, no explanation:
{
  "CHARACTERS": {
    "[CHAR_ID]": "Detailed physical description (hair, clothing, height, distinguishing marks)."
  },
  "LOCATIONS": {
    "[LOC_ID]": "Distinctive features of the setting (lighting, furniture, background elements)."
  },
  "PROPS": {
    "[PROP_ID]": "Specific objects of importance (e.g., a specific award, a phone, a trophy)."
  }
}"""


# ---------------------------------------------------------------------------
# Primary prompt — dialogue pre-filled from OpenAI transcript
# ---------------------------------------------------------------------------
SHOT_ANALYSIS_PROMPT = """\
You are a professional cinematographer and editor building a shot list for a video production database.

{global_registry}

RECENT CONTEXT (last few shots):
{rolling_context}

INSTRUCTION FOR COMPOSITION:
Look at the PERSISTENT STORY LEGEND above.
If a character, location, or prop from the Legend is on screen, use their [ID].
STRICT RULE: Do not repeat descriptions from the Legend (e.g., do NOT say "[CHAR_1] is wearing a blue suit").
ONLY describe what has CHANGED since the RECENT CONTEXT (e.g., "[CHAR_1] is now shouting into [PROP_1]").

OPTICAL FLOW MOTION HINT (measured from actual video frames — use as a strong prior for the `movement` field):
{motion_hint}

DIALOGUE TRANSCRIPT (speech recognition for this clip):
{transcript}

Analyze this video clip. This clip is a SINGLE shot — shot boundaries have already been detected. Your job is only to describe what you see.
Output exactly one shot object.

For EACH shot output every field below:

• id             — integer starting at {start_id}
• startTime      — absolute timestamp MM:SS.mmm  (this clip starts at {offset_timecode} in the full video — add {offset_seconds:.3f}s to every relative time you observe)
• endTime        — absolute timestamp MM:SS.mmm
• cut            — transition INTO this shot: hard cut | dissolve | match cut | whip pan | wipe | fade in | smash cut | jump cut | etc.
• shotType       — extreme wide shot | wide shot | medium wide | medium | medium close-up | close-up | extreme close-up | insert | POV | over-the-shoulder | two-shot | etc.
• angle          — eye-level | low angle | high angle | dutch/canted | bird's eye | worm's eye
• lens           — wide/wide-angle | normal | telephoto | macro | anamorphic
• focus          — what is sharp and depth-of-field character (e.g. "shallow DOF, subject sharp, background bokeh")
• movement       — static/locked | pan left/right | tilt up/down | dolly in/out | tracking/follow | crane up/down | handheld | zoom in/out | drone/aerial | etc.
• composition    — 4–7 sentences using [IDs] from the Legend for known entities. Describe positions, props, background, colors, lighting, mood, and ONLY changes or new elements not already captured in RECENT CONTEXT. Specific enough to recreate the frame.
• graphicsOverlays — on-screen text, titles, lower thirds, logos, captions, VFX. Write exactly "None" if absent.
• dialogue       — using the transcript above, quote the exact words spoken during this shot's time window. Write exactly "None" if no speech occurs.
• sfx            — describe music (genre/mood/instrumentation), ambient sound, and notable sound effects. Do NOT include spoken words. Write exactly "None" if no non-speech audio.
• sceneSummary   — HIDDEN FIELD: one sentence summarising what happens in this shot (used internally for rolling context — will not appear in final output). Be concise; reference [IDs] where relevant.

STRICT RULES:
1. startTime / endTime must be ABSOLUTE (full-video time). Add {offset_seconds:.3f}s to every clip-relative timestamp.
2. Timestamps accurate to ~50ms. Use the exact cut frame, not a round number.
3. dialogue: only words spoken within this shot's time window — write "None" if silent.
4. sfx: music, ambience, sound effects only — no spoken words.
5. composition: use [IDs] from the Legend; never re-describe a known entity from scratch.

⚠️ TIMESTAMP WARNING: This clip was extracted from a longer video. The clip file starts at 00:00 internally, but in the FULL VIDEO it begins at {offset_timecode} ({offset_seconds:.3f}s). Your first shot's startTime MUST be approximately {offset_timecode} — NEVER 00:00.000 unless the full video itself starts at zero.

Output ONLY a valid JSON object — no markdown fences, no explanation:
{{
  "shots": [
    {{
      "id": {start_id},
      "startTime": "MM:SS.mmm",
      "endTime": "MM:SS.mmm",
      "cut": "...",
      "shotType": "...",
      "angle": "...",
      "lens": "...",
      "focus": "...",
      "movement": "...",
      "composition": "...",
      "graphicsOverlays": "...",
      "dialogue": "...",
      "sfx": "...",
      "sceneSummary": "..."
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Fallback prompt — no transcript available
# ---------------------------------------------------------------------------
NO_TRANSCRIPT_PROMPT = """\
You are a professional cinematographer and editor building a shot list for a video production database.

{global_registry}

RECENT CONTEXT (last few shots):
{rolling_context}

INSTRUCTION FOR COMPOSITION:
Look at the PERSISTENT STORY LEGEND above.
If a character, location, or prop from the Legend is on screen, use their [ID].
STRICT RULE: Do not repeat descriptions from the Legend (e.g., do NOT say "[CHAR_1] is wearing a blue suit").
ONLY describe what has CHANGED since the RECENT CONTEXT (e.g., "[CHAR_1] is now shouting into [PROP_1]").

OPTICAL FLOW MOTION HINT (measured from actual video frames — use as a strong prior for the `movement` field):
{motion_hint}

Analyze this video clip. This clip is a SINGLE shot — shot boundaries have already been detected. Your job is only to describe what you see.
Output exactly one shot object.

For EACH shot output every field below:

• id             — integer starting at {start_id}
• startTime      — absolute timestamp MM:SS.mmm  (this clip starts at {offset_timecode} in the full video — add {offset_seconds:.3f}s to every relative time you observe)
• endTime        — absolute timestamp MM:SS.mmm
• cut            — transition INTO this shot: hard cut | dissolve | match cut | whip pan | wipe | fade in | smash cut | jump cut | etc.
• shotType       — extreme wide shot | wide shot | medium wide | medium | medium close-up | close-up | extreme close-up | insert | POV | over-the-shoulder | two-shot | etc.
• angle          — eye-level | low angle | high angle | dutch/canted | bird's eye | worm's eye
• lens           — wide/wide-angle | normal | telephoto | macro | anamorphic
• focus          — what is sharp and depth-of-field character
• movement       — Analyze motion across ALL frames before deciding. Check for:
                   (1) subject growing/shrinking in frame → zoom in/out or dolly in/out
                   (2) entire frame shifting laterally/vertically → pan or tilt
                   (3) background parallax with stable subject → dolly or tracking
                   (4) edge motion blur or slight shakiness → handheld
                   (5) slow push toward subject → push-in / dolly in
                   Do NOT default to "static" — actively verify. Only label static/locked
                   if the frame is genuinely completely stable throughout.
                   Format: one of — static/locked | pan left | pan right | tilt up | tilt down |
                   dolly in | dolly out | tracking | handheld | zoom in | zoom out | crane up |
                   crane down | drone/aerial | push-in | pull-out
• composition    — 4–7 sentences using [IDs] from the Legend for known entities. Describe positions, props, background, colors, lighting, mood, and ONLY changes or new elements not already captured in RECENT CONTEXT. Specific enough to recreate the frame.
• graphicsOverlays — on-screen text, titles, logos, VFX. Write exactly "None" if absent.
• dialogue       — transcribe any spoken words or voiceover as accurately as possible. Write exactly "None" if silent.
• sfx            — describe music (genre/mood/instrumentation), ambient sound, and notable sound effects. Do NOT include spoken words. Write exactly "None" if no non-speech audio.
• sceneSummary   — HIDDEN FIELD: one sentence summarising what happens in this shot (used internally for rolling context — will not appear in final output). Be concise; reference [IDs] where relevant.

STRICT RULES:
1. startTime / endTime must be ABSOLUTE. Add {offset_seconds:.3f}s to every clip-relative timestamp.
2. Timestamps accurate to ~50ms. Use the exact cut frame, not a round number.
3. composition: use [IDs] from the Legend; never re-describe a known entity from scratch.

⚠️ TIMESTAMP WARNING: This clip was extracted from a longer video. The clip file starts at 00:00 internally, but in the FULL VIDEO it begins at {offset_timecode} ({offset_seconds:.3f}s). Your first shot's startTime MUST be approximately {offset_timecode} — NEVER 00:00.000 unless the full video itself starts at zero.

Output ONLY a valid JSON object — no markdown fences, no explanation:
{{
  "shots": [
    {{
      "id": {start_id},
      "startTime": "MM:SS.mmm",
      "endTime": "MM:SS.mmm",
      "cut": "...",
      "shotType": "...",
      "angle": "...",
      "lens": "...",
      "focus": "...",
      "movement": "...",
      "composition": "...",
      "graphicsOverlays": "...",
      "dialogue": "...",
      "sfx": "...",
      "sceneSummary": "..."
    }}
  ]
}}"""
