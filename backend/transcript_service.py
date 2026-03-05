"""
Audio transcription utilities.

This module provides:
- transcribe_wav_bytes: simple ASR dictation (no diarization).
- diarize_and_transcribe: speaker diarization + per-segment ASR, returned as dialog lines.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import torch
import torchaudio
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ASR_TARGET_SR: int = 16000
DICTATION_CHUNK_S: int = 20
DICTATION_STRIDE_S: int = 4

SEGMENT_CHUNK_S: int = 10
SEGMENT_STRIDE_S: int = 2

MERGE_MAX_GAP_S: float = 0.10
MIN_SEGMENT_DUR_S: float = 0.30

DEFAULT_NUM_SPEAKERS: int = 2


# -----------------------------------------------------------------------------
# ASR text cleaning
# -----------------------------------------------------------------------------

_END_TOKEN_RE = re.compile(r"</s>\s*")


def clean_asr_text(text: str) -> str:
    """
    Remove unwanted special tokens from ASR output and normalize whitespace.
    """
    if not text:
        return ""
    text = _END_TOKEN_RE.sub("", text)  # remove one or many occurrences of </s>
    text = text.replace("<s>", "")      # remove <s> if it appears
    text = " ".join(text.split())       # normalize whitespace
    return text.strip()


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Segment:
    """A diarization segment with a speaker label and time boundaries (seconds)."""
    speaker: str
    start: float
    end: float


# -----------------------------------------------------------------------------
# RTTM helpers
# -----------------------------------------------------------------------------

def read_rttm_to_segments(rttm_path: str) -> List[Segment]:
    """
    Parse an RTTM file (SPEAKER lines) into a sorted list of Segment objects.

    Expected RTTM columns (common format):
      parts[3] -> start time (sec)
      parts[4] -> duration (sec)
      parts[7] -> speaker label
    """
    segments: List[Segment] = []

    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue

            start = float(parts[3])
            dur = float(parts[4])
            end = start + dur
            speaker = parts[7]

            segments.append(Segment(speaker=speaker, start=start, end=end))

    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def merge_adjacent_segments(segments: List[Segment], max_gap: float = MERGE_MAX_GAP_S) -> List[Segment]:
    """
    Merge consecutive segments from the same speaker if the gap between them <= max_gap.
    """
    if not segments:
        return []

    merged: List[Segment] = [segments[0]]

    for seg in segments[1:]:
        last = merged[-1]
        if seg.speaker == last.speaker and (seg.start - last.end) <= max_gap:
            last.end = max(last.end, seg.end)
        else:
            merged.append(seg)

    return merged


# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------

def clamp(v: int, lo: int, hi: int) -> int:
    """Clamp integer v to [lo, hi]."""
    return max(lo, min(hi, v))

def _asr_input_resampled(
    waveform: torch.Tensor,
    sr: int,
    target_sr: int = ASR_TARGET_SR,
) -> Dict[str, object]:
    """
    Convert waveform to mono, resample to target_sr if needed,
    and return HuggingFace ASR pipeline input dict:
      {"array": np.ndarray(float32), "sampling_rate": int}
    """
    # Convert to mono (1D)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        mono = waveform.mean(dim=0)
    else:
        mono = waveform.squeeze(0) if waveform.ndim == 2 else waveform

    # Resample if needed
    if sr != target_sr:
        mono = torchaudio.functional.resample(mono, sr, target_sr)
        sr = target_sr

    arr = mono.detach().cpu().to(torch.float32).numpy()
    return {"array": arr, "sampling_rate": sr}


def chunk_to_asr_input(waveform: torch.Tensor, sr: int) -> Dict[str, object]:
    """Prepare ASR input dict and resample to ASR_TARGET_SR for model compatibility."""
    return _asr_input_resampled(waveform, sr, target_sr=ASR_TARGET_SR)

def build_dialog_lines(items: List[Dict[str, str]]) -> str:
    """
    Merge consecutive entries with the same tag and format as dialog lines:
      A: ...
      B: ...
    """
    merged: List[Dict[str, str]] = []

    for it in items:
        tag = it.get("tag", "")
        text = (it.get("text", "") or "").strip()
        if not text:
            continue

        if merged and merged[-1]["tag"] == tag:
            merged[-1]["text"] = (merged[-1]["text"] + " " + text).strip()
        else:
            merged.append({"tag": tag, "text": text})

    return "\n".join(f"{m['tag']}: {m['text']}" for m in merged) if merged else ""


# -----------------------------------------------------------------------------
# Cached model loaders
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_diarization_pipeline() -> Pipeline:
    """Load and cache pyannote diarization pipeline."""
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

@lru_cache(maxsize=1)
def get_asr_pipeline():
    """Load and cache HuggingFace ASR pipeline."""
    return hf_pipeline("automatic-speech-recognition", model="google/medasr")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    """
    Simple dictation ASR. Expects WAV bytes.
    Returns the transcribed text.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "dictation.wav")
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)

        waveform, sr = torchaudio.load(audio_path)

        asr_pipe = get_asr_pipeline()
        asr_inp = _asr_input_resampled(waveform, sr, target_sr=ASR_TARGET_SR)

        res = asr_pipe(asr_inp, chunk_length_s=DICTATION_CHUNK_S, stride_length_s=DICTATION_STRIDE_S)
        raw = (res.get("text", "") or "")
        return clean_asr_text(raw)

def diarize_and_transcribe(wav_bytes: bytes) -> str:
    """
    Run speaker diarization and then ASR per speaker segment.
    Returns dialog-style text:
      A: ...
      B: ...
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "input.wav")
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        _, num_samples = waveform.shape

        # Diarization
        diar_pipe = get_diarization_pipeline()
        out = diar_pipe({"waveform": waveform, "sample_rate": sr}, num_speakers=DEFAULT_NUM_SPEAKERS)
        diarization = out.speaker_diarization

        # Write RTTM to temp file
        rttm_path = os.path.join(tmpdir, "audio.rttm")
        with open(rttm_path, "w", encoding="utf-8") as rttm:
            diarization.write_rttm(rttm)

        segments = read_rttm_to_segments(rttm_path)
        segments = merge_adjacent_segments(segments, max_gap=MERGE_MAX_GAP_S)

        # ASR per segment
        asr_pipe = get_asr_pipeline()

        label2tag: Dict[str, str] = {}
        next_tag = "A"  # kept as-is to avoid changing behavior

        dialog_items: List[Dict[str, str]] = []

        for seg in segments:
            dur = seg.end - seg.start
            if dur < MIN_SEGMENT_DUR_S:
                continue

            if seg.speaker not in label2tag:
                label2tag[seg.speaker] = next_tag
                # kept as-is to avoid changing behavior
                next_tag = "B" if next_tag == "A" else next_tag

            start_sample = clamp(int(seg.start * sr), 0, num_samples)
            end_sample = clamp(int(seg.end * sr), 0, num_samples)
            if end_sample <= start_sample:
                continue

            chunk = waveform[:, start_sample:end_sample]
            asr_inp = chunk_to_asr_input(chunk, sr)

            try:
                res = asr_pipe(asr_inp, chunk_length_s=SEGMENT_CHUNK_S, stride_length_s=SEGMENT_STRIDE_S)
                raw = (res.get("text", "") or "")
                text = clean_asr_text(raw)
            except Exception as e:
                text = f"[ASR ERROR] {e}"

            dialog_items.append({"tag": label2tag[seg.speaker], "text": text})

        return build_dialog_lines(dialog_items)
