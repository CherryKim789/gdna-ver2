#!/usr/bin/env python3
"""
Graceful-Degradation DNA Data Analysis v17
==========================================

Architecture
------------
    H_critical(144 bits, recoverable)
    || H_extended(160 bits, nonblocking)
    || C_canonical(primary graceful content)

v17 keeps the v14 architecture as the foundation and extends it to three media families:
    image -> primary canonical pixels -> fresh PNG
    text  -> canonical UTF-8 bytes -> fresh UTF-8 text
    audio -> canonical PCM16 -> fresh WAV

Core goals
----------
1. Keep Seo R∞-P8 local mapping (4 bits -> 2 nt, 8 periodic mapping columns).
2. Keep v14 critical-header recovery: known MAGIC/VERSION repair + CRC32
   syndrome + meet-in-the-middle + semantic filtering, then labeled best-effort
   structural guessing if strict recovery is AMBIGUOUS/FAIL.
3. Keep 160-bit extended metadata nonblocking. DAMAGED extended metadata never
   blocks reconstruction.
4. Preserve corruption inside reconstructed content whenever framing is still
   structurally usable.
5. Output should remain openable whenever the canonical representation can be
   interpreted structurally.

Important scope
---------------
* Substitution-only channel. Insertions/deletions/truncation are not synchronized.
* Image cap: 256 x 256, aspect-ratio preserving.
* Audio canonical form: PCM16, 22.05 kHz, stereo, first 20 s maximum.
* Text canonical form: UTF-8. Non-UTF-8 text is converted deterministically to
  UTF-8 for openable reconstruction.
* Clean recovery is exact relative to the canonical representation, not
  byte-identical to the uploaded compressed container.
* CRC32 is metadata redundancy. No BCH/RS/repetition ECC is added.
* Decode UI accepts Source bits / Source DNA as well as full bitstream / full DNA artifacts.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import mimetypes
import random
import subprocess
import tempfile
import wave
import zipfile
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# =============================================================================
# Protocol constants
# =============================================================================

APP_VERSION = "17"

CRITICAL_MAGIC = b"GD17"
CRITICAL_VERSION = 1
CRITICAL_CORE_BYTES = 14
CRITICAL_BYTES = 18
CRITICAL_BITS = 144
CRITICAL_DNA_NT = 72

EXT_VERSION = 1
EXT_CORE_BYTES = 16
EXT_BYTES = 20
EXT_BITS = 160
EXT_DNA_NT = 80

FIXED_META_BITS = CRITICAL_BITS + EXT_BITS
FIXED_META_DNA_NT = CRITICAL_DNA_NT + EXT_DNA_NT

# Critical PROFILE values.
PROFILE_BINARY1 = 1
PROFILE_L8 = 2
PROFILE_RGB8 = 3
PROFILE_RGBA8 = 4
PROFILE_TEXT_UTF8 = 10
PROFILE_AUDIO_PCM16 = 20

PROFILE_NAMES = {
    PROFILE_BINARY1: "Image Binary1",
    PROFILE_L8: "Image L8 grayscale",
    PROFILE_RGB8: "Image RGB8",
    PROFILE_RGBA8: "Image RGBA8",
    PROFILE_TEXT_UTF8: "Text UTF-8",
    PROFILE_AUDIO_PCM16: "Audio PCM16",
}

IMAGE_PROFILES = {PROFILE_BINARY1, PROFILE_L8, PROFILE_RGB8, PROFILE_RGBA8}
BITS_PER_PIXEL = {
    PROFILE_BINARY1: 1,
    PROFILE_L8: 8,
    PROFILE_RGB8: 24,
    PROFILE_RGBA8: 32,
}
PROFILE_MEDIA_KIND = {
    PROFILE_BINARY1: "image",
    PROFILE_L8: "image",
    PROFILE_RGB8: "image",
    PROFILE_RGBA8: "image",
    PROFILE_TEXT_UTF8: "text",
    PROFILE_AUDIO_PCM16: "audio",
}

DEMO_MAX_WIDTH = 256
DEMO_MAX_HEIGHT = 256

AUDIO_RATE = 22050
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_WIDTH = 2
AUDIO_MAX_SECONDS = 20

MAX_CRITICAL_UNKNOWN_SEARCH_BITS = 6
MAX_MITM_PATTERNS = 50000

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
TEXT_EXTENSIONS = {
    ".txt", ".csv", ".tsv", ".json", ".xml", ".md", ".py", ".html", ".htm",
    ".css", ".js", ".yaml", ".yml", ".log",
}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus", ".wma"}
ALL_UPLOAD_EXTENSIONS = sorted({x.lstrip(".") for x in IMAGE_EXTENSIONS | TEXT_EXTENSIONS | AUDIO_EXTENSIONS})

MEDIA_KIND_CODES = {"unknown": 0, "image": 1, "text": 2, "audio": 3}
MEDIA_KIND_NAMES = {v: k for k, v in MEDIA_KIND_CODES.items()}

FORMAT_CODES = {
    "UNKNOWN": 0,
    "PNG": 1, "JPEG": 2, "WEBP": 3, "BMP": 4, "TIFF": 5, "GIF": 6,
    "TXT": 20, "CSV": 21, "TSV": 22, "JSON": 23, "XML": 24, "MD": 25,
    "PY": 26, "HTML": 27, "HTM": 28, "CSS": 29, "JS": 30, "YAML": 31,
    "YML": 32, "LOG": 33,
    "WAV": 40, "MP3": 41, "AAC": 42, "M4A": 43, "FLAC": 44, "OGG": 45,
    "OPUS": 46, "WMA": 47,
}
FORMAT_NAMES = {v: k for k, v in FORMAT_CODES.items()}

IMAGE_MODE_CODES = {
    "UNKNOWN": 0, "1": 1, "L": 2, "LA": 3, "P": 4, "RGB": 5, "RGBA": 6,
    "CMYK": 7, "I": 8, "F": 9, "I;16": 10,
}
IMAGE_MODE_NAMES = {v: k for k, v in IMAGE_MODE_CODES.items()}

ENCODING_CODES = {"UNKNOWN": 0, "UTF-8": 1, "UTF-8-SIG": 2, "LATIN-1->UTF-8": 3}
ENCODING_NAMES = {v: k for k, v in ENCODING_CODES.items()}

COLORSPACE_UNKNOWN = 0
COLORSPACE_SRGB = 1
COLORSPACE_ICC_TO_SRGB = 2
COLORSPACE_GRAYSCALE = 3
COLORSPACE_BINARY = 4
COLORSPACE_NAMES = {
    COLORSPACE_UNKNOWN: "unknown/native",
    COLORSPACE_SRGB: "sRGB / assumed sRGB",
    COLORSPACE_ICC_TO_SRGB: "ICC-normalized to sRGB",
    COLORSPACE_GRAYSCALE: "grayscale intensity",
    COLORSPACE_BINARY: "binary black/white",
}

RESAMPLE_NONE = 0
RESAMPLE_LANCZOS = 1
RESAMPLE_NEAREST = 2
RESAMPLE_NAMES = {RESAMPLE_NONE: "none", RESAMPLE_LANCZOS: "LANCZOS", RESAMPLE_NEAREST: "NEAREST"}

# Generic extended flags.
FLAG_RESIZED = 1 << 0
FLAG_ORIENTATION_APPLIED = 1 << 1
FLAG_INPUT_ALPHA = 1 << 2
FLAG_USEFUL_ALPHA = 1 << 3
FLAG_THRESHOLD_APPLIED = 1 << 4
FLAG_BINARY_EXACT = 1 << 5
FLAG_ICC_PRESENT = 1 << 6
FLAG_ICC_TO_SRGB = 1 << 7
FLAG_TRUNCATED = 1 << 8
FLAG_TRANSCODED = 1 << 9
FLAG_ANIMATED_INPUT = 1 << 11
FLAG_GRAYSCALE_CONTENT = 1 << 12
FLAG_TEXT_REENCODED = 1 << 13
FLAG_PALETTE_INPUT = 1 << 14
FLAG_GAMMA_PRESENT = 1 << 15

FLAG_LABELS = [
    (FLAG_RESIZED, "resized"),
    (FLAG_ORIENTATION_APPLIED, "orientation_applied"),
    (FLAG_INPUT_ALPHA, "input_alpha_or_transparency"),
    (FLAG_USEFUL_ALPHA, "useful_alpha"),
    (FLAG_THRESHOLD_APPLIED, "threshold_applied"),
    (FLAG_BINARY_EXACT, "exact_binary_content"),
    (FLAG_ICC_PRESENT, "icc_present"),
    (FLAG_ICC_TO_SRGB, "icc_converted_to_srgb"),
    (FLAG_TRUNCATED, "duration_truncated"),
    (FLAG_TRANSCODED, "canonical_transcode"),
    (FLAG_ANIMATED_INPUT, "animated_input"),
    (FLAG_GRAYSCALE_CONTENT, "grayscale_content"),
    (FLAG_TEXT_REENCODED, "text_reencoded_to_utf8"),
    (FLAG_PALETTE_INPUT, "palette_input"),
    (FLAG_GAMMA_PRESENT, "gamma_present"),
]


# =============================================================================
# Seo R∞-P8 local mapping
# =============================================================================

SEO_RINF_P8 = {
    "0000": ["CA", "AT", "AG", "AC", "AA", "TT", "TG", "TC"],
    "0001": ["CC", "CA", "AT", "AG", "AC", "AA", "TT", "TG"],
    "0010": ["CG", "CC", "CA", "AT", "AG", "AC", "AA", "TT"],
    "0011": ["CT", "CG", "CC", "CA", "AT", "AG", "AC", "AA"],
    "0100": ["GA", "CT", "CG", "CC", "CA", "AT", "AG", "AC"],
    "0101": ["GC", "GA", "CT", "CG", "CC", "CA", "AT", "AG"],
    "0110": ["GG", "GC", "GA", "CT", "CG", "CC", "CA", "AT"],
    "0111": ["GT", "GG", "GC", "GA", "CT", "CG", "CC", "CA"],
    "1000": ["TA", "GT", "GG", "GC", "GA", "CT", "CG", "CC"],
    "1001": ["TC", "TA", "GT", "GG", "GC", "GA", "CT", "CG"],
    "1010": ["TG", "TC", "TA", "GT", "GG", "GC", "GA", "CT"],
    "1011": ["TT", "TG", "TC", "TA", "GT", "GG", "GC", "GA"],
    "1100": ["AA", "TT", "TG", "TC", "TA", "GT", "GG", "GC"],
    "1101": ["AC", "AA", "TT", "TG", "TC", "TA", "GT", "GG"],
    "1110": ["AG", "AC", "AA", "TT", "TG", "TC", "TA", "GT"],
    "1111": ["AT", "AG", "AC", "AA", "TT", "TG", "TC", "TA"],
}
SEO_INV: List[Dict[str, str]] = []
for _col in range(8):
    _inv = {SEO_RINF_P8[b][_col]: b for b in SEO_RINF_P8}
    if len(_inv) != 16:
        raise RuntimeError(f"R∞-P8 table is not bijective in column {_col}.")
    SEO_INV.append(_inv)


def seo_encode_bits(bits: str) -> Tuple[str, int]:
    bits = "".join(bits.split())
    if any(c not in "01" for c in bits):
        raise ValueError("Input must contain only 0/1.")
    pad = (-len(bits)) % 4
    padded = bits + "0" * pad
    out: List[str] = []
    for i in range(0, len(padded), 4):
        unit = i // 4
        out.append(SEO_RINF_P8[padded[i:i + 4]][unit % 8])
    return "".join(out), pad


def seo_decode_dna(dna: str, nbits: Optional[int] = None) -> str:
    dna = "".join(dna.split()).upper()
    if len(dna) % 2:
        raise ValueError("R∞-P8 DNA length must be even.")
    if any(c not in "ACGT" for c in dna):
        raise ValueError("DNA contains non-ACGT characters.")
    out: List[str] = []
    for i in range(0, len(dna), 2):
        unit = i // 2
        out.append(SEO_INV[unit % 8][dna[i:i + 2]])
    bits = "".join(out)
    return bits if nbits is None else bits[:nbits]


# =============================================================================
# Generic utilities
# =============================================================================

_BYTE_BITS = tuple(f"{x:08b}" for x in range(256))


def bytes_to_bits(data: bytes) -> str:
    return "".join(_BYTE_BITS[x] for x in data)


def bits_to_bytes(bits: str) -> bytes:
    bits = "".join(bits.split())
    if len(bits) % 8:
        raise ValueError("Bit length must be divisible by 8.")
    if any(c not in "01" for c in bits):
        raise ValueError("Bit string contains non-binary characters.")
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_bit_text(data: bytes | str) -> str:
    text = data.decode("utf-8", errors="strict") if isinstance(data, bytes) else data
    out = "".join(text.split())
    if not out or any(c not in "01" for c in out):
        raise ValueError("Bitstream must contain only 0 and 1; whitespace is ignored.")
    return out


def normalize_dna_text(data: bytes | str) -> str:
    text = data.decode("utf-8", errors="strict") if isinstance(data, bytes) else data
    out = "".join(text.split()).upper()
    if not out or any(c not in "ACGT" for c in out):
        raise ValueError("DNA must contain only A/C/G/T; whitespace is ignored.")
    return out


def bit_at(data: bytes, bit_index: int) -> int:
    return (data[bit_index // 8] >> (7 - bit_index % 8)) & 1


def flip_bit(data: bytes, bit_index: int) -> bytes:
    if not 0 <= bit_index < len(data) * 8:
        raise IndexError(bit_index)
    out = bytearray(data)
    out[bit_index // 8] ^= 1 << (7 - bit_index % 8)
    return bytes(out)


def flip_many_bytes(data: bytes, positions: Sequence[int]) -> bytes:
    out = bytearray(data)
    for p in positions:
        if not 0 <= p < len(out) * 8:
            raise IndexError(p)
        out[p // 8] ^= 1 << (7 - p % 8)
    return bytes(out)


def flip_bit_string(bits: str, positions: Sequence[int]) -> str:
    chars = list(bits)
    for p in positions:
        if not 0 <= p < len(chars):
            raise IndexError(p)
        chars[p] = "1" if chars[p] == "0" else "0"
    return "".join(chars)


def random_bit_flips(bits: str, n_errors: int, seed: int) -> Tuple[str, List[int]]:
    if not 0 <= n_errors <= len(bits):
        raise ValueError("Invalid number of bit flips.")
    if n_errors == 0:
        return bits, []
    rng = random.Random(seed)
    positions = sorted(rng.sample(range(len(bits)), n_errors))
    return flip_bit_string(bits, positions), positions


def random_dna_substitutions(dna: str, n_errors: int, seed: int) -> Tuple[str, List[Tuple[int, str, str]]]:
    if not 0 <= n_errors <= len(dna):
        raise ValueError("Invalid number of DNA substitutions.")
    if n_errors == 0:
        return dna, []
    rng = random.Random(seed)
    positions = sorted(rng.sample(range(len(dna)), n_errors))
    chars = list(dna)
    rows: List[Tuple[int, str, str]] = []
    for p in positions:
        old = chars[p]
        new = rng.choice([b for b in "ACGT" if b != old])
        chars[p] = new
        rows.append((p, old, new))
    return "".join(chars), rows


def diff_positions(a: str, b: str) -> List[int]:
    if len(a) != len(b):
        raise ValueError("Length mismatch while computing differences.")
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


def short_preview(text: str, n: int = 256) -> str:
    return text if len(text) <= n else text[:n] + f" ... [{len(text)-n:,} more characters]"


def flag_names(flags: int) -> List[str]:
    return [label for bit, label in FLAG_LABELS if flags & bit]


def format_code_from_name(name: str, pil_format: Optional[str] = None) -> int:
    if pil_format:
        key = pil_format.upper()
        if key == "JPG":
            key = "JPEG"
        if key in FORMAT_CODES:
            return FORMAT_CODES[key]
    ext = Path(name).suffix.lower().lstrip(".").upper()
    if ext == "JPG":
        ext = "JPEG"
    return FORMAT_CODES.get(ext, FORMAT_CODES["UNKNOWN"])


def media_kind_from_name(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


# =============================================================================
# External media helpers
# =============================================================================


def ffmpeg_available() -> bool:
    try:
        p = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return p.returncode == 0
    except Exception:
        return False


def ffprobe_available() -> bool:
    try:
        p = subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return p.returncode == 0
    except Exception:
        return False


def run_ffmpeg_input_bytes(data: bytes, suffix: str, args_after_input: List[str], timeout: int = 180) -> bytes:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required for audio canonicalization but is not available.")
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / f"input{suffix or '.bin'}"
        inp.write_bytes(data)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(inp)] + list(args_after_input)
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        if p.returncode != 0:
            msg = p.stderr.decode("utf-8", errors="replace")[-2500:]
            raise RuntimeError(f"ffmpeg failed: {msg}")
        return p.stdout


def ffprobe_json(data: bytes, suffix: str) -> Dict[str, object]:
    if not ffprobe_available():
        return {}
    try:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / f"input{suffix or '.bin'}"
            inp.write_bytes(data)
            p = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(inp)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False,
            )
            if p.returncode != 0:
                return {}
            return json.loads(p.stdout.decode("utf-8", errors="replace") or "{}")
    except Exception:
        return {}


def probe_duration(probe: Dict[str, object]) -> Optional[float]:
    try:
        fmt = probe.get("format", {}) if isinstance(probe, dict) else {}
        if isinstance(fmt, dict) and fmt.get("duration") is not None:
            return float(fmt["duration"])
    except Exception:
        pass
    for s in probe.get("streams", []) if isinstance(probe, dict) else []:
        try:
            if s.get("duration") is not None:
                return float(s["duration"])
        except Exception:
            pass
    return None


def parse_fraction(value: object) -> Optional[float]:
    try:
        text = str(value)
        if "/" in text:
            a, b = text.split("/", 1)
            bval = float(b)
            return float(a) / bval if bval else None
        return float(text)
    except Exception:
        return None


def wav_bytes_from_pcm(raw: bytes, rate: int, channels: int, sample_width: int) -> bytes:
    if channels < 1 or sample_width < 1:
        raise ValueError("Invalid WAV parameters.")
    frame_bytes = channels * sample_width
    if len(raw) % frame_bytes:
        raise ValueError("PCM source is not frame-aligned.")
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(rate)
        w.writeframes(raw)
    return bio.getvalue()




# =============================================================================
# Critical 144-bit header
# =============================================================================
# MAGIC32 | VERSION8 | CONTENT_BITS40 | PROFILE8 | AUX24 | CRC32
# AUX semantics:
#   image: WIDTH_MINUS1[8] || HEIGHT_MINUS1[8] || RESERVED[8]
#   text : 0
#   audio: SAMPLE_RATE[18] || CHANNELS_MINUS1[3] || SAMPLE_WIDTH_MINUS1[3]


def pack_image_aux(width: int, height: int, reserved: int = 0) -> int:
    if not (1 <= width <= 256 and 1 <= height <= 256):
        raise ValueError("Canonical image dimensions must be in 1..256.")
    return ((width - 1) << 16) | ((height - 1) << 8) | (reserved & 0xFF)


def unpack_image_aux(aux: int) -> Tuple[int, int, int]:
    return ((aux >> 16) & 0xFF) + 1, ((aux >> 8) & 0xFF) + 1, aux & 0xFF


def pack_audio_aux(rate: int, channels: int, sample_width: int) -> int:
    if not 1 <= rate <= 262143:
        raise ValueError("Audio sample rate does not fit 18 bits.")
    if not 1 <= channels <= 8:
        raise ValueError("Audio channels must be 1..8.")
    if not 1 <= sample_width <= 8:
        raise ValueError("Audio sample width must be 1..8 bytes.")
    return (rate << 6) | ((channels - 1) << 3) | (sample_width - 1)


def unpack_audio_aux(aux: int) -> Tuple[int, int, int]:
    return (aux >> 6) & 0x3FFFF, ((aux >> 3) & 0x7) + 1, (aux & 0x7) + 1




def build_critical_header(content_bits: int, profile: int, aux: int) -> bytes:
    if not 0 <= content_bits < (1 << 40):
        raise ValueError("CONTENT_BITS does not fit 40 bits.")
    if profile not in PROFILE_NAMES:
        raise ValueError("Unknown canonical profile.")
    if not 0 <= aux < (1 << 24):
        raise ValueError("AUX does not fit 24 bits.")
    core = (
        CRITICAL_MAGIC
        + bytes([CRITICAL_VERSION])
        + int(content_bits).to_bytes(5, "big")
        + bytes([profile])
        + int(aux).to_bytes(3, "big")
    )
    if len(core) != CRITICAL_CORE_BYTES:
        raise AssertionError("Unexpected critical core length.")
    crc = zlib.crc32(core) & 0xFFFFFFFF
    return core + crc.to_bytes(4, "big")


def parse_critical_header(header: bytes) -> Dict[str, object]:
    if len(header) != CRITICAL_BYTES:
        raise ValueError("Critical header must be exactly 18 bytes / 144 bits.")
    core = header[:CRITICAL_CORE_BYTES]
    stored_crc = int.from_bytes(header[14:18], "big")
    computed_crc = zlib.crc32(core) & 0xFFFFFFFF
    profile = header[10]
    aux = int.from_bytes(header[11:14], "big")
    result: Dict[str, object] = {
        "magic": header[:4].decode("ascii", errors="replace"),
        "version": header[4],
        "content_bits": int.from_bytes(header[5:10], "big"),
        "profile": profile,
        "profile_name": PROFILE_NAMES.get(profile, "INVALID"),
        "media_kind": PROFILE_MEDIA_KIND.get(profile, "unknown"),
        "aux": aux,
        "stored_crc32": f"{stored_crc:08x}",
        "computed_crc32": f"{computed_crc:08x}",
        "crc_ok": stored_crc == computed_crc,
        "magic_ok": header[:4] == CRITICAL_MAGIC,
        "version_ok": header[4] == CRITICAL_VERSION,
    }
    if profile in IMAGE_PROFILES:
        w, h, reserved = unpack_image_aux(aux)
        result.update({"width": w, "height": h, "reserved": reserved, "aux_interpretation": f"{w}×{h}, reserved={reserved}"})
    elif profile == PROFILE_TEXT_UTF8:
        result.update({"aux_interpretation": "reserved=0"})
    elif profile == PROFILE_AUDIO_PCM16:
        rate, ch, sw = unpack_audio_aux(aux)
        result.update({"sample_rate": rate, "channels": ch, "sample_width": sw,
                       "aux_interpretation": f"{rate} Hz, {ch} ch, {sw * 8}-bit"})
    else:
        result["aux_interpretation"] = "unknown"
    return result


def critical_syndrome(header: bytes) -> int:
    if len(header) != CRITICAL_BYTES:
        raise ValueError("Critical header must be 18 bytes.")
    return (zlib.crc32(header[:14]) & 0xFFFFFFFF) ^ int.from_bytes(header[14:18], "big")


def expected_profile_content_bits(profile: int, aux: int) -> Optional[int]:
    if profile in IMAGE_PROFILES:
        w, h, reserved = unpack_image_aux(aux)
        if reserved != 0:
            return None
        return w * h * BITS_PER_PIXEL[profile]
    if profile == PROFILE_TEXT_UTF8:
        return None  # variable; validated by byte alignment + observed length
    if profile == PROFILE_AUDIO_PCM16:
        rate, ch, sw = unpack_audio_aux(aux)
        if (rate, ch, sw) != (AUDIO_RATE, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH):
            return None
        return None
    return None


def critical_semantic_valid(
    header: bytes,
    *,
    observed_source_bits: Optional[int] = None,
    observed_source_nt: Optional[int] = None,
) -> bool:
    if len(header) != CRITICAL_BYTES:
        return False
    if header[:4] != CRITICAL_MAGIC or header[4] != CRITICAL_VERSION:
        return False
    if critical_syndrome(header) != 0:
        return False
    content_bits = int.from_bytes(header[5:10], "big")
    profile = header[10]
    aux = int.from_bytes(header[11:14], "big")
    if profile not in PROFILE_NAMES:
        return False

    if profile in IMAGE_PROFILES:
        w, h, reserved = unpack_image_aux(aux)
        if reserved != 0 or not (1 <= w <= 256 and 1 <= h <= 256):
            return False
        if content_bits != w * h * BITS_PER_PIXEL[profile]:
            return False
    elif profile == PROFILE_TEXT_UTF8:
        if aux != 0 or content_bits % 8 != 0:
            return False
    elif profile == PROFILE_AUDIO_PCM16:
        rate, ch, sw = unpack_audio_aux(aux)
        if (rate, ch, sw) != (AUDIO_RATE, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH):
            return False
        if content_bits % (8 * ch * sw) != 0:
            return False
    else:
        return False

    if observed_source_bits is not None and content_bits != int(observed_source_bits):
        return False
    if observed_source_nt is not None:
        if 2 * math.ceil(content_bits / 4) != int(observed_source_nt):
            return False
    return True


# =============================================================================
# Critical metadata recovery: v11/v14 strategy
# =============================================================================

@dataclass
class CriticalRecovery:
    status: str  # UNIQUE / AMBIGUOUS / FAIL / *_GUESSED
    received_header: bytes
    recovered_header: Optional[bytes]
    known_error_positions: List[int]
    searched_error_positions: List[int]
    corrected_positions: List[int]
    candidate_count: int
    distance: Optional[int]
    message: str


def repair_critical_known_fields(received_header: bytes) -> Tuple[bytes, List[int]]:
    if len(received_header) != CRITICAL_BYTES:
        raise ValueError("Critical header must be exactly 18 bytes.")
    expected = CRITICAL_MAGIC + bytes([CRITICAL_VERSION])
    wrong = [i for i in range(40) if bit_at(received_header, i) != bit_at(expected, i)]
    patched = bytearray(received_header)
    patched[:5] = expected
    return bytes(patched), wrong


def critical_signature_table(header: bytes, positions: Sequence[int]) -> Dict[int, int]:
    base = critical_syndrome(header)
    return {p: critical_syndrome(flip_bit(header, p)) ^ base for p in positions}


def combo_xor(signatures: Dict[int, int], comb: Sequence[int]) -> int:
    x = 0
    for p in comb:
        x ^= signatures[p]
    return x


def find_syndrome_patterns_mitm(
    signatures: Dict[int, int], positions: Sequence[int], target: int, weight: int,
    max_patterns: int = MAX_MITM_PATTERNS,
) -> List[Tuple[int, ...]]:
    positions = tuple(positions)
    if weight == 0:
        return [tuple()] if target == 0 else []
    left_w = weight // 2
    right_w = weight - left_w
    left: Dict[int, List[Tuple[int, ...]]] = defaultdict(list)
    for comb in itertools.combinations(positions, left_w):
        left[combo_xor(signatures, comb)].append(comb)
    found = set()
    capped = False
    for right in itertools.combinations(positions, right_w):
        xr = combo_xor(signatures, right)
        need = target ^ xr
        for left_comb in left.get(need, ()):
            if set(left_comb).isdisjoint(right):
                found.add(tuple(sorted(left_comb + right)))
                if len(found) >= max_patterns:
                    capped = True
                    break
        if capped:
            break
    result = sorted(found)
    if capped:
        result.append((-1,))
    return result


def recover_critical_header(
    received_header: bytes,
    *,
    observed_source_bits: Optional[int] = None,
    observed_source_nt: Optional[int] = None,
    max_unknown_bit_errors: int = MAX_CRITICAL_UNKNOWN_SEARCH_BITS,
) -> CriticalRecovery:
    if len(received_header) != CRITICAL_BYTES:
        return CriticalRecovery("FAIL", received_header, None, [], [], [], 0, None,
                                "Received critical metadata is not exactly 144 bits.")
    patched, known_errors = repair_critical_known_fields(received_header)
    target = critical_syndrome(patched)
    positions = tuple(range(40, 144))
    signatures = critical_signature_table(patched, positions)
    for w in range(max_unknown_bit_errors + 1):
        patterns = find_syndrome_patterns_mitm(signatures, positions, target, w)
        capped = bool(patterns and patterns[-1] == (-1,))
        if capped:
            patterns = patterns[:-1]
        valid: Dict[bytes, Tuple[int, ...]] = {}
        for pat in patterns:
            cand = flip_many_bytes(patched, pat)
            if critical_semantic_valid(cand, observed_source_bits=observed_source_bits,
                                       observed_source_nt=observed_source_nt):
                valid[cand] = pat
        if capped and valid:
            return CriticalRecovery("AMBIGUOUS", received_header, None, known_errors, [], known_errors,
                                    max(2, len(valid)), len(known_errors) + w,
                                    "Nearest strict candidate enumeration reached the safety cap.")
        if len(valid) == 1:
            cand, pat = next(iter(valid.items()))
            corrected = sorted(known_errors + list(pat))
            return CriticalRecovery("UNIQUE", received_header, cand, known_errors, list(pat), corrected,
                                    1, len(corrected),
                                    f"Unique nearest semantically valid critical header found at unknown-region weight {w}.")
        if len(valid) > 1:
            return CriticalRecovery("AMBIGUOUS", received_header, None, known_errors, [], known_errors,
                                    len(valid), len(known_errors) + w,
                                    f"{len(valid)} equally-near semantically valid critical headers found.")
    return CriticalRecovery("FAIL", received_header, None, known_errors, [], known_errors, 0, None,
                            f"No semantically valid critical header found within {max_unknown_bit_errors} searched unknown-region bit flips.")


def header_hamming_distance(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError("Header lengths differ.")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def _candidate_from_inferred_bits(profile: int, nbits: int) -> List[bytes]:
    out: List[bytes] = []
    if profile in IMAGE_PROFILES:
        bpp = BITS_PER_PIXEL[profile]
        if nbits <= 0 or nbits % bpp:
            return out
        pixels = nbits // bpp
        if not 1 <= pixels <= DEMO_MAX_WIDTH * DEMO_MAX_HEIGHT:
            return out
        for w in range(1, DEMO_MAX_WIDTH + 1):
            if pixels % w:
                continue
            h = pixels // w
            if 1 <= h <= DEMO_MAX_HEIGHT:
                out.append(build_critical_header(nbits, profile, pack_image_aux(w, h, 0)))
    elif profile == PROFILE_TEXT_UTF8:
        if nbits >= 0 and nbits % 8 == 0:
            out.append(build_critical_header(nbits, profile, 0))
    elif profile == PROFILE_AUDIO_PCM16:
        if nbits >= 0 and nbits % (8 * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH) == 0:
            out.append(build_critical_header(nbits, profile, pack_audio_aux(AUDIO_RATE, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH)))
    return out


def structural_critical_candidates(
    *, observed_source_bits: Optional[int] = None, observed_source_nt: Optional[int] = None,
    preferred_media_kind: Optional[str] = None,
) -> List[bytes]:
    out: List[bytes] = []
    if observed_source_bits is not None:
        nbits = int(observed_source_bits)
        for profile in PROFILE_NAMES:
            out.extend(_candidate_from_inferred_bits(profile, nbits))
    elif observed_source_nt is not None:
        nt = int(observed_source_nt)
        if nt <= 0 or nt % 2:
            return []
        # Byte-aligned profiles have exact nbits = 2*nt.
        inferred = 2 * nt
        for profile in (PROFILE_L8, PROFILE_RGB8, PROFILE_RGBA8, PROFILE_TEXT_UTF8, PROFILE_AUDIO_PCM16):
            out.extend(_candidate_from_inferred_bits(profile, inferred))
        # Binary1 may have 0..3 right-zero pad bits. Enumerate dimensions directly.
        for w in range(1, DEMO_MAX_WIDTH + 1):
            for h in range(1, DEMO_MAX_HEIGHT + 1):
                nbits = w * h
                if 2 * math.ceil(nbits / 4) == nt:
                    out.append(build_critical_header(nbits, PROFILE_BINARY1, pack_image_aux(w, h, 0)))
    out = list(dict.fromkeys(out))
    if preferred_media_kind in MEDIA_KIND_CODES and preferred_media_kind != "unknown":
        preferred = [h for h in out if PROFILE_MEDIA_KIND.get(h[10]) == preferred_media_kind]
        if preferred:
            return preferred
    return out


def best_effort_guess_critical_header(
    received_header: bytes, prior: CriticalRecovery, *, observed_source_bits: Optional[int] = None,
    observed_source_nt: Optional[int] = None, preferred_media_kind: Optional[str] = None,
) -> CriticalRecovery:
    candidates = structural_critical_candidates(
        observed_source_bits=observed_source_bits, observed_source_nt=observed_source_nt,
        preferred_media_kind=preferred_media_kind,
    )
    if not candidates:
        return CriticalRecovery(f"{prior.status}_NO_GUESS", received_header, None,
                                prior.known_error_positions, prior.searched_error_positions,
                                prior.corrected_positions, 0, prior.distance,
                                prior.message + " No structurally compatible fallback header exists.")
    ranked = []
    for cand in candidates:
        meta = parse_critical_header(cand)
        dist = header_hamming_distance(received_header, cand)
        ranked.append((dist, int(meta["profile"]), int(meta["aux"]), cand))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    best_dist = ranked[0][0]
    tied = [r for r in ranked if r[0] == best_dist]
    chosen = tied[0][3]
    corrected = [i for i in range(CRITICAL_BITS) if bit_at(received_header, i) != bit_at(chosen, i)]
    tie_note = f" {len(tied)} equal-distance candidates existed; deterministic tie-break was used." if len(tied) > 1 else ""
    return CriticalRecovery(
        f"{prior.status}_GUESSED", received_header, chosen,
        prior.known_error_positions, prior.searched_error_positions, corrected,
        len(tied), best_dist,
        prior.message + f" Best-effort structural salvage selected a header at Hamming distance {best_dist}." + tie_note
        + " Reconstruction continues with this structural salvage candidate.",
    )


# Same known witness geometry as the v11 144-bit CRC layout. The helper below
# verifies the witness under the current v17 header convention before exposing it.
CRC_DMIN8_WITNESS = (40, 41, 89, 100, 120, 125, 133, 136)


def verify_embedded_crc_witness(reference_header: bytes) -> Dict[str, object]:
    patched, _ = repair_critical_known_fields(reference_header)
    positions = tuple(range(40, 144))
    signatures = critical_signature_table(patched, positions)
    syn = 0
    for p in CRC_DMIN8_WITNESS:
        syn ^= signatures[p]
    return {
        "witness_positions": list(CRC_DMIN8_WITNESS),
        "weight": len(CRC_DMIN8_WITNESS),
        "zero_syndrome": syn == 0,
        "note": (
            "This verifies the embedded weight-8 witness only. It does not by itself re-run the exhaustive "
            "weight-1..7 minimum-distance proof."
        ),
    }


def any_zero_syndrome_pattern(
    signatures: Dict[int, int],
    positions: Sequence[int],
    weight: int,
) -> Optional[Tuple[int, ...]]:
    """Exact MITM search for any nonzero pattern of a given weight with zero CRC-syndrome contribution."""
    if weight == 0:
        return tuple()
    left_w = weight // 2
    right_w = weight - left_w
    table: Dict[int, List[Tuple[int, ...]]] = defaultdict(list)
    for comb in itertools.combinations(positions, left_w):
        table[combo_xor(signatures, comb)].append(comb)
    for right in itertools.combinations(positions, right_w):
        xr = combo_xor(signatures, right)
        for left in table.get(xr, ()):
            if set(left).isdisjoint(right):
                return tuple(sorted(left + right))
    return None


def exact_critical_crc_distance_certificate(reference_header: bytes) -> Dict[str, object]:
    """
    Re-run the v11-style exact distance certificate for the v17 144-bit critical block.

    This checks weights 1..7 exhaustively with MITM and then validates the embedded
    weight-8 zero-syndrome witness. It can be computationally expensive and is not
    executed automatically by the Streamlit UI.
    """
    patched, _ = repair_critical_known_fields(reference_header)
    positions = tuple(range(40, 144))
    signatures = critical_signature_table(patched, positions)
    checked = []
    for w in range(1, 8):
        hit = any_zero_syndrome_pattern(signatures, positions, w)
        checked.append({"weight": w, "zero_syndrome_found": hit is not None})
        if hit is not None:
            return {
                "exact": True,
                "minimum_distance": w,
                "guaranteed_unknown_region_bit_errors": (w - 1) // 2,
                "first_witness": list(hit),
                "checked": checked,
                "interpretation": "A zero-syndrome pattern exists below weight 8 for this convention.",
            }

    witness_syn = 0
    for p in CRC_DMIN8_WITNESS:
        witness_syn ^= signatures[p]
    if witness_syn != 0:
        return {
            "exact": False,
            "minimum_distance": None,
            "guaranteed_unknown_region_bit_errors": None,
            "first_witness": None,
            "checked": checked,
            "interpretation": "Embedded weight-8 witness failed; CRC convention/layout differs from the expected geometry.",
        }
    return {
        "exact": True,
        "minimum_distance": 8,
        "guaranteed_unknown_region_bit_errors": 3,
        "first_witness": list(CRC_DMIN8_WITNESS),
        "checked": checked,
        "interpretation": (
            "No zero-syndrome pattern exists at weights 1..7 and a weight-8 witness exists. "
            "Thus d_min=8 for the unknown 104-bit region and the universal nearest-codeword "
            "guarantee is 3 unknown-region bit errors before semantic filtering. Recovery at "
            "higher weights can occur empirically but is not an unconditional guarantee."
        ),
    }



# =============================================================================
# Extended 160-bit nonblocking metadata
# =============================================================================
# Generic layout (20 bytes):
# 0       EXT_VERSION
# 1       MEDIA_KIND
# 2..3    FLAGS
# 4       PARAM0
# 5       PARAM1
# 6..7    ORIGINAL_A
# 8..9    ORIGINAL_B
# 10      FORMAT_CODE
# 11      MODE_OR_CODEC_CODE
# 12..13  PARAM2_Q16
# 14..15  PARAM3_Q16
# 16..19  CRC32 over bytes 0..15

@dataclass
class ExtendedFields:
    media_kind: str
    flags: int = 0
    param0: int = 0
    param1: int = 0
    original_a: int = 0
    original_b: int = 0
    format_code: int = 0
    mode_codec_code: int = 0
    param2_q16: int = 0
    param3_q16: int = 0


def build_extended_header(fields: ExtendedFields) -> bytes:
    core = (
        bytes([EXT_VERSION])
        + bytes([MEDIA_KIND_CODES.get(fields.media_kind, 0) & 0xFF])
        + int(fields.flags & 0xFFFF).to_bytes(2, "big")
        + bytes([fields.param0 & 0xFF])
        + bytes([fields.param1 & 0xFF])
        + int(max(0, min(65535, fields.original_a))).to_bytes(2, "big")
        + int(max(0, min(65535, fields.original_b))).to_bytes(2, "big")
        + bytes([fields.format_code & 0xFF])
        + bytes([fields.mode_codec_code & 0xFF])
        + int(max(0, min(65535, fields.param2_q16))).to_bytes(2, "big")
        + int(max(0, min(65535, fields.param3_q16))).to_bytes(2, "big")
    )
    if len(core) != EXT_CORE_BYTES:
        raise AssertionError(f"Unexpected extended core length: {len(core)}")
    crc = zlib.crc32(core) & 0xFFFFFFFF
    return core + crc.to_bytes(4, "big")


def parse_extended_header(header: bytes) -> Dict[str, object]:
    if len(header) != EXT_BYTES:
        raise ValueError("Extended metadata must be exactly 20 bytes / 160 bits.")
    core = header[:16]
    stored = int.from_bytes(header[16:20], "big")
    computed = zlib.crc32(core) & 0xFFFFFFFF
    media_code = header[1]
    flags = int.from_bytes(header[2:4], "big")
    result = {
        "version": header[0],
        "media_kind_code": media_code,
        "media_kind": MEDIA_KIND_NAMES.get(media_code, "unknown"),
        "flags": flags,
        "flag_names": flag_names(flags),
        "param0": header[4],
        "param1": header[5],
        "original_a": int.from_bytes(header[6:8], "big"),
        "original_b": int.from_bytes(header[8:10], "big"),
        "format_code": header[10],
        "format_name": FORMAT_NAMES.get(header[10], "UNKNOWN"),
        "mode_codec_code": header[11],
        "param2_q16": int.from_bytes(header[12:14], "big"),
        "param3_q16": int.from_bytes(header[14:16], "big"),
        "stored_crc32": f"{stored:08x}",
        "computed_crc32": f"{computed:08x}",
        "crc_ok": stored == computed,
        "version_ok": header[0] == EXT_VERSION,
    }
    result["status"] = "VALID" if result["crc_ok"] and result["version_ok"] else "DAMAGED"
    return result


# =============================================================================
# Canonical content builders
# =============================================================================

@dataclass
class CanonicalContent:
    kind: str
    profile: int
    aux: int
    bits: str
    original_name: str
    original_bytes: int
    canonical_sha256: str
    extended_fields: ExtendedFields
    canonical_description: str
    input_preview_kind: str
    input_preview_payload: object
    encoded_preview_kind: str
    encoded_preview_payload: object
    reference_bytes: bytes
    reference_name: str
    reference_mime: str
    reference_kind: str
    analysis: Dict[str, object] = field(default_factory=dict)

    @property
    def content_bits(self) -> int:
        return len(self.bits)


def image_to_png_bytes(image: Image.Image) -> bytes:
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def is_grayscale_content(im: Image.Image) -> bool:
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))


def useful_alpha(im: Image.Image) -> bool:
    a = np.asarray(im.convert("RGBA").getchannel("A"), dtype=np.uint8)
    return bool(np.any(a != 255))


def auto_threshold_otsu(gray: np.ndarray) -> int:
    arr = np.asarray(gray, dtype=np.uint8)
    unique = np.unique(arr)
    if len(unique) == 0:
        return 128
    if set(unique.tolist()).issubset({0, 255}):
        return 127
    if len(unique) == 1:
        return max(0, min(254, int(unique[0])))
    hist = np.bincount(arr.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    bins = np.arange(256, dtype=np.float64)
    sum_total = float(np.dot(hist, bins))
    weight_bg = 0.0
    sum_bg = 0.0
    best_t = 128
    best_var = -1.0
    for t in range(255):
        weight_bg += hist[t]
        if weight_bg <= 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            best_t = t
    return int(best_t)


def decode_normalize_image(data: bytes) -> Tuple[Image.Image, Dict[str, object]]:
    with Image.open(io.BytesIO(data)) as src:
        pil_format = (src.format or "UNKNOWN").upper()
        source_mode = src.mode
        original_size = src.size
        info = dict(src.info)
        animated = bool(getattr(src, "is_animated", False))
        try:
            exif = src.getexif()
            orientation = int(exif.get(274, 1)) if exif else 1
        except Exception:
            orientation = 1
        src.seek(0)
        src.load()
        base = ImageOps.exif_transpose(src.copy())
    orientation_applied = orientation not in (0, 1)
    palette_input = source_mode == "P"
    has_alpha_metadata = bool("transparency" in info or "A" in source_mode or source_mode in ("LA", "RGBA"))
    has_useful_alpha = useful_alpha(base)
    icc_blob = info.get("icc_profile")
    icc_present = bool(icc_blob)
    icc_to_srgb = False
    gamma = info.get("gamma")
    try:
        gamma = None if gamma is None else float(gamma)
    except Exception:
        gamma = None
    working = base.convert("RGBA" if has_useful_alpha else "RGB")
    if icc_present:
        try:
            from PIL import ImageCms
            in_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_blob))
            out_profile = ImageCms.createProfile("sRGB")
            if has_useful_alpha:
                alpha = working.getchannel("A")
                rgb = ImageCms.profileToProfile(working.convert("RGB"), in_profile, out_profile, outputMode="RGB")
                working = rgb.convert("RGBA"); working.putalpha(alpha)
            else:
                working = ImageCms.profileToProfile(working.convert("RGB"), in_profile, out_profile, outputMode="RGB")
            icc_to_srgb = True
        except Exception:
            pass
    grayscale = (not has_useful_alpha) and is_grayscale_content(working)
    normalized = working.convert("RGBA" if has_useful_alpha else ("L" if grayscale else "RGB"))
    return normalized, {
        "pil_format": pil_format, "source_mode": source_mode, "original_size": original_size,
        "orientation_applied": orientation_applied, "palette_input": palette_input,
        "input_alpha": has_alpha_metadata, "useful_alpha": has_useful_alpha,
        "icc_present": icc_present, "icc_to_srgb": icc_to_srgb, "gamma": gamma,
        "animated": animated, "grayscale": grayscale,
    }


def resize_image_cap(im: Image.Image, binary_before: bool) -> Tuple[Image.Image, bool, int]:
    if im.width <= DEMO_MAX_WIDTH and im.height <= DEMO_MAX_HEIGHT:
        return im, False, RESAMPLE_NONE
    scale = min(DEMO_MAX_WIDTH / im.width, DEMO_MAX_HEIGHT / im.height)
    size = (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale))))
    code = RESAMPLE_NEAREST if binary_before else RESAMPLE_LANCZOS
    method = Image.Resampling.NEAREST if code == RESAMPLE_NEAREST else Image.Resampling.LANCZOS
    return im.resize(size, method), True, code


def build_canonical_image(data: bytes, name: str) -> CanonicalContent:
    normalized, meta = decode_normalize_image(data)
    pre_gray = np.asarray(normalized.convert("L"), dtype=np.uint8)
    pre_binary = (not meta["useful_alpha"] and meta["grayscale"] and set(np.unique(pre_gray).tolist()).issubset({0, 255}))
    normalized, resized, resample = resize_image_cap(normalized, bool(pre_binary))
    gray = np.asarray(normalized.convert("L"), dtype=np.uint8)
    threshold = auto_threshold_otsu(gray)
    alpha = useful_alpha(normalized)
    grayscale = (not alpha) and is_grayscale_content(normalized)
    binary = grayscale and set(np.unique(gray).tolist()).issubset({0, 255})
    if alpha:
        arr = np.asarray(normalized.convert("RGBA"), dtype=np.uint8)
        raw = arr.tobytes(); bits = bytes_to_bits(raw); profile = PROFILE_RGBA8
        preview = Image.fromarray(arr, mode="RGBA"); mode = "RGBA8"; threshold_applied = False
        colorspace = COLORSPACE_ICC_TO_SRGB if meta["icc_to_srgb"] else COLORSPACE_SRGB
        hash_bytes = raw
    elif binary:
        b = (gray > threshold).astype(np.uint8); flat = b.reshape(-1)
        bits = "".join("1" if x else "0" for x in flat.tolist()); profile = PROFILE_BINARY1
        preview = Image.fromarray((b * 255).astype(np.uint8), mode="L"); mode = "Binary1"; threshold_applied = True
        colorspace = COLORSPACE_BINARY; hash_bytes = np.packbits(flat, bitorder="big").tobytes()
    elif grayscale:
        arr = np.asarray(normalized.convert("L"), dtype=np.uint8); raw = arr.tobytes()
        bits = bytes_to_bits(raw); profile = PROFILE_L8; preview = Image.fromarray(arr, mode="L")
        mode = "L8"; threshold_applied = False; colorspace = COLORSPACE_GRAYSCALE; hash_bytes = raw
    else:
        arr = np.asarray(normalized.convert("RGB"), dtype=np.uint8); raw = arr.tobytes()
        bits = bytes_to_bits(raw); profile = PROFILE_RGB8; preview = Image.fromarray(arr, mode="RGB")
        mode = "RGB8"; threshold_applied = False
        colorspace = COLORSPACE_ICC_TO_SRGB if meta["icc_to_srgb"] else COLORSPACE_SRGB; hash_bytes = raw
    flags = 0
    for cond, bit in [
        (resized, FLAG_RESIZED), (meta["orientation_applied"], FLAG_ORIENTATION_APPLIED),
        (meta["input_alpha"], FLAG_INPUT_ALPHA), (alpha, FLAG_USEFUL_ALPHA),
        (threshold_applied, FLAG_THRESHOLD_APPLIED), (binary, FLAG_BINARY_EXACT),
        (meta["icc_present"], FLAG_ICC_PRESENT), (meta["icc_to_srgb"], FLAG_ICC_TO_SRGB),
        (meta["animated"], FLAG_ANIMATED_INPUT), (grayscale, FLAG_GRAYSCALE_CONTENT),
        (meta["palette_input"], FLAG_PALETTE_INPUT), (meta["gamma"] is not None, FLAG_GAMMA_PRESENT),
    ]:
        if cond: flags |= bit
    ow, oh = meta["original_size"]
    gamma_q = 0 if meta["gamma"] is None else max(0, min(65535, int(round(float(meta["gamma"]) * 10000))))
    ext = ExtendedFields(
        media_kind="image", flags=flags, param0=threshold, param1=resample,
        original_a=min(65535, int(ow)), original_b=min(65535, int(oh)),
        format_code=format_code_from_name(name, str(meta["pil_format"])),
        mode_codec_code=IMAGE_MODE_CODES.get(str(meta["source_mode"]), 0),
        param2_q16=colorspace, param3_q16=gamma_q,
    )
    ref = image_to_png_bytes(preview)
    input_png = image_to_png_bytes(normalized)
    aux = pack_image_aux(preview.width, preview.height, 0)
    analysis = {
        "policy": "Content-preserving Auto", "original_dimensions": f"{ow}×{oh}",
        "canonical_dimensions": f"{preview.width}×{preview.height}", "canonical_profile": mode,
        "threshold": threshold, "threshold_applied": threshold_applied, "resized": resized,
        "resample_method": RESAMPLE_NAMES[resample], "colorspace": COLORSPACE_NAMES[colorspace],
        "unique_gray_levels": int(len(np.unique(gray))), "original_format": meta["pil_format"],
        "source_mode": meta["source_mode"],
    }
    return CanonicalContent(
        kind="image", profile=profile, aux=aux, bits=bits, original_name=name, original_bytes=len(data),
        canonical_sha256=sha256_hex(hash_bytes), extended_fields=ext,
        canonical_description=f"{mode} primary pixels, {preview.width}×{preview.height}",
        input_preview_kind="image", input_preview_payload=input_png,
        encoded_preview_kind="image", encoded_preview_payload=ref,
        reference_bytes=ref, reference_name=f"{Path(name).stem}_encoded_reference.png",
        reference_mime="image/png", reference_kind="image", analysis=analysis,
    )


def build_canonical_text(data: bytes, name: str) -> CanonicalContent:
    flags = 0
    encoding_code = ENCODING_CODES["UTF-8"]
    try:
        text = data.decode("utf-8", errors="strict")
        encoding_name = "UTF-8"
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="strict")
        encoding_name = "LATIN-1->UTF-8"
        encoding_code = ENCODING_CODES[encoding_name]
        flags |= FLAG_TEXT_REENCODED
    canonical_bytes = text.encode("utf-8")
    bits = bytes_to_bits(canonical_bytes)
    newline_code = 0
    if "\r\n" in text: newline_code = 1
    elif "\r" in text: newline_code = 2
    elif "\n" in text: newline_code = 3
    n = min(len(data), 0xFFFFFFFF)
    ext = ExtendedFields(
        media_kind="text", flags=flags, param0=newline_code, param1=0,
        original_a=(n >> 16) & 0xFFFF, original_b=n & 0xFFFF,
        format_code=format_code_from_name(name), mode_codec_code=encoding_code,
        param2_q16=0, param3_q16=0,
    )
    preview = text[:12000]
    analysis = {
        "canonical_encoding": "UTF-8", "source_decoding": encoding_name,
        "original_bytes": len(data), "canonical_bytes": len(canonical_bytes),
        "characters": len(text), "newline_code": newline_code,
    }
    return CanonicalContent(
        kind="text", profile=PROFILE_TEXT_UTF8, aux=0, bits=bits,
        original_name=name, original_bytes=len(data), canonical_sha256=sha256_hex(canonical_bytes),
        extended_fields=ext, canonical_description="UTF-8 primary text bytes",
        input_preview_kind="text", input_preview_payload=preview,
        encoded_preview_kind="text", encoded_preview_payload=preview,
        reference_bytes=canonical_bytes, reference_name=f"{Path(name).stem}_encoded_reference.txt",
        reference_mime="text/plain; charset=utf-8", reference_kind="text", analysis=analysis,
    )


def build_canonical_audio(data: bytes, name: str) -> CanonicalContent:
    suffix = Path(name).suffix.lower()
    probe = ffprobe_json(data, suffix)
    duration = probe_duration(probe)
    streams = probe.get("streams", []) if isinstance(probe, dict) else []
    astream = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"), {})
    orig_rate = int(astream.get("sample_rate") or 0) if str(astream.get("sample_rate") or "").isdigit() else 0
    orig_ch = int(astream.get("channels") or 0) if str(astream.get("channels") or "").isdigit() else 0
    args = ["-vn", "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_RATE)]
    if duration is not None and duration > AUDIO_MAX_SECONDS:
        args += ["-t", str(AUDIO_MAX_SECONDS)]
    args += ["-f", "s16le", "pipe:1"]
    raw = run_ffmpeg_input_bytes(data, suffix, args, timeout=180)
    if not raw or len(raw) % (AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH):
        raise RuntimeError("Audio canonical PCM is empty or not frame-aligned.")
    flags = FLAG_TRANSCODED
    if duration is not None and duration > AUDIO_MAX_SECONDS: flags |= FLAG_TRUNCATED
    canonical_duration = len(raw) / float(AUDIO_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH)
    ext = ExtendedFields(
        media_kind="audio", flags=flags, param0=min(orig_ch, 255), param1=0,
        original_a=min(orig_rate, 65535),
        original_b=min(65535, int(round((duration or 0.0) * 100))),
        format_code=format_code_from_name(name), mode_codec_code=0,
        param2_q16=min(65535, AUDIO_RATE), param3_q16=min(65535, int(round(canonical_duration * 100))),
    )
    aux = pack_audio_aux(AUDIO_RATE, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH)
    wav = wav_bytes_from_pcm(raw, AUDIO_RATE, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH)
    analysis = {
        "canonical_format": "PCM16 WAV reconstruction", "canonical_sample_rate_hz": AUDIO_RATE,
        "canonical_channels": AUDIO_CHANNELS, "canonical_sample_width_bits": AUDIO_SAMPLE_WIDTH * 8,
        "original_sample_rate_hz": orig_rate or None, "original_channels": orig_ch or None,
        "original_duration_s": duration, "canonical_duration_s": canonical_duration,
        "duration_truncated": bool(flags & FLAG_TRUNCATED),
    }
    return CanonicalContent(
        kind="audio", profile=PROFILE_AUDIO_PCM16, aux=aux, bits=bytes_to_bits(raw),
        original_name=name, original_bytes=len(data), canonical_sha256=sha256_hex(raw),
        extended_fields=ext,
        canonical_description=f"PCM16 {AUDIO_RATE} Hz, {AUDIO_CHANNELS} ch, {canonical_duration:.2f} s",
        input_preview_kind="audio", input_preview_payload=wav,
        encoded_preview_kind="audio", encoded_preview_payload=wav,
        reference_bytes=wav, reference_name=f"{Path(name).stem}_encoded_reference.wav",
        reference_mime="audio/wav", reference_kind="audio", analysis=analysis,
    )




def build_canonical_content(data: bytes, name: str) -> CanonicalContent:
    kind = media_kind_from_name(name)
    if kind == "image": return build_canonical_image(data, name)
    if kind == "text": return build_canonical_text(data, name)
    if kind == "audio": return build_canonical_audio(data, name)
    raise ValueError("Unsupported file type. Use a supported image, text, or audio extension.")


# =============================================================================
# Reconstruction from critical metadata + source
# =============================================================================

@dataclass
class ReconstructedMedia:
    data: bytes
    name: str
    mime: str
    kind: str
    semantic_note: str


def reconstruct_from_source(critical_header: bytes, source_bits: str, stem: str = "reconstructed") -> ReconstructedMedia:
    if not critical_semantic_valid(critical_header, observed_source_bits=len(source_bits)):
        raise ValueError("Recovered critical header is not semantically valid for the received source length.")
    meta = parse_critical_header(critical_header)
    profile = int(meta["profile"])
    aux = int(meta["aux"])
    if profile in IMAGE_PROFILES:
        w, h, _ = unpack_image_aux(aux)
        if profile == PROFILE_BINARY1:
            arr = np.fromiter((255 if c == "1" else 0 for c in source_bits), dtype=np.uint8, count=len(source_bits)).reshape((h, w))
            image = Image.fromarray(arr, mode="L")
        elif profile == PROFILE_L8:
            arr = np.frombuffer(bits_to_bytes(source_bits), dtype=np.uint8).reshape((h, w))
            image = Image.fromarray(arr.copy(), mode="L")
        elif profile == PROFILE_RGB8:
            arr = np.frombuffer(bits_to_bytes(source_bits), dtype=np.uint8).reshape((h, w, 3))
            image = Image.fromarray(arr.copy(), mode="RGB")
        else:
            arr = np.frombuffer(bits_to_bytes(source_bits), dtype=np.uint8).reshape((h, w, 4))
            image = Image.fromarray(arr.copy(), mode="RGBA")
        return ReconstructedMedia(image_to_png_bytes(image), f"{stem}_reconstructed.png", "image/png", "image",
                                  "Fresh PNG serialized from decoded canonical pixels.")
    if profile == PROFILE_TEXT_UTF8:
        raw = bits_to_bytes(source_bits)
        text = raw.decode("utf-8", errors="replace")
        out = text.encode("utf-8")
        return ReconstructedMedia(out, f"{stem}_reconstructed.txt", "text/plain; charset=utf-8", "text",
                                  "Fresh valid UTF-8 text; invalid noisy byte sequences are replaced during reconstruction.")
    if profile == PROFILE_AUDIO_PCM16:
        rate, ch, sw = unpack_audio_aux(aux)
        raw = bits_to_bytes(source_bits)
        wav = wav_bytes_from_pcm(raw, rate, ch, sw)
        return ReconstructedMedia(wav, f"{stem}_reconstructed.wav", "audio/wav", "audio",
                                  "Fresh WAV header wrapped around decoded PCM samples.")
    raise ValueError("Unsupported canonical profile.")


# =============================================================================
# Full encode/decode artifacts
# =============================================================================

@dataclass
class EncodedArtifact:
    canonical: CanonicalContent
    critical_header: bytes
    extended_header: bytes
    critical_bits: str
    extended_bits: str
    source_bits: str
    full_bits: str
    critical_dna: str
    extended_dna: str
    source_dna: str
    full_dna: str
    source_pad_bits: int
    reference: ReconstructedMedia


@dataclass
class DecodedArtifact:
    input_kind: str
    critical_recovery: CriticalRecovery
    critical_header: Optional[bytes]
    critical_metadata: Optional[Dict[str, object]]
    extended_header_received: bytes
    extended_metadata: Dict[str, object]
    source_bits: Optional[str]
    source_dna: Optional[str]
    reconstructed: Optional[ReconstructedMedia]
    status: str
    message: str


def encode_canonical(c: CanonicalContent) -> EncodedArtifact:
    critical = build_critical_header(c.content_bits, c.profile, c.aux)
    extended = build_extended_header(c.extended_fields)
    critical_bits = bytes_to_bits(critical)
    extended_bits = bytes_to_bits(extended)
    full_bits = critical_bits + extended_bits + c.bits
    critical_dna, p1 = seo_encode_bits(critical_bits)
    extended_dna, p2 = seo_encode_bits(extended_bits)
    if p1 or p2:
        raise AssertionError("Fixed metadata blocks must be divisible by four bits.")
    source_dna, source_pad = seo_encode_bits(c.bits)
    full_dna = critical_dna + extended_dna + source_dna
    ref = reconstruct_from_source(critical, c.bits, Path(c.original_name).stem)
    return EncodedArtifact(c, critical, extended, critical_bits, extended_bits, c.bits, full_bits,
                           critical_dna, extended_dna, source_dna, full_dna, source_pad, ref)


def _decode_common(
    input_kind: str, critical_rx: bytes, extended_rx: bytes, source_bits: Optional[str], source_dna: Optional[str],
    *, observed_source_bits: Optional[int], observed_source_nt: Optional[int], max_recovery_bits: int,
) -> DecodedArtifact:
    ext_meta = parse_extended_header(extended_rx)
    preferred = ext_meta["media_kind"] if ext_meta["status"] == "VALID" else None
    rec = recover_critical_header(
        critical_rx, observed_source_bits=observed_source_bits, observed_source_nt=observed_source_nt,
        max_unknown_bit_errors=max_recovery_bits,
    )
    if rec.recovered_header is None:
        rec = best_effort_guess_critical_header(
            critical_rx, rec, observed_source_bits=observed_source_bits, observed_source_nt=observed_source_nt,
            preferred_media_kind=preferred,
        )
    if rec.recovered_header is None:
        return DecodedArtifact(input_kind, rec, None, None, extended_rx, ext_meta, source_bits, source_dna,
                               None, rec.status,
                               "No structurally compatible critical-header interpretation was available.")
    meta = parse_critical_header(rec.recovered_header)
    nbits = int(meta["content_bits"])
    if source_bits is None:
        if source_dna is None:
            raise AssertionError("Missing source artifact.")
        expected_nt = 2 * math.ceil(nbits / 4)
        if len(source_dna) != expected_nt:
            return DecodedArtifact(input_kind, rec, rec.recovered_header, meta, extended_rx, ext_meta,
                                   None, source_dna, None, "FAIL",
                                   "Source-DNA length is inconsistent with recovered metadata; indels are unsupported.")
        source_bits = seo_decode_dna(source_dna, nbits)
    stem = "decoded"
    reconstructed = reconstruct_from_source(rec.recovered_header, source_bits, stem)
    crit_text = "Critical metadata uniquely recovered." if rec.status == "UNIQUE" else f"Critical metadata used best-effort structural salvage ({rec.status})."
    ext_text = "Extended metadata: VALID." if ext_meta["status"] == "VALID" else f"Extended metadata: {ext_meta["status"]}. Reconstruction continues."
    return DecodedArtifact(input_kind, rec, rec.recovered_header, meta, extended_rx, ext_meta,
                           source_bits, source_dna, reconstructed, "RECONSTRUCTED",
                           f"Reconstruction completed. {crit_text} {ext_text}")


def decode_full_bitstream(full_bits: str, max_recovery_bits: int = MAX_CRITICAL_UNKNOWN_SEARCH_BITS) -> DecodedArtifact:
    full_bits = normalize_bit_text(full_bits)
    if len(full_bits) < FIXED_META_BITS:
        raise ValueError(f"Full bitstream must contain at least {FIXED_META_BITS} metadata bits.")
    critical_rx = bits_to_bytes(full_bits[:CRITICAL_BITS])
    extended_rx = bits_to_bytes(full_bits[CRITICAL_BITS:FIXED_META_BITS])
    source_bits = full_bits[FIXED_META_BITS:]
    return _decode_common("Full bitstream", critical_rx, extended_rx, source_bits, None,
                          observed_source_bits=len(source_bits), observed_source_nt=None,
                          max_recovery_bits=max_recovery_bits)


def decode_full_dna(full_dna: str, max_recovery_bits: int = MAX_CRITICAL_UNKNOWN_SEARCH_BITS) -> DecodedArtifact:
    full_dna = normalize_dna_text(full_dna)
    if len(full_dna) < FIXED_META_DNA_NT:
        raise ValueError(f"Full DNA must contain at least {FIXED_META_DNA_NT} fixed metadata nt.")
    critical_dna = full_dna[:CRITICAL_DNA_NT]
    extended_dna = full_dna[CRITICAL_DNA_NT:FIXED_META_DNA_NT]
    source_dna = full_dna[FIXED_META_DNA_NT:]
    critical_rx = bits_to_bytes(seo_decode_dna(critical_dna, CRITICAL_BITS))
    extended_rx = bits_to_bytes(seo_decode_dna(extended_dna, EXT_BITS))
    return _decode_common("Full DNA", critical_rx, extended_rx, None, source_dna,
                          observed_source_bits=None, observed_source_nt=len(source_dna),
                          max_recovery_bits=max_recovery_bits)


# =============================================================================
# Analysis helpers
# =============================================================================


def dna_stats(dna: str) -> Dict[str, object]:
    n = len(dna)
    counts = {b: dna.count(b) for b in "ACGT"}
    fracs = {b: (counts[b] / n if n else 0.0) for b in "ACGT"}
    gc = fracs["G"] + fracs["C"]
    h0 = 0.0
    for p in fracs.values():
        if p > 0: h0 -= p * math.log2(p)
    max_run = 0
    if dna:
        max_run = cur = 1
        for i in range(1, n):
            if dna[i] == dna[i - 1]:
                cur += 1; max_run = max(max_run, cur)
            else:
                cur = 1
    return {
        "length_nt": n, "A": counts["A"], "C": counts["C"], "G": counts["G"], "T": counts["T"],
        "A_fraction": fracs["A"], "C_fraction": fracs["C"], "G_fraction": fracs["G"], "T_fraction": fracs["T"],
        "GC_fraction": gc, "H0_bits_per_nt": h0, "max_homopolymer": max_run,
    }


def dna_transition_probability_df(dna: str) -> pd.DataFrame:
    bases = "ACGT"
    counts = {a: {b: 0 for b in bases} for a in bases}
    for a, b in zip(dna, dna[1:]):
        if a in counts and b in counts[a]: counts[a][b] += 1
    rows = []
    for a in bases:
        total = sum(counts[a].values())
        rows.append([a] + [(counts[a][b] / total if total else 0.0) for b in bases])
    return pd.DataFrame(rows, columns=["from", "to_A", "to_C", "to_G", "to_T"]).set_index("from")


def bitstream_stats(bits: str) -> Dict[str, object]:
    n = len(bits); ones = bits.count("1"); zeros = n - ones
    p1 = ones / n if n else 0.0; p0 = zeros / n if n else 0.0
    h = 0.0
    for p in (p0, p1):
        if p > 0: h -= p * math.log2(p)
    return {"length_bits": n, "zeros": zeros, "ones": ones, "zero_fraction": p0, "one_fraction": p1, "H0_bits_per_bit": h}


def source_bit_difference_df(reference: str, observed: str, profile: int, aux: int) -> pd.DataFrame:
    pos = diff_positions(reference, observed)
    rows = []
    if profile in IMAGE_PROFILES:
        w, _, _ = unpack_image_aux(aux); bpp = BITS_PER_PIXEL[profile]
        for p in pos:
            pixel = p // bpp; local = p % bpp; y, x = divmod(pixel, w)
            if profile == PROFILE_BINARY1: channel, bitc = "BINARY", 0
            elif profile == PROFILE_L8: channel, bitc = "L", local
            elif profile == PROFILE_RGB8: channel, bitc = ["R", "G", "B"][local // 8], local % 8
            else: channel, bitc = ["R", "G", "B", "A"][local // 8], local % 8
            rows.append([p, p // 8, "image", pixel, x, y, channel, bitc, reference[p], observed[p]])
    elif profile == PROFILE_TEXT_UTF8:
        for p in pos: rows.append([p, p // 8, "text", p // 8, None, None, "byte", p % 8, reference[p], observed[p]])
    elif profile == PROFILE_AUDIO_PCM16:
        sample_bits = AUDIO_SAMPLE_WIDTH * 8
        for p in pos:
            sample_value = p // sample_bits; ch = sample_value % AUDIO_CHANNELS; frame = sample_value // AUDIO_CHANNELS
            rows.append([p, p // 8, "audio", frame, None, None, f"ch{ch}", p % sample_bits, reference[p], observed[p]])
    return pd.DataFrame(rows, columns=["source_bit_index", "byte_index", "media", "unit_index", "x", "y", "channel", "bit_in_unit", "reference_bit", "observed_bit"])


def image_error_mask_png(reference: str, observed: str, profile: int, aux: int) -> Optional[bytes]:
    if profile not in IMAGE_PROFILES or len(reference) != len(observed):
        return None
    w, h, _ = unpack_image_aux(aux); bpp = BITS_PER_PIXEL[profile]
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in diff_positions(reference, observed):
        idx = p // bpp
        if 0 <= idx < w * h:
            y, x = divmod(idx, w); mask[y, x] = 255
    return image_to_png_bytes(Image.fromarray(mask, mode="L"))


def numeric_source_metrics(reference_bits: str, observed_bits: str, profile: int, aux: int) -> Dict[str, object]:
    diffs = diff_positions(reference_bits, observed_bits)
    changed_bytes = len({p // 8 for p in diffs})
    out: Dict[str, object] = {
        "source_bit_errors": len(diffs), "changed_source_bytes": changed_bytes,
        "source_bit_error_fraction": len(diffs) / len(reference_bits) if reference_bits else 0.0,
    }
    if profile in IMAGE_PROFILES:
        w, h, _ = unpack_image_aux(aux); bpp = BITS_PER_PIXEL[profile]
        pixels = {p // bpp for p in diffs}
        out["affected_pixels"] = len(pixels)
        out["affected_pixel_fraction"] = len(pixels) / (w * h)
        # Numeric pixel MAE/MSE/PSNR.
        try:
            if profile == PROFILE_BINARY1:
                a = np.fromiter((255 if c == "1" else 0 for c in reference_bits), dtype=np.float64)
                b = np.fromiter((255 if c == "1" else 0 for c in observed_bits), dtype=np.float64)
            else:
                a = np.frombuffer(bits_to_bytes(reference_bits), dtype=np.uint8).astype(np.float64)
                b = np.frombuffer(bits_to_bytes(observed_bits), dtype=np.uint8).astype(np.float64)
            d = a - b; mse = float(np.mean(d * d)); mae = float(np.mean(np.abs(d)))
            out.update({"MAE": mae, "MSE": mse, "PSNR_dB": (float("inf") if mse == 0 else 10 * math.log10((255.0 ** 2) / mse))})
        except Exception:
            pass
    elif profile == PROFILE_AUDIO_PCM16:
        try:
            a = np.frombuffer(bits_to_bytes(reference_bits), dtype="<i2").astype(np.float64)
            b = np.frombuffer(bits_to_bytes(observed_bits), dtype="<i2").astype(np.float64)
            d = a - b; mse = float(np.mean(d * d)); mae = float(np.mean(np.abs(d)))
            sig = float(np.mean(a * a))
            out.update({"sample_MAE": mae, "sample_MSE": mse,
                        "SNR_dB": (float("inf") if mse == 0 else (10 * math.log10(sig / mse) if sig > 0 else float("-inf")))})
        except Exception:
            pass
    return out


def critical_recovery_df(rec: CriticalRecovery) -> pd.DataFrame:
    rows = [
        ["Status", rec.status], ["Candidate count", rec.candidate_count], ["Distance (bits)", rec.distance],
        ["Known MAGIC/VERSION errors", ", ".join(map(str, rec.known_error_positions)) or "none"],
        ["Corrected/guessed positions", ", ".join(map(str, rec.corrected_positions[:200])) or "none"],
        ["Message", rec.message],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def critical_metadata_df(header: bytes) -> pd.DataFrame:
    m = parse_critical_header(header)
    return pd.DataFrame([
        ["MAGIC", "0–31", m["magic"]], ["VERSION", "32–39", m["version"]],
        ["CONTENT_BITS", "40–79", m["content_bits"]], ["PROFILE", "80–87", f"{m['profile']} — {m['profile_name']}"],
        ["AUX", "88–111", m["aux_interpretation"]], ["CRC32", "112–143", f"{m['stored_crc32']} (valid={m['crc_ok']})"],
    ], columns=["Field", "Bit range", "Value"])


def extended_metadata_df(header: bytes) -> pd.DataFrame:
    m = parse_extended_header(header)
    rows = [[k, (json.dumps(v) if isinstance(v, list) else v)] for k, v in m.items()]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def representation_size_df(c: CanonicalContent) -> pd.DataFrame:
    return pd.DataFrame([
        ["Original uploaded container", c.original_bytes * 8, c.original_bytes],
        ["Critical Header", CRITICAL_BITS, CRITICAL_BYTES],
        ["Extended Metadata", EXT_BITS, EXT_BYTES],
        ["Primary canonical source", c.content_bits, math.ceil(c.content_bits / 8)],
        ["Full bitstream", FIXED_META_BITS + c.content_bits, math.ceil((FIXED_META_BITS + c.content_bits) / 8)],
        ["Full DNA", 2 * math.ceil((FIXED_META_BITS + c.content_bits) / 4), None],
    ], columns=["Region", "Bits / nt", "Approx. bytes"])


def build_manifest(encoded: EncodedArtifact) -> Dict[str, object]:
    c = encoded.canonical
    return {
        "app_version": APP_VERSION,
        "architecture": "144-bit recoverable critical + 160-bit nonblocking extended + primary canonical source",
        "media_kind": c.kind, "original_name": c.original_name, "original_file_bytes": c.original_bytes,
        "canonical_profile": PROFILE_NAMES[c.profile], "canonical_description": c.canonical_description,
        "canonical_sha256": c.canonical_sha256, "critical_bits": CRITICAL_BITS, "extended_bits": EXT_BITS,
        "source_bits": len(encoded.source_bits), "full_bits": len(encoded.full_bits),
        "source_dna_nt": len(encoded.source_dna), "full_dna_nt": len(encoded.full_dna),
        "source_pad_bits": encoded.source_pad_bits,
        "critical_metadata": parse_critical_header(encoded.critical_header),
        "extended_metadata": parse_extended_header(encoded.extended_header),
        "canonicalization": c.analysis,
        "channel": "substitutions only; no indel synchronization",
        "clean_recovery": "exact relative to canonical representation; fresh openable output container",
    }


def stress_bits(encoded: EncodedArtifact, n_critical: int, n_extended: int, n_source: int, seed: int) -> Tuple[str, Dict[str, object]]:
    c_bits, cp = random_bit_flips(encoded.critical_bits, n_critical, seed)
    e_bits, ep = random_bit_flips(encoded.extended_bits, n_extended, seed + 1)
    s_bits, sp = random_bit_flips(encoded.source_bits, n_source, seed + 2)
    return c_bits + e_bits + s_bits, {"critical": cp, "extended": ep, "source": sp}


def stress_dna(encoded: EncodedArtifact, n_critical: int, n_extended: int, n_source: int, seed: int) -> Tuple[str, Dict[str, object]]:
    c, cp = random_dna_substitutions(encoded.critical_dna, n_critical, seed)
    e, ep = random_dna_substitutions(encoded.extended_dna, n_extended, seed + 1)
    s, sp = random_dna_substitutions(encoded.source_dna, n_source, seed + 2)
    return c + e + s, {"critical": cp, "extended": ep, "source": sp}


def full_dna_difference_df(reference: str, observed: str) -> pd.DataFrame:
    if len(reference) != len(observed):
        return pd.DataFrame(columns=["nt_index", "region", "reference_base", "observed_base"])
    rows = []
    for i, (a, b) in enumerate(zip(reference, observed)):
        if a == b: continue
        region = "critical" if i < CRITICAL_DNA_NT else ("extended" if i < FIXED_META_DNA_NT else "source")
        rows.append([i, region, a, b])
    return pd.DataFrame(rows, columns=["nt_index", "region", "reference_base", "observed_base"])


def make_analysis_bundle(encoded: EncodedArtifact, dec: Optional[DecodedArtifact], diff_df: Optional[pd.DataFrame],
                         dna_diff_df: Optional[pd.DataFrame], comparison: Optional[Dict[str, object]],
                         error_mask: Optional[bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        manifest = build_manifest(encoded)
        z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        z.writestr("critical_header.csv", critical_metadata_df(encoded.critical_header).to_csv(index=False))
        z.writestr("extended_metadata.csv", extended_metadata_df(encoded.extended_header).to_csv(index=False))
        z.writestr("dna_transition_probabilities.csv", dna_transition_probability_df(encoded.source_dna).to_csv())
        z.writestr("representation_sizes.csv", representation_size_df(encoded.canonical).to_csv(index=False))
        z.writestr(encoded.canonical.reference_name, encoded.canonical.reference_bytes)
        if dec is not None:
            z.writestr("latest_critical_recovery.csv", critical_recovery_df(dec.critical_recovery).to_csv(index=False))
            if dec.reconstructed is not None:
                z.writestr(dec.reconstructed.name, dec.reconstructed.data)
        if diff_df is not None: z.writestr("latest_source_differences.csv", diff_df.to_csv(index=False))
        if dna_diff_df is not None: z.writestr("latest_DNA_substitutions.csv", dna_diff_df.to_csv(index=False))
        if comparison is not None: z.writestr("latest_comparison.json", json.dumps(comparison, indent=2, default=str))
        if error_mask is not None: z.writestr("latest_error_mask.png", error_mask)
    return out.getvalue()


# =============================================================================
# Streamlit UI helpers
# =============================================================================


def init_state() -> None:
    defaults = {
        "v17_ready": False, "v17_encoded": None, "v17_name": None, "v17_original": None,
        "v17_last_decoded": None, "v17_last_diff": None, "v17_last_dna_diff": None,
        "v17_last_comparison": None, "v17_last_mask": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


def render_media(
    label: str,
    kind: str,
    payload: object,
    mime: Optional[str] = None,
    *,
    key: Optional[str] = None,
) -> None:
    """Render a media preview.

    ``key`` is used by widget-backed previews (currently text areas) so the same
    logical preview can safely appear in Encode, Decode, and Stress Test on one
    Streamlit page without duplicate-element IDs.
    """
    st.markdown(f"**{label}**")
    if kind == "image":
        st.image(payload, width=320)
    elif kind == "text":
        text = payload.decode("utf-8", errors="replace") if isinstance(payload, (bytes, bytearray)) else str(payload)
        st.text_area(
            f"{label} preview",
            text[:12000],
            height=220,
            disabled=True,
            label_visibility="collapsed",
            key=key,
        )
    elif kind == "audio":
        st.audio(payload)
    else:
        st.caption("Preview unavailable.")


def _render_reconstruction(
    dec: DecodedArtifact,
    encoded: Optional[EncodedArtifact],
    *,
    key_prefix: str,
) -> None:
    """Render reconstructed content, keeping image reference/noisy/mask on one row."""
    st.markdown("#### Reconstruction")
    st.write(dec.message)
    if dec.reconstructed is None:
        return

    if encoded is not None and dec.source_bits is not None:
        if encoded.canonical.kind == "image" and dec.reconstructed.kind == "image":
            mask = image_error_mask_png(
                encoded.source_bits, dec.source_bits, encoded.canonical.profile, encoded.canonical.aux
            )
            changed = 0
            if len(encoded.source_bits) == len(dec.source_bits):
                changed = sum(a != b for a, b in zip(encoded.source_bits, dec.source_bits))
            reconstructed_label = "Noisy reconstructed image" if changed else "Reconstructed image"

            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                render_media(
                    "Encoded Reference", encoded.canonical.reference_kind,
                    encoded.canonical.reference_bytes, encoded.canonical.reference_mime,
                    key=f"{key_prefix}_reference_text",
                )
            with c2:
                render_media(
                    reconstructed_label, dec.reconstructed.kind,
                    dec.reconstructed.data, dec.reconstructed.mime,
                    key=f"{key_prefix}_reconstructed_text",
                )
            with c3:
                if mask is not None:
                    render_media("Binary error mask", "image", mask, key=f"{key_prefix}_mask_text")
                    st.caption("White = at least one decoded source bit changed in that pixel.")
                else:
                    st.markdown("**Binary error mask**")
                    st.caption("Error mask unavailable for this decoded image.")
            return

        c1, c2 = st.columns(2)
        with c1:
            render_media(
                "Encoded Reference", encoded.canonical.reference_kind,
                encoded.canonical.reference_bytes, encoded.canonical.reference_mime,
                key=f"{key_prefix}_reference_text",
            )
        with c2:
            render_media(
                "Reconstructed content", dec.reconstructed.kind,
                dec.reconstructed.data, dec.reconstructed.mime,
                key=f"{key_prefix}_reconstructed_text",
            )
    else:
        render_media(
            "Reconstructed content", dec.reconstructed.kind,
            dec.reconstructed.data, dec.reconstructed.mime,
            key=f"{key_prefix}_reconstructed_text",
        )


def _render_recovery_tables(dec: DecodedArtifact, *, compact: bool) -> None:
    """Render recovery/metadata tables either normally or in compact expanders."""
    if compact:
        with st.expander("Critical Header Recovery", expanded=False):
            st.dataframe(critical_recovery_df(dec.critical_recovery), use_container_width=True, hide_index=True)
            if dec.critical_header is not None:
                st.dataframe(critical_metadata_df(dec.critical_header), use_container_width=True, hide_index=True)

        with st.expander("Extended Metadata", expanded=False):
            ext_status = str(dec.extended_metadata.get("status", "FAIL"))
            if ext_status == "VALID":
                st.success("Extended metadata: VALID.")
            else:
                st.warning(f"Extended metadata: {ext_status}. Reconstruction continues.")
            st.dataframe(extended_metadata_df(dec.extended_header_received), use_container_width=True, hide_index=True)
        return

    st.markdown("#### Critical Header Recovery")
    st.dataframe(critical_recovery_df(dec.critical_recovery), use_container_width=True, hide_index=True)
    if dec.critical_header is not None:
        st.dataframe(critical_metadata_df(dec.critical_header), use_container_width=True, hide_index=True)

    st.markdown("#### Extended Metadata")
    ext_status = str(dec.extended_metadata.get("status", "FAIL"))
    if ext_status == "VALID":
        st.success("Extended metadata: VALID.")
    else:
        st.warning(f"Extended metadata: {ext_status}. Reconstruction continues.")
    st.dataframe(extended_metadata_df(dec.extended_header_received), use_container_width=True, hide_index=True)


def show_decoded_result(
    dec: DecodedArtifact,
    encoded: Optional[EncodedArtifact] = None,
    *,
    reconstruction_first: bool = False,
    compact_tables: bool = False,
    key_prefix: str = "v17_result",
) -> None:
    """Render decode/stress results.

    Decode keeps the original ordering by using the defaults. Stress Test can set
    ``reconstruction_first=True`` and ``compact_tables=True`` so the media result
    is immediately visible and metadata tables are collapsed to save space.
    """
    if reconstruction_first:
        _render_reconstruction(dec, encoded, key_prefix=key_prefix)
        _render_recovery_tables(dec, compact=compact_tables)
    else:
        _render_recovery_tables(dec, compact=compact_tables)
        _render_reconstruction(dec, encoded, key_prefix=key_prefix)

def update_latest_analysis(dec: DecodedArtifact, encoded: EncodedArtifact, context: str) -> None:
    st.session_state.v17_last_decoded = dec
    st.session_state.v17_last_diff = None
    st.session_state.v17_last_comparison = None
    st.session_state.v17_last_mask = None
    if dec.source_bits is None:
        return
    try:
        diff_df = source_bit_difference_df(encoded.source_bits, dec.source_bits, encoded.canonical.profile, encoded.canonical.aux)
        metrics = numeric_source_metrics(encoded.source_bits, dec.source_bits, encoded.canonical.profile, encoded.canonical.aux)
        comp = {
            "context": context, "critical_recovery_status": dec.critical_recovery.status,
            "critical_recovery_distance_bits": dec.critical_recovery.distance,
            "extended_metadata_status": dec.extended_metadata["status"], **metrics,
        }
        st.session_state.v17_last_diff = diff_df
        st.session_state.v17_last_comparison = comp
        if encoded.canonical.kind == "image":
            st.session_state.v17_last_mask = image_error_mask_png(encoded.source_bits, dec.source_bits, encoded.canonical.profile, encoded.canonical.aux)
    except Exception:
        pass


# =============================================================================
# Streamlit app
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="Graceful DNA Data Analysis v17", layout="wide")
    init_state()
    st.title("Graceful-Degradation DNA Data Analysis — v17")
    st.caption("Recoverable 144-bit critical header + nonblocking 160-bit extended metadata + primary canonical image/text/audio source | Seo R∞-P8")

    st.divider(); st.header("0. Overview")
    st.code("H_critical (144 bits, recoverable) || H_extended (160 bits, nonblocking) || C_canonical (primary graceful content)", language=None)
    st.dataframe(pd.DataFrame([
        ["Critical Header", "144 bits / 72 nt", "Required for reconstruction", "CRC/MITM recovery first; deterministic structural guess if AMBIGUOUS/FAIL"],
        ["Extended Metadata", "160 bits / 80 nt", "Not required for reconstruction", "VALID/DAMAGED; never blocks reconstruction"],
        ["Primary Canonical Source", "variable", "Media content", "Image pixels / UTF-8 text / PCM16 audio"],
        ["Channel model", "substitutions only", "No indel synchronization", "Bit/base insertion or deletion is unsupported"],
        ["Output", "fresh media container", "Canonical-content reconstruction", "PNG / UTF-8 text / WAV"],
    ], columns=["Component", "Size", "Role", "Behavior"]), use_container_width=True, hide_index=True)
    st.markdown("#### Supported Primary Canonical Media")
    st.dataframe(pd.DataFrame([
        ["Image", "Binary1 / L8 / RGB8 / RGBA8", "PNG", "256×256 maximum bounding box"],
        ["Text", "UTF-8 bytes", "UTF-8 text", "content remains readable/openable after noisy reconstruction when structurally possible"],
        ["Audio", f"PCM16 {AUDIO_RATE} Hz stereo", "WAV", f"first {AUDIO_MAX_SECONDS} s maximum"],
    ], columns=["Media", "Primary canonical source", "Fresh reconstruction", "Demo policy"]), use_container_width=True, hide_index=True)
    st.info("Critical metadata is recovered first. If strict recovery is AMBIGUOUS/FAIL, v17 attempts deterministic structural salvage. Extended metadata remains nonblocking.")

    st.divider(); st.header("1. Encode")
    st.caption("Upload a supported image, text, or audio file → primary canonical source → fixed metadata → Seo R∞-P8 DNA.")
    upload = st.file_uploader("Upload file", type=ALL_UPLOAD_EXTENSIONS, key="v17_input")
    if upload is not None:
        try:
            original = upload.getvalue()
            canonical = build_canonical_content(original, upload.name)
            encoded = encode_canonical(canonical)
            st.session_state.v17_ready = True; st.session_state.v17_encoded = encoded
            st.session_state.v17_name = upload.name; st.session_state.v17_original = original
            st.dataframe(pd.DataFrame([
                ["Media type", canonical.kind, "auto-detected from supported extension"],
                ["Original file", f"{len(original):,} bytes", "input container"],
                ["Canonical profile", PROFILE_NAMES[canonical.profile], canonical.canonical_description],
                ["Critical Header", f"{CRITICAL_BITS} bits", f"{CRITICAL_DNA_NT} nt"],
                ["Extended Metadata", f"{EXT_BITS} bits", f"{EXT_DNA_NT} nt"],
                ["Canonical source", f"{len(encoded.source_bits):,} bits", f"{len(encoded.source_dna):,} nt"],
                ["Full bitstream", f"{len(encoded.full_bits):,} bits", "critical + extended + source"],
                ["Full DNA", f"{len(encoded.full_dna):,} nt", "critical + extended + source DNA"],
            ], columns=["Item", "Value", "Interpretation"]), use_container_width=True, hide_index=True)
            p1, p2 = st.columns(2)
            with p1: render_media("Input", canonical.input_preview_kind, canonical.input_preview_payload, key="v17_encode_input_text")
            with p2:
                render_media("Encoded image" if canonical.kind == "image" else "Encoded content", canonical.encoded_preview_kind, canonical.encoded_preview_payload, key="v17_encode_content_text")
            st.markdown("#### Canonicalization")
            st.dataframe(pd.DataFrame([[k, v] for k, v in canonical.analysis.items()], columns=["Parameter", "Value"]), use_container_width=True, hide_index=True)
            st.markdown("#### Critical Header")
            st.dataframe(critical_metadata_df(encoded.critical_header), use_container_width=True, hide_index=True)
            st.markdown("#### Extended Metadata")
            st.dataframe(extended_metadata_df(encoded.extended_header), use_container_width=True, hide_index=True)
            st.markdown("#### Bitstream Structure")
            st.dataframe(pd.DataFrame([
                ["Critical Header", 0, CRITICAL_BITS - 1, CRITICAL_BITS, "recoverable / reconstruction-critical"],
                ["Extended Metadata", CRITICAL_BITS, FIXED_META_BITS - 1, EXT_BITS, "nonblocking descriptive metadata"],
                ["Primary Canonical Source", FIXED_META_BITS, len(encoded.full_bits) - 1, len(encoded.source_bits), f"graceful {canonical.kind} content"],
            ], columns=["Region", "Start bit", "End bit", "Length", "Role"]), use_container_width=True, hide_index=True)
            d1, d2, d3, d4 = st.columns(4)
            stem = Path(upload.name).stem
            d1.download_button("Download full bitstream", (encoded.full_bits + "\n").encode("ascii"), file_name=f"{stem}_full_bitstream_v17.txt", mime="text/plain", key="v17_encode_full_bits")
            d2.download_button("Download source bits", (encoded.source_bits + "\n").encode("ascii"), file_name=f"{stem}_source_bits_v17.txt", mime="text/plain", key="v17_encode_source_bits")
            d3.download_button("Download full DNA", (encoded.full_dna + "\n").encode("ascii"), file_name=f"{stem}_full_DNA_v17.txt", mime="text/plain", key="v17_encode_full_dna")
            d4.download_button("Download source DNA", (encoded.source_dna + "\n").encode("ascii"), file_name=f"{stem}_source_DNA_v17.txt", mime="text/plain", key="v17_encode_source_dna")
            st.download_button("Download Encoded Reference", canonical.reference_bytes, file_name=canonical.reference_name, mime=canonical.reference_mime, key="v17_encode_reference")
            with st.expander("Inspect clean metadata / source preview"):
                st.code("CRITICAL BITS:\n" + encoded.critical_bits, language=None)
                st.code("EXTENDED BITS:\n" + encoded.extended_bits, language=None)
                st.code("SOURCE BITS:\n" + short_preview(encoded.source_bits), language=None)
                st.code("SOURCE DNA:\n" + short_preview(encoded.source_dna), language=None)
        except Exception as exc:
            st.session_state.v17_ready = False
            st.error(str(exc))

    st.divider(); st.header("2. Decode")
    if not st.session_state.v17_ready:
        st.info("Encode a file first so Decode can compare against the current reference.")
    else:
        encoded: EncodedArtifact = st.session_state.v17_encoded
        st.caption(
            "Upload source-only bits/DNA to decode against the current clean metadata, "
            "or upload a complete bitstream/DNA artifact that contains metadata + source."
        )
        kind = st.radio(
            "Decode mode",
            ["Source bits", "Source DNA", "Full bitstream", "Full DNA"],
            horizontal=True,
            key="v17_decode_kind",
        )
        upload_label = {
            "Source bits": "Upload source-bit file",
            "Source DNA": "Upload source-DNA file",
            "Full bitstream": "Upload full-bitstream file",
            "Full DNA": "Upload full-DNA file",
        }[kind]
        f = st.file_uploader(upload_label, type=["txt"], key="v17_decode_file")

        if f is not None:
            try:
                raw = f.getvalue()
                st.session_state.v17_last_dna_diff = None

                if kind == "Source bits":
                    source_bits = normalize_bit_text(raw)
                    if len(source_bits) != len(encoded.source_bits):
                        raise ValueError(
                            f"Uploaded source-bit length is {len(source_bits):,} bits; "
                            f"current encoded source is {len(encoded.source_bits):,} bits. "
                            "Source-only Decode supports substitutions only; insertion/deletion/truncation is unsupported."
                        )
                    dec = _decode_common(
                        "Source bits", encoded.critical_header, encoded.extended_header,
                        source_bits, None, observed_source_bits=len(source_bits),
                        observed_source_nt=None, max_recovery_bits=MAX_CRITICAL_UNKNOWN_SEARCH_BITS,
                    )

                elif kind == "Source DNA":
                    source_dna = normalize_dna_text(raw)
                    if len(source_dna) != len(encoded.source_dna):
                        raise ValueError(
                            f"Uploaded source-DNA length is {len(source_dna):,} nt; "
                            f"current encoded source is {len(encoded.source_dna):,} nt. "
                            "Source-only Decode supports substitutions only; insertion/deletion/truncation is unsupported."
                        )
                    dec = _decode_common(
                        "Source DNA", encoded.critical_header, encoded.extended_header,
                        None, source_dna, observed_source_bits=None,
                        observed_source_nt=len(source_dna), max_recovery_bits=MAX_CRITICAL_UNKNOWN_SEARCH_BITS,
                    )
                    st.session_state.v17_last_dna_diff = full_dna_difference_df(
                        encoded.source_dna, source_dna
                    ) if len(source_dna) == len(encoded.source_dna) else None

                elif kind == "Full bitstream":
                    dec = decode_full_bitstream(raw)

                else:  # Full DNA
                    dec = decode_full_dna(raw)
                    obs = normalize_dna_text(raw)
                    if len(obs) == len(encoded.full_dna):
                        st.session_state.v17_last_dna_diff = full_dna_difference_df(encoded.full_dna, obs)

                show_decoded_result(dec, encoded, key_prefix="v17_decode_result")
                update_latest_analysis(dec, encoded, "Decode")

                if dec.reconstructed is not None:
                    st.download_button(
                        "Download Reconstructed file", dec.reconstructed.data,
                        file_name=dec.reconstructed.name, mime=dec.reconstructed.mime,
                        key="v17_decode_reconstructed",
                    )
            except Exception as exc:
                st.error(str(exc))

    st.divider(); st.header("3. Stress Test")
    if not st.session_state.v17_ready:
        st.info("Encode a file first.")
    else:
        encoded: EncodedArtifact = st.session_state.v17_encoded

        stress_mode = st.radio(
            "Stress Test mode",
            ["Current bitstream/DNA from Encode", "Upload bitstream/DNA from user"],
            horizontal=True,
            key="v17_stress_mode",
        )

        if stress_mode == "Current bitstream/DNA from Encode":
            domain = st.radio(
                "Inject substitutions in", ["Bits", "DNA"], horizontal=True,
                key="v17_stress_current_domain",
            )
            a, b, c, d = st.columns(4)
            if domain == "Bits":
                maxc, maxe, maxs = CRITICAL_BITS, EXT_BITS, len(encoded.source_bits)
            else:
                maxc, maxe, maxs = CRITICAL_DNA_NT, EXT_DNA_NT, len(encoded.source_dna)

            ncrit = a.number_input(
                "Critical errors", 0, int(min(maxc, 100)), 0, 1,
                key="v17_stress_current_critical_errors",
            )
            nextd = b.number_input(
                "Extended errors", 0, int(min(maxe, 100)), 0, 1,
                key="v17_stress_current_extended_errors",
            )
            nsrc = c.number_input(
                "Source errors", 0, int(min(maxs, 10000)), min(10, int(maxs)), 1,
                key="v17_stress_current_source_errors",
            )
            seed = d.number_input(
                "Random seed", 0, 10_000_000, 15, 1,
                key="v17_stress_current_seed",
            )

            if st.button("Run Stress Test", type="primary", key="v17_run_current_stress"):
                try:
                    if domain == "Bits":
                        modified, _ = stress_bits(encoded, int(ncrit), int(nextd), int(nsrc), int(seed))
                        dec = decode_full_bitstream(modified)
                        st.session_state.v17_last_dna_diff = None
                    else:
                        modified, _ = stress_dna(encoded, int(ncrit), int(nextd), int(nsrc), int(seed))
                        dec = decode_full_dna(modified)
                        st.session_state.v17_last_dna_diff = full_dna_difference_df(encoded.full_dna, modified)

                    # Stress Test UI: reconstruction first, then compact metadata tables.
                    show_decoded_result(
                        dec, encoded,
                        reconstruction_first=True,
                        compact_tables=True,
                        key_prefix="v17_stress_current_result",
                    )
                    update_latest_analysis(dec, encoded, "Stress Test — current Encode artifact")
                    if dec.reconstructed is not None:
                        st.download_button(
                            "Download Reconstructed file", dec.reconstructed.data,
                            file_name=dec.reconstructed.name, mime=dec.reconstructed.mime,
                            key="v17_stress_current_reconstructed",
                        )
                except Exception as exc:
                    st.error(str(exc))

        else:  # Upload bitstream/DNA from user
            st.caption(
                "Upload a modified full bitstream or full DNA artifact. The substitution-only model requires "
                "the uploaded artifact to keep the same length as the current Encode reference."
            )

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Download clean full bitstream", (encoded.full_bits + "\n").encode("ascii"),
                file_name="v17_stress_clean_bitstream.txt", mime="text/plain",
                key="v17_stress_upload_clean_bits",
            )
            dl2.download_button(
                "Download clean full DNA", (encoded.full_dna + "\n").encode("ascii"),
                file_name="v17_stress_clean_DNA.txt", mime="text/plain",
                key="v17_stress_upload_clean_dna",
            )

            upload_kind = st.radio(
                "Uploaded artifact type", ["Full bitstream", "Full DNA"], horizontal=True,
                key="v17_stress_upload_kind",
            )
            uploaded_file = st.file_uploader(
                "Upload modified artifact", type=["txt"], key="v17_stress_upload_file",
            )

            if uploaded_file is not None and st.button(
                "Run Uploaded Stress Test", type="primary", key="v17_run_uploaded_stress"
            ):
                try:
                    uploaded_raw = uploaded_file.getvalue()
                    if upload_kind == "Full bitstream":
                        observed = normalize_bit_text(uploaded_raw)
                        if len(observed) != len(encoded.full_bits):
                            raise ValueError(
                                f"Uploaded bitstream length is {len(observed):,} bits; clean reference is "
                                f"{len(encoded.full_bits):,} bits. Stress Test supports substitutions only; "
                                "insertion/deletion/truncation is unsupported."
                            )
                        dec = decode_full_bitstream(observed)
                        st.session_state.v17_last_dna_diff = None
                    else:
                        observed = normalize_dna_text(uploaded_raw)
                        if len(observed) != len(encoded.full_dna):
                            raise ValueError(
                                f"Uploaded DNA length is {len(observed):,} nt; clean reference is "
                                f"{len(encoded.full_dna):,} nt. Stress Test supports substitutions only; "
                                "insertion/deletion/truncation is unsupported."
                            )
                        dec = decode_full_dna(observed)
                        st.session_state.v17_last_dna_diff = full_dna_difference_df(encoded.full_dna, observed)

                    show_decoded_result(
                        dec, encoded,
                        reconstruction_first=True,
                        compact_tables=True,
                        key_prefix="v17_stress_upload_result",
                    )
                    update_latest_analysis(dec, encoded, "Stress Test — uploaded user artifact")
                    if dec.reconstructed is not None:
                        st.download_button(
                            "Download Reconstructed file", dec.reconstructed.data,
                            file_name=dec.reconstructed.name, mime=dec.reconstructed.mime,
                            key="v17_stress_upload_reconstructed",
                        )
                except Exception as exc:
                    st.error(str(exc))

    st.divider(); st.header("4. Analysis")
    if not st.session_state.v17_ready:
        st.info("Encode a file first.")
    else:
        encoded: EncodedArtifact = st.session_state.v17_encoded
        stats = dna_stats(encoded.source_dna)
        st.markdown("#### DNA Base Composition")
        st.dataframe(pd.DataFrame({
            "Base": ["A", "C", "G", "T"],
            "Count": [stats["A"], stats["C"], stats["G"], stats["T"]],
            "Fraction": [stats["A_fraction"], stats["C_fraction"], stats["G_fraction"], stats["T_fraction"]],
        }), width=620, hide_index=True)
        st.markdown("#### Sequence and Bitstream Statistics")
        st.dataframe(pd.DataFrame([
            ["Source DNA length", stats["length_nt"]], ["GC fraction", stats["GC_fraction"]],
            ["Zero-order entropy H0 (bits/nt)", stats["H0_bits_per_nt"]], ["Maximum homopolymer", stats["max_homopolymer"]],
            ["Source bits", len(encoded.source_bits)], ["Critical metadata nt", CRITICAL_DNA_NT],
            ["Extended metadata nt", EXT_DNA_NT], ["Full DNA nt", len(encoded.full_dna)],
        ], columns=["Metric", "Value"]), width=760, hide_index=True)
        st.markdown("#### DNA Transition Probabilities")
        st.dataframe(dna_transition_probability_df(encoded.source_dna), width=660)
        st.markdown("#### Canonical Representation Size")
        st.dataframe(representation_size_df(encoded.canonical), width=780, hide_index=True)
        st.markdown("#### Media Canonicalization")
        st.dataframe(pd.DataFrame([[k, v] for k, v in encoded.canonical.analysis.items()], columns=["Parameter", "Value"]), use_container_width=True, hide_index=True)
        st.markdown("#### Latest Recovery / Reconstruction Metrics")
        if st.session_state.v17_last_comparison:
            st.dataframe(pd.DataFrame([[k, v] for k, v in st.session_state.v17_last_comparison.items()], columns=["Metric", "Value"]), use_container_width=True, hide_index=True)
            if st.session_state.v17_last_diff is not None and not st.session_state.v17_last_diff.empty:
                st.markdown("#### Source Error Positions")
                st.dataframe(st.session_state.v17_last_diff.head(3000), use_container_width=True, hide_index=True)
        else:
            st.caption("Run Decode or Stress Test to populate recovery/noisy-comparison metrics.")
        st.caption("DNA base composition and source error positions are shown as tables only; no charts are drawn.")

    st.divider(); st.header("5. Analysis Export")
    if not st.session_state.v17_ready:
        st.info("Encode a file first.")
    else:
        encoded: EncodedArtifact = st.session_state.v17_encoded
        manifest = build_manifest(encoded)
        dna_s = dna_stats(encoded.source_dna); bit_s = bitstream_stats(encoded.source_bits)
        transition_df = dna_transition_probability_df(encoded.source_dna); rep_df = representation_size_df(encoded.canonical)
        rows = []
        for group, obj in (("dna", dna_s), ("bitstream", bit_s), ("canonicalization", encoded.canonical.analysis),
                           ("latest_comparison", st.session_state.v17_last_comparison or {})):
            for k, v in obj.items():
                rows.append([group, k, json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v])
        summary_df = pd.DataFrame(rows, columns=["group", "metric", "value"])
        st.dataframe(pd.DataFrame([
            ["Manifest", "JSON"], ["Analysis summary", "CSV"], ["Critical metadata", "CSV"],
            ["Extended metadata", "CSV"], ["DNA transition matrix", "CSV"], ["Representation sizes", "CSV"],
            ["Latest critical recovery", "CSV"], ["Latest source differences", "CSV"], ["Latest DNA substitutions", "CSV"],
            ["Encoded Reference", Path(encoded.canonical.reference_name).suffix.lstrip(".").upper()],
            ["Reconstructed file", "media-specific"], ["Error mask", "PNG, image only"], ["Complete analysis bundle", "ZIP"],
        ], columns=["Artifact", "Format"]), use_container_width=True, hide_index=True)
        stem = Path(encoded.canonical.original_name).stem
        x1, x2, x3 = st.columns(3)
        x1.download_button("Download manifest JSON", json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"), file_name=f"{stem}_v17_manifest.json", mime="application/json", key="v17_export_manifest")
        x2.download_button("Download analysis summary CSV", summary_df.to_csv(index=False).encode("utf-8"), file_name=f"{stem}_v17_analysis_summary.csv", mime="text/csv", key="v17_export_summary")
        x3.download_button("Download transition CSV", transition_df.to_csv().encode("utf-8"), file_name=f"{stem}_v17_transition_probabilities.csv", mime="text/csv", key="v17_export_transition")
        y1, y2, y3 = st.columns(3)
        y1.download_button("Download Critical Header CSV", critical_metadata_df(encoded.critical_header).to_csv(index=False).encode("utf-8"), file_name=f"{stem}_v17_critical_header.csv", mime="text/csv", key="v17_export_critical")
        y2.download_button("Download Extended Metadata CSV", extended_metadata_df(encoded.extended_header).to_csv(index=False).encode("utf-8"), file_name=f"{stem}_v17_extended_metadata.csv", mime="text/csv", key="v17_export_extended")
        y3.download_button("Download representation-size CSV", rep_df.to_csv(index=False).encode("utf-8"), file_name=f"{stem}_v17_representation_sizes.csv", mime="text/csv", key="v17_export_sizes")
        z1, z2, z3 = st.columns(3)
        z1.download_button("Download Encoded Reference", encoded.canonical.reference_bytes, file_name=encoded.canonical.reference_name, mime=encoded.canonical.reference_mime, key="v17_export_reference")
        if st.session_state.v17_last_diff is not None:
            z2.download_button("Download latest source differences CSV", st.session_state.v17_last_diff.to_csv(index=False).encode("utf-8"), file_name=f"{stem}_v17_latest_source_differences.csv", mime="text/csv", key="v17_export_source_diff")
        if st.session_state.v17_last_dna_diff is not None:
            z3.download_button("Download latest DNA substitutions CSV", st.session_state.v17_last_dna_diff.to_csv(index=False).encode("utf-8"), file_name=f"{stem}_v17_latest_DNA_substitutions.csv", mime="text/csv", key="v17_export_dna_diff")
        if st.session_state.v17_last_decoded is not None and st.session_state.v17_last_decoded.reconstructed is not None:
            r = st.session_state.v17_last_decoded.reconstructed
            st.download_button("Download latest Reconstructed file", r.data, file_name=r.name, mime=r.mime, key="v17_export_reconstructed")
        if st.session_state.v17_last_mask is not None:
            st.download_button("Download latest error mask PNG", st.session_state.v17_last_mask, file_name=f"{stem}_v17_latest_error_mask.png", mime="image/png", key="v17_export_mask")
        bundle = make_analysis_bundle(encoded, st.session_state.v17_last_decoded, st.session_state.v17_last_diff,
                                      st.session_state.v17_last_dna_diff, st.session_state.v17_last_comparison,
                                      st.session_state.v17_last_mask)
        st.download_button("Download complete analysis bundle ZIP", bundle, file_name=f"{stem}_v17_analysis_bundle.zip", mime="application/zip", key="v17_export_bundle")


if __name__ == "__main__":
    main()
