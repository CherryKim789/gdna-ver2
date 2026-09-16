Graceful-Degradation DNA Data Analysis v17
==========================================

Foundation
----------
v17 keeps the v14 architecture and UI structure:

H_critical (144 bits, recoverable)
|| H_extended (160 bits, nonblocking)
|| C_canonical (primary graceful content)

UI sections are retained:
0. Overview
1. Encode
2. Decode
3. Stress Test
4. Analysis
5. Analysis Export

Supported media
---------------
Image
- PNG, JPEG, WebP, BMP, TIFF, GIF
- automatic content-preserving canonicalization
- Binary1 / L8 / RGB8 / RGBA8
- maximum canonical bounding box 256 x 256
- fresh reconstruction: PNG

Text
- TXT, CSV, TSV, JSON, XML, MD, PY, HTML/HTM, CSS, JS, YAML/YML, LOG
- primary canonical source: UTF-8 bytes
- non-UTF-8 input falls back through Latin-1 decoding and is re-serialized as UTF-8
- fresh reconstruction: UTF-8 text
- noisy invalid UTF-8 byte sequences are decoded with replacement characters so the output remains valid UTF-8

Audio
- WAV, MP3, AAC, M4A, FLAC, OGG, OPUS, WMA
- primary canonical source: PCM16, 22.05 kHz, stereo
- demo coverage: first 20 s maximum
- fresh reconstruction: WAV
- ffmpeg/ffprobe are required for audio canonicalization

Metadata
--------
Critical Header: 144 bits
MAGIC32 | VERSION8 | CONTENT_BITS40 | PROFILE8 | AUX24 | CRC32

Recovery:
- known MAGIC/VERSION repair
- CRC32 syndrome
- meet-in-the-middle search
- media-aware semantic filtering
- UNIQUE / AMBIGUOUS / FAIL strict result
- deterministic structural salvage when strict recovery cannot produce a unique header

Extended Metadata: 160 bits
- descriptive/canonicalization information
- VALID / DAMAGED
- never blocks reconstruction

Stress Test
-----------
Internal:
- bit or DNA substitutions can be injected independently into Critical Header, Extended Metadata and Primary Canonical Source.

External bitstream test:
1. Download the clean full bitstream.
2. Flip 0<->1 bits externally without inserting or deleting symbols.
3. Upload the modified .txt bitstream in 3. Stress Test.
4. Run Uploaded Bitstream Stress Test.
5. Inspect metadata recovery, reconstructed content and analysis tables.

Analysis
--------
DNA Base Composition and Source Error Positions are table-only in v17.
No matplotlib/pyplot charts are used.

Scope
-----
- substitution-only channel model
- no insertion/deletion/truncation synchronization
- Seo R∞-P8 local mapping: 4 bits -> 2 nt, 8 periodic mapping columns
- CRC32 is metadata redundancy; no additional BCH/RS/repetition ECC layer
- clean recovery is exact relative to the canonical representation, not byte-identical to compressed image/audio input containers

Run
---
pip install -r requirements_graceful_seo_v17.txt
streamlit run graceful_seo_architecture_v17.py

System dependency for audio:
ffmpeg (including ffprobe)
