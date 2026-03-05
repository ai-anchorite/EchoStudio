"""Text preprocessing and chunking utilities for Echo TTS.

Ported and adapted from echo-tts-api fork with improvements for
long-form generation with automatic sentence splitting and time-based chunking.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Match, Tuple

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
SPEAKER_TAG_PATTERN = re.compile(r"(\[S\d+\])")
CHUNK_MARKER_PATTERN = re.compile(r"\*\*Chunk\s+\d+:\*\*\s*")
POTENTIAL_END_PATTERN = re.compile(r'([.!?])(["\']?)(\s+|$)')
BULLET_POINT_PATTERN = re.compile(r"(?:^|\n)\s*([-\u2022*]|\d+\.)\s+")
ALL_CAPS_WORD_PATTERN = re.compile(r"\b([A-Z]{2,})(?:'S|'s)?\b")
EXCLAMATION_PATTERN = re.compile(r"!+")
SENTENCE_START_PATTERN = re.compile(r"(?:(?<=^)|(?<=[.!?]\s)|(?<=\]\s))([a-z])")

ABBREVIATIONS_FILE = Path(__file__).resolve().parent / "abbreviations.txt"

# Basic abbreviations to avoid false sentence splits.
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "rev.", "hon.", "st.",
    "etc.", "e.g.", "i.e.", "vs.", "approx.", "dept.", "jr.", "sr.",
    "u.s.", "u.k.", "a.m.", "p.m.",
}

# ---------------------------------------------------------------------------
# Abbreviation loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_uppercase_abbreviations() -> set:
    preserved: set = set()
    try:
        with ABBREVIATIONS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if not word or word.startswith("#"):
                    continue
                preserved.add(word.upper())
    except (FileNotFoundError, OSError):
        return set()
    return preserved


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def _lowercase_caps_except_abbreviations(text: str) -> str:
    preserved = _load_uppercase_abbreviations()

    def repl(match: Match) -> str:
        word = match.group(1)
        suffix = match.group(0)[len(word):]
        if word in preserved:
            return f"{word}{suffix}"
        return f"{word.lower()}{suffix}"

    return ALL_CAPS_WORD_PATTERN.sub(repl, text)


def _normalize_exclamation(text: str) -> str:
    def repl(match: Match) -> str:
        span = match.group(0)
        return "." if len(span) == 1 else "!"

    return EXCLAMATION_PATTERN.sub(repl, text)


def _capitalize_sentence_starts(text: str) -> str:
    def repl(match: Match) -> str:
        return match.group(1).upper()

    return SENTENCE_START_PATTERN.sub(repl, text)


def preprocess_text(text: str, normalize_exclamation: bool = True) -> str:
    """Normalize text: fix unicode, collapse whitespace, ensure speaker tag."""
    replacements = [
        ("\u2026", ","), ("\u2014", ", "), ("\u2013", ", "),
        (":", ","), (";", ","), ("\n", " "),
        ('"', ""), ("\u201c", ""), ("\u201d", ""),
        ("\u2018", "'"), ("\u2019", "'"), ("\u201a", "'"), ("\u201b", "'"),
        ("\u02bc", "'"), ("\u02b9", "'"), ("\u02bb", "'"), ("\u02be", "'"),
        ("\u02bf", "'"), ("\u2032", "'"), ("\u2035", "'"), ("\uff07", "'"),
        ("\ua78c", "'"), ("*", ""),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    if normalize_exclamation:
        text = _normalize_exclamation(text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = " ".join(text.split())
    text = _lowercase_caps_except_abbreviations(text)
    text = _capitalize_sentence_starts(text)

    if text and not text.startswith("[") and not text.startswith("(") and "S1" not in text and "S2" not in text:
        text = f"[S1] {text}"
    return text


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _is_valid_sentence_end(text: str, index: int) -> bool:
    """Heuristic: is the period at *index* an end-of-sentence marker?"""
    if index > 0 and text[index - 1].isdigit():
        if index + 1 < len(text) and (text[index + 1].isdigit() or text[index + 1] == "."):
            return False
    window_start = max(0, index - 10)
    segment = text[window_start: index + 1].lower()
    words = segment.strip().split()
    if words:
        last = words[-1]
        if not last.endswith("."):
            last += "."
        if last in ABBREVIATIONS:
            return False
    return True


def _split_text_by_punctuation(text: str) -> List[str]:
    sentences: List[str] = []
    last_split, text_len = 0, len(text)
    for match in POTENTIAL_END_PATTERN.finditer(text):
        punct_index, punct_char = match.start(1), text[match.start(1)]
        quote_match = match.group(2)
        quote_len = len(quote_match) if quote_match is not None else 0
        slice_end_index = punct_index + 1 + quote_len

        if punct_char in {"!", "?"}:
            chunk = text[last_split:slice_end_index].strip()
            if chunk:
                sentences.append(chunk)
            last_split = match.end()
            continue

        if punct_char == ".":
            if (punct_index > 0 and text[punct_index - 1] == ".") or (
                punct_index < text_len - 1 and text[punct_index + 1] == "."
            ):
                continue
            if _is_valid_sentence_end(text, punct_index):
                chunk = text[last_split:slice_end_index].strip()
                if chunk:
                    sentences.append(chunk)
                last_split = match.end()

    tail = text[last_split:].strip()
    if tail:
        sentences.append(tail)
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling bullet lists and punctuation."""
    if not text or text.isspace():
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    bullet_matches = list(BULLET_POINT_PATTERN.finditer(text))
    if bullet_matches:
        sentences: List[str] = []
        start_pos = 0
        for i, match in enumerate(bullet_matches):
            bullet_start = match.start()
            if i == 0 and bullet_start > start_pos:
                pre = text[start_pos:bullet_start].strip()
                if pre:
                    sentences.extend(_split_text_by_punctuation(pre))
            bullet_end = bullet_matches[i + 1].start() if i + 1 < len(bullet_matches) else len(text)
            item = text[bullet_start:bullet_end].strip()
            if item:
                sentences.append(item)
            start_pos = bullet_end
        if start_pos < len(text):
            post = text[start_pos:].strip()
            if post:
                sentences.extend(_split_text_by_punctuation(post))
        return [s for s in sentences if s]
    return _split_text_by_punctuation(text)


# ---------------------------------------------------------------------------
# Time-based text chunking for long-form generation
# ---------------------------------------------------------------------------

def _preprocess_and_tag_sentences(full_text: str) -> List[Tuple[str, str]]:
    """Associate sentences with speaker tags (e.g., [S1])."""
    if not full_text or full_text.isspace():
        return []
    tagged: List[Tuple[str, str]] = []
    segments = SPEAKER_TAG_PATTERN.split(full_text)
    current_tag = "[S1]"

    if segments and SPEAKER_TAG_PATTERN.fullmatch(segments[0]):
        current_tag, segments = segments[0], segments[1:]
    elif segments and segments[0] == "" and len(segments) > 1 and SPEAKER_TAG_PATTERN.fullmatch(segments[1]):
        current_tag, segments = segments[1], segments[2:]
    elif segments and segments[0].isspace() and len(segments) > 1 and SPEAKER_TAG_PATTERN.fullmatch(segments[1]):
        current_tag, segments = segments[1], segments[2:]

    buffer = ""
    for segment in segments:
        if not segment:
            continue
        if SPEAKER_TAG_PATTERN.fullmatch(segment):
            if buffer.strip():
                for sentence in split_into_sentences(buffer.strip()):
                    tagged.append((current_tag, sentence))
            current_tag, buffer = segment, ""
        else:
            buffer += segment

    if buffer.strip():
        for sentence in split_into_sentences(buffer.strip()):
            tagged.append((current_tag, sentence))

    if not tagged and full_text.strip():
        cleaned = full_text.strip()
        match = SPEAKER_TAG_PATTERN.match(cleaned)
        start_tag = match.group(1) if match else current_tag
        rest = SPEAKER_TAG_PATTERN.sub("", cleaned, count=1).strip()
        if rest:
            tagged.append((start_tag, rest))
    return tagged


def _estimate_seconds(text: str, chars_per_second: float = 14.0, words_per_second: float = 2.7) -> float:
    """Estimate spoken duration using char and word rate heuristics."""
    chars = len(text)
    words = max(1, len(text.split()))
    char_est = chars / chars_per_second if chars_per_second > 0 else chars / 4.0
    word_est = words / words_per_second if words_per_second > 0 else words / 2.5
    return max(char_est, word_est)


def format_chunks(chunks: List[str]) -> str:
    """Format chunk list into the display format with **Chunk N:** markers."""
    if len(chunks) == 1:
        return chunks[0]
    return "\n\n".join(f"**Chunk {i+1}:** {c}" for i, c in enumerate(chunks))


def parse_chunked_text(text: str) -> List[str] | None:
    """If text contains **Chunk N:** markers, split on them and return chunks.

    Returns None if the text is not pre-chunked.
    """
    if not CHUNK_MARKER_PATTERN.search(text):
        return None
    parts = CHUNK_MARKER_PATTERN.split(text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks if chunks else None


def chunk_text_by_time(
    full_text: str,
    target_seconds: float = 28.0,
    min_seconds: float = 18.0,
    max_seconds: float = 35.0,
    chars_per_second: float = 14.0,
    words_per_second: float = 2.7,
) -> List[str]:
    """Chunk text into ~time-sized pieces suitable for single-shot generation.

    Each chunk is preprocessed and tagged with speaker markers.
    Target ~28s keeps us well within the model's 30s training window.
    """
    # If the text already has chunk markers (user pre-processed), use those directly
    pre_chunked = parse_chunked_text(full_text)
    if pre_chunked is not None:
        return pre_chunked

    cleaned = preprocess_text(full_text)
    tagged_sentences = _preprocess_and_tag_sentences(cleaned)
    if not tagged_sentences:
        return []

    chunks: List[str] = []
    chunk_sentences: List[str] = []
    chunk_secs = 0.0
    first_tag: str | None = None
    last_tag: str | None = None

    def flush_chunk() -> None:
        nonlocal chunk_sentences, chunk_secs, first_tag, last_tag
        if chunk_sentences and first_tag:
            chunks.append(" ".join(chunk_sentences))
        chunk_sentences = []
        chunk_secs = 0.0
        first_tag = None
        last_tag = None

    for sentence_tag, sentence_text in tagged_sentences:
        sent_secs = _estimate_seconds(sentence_text, chars_per_second, words_per_second)

        if not chunk_sentences:
            first_tag = sentence_tag
            last_tag = sentence_tag
            chunk_sentences = [f"{sentence_tag} {sentence_text}"]
            chunk_secs = sent_secs
            if sent_secs >= max_seconds:
                flush_chunk()
            continue

        projected = chunk_secs + sent_secs
        if (projected > target_seconds and chunk_secs >= min_seconds) or projected > max_seconds:
            flush_chunk()
            first_tag = sentence_tag
            last_tag = sentence_tag
            chunk_sentences = [f"{sentence_tag} {sentence_text}"]
            chunk_secs = sent_secs
            if sent_secs >= max_seconds:
                flush_chunk()
            continue

        if sentence_tag != last_tag:
            chunk_sentences.append(f"{sentence_tag} {sentence_text}")
        else:
            chunk_sentences.append(sentence_text)
        last_tag = sentence_tag
        chunk_secs = projected

    flush_chunk()
    return chunks
