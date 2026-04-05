"""
text_utils.py

Pure text transformation functions with no domain-specific imports.
Extracted from tools/text_extractor.py and extraction/trace_capturer.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from utils.data_structures import Block, TransformConfig


# ======================================================================
# Constants
# ======================================================================

COMPOUND_PREFIXES = {
    "self", "well", "non", "pre", "post", "co", "re", "anti", "semi",
    "multi", "inter", "intra", "cross", "over", "under", "out", "up",
    "all", "ex", "half", "mid", "sub", "super", "ultra", "vice",
}


# ======================================================================
# Markdown parsing
# ======================================================================

def parse_markdown_blocks(md_text: str) -> list:
    """Parse markdown into structured blocks."""
    blocks = []
    lines = md_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            blocks.append(Block(block_type="heading", level=level, text=heading_match.group(2).strip(), raw=line))
            i += 1
            continue

        if line.startswith('```'):
            code_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])
                i += 1
            raw = '\n'.join(code_lines)
            content = '\n'.join(code_lines[1:-1]) if len(code_lines) > 2 else ""
            blocks.append(Block(block_type="code", level=None, text=content, raw=raw))
            continue

        img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$', line)
        if img_match:
            blocks.append(Block(block_type="image", level=None, text=img_match.group(1), raw=line))
            i += 1
            continue

        list_match = re.match(r'^(\s*)([-*+])\s+(.+)$', line)
        if list_match:
            indent = len(list_match.group(1))
            level = indent // 2 + 1
            blocks.append(Block(block_type="list_item", level=level, text=list_match.group(3).strip(), raw=line))
            i += 1
            continue

        olist_match = re.match(r'^(\s*)(\d+)\.\s+(.+)$', line)
        if olist_match:
            indent = len(olist_match.group(1))
            level = indent // 2 + 1
            blocks.append(Block(block_type="ordered_list_item", level=level, text=olist_match.group(3).strip(), raw=line))
            i += 1
            continue

        if re.match(r'^(-{3,}|_{3,}|\*{3,})\s*$', line):
            blocks.append(Block(block_type="horizontal_rule", level=None, text="", raw=line))
            i += 1
            continue

        if line.startswith('>'):
            quote_lines = []
            quote_start = i
            while i < len(lines) and lines[i].startswith('>'):
                quote_lines.append(lines[i].lstrip('> '))
                i += 1
            raw_lines = lines[quote_start:i] if quote_lines else []
            blocks.append(Block(block_type="blockquote", level=None, text='\n'.join(quote_lines), raw='\n'.join(raw_lines)))
            continue

        if not line.strip():
            i += 1
            continue

        para_lines = []
        while i < len(lines):
            l = lines[i]
            if (not l.strip() or
                re.match(r'^#{1,6}\s+', l) or
                l.startswith('```') or
                re.match(r'^!\[', l) or
                re.match(r'^\s*[-*+]\s+', l) or
                re.match(r'^\s*\d+\.\s+', l) or
                re.match(r'^(-{3,}|_{3,}|\*{3,})\s*$', l) or
                l.startswith('>')):
                break
            para_lines.append(l)
            i += 1

        if para_lines:
            raw = '\n'.join(para_lines)
            blocks.append(Block(block_type="paragraph", level=None, text=raw, raw=raw))

    return blocks


def markdown_to_plain_text(md: str) -> str:
    """Strip Markdown formatting to get plain text."""
    text = md
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'___([^_]+)___', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```[^\n]*\n?', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^(-{3,}|_{3,}|\*{3,})\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ======================================================================
# Text cleanup transforms
# ======================================================================

def _load_dictionary(config: TransformConfig) -> set:
    """Load a word list for dictionary-based hyphen validation."""
    if config.dictionary_path:
        dict_path = Path(config.dictionary_path)
        if dict_path.exists():
            return {w.strip().lower() for w in dict_path.read_text().splitlines()}
    system_dict = Path("/usr/share/dict/words")
    if system_dict.exists():
        return {w.strip().lower() for w in system_dict.read_text().splitlines()}
    return set()


def collapse_lettertracks(text: str, min_length: int = 4) -> str:
    """Collapse lettertracked text (e.g., 'C o m p r e h e n s i o n' -> 'Comprehension')."""
    def _collapse_segment(segment: str) -> str:
        chars = segment.split(" ")
        if all(len(c) == 1 and c.isalpha() for c in chars) and len(chars) >= min_length:
            return "".join(chars)
        return segment

    def _collapse_run(match: re.Match) -> str:
        full = match.group(0)
        segments = full.split("  ")
        collapsed = [_collapse_segment(seg.strip()) for seg in segments]
        result = " ".join(collapsed)
        if result != full:
            return result
        return full

    pattern = r"(?<![a-zA-Z])(?:[A-Za-z] ){3,}[A-Za-z](?:  (?:[A-Za-z] ){2,}[A-Za-z])*(?![a-zA-Z])"
    return re.sub(pattern, _collapse_run, text)


def rejoin_hyphens(text: str, config: TransformConfig) -> str:
    """Rejoin words split by soft hyphens (line-break artifacts)."""
    dictionary = _load_dictionary(config) if config.should_use_dictionary else set()

    def _should_rejoin(left: str, right: str) -> bool:
        if right[0].isupper():
            return False
        if left.lower() in COMPOUND_PREFIXES:
            return False
        if dictionary:
            joined = (left + right).lower()
            return joined in dictionary
        return True

    def _rejoin_match(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if _should_rejoin(left, right):
            return left + right
        return match.group(0)

    text = re.sub(r"(\w{2,})-\s*\n\s*(\w{2,})(?!-)", _rejoin_match, text)
    text = re.sub(r"(\w{2,})-([a-z]\w+)(?!-)", _rejoin_match, text)
    return text


def normalize_block_spacing(text: str) -> str:
    """Insert missing spaces between concatenated text blocks."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.!?;:)])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d+)([A-Z])", r"\1 \2 \3", text)
    text = re.sub(r"(\d)([A-Z])(?=\s)", r"\1 \2", text)
    return text


def clean_markdown_artifacts(text: str) -> str:
    """Fix markdown-specific extraction artifacts."""
    text = re.sub(r"\*\*\*\*", "** **", text)
    text = re.sub(r"\*\*\s+\*\*", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Final whitespace cleanup."""
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def apply_transforms(text: str, config: TransformConfig) -> str:
    """Apply the full text cleanup pipeline."""
    if config.should_collapse_lettertracks:
        text = collapse_lettertracks(text, config.min_lettertrack_length)
    if config.should_rejoin_hyphens:
        text = rejoin_hyphens(text, config)
    if config.should_normalize_block_spacing:
        text = normalize_block_spacing(text)
    if config.should_clean_markdown_artifacts and config.is_markdown:
        text = clean_markdown_artifacts(text)
    if config.should_normalize_whitespace:
        text = normalize_whitespace(text)
    return text


# ======================================================================
# Response text utilities (extracted from trace_capturer.py)
# ======================================================================

def extract_thinking(text: str) -> tuple:
    """Extract <think>...</think> reasoning block and remaining text.

    Args:
        text: Raw model response that may contain <think>...</think> blocks.

    Returns:
        Tuple of (thinking_text, remaining_text). If no think block is found,
        thinking_text is empty and remaining_text is the original text.
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        remaining = text[:think_match.start()] + text[think_match.end():]
        return thinking, remaining.strip()
    return "", text


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return text.strip()
