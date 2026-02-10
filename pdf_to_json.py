#!/usr/bin/env python3
"""
PDF to Structured JSON Converter for Savarkar GPT Project
==========================================================
Converts 6 PDF books into clean, structured JSON files with preprocessing:
  - OCR artifact cleanup (spaced characters, misrecognitions)
  - Page marker removal
  - Blank page/line cleanup
  - Chapter-level structuring
  - Metadata extraction
  - Cross-document duplicate detection

Usage:
    python3 pdf_to_json.py
    python3 pdf_to_json.py --check-duplicates   # Only run duplicate check
    python3 pdf_to_json.py --skip-duplicates     # Skip duplicate check
"""

import fitz  # PyMuPDF
import json
import re
import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


# ============================================================
# CONFIGURATION
# ============================================================

PDF_DIR = Path(__file__).parent
OUTPUT_DIR = PDF_DIR / "json_output"

# Manual chapter title corrections for books with OCR-garbled titles
KEER_CHAPTER_TITLES = {
    1: "Childhood and Youth",
    2: "The Rising Leader",
    3: "Revolutionary Activities",
    4: "The Storm Breaks",
    5: "Epic Escape and Trials",
    6: "The Indian Bastille",
    7: "Genius Thrives in Jail",
    8: "Release and Internment",
    9: "Social Revolution",
    10: "Nationalist and Author",
    11: "Back to Freedom",
    12: "Whirlwind Propaganda",
    13: "War and Militarisation",
    14: "Hindu Manifesto",
    15: "Attacks Gandhi and Jinnah",
    16: "Cripps Mission",
    17: "Mahasabha Marches On",
    18: "The Writing on the Wall",
    19: "Fight for a United India",
    20: "From Parity to Pakistan",
    21: "The Red Fort Trial",
    22: "Detention and Internment",
    23: "Memorial to Martyrs",
    24: "The Menace of Christianstans",
    25: "Old Age",
    26: "Warning Against Aggression",
    27: "Nation Pays Homage",
    28: "The Eternal Hero",
}

# PDF metadata configuration
PDF_CONFIGS = {
    "6_Glorious_Epochs_of_Indian_History.pdf": {
        "title": "Six Glorious Epochs of Indian History",
        "author": "Vinayak Damodar Savarkar",
        "translator": "S. T. Godbole",
        "year": 1971,
        "language": "English (translated from Marathi)",
        "category": "history",
        "description": "A commentary on six significant epochs of Indian history from Chandragupta Maurya to the end of British dominance.",
        "ocr_quality": "poor",  # needs heavy preprocessing
    },
    "Hindupadpatshahi Eng.pdf": {
        "title": "Hindu Pad-Padashahi",
        "author": "Vinayak Damodar Savarkar",
        "translator": None,
        "year": None,
        "language": "English",
        "category": "history",
        "description": "History of the rise and fall of the Maratha Hindu Empire, focusing on the period from Shivaji to the fall of the Peshwas.",
        "ocr_quality": "moderate",
    },
    "Hindurashtra Darshan.pdf": {
        "title": "Hindu Rashtra Darshan",
        "author": "Vinayak Damodar Savarkar",
        "translator": None,
        "year": None,
        "language": "English",
        "category": "political_speeches",
        "description": "Collection of presidential addresses delivered at sessions of the Akhil Bharatiya Hindu Mahasabha.",
        "ocr_quality": "good",
    },
    "Savarkar (part 2) A Contested Legacy, 1924-1966 by Vikram Sampath (z-lib.org).pdf": {
        "title": "Savarkar: A Contested Legacy, 1924-1966",
        "author": "Vikram Sampath",
        "translator": None,
        "year": 2021,
        "language": "English",
        "category": "biography",
        "description": "Second volume of Vikram Sampath's biography of Savarkar, covering 1924-1966.",
        "ocr_quality": "excellent",
    },
    "Savarkar_-_Vikram_Sampath.pdf": {
        "title": "Savarkar: Echoes from a Forgotten Past, 1883-1924",
        "author": "Vikram Sampath",
        "translator": None,
        "year": 2019,
        "language": "English",
        "category": "biography",
        "description": "First volume of Vikram Sampath's biography of Savarkar, covering 1883-1924.",
        "ocr_quality": "excellent",
    },
    "veer savarkar dhananjay keer.pdf": {
        "title": "Veer Savarkar",
        "author": "Dhananjay Keer",
        "translator": None,
        "year": 1966,
        "language": "English",
        "category": "biography",
        "description": "Complete biography of Veer Savarkar by historian Dhananjay Keer.",
        "ocr_quality": "moderate",
    },
}


# ============================================================
# TEXT PREPROCESSING
# ============================================================

class TextPreprocessor:
    """Handles all text cleaning and OCR artifact removal."""

    # Pattern: "-- X of Y --" page markers
    PAGE_MARKER_RE = re.compile(r'\n*--\s*\d+\s+of\s+\d+\s*--\n*')

    # Pattern: Heavily spaced text like "V . D . S a v a r k a r"
    # Matches sequences where single chars are separated by spaces/dots
    SPACED_CHARS_RE = re.compile(r'(?<!\w)([A-Za-z])\s*\.\s*(?=[A-Za-z]\s*\.)')

    # Pattern: Lines that are just page numbers
    PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)

    # Pattern: Multiple blank lines
    MULTI_BLANK_RE = re.compile(r'\n{3,}')

    # Pattern: Lines with heavy character spacing (e.g., "C O N T E N T S")
    HEAVY_SPACING_RE = re.compile(
        r'\b([A-Z])\s+([A-Z])\s+([A-Z])\s+([A-Z])(?:\s+([A-Z]))?(?:\s+([A-Z]))?(?:\s+([A-Z]))?(?:\s+([A-Z]))?(?:\s+([A-Z]))?(?:\s+([A-Z]))?(?:\s+([A-Z]))?(?:\s+([A-Z]))?(?:\s+([A-Z]))?\b'
    )

    @classmethod
    def clean_text(cls, text: str, ocr_quality: str = "good") -> str:
        """Main cleaning pipeline."""
        # Step 1: Remove page markers
        text = cls.remove_page_markers(text)

        # Step 2: Fix character spacing (for poor OCR)
        if ocr_quality in ("poor", "moderate"):
            text = cls.fix_character_spacing(text)

        # Step 3: Remove standalone page numbers
        text = cls.PAGE_NUM_RE.sub('', text)

        # Step 4: Fix common OCR errors
        text = cls.fix_ocr_errors(text)

        # Step 5: Normalize whitespace
        text = cls.normalize_whitespace(text)

        # Step 6: Fix broken words across lines
        text = cls.fix_broken_words(text)

        # Step 7: Final cleanup
        text = cls.final_cleanup(text)

        return text.strip()

    @classmethod
    def remove_page_markers(cls, text: str) -> str:
        """Remove '-- X of Y --' page markers."""
        return cls.PAGE_MARKER_RE.sub('\n', text)

    @classmethod
    def fix_character_spacing(cls, text: str) -> str:
        """
        Fix text with excessive character spacing.
        E.g., "C O N T E N T S" -> "CONTENTS"
             "V . D . S a v a r k a r" -> "V.D. Savarkar"
             "W h e n I saw them" -> "When I saw them"
        """
        lines = text.split('\n')
        fixed_lines = []

        for line in lines:
            stripped = line.strip()

            # Detect lines where most characters are separated by single spaces
            if cls._is_heavily_spaced(stripped):
                fixed = cls._collapse_spaced_text(stripped)
                fixed_lines.append(fixed)
            else:
                # Even for non-heavily-spaced lines, fix inline spaced words
                # Pattern: 3+ single letters separated by spaces (e.g., "W h e n", "T h e n")
                fixed = cls._fix_inline_spaced_words(stripped)
                fixed_lines.append(fixed)

        return '\n'.join(fixed_lines)

    @classmethod
    def _fix_inline_spaced_words(cls, line: str) -> str:
        """
        Fix individual spaced words within an otherwise normal line.
        E.g., "W h e n I saw them" -> "When I saw them"
              "the H i n d u society" -> "the Hindu society"
              "a n d power" -> "and power"
        """
        # Match sequences of 3+ single letters separated by single spaces.
        # We require at least 3 letters to avoid false positives with real
        # two-letter combinations like "a I" or "I a".
        def collapse_match(m):
            chars = m.group(0).split()
            collapsed = ''.join(chars)
            # Preserve trailing space if the match was followed by more text
            return collapsed

        # Pattern: 3 or more single alpha chars separated by exactly one space each
        # Lookahead/behind ensure we don't eat into real words
        result = re.sub(
            r'(?<![a-zA-Z])([a-zA-Z] ){2,}[a-zA-Z](?![a-zA-Z])',
            collapse_match,
            line
        )

        # Fix any collapsed words that got stuck to the next word (e.g., "WhenI" -> "When I")
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)

        return result

    @classmethod
    def _is_heavily_spaced(cls, line: str) -> bool:
        """Check if a line has heavy character spacing from OCR."""
        if len(line) < 5:
            return False

        # Count single-char words (letters separated by spaces)
        words = line.split()
        if not words:
            return False

        single_char_words = sum(1 for w in words if len(w) == 1 and w.isalpha())
        ratio = single_char_words / len(words)

        # If more than 40% of "words" are single characters, it's likely spaced
        return ratio > 0.4 and len(words) > 3

    @classmethod
    def _collapse_spaced_text(cls, line: str) -> str:
        """Collapse heavily spaced text back to normal."""
        result = []
        words = line.split()
        i = 0
        current_word = []

        while i < len(words):
            token = words[i]

            if len(token) == 1 and token.isalpha():
                current_word.append(token)
            else:
                if current_word:
                    result.append(''.join(current_word))
                    current_word = []
                result.append(token)
            i += 1

        if current_word:
            result.append(''.join(current_word))

        return ' '.join(result)

    @classmethod
    def fix_ocr_errors(cls, text: str) -> str:
        """Fix common OCR misrecognitions."""
        replacements = [
            # Common OCR substitutions
            (r'(?<!\w)0(?=\w)', 'O'),     # 0 -> O at word start
            (r'(?<=\w)0(?!\w)', 'o'),     # 0 -> o at word end (context-dependent)
            (r'\bl\b', 'I'),               # standalone l -> I
            (r'(?<=[a-z])I(?=[a-z])', 'l'), # I between lowercase -> l
            # Fix broken hyphens
            (r'(\w)¬\s*\n\s*(\w)', r'\1\2'),  # word¬\nword -> word+word
        ]

        for pattern, replacement in replacements:
            try:
                text = re.sub(pattern, replacement, text)
            except re.error:
                pass

        return text

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Normalize excessive whitespace."""
        # Replace multiple spaces with single space (but preserve newlines)
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Replace 3+ newlines with 2
        text = cls.MULTI_BLANK_RE.sub('\n\n', text)
        return text

    @classmethod
    def fix_broken_words(cls, text: str) -> str:
        """Rejoin words broken across lines by hyphens."""
        # Pattern: word ending with hyphen at end of line, continued on next line
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text

    @classmethod
    def final_cleanup(cls, text: str) -> str:
        """Final pass of cleanup."""
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # Remove multiple consecutive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text


# ============================================================
# CHAPTER DETECTION
# ============================================================

class ChapterDetector:
    """Detects chapter boundaries in text."""

    # Pattern for TOC entries like "1. Title", "2. Title"
    TOC_ENTRY_RE = re.compile(r'^(\d{1,2})\.\s+(.+)', re.MULTILINE)

    # Pattern for "CHAPTER X" followed by title on next line
    CHAPTER_BLOCK_RE = re.compile(
        r'^(?:CHAPTER|Chapter)\s+(\d+[o]?)\s*\n\s*(.+?)(?:\n|$)',
        re.MULTILINE
    )

    # Pattern for "Chapter X. Title" on same line
    CHAPTER_INLINE_RE = re.compile(
        r'^(?:CHAPTER|Chapter)\s+([IVXLCDM]+|\d+)[.:\s]+([A-Z][\w\s,\-\'\"]+)',
        re.MULTILINE
    )

    @classmethod
    def detect_chapters(cls, text: str, category: str = "general",
                        title_corrections: dict = None, source_file: str = "") -> list:
        """
        Detect chapter boundaries and return structured chapters.

        Strategy:
        1. Try "CHAPTER X\\nTitle" block pattern (Keer-style)
        2. Try TOC-based detection with body title matching
        3. Try "Chapter X. Title" inline pattern
        4. Fall back to single document
        """
        # Strategy 1: "CHAPTER X\nTitle" block pattern (common in older books)
        chapters = cls._try_chapter_block_headings(text, title_corrections)
        if chapters and cls._chapters_have_content(chapters):
            return chapters

        # Strategy 2: TOC-based detection - find titles in body text
        chapters = cls._try_toc_based_detection(text)
        if chapters and cls._chapters_have_content(chapters):
            return chapters

        # Strategy 3: Inline chapter headings "Chapter X. Title"
        chapters = cls._try_chapter_inline_headings(text)
        if chapters and cls._chapters_have_content(chapters):
            return chapters

        # Strategy 4: Section-based detection (numbered sections in body)
        chapters = cls._try_section_headings(text)
        if chapters and cls._chapters_have_content(chapters):
            return chapters

        # Fallback: single document
        return [{
            "chapter_number": 1,
            "chapter_title": "Full Text",
            "content": text
        }]

    @classmethod
    def _chapters_have_content(cls, chapters: list) -> bool:
        """Verify that detected chapters actually have meaningful content distribution."""
        if not chapters:
            return False
        content_chapters = [ch for ch in chapters if len(ch.get("content", "")) > 500]
        # At least half the chapters should have substantial content
        return len(content_chapters) >= max(2, len(chapters) // 3)

    @classmethod
    def _try_chapter_block_headings(cls, text: str, title_corrections: dict = None) -> list:
        """
        Detect 'CHAPTER X\\nTitle' style headings (number on one line, title on next).
        Used by Dhananjay Keer and similar older books.
        """
        matches = list(cls.CHAPTER_BLOCK_RE.finditer(text))
        if len(matches) < 3:
            return []

        chapters = []

        # Add preamble
        first_start = matches[0].start()
        if first_start > 100:
            preamble = text[:first_start].strip()
            if len(preamble) > 50:
                chapters.append({
                    "chapter_number": 0,
                    "chapter_title": "Preamble / Front Matter",
                    "content": preamble
                })

        for i, match in enumerate(matches):
            raw_num = match.group(1).replace('o', '0')  # OCR: 'o' -> '0'
            try:
                chapter_num = int(raw_num)
            except ValueError:
                chapter_num = i + 1

            raw_title = match.group(2).strip()

            # Apply title corrections if available
            if title_corrections and chapter_num in title_corrections:
                chapter_title = title_corrections[chapter_num]
            else:
                chapter_title = raw_title

            # Content is from after this heading to the start of the next chapter
            content_start = match.end()
            if i + 1 < len(matches):
                content_end = matches[i + 1].start()
            else:
                content_end = len(text)

            content = text[content_start:content_end].strip()

            chapters.append({
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "content": content
            })

        return chapters

    @classmethod
    def _try_chapter_inline_headings(cls, text: str) -> list:
        """Try to detect 'Chapter I/1. Title' style headings on same line."""
        matches = list(cls.CHAPTER_INLINE_RE.finditer(text))
        if len(matches) < 3:
            return []

        chapters = []
        first_start = matches[0].start()
        if first_start > 100:
            preamble = text[:first_start].strip()
            if len(preamble) > 50:
                chapters.append({
                    "chapter_number": 0,
                    "chapter_title": "Preamble / Front Matter",
                    "content": preamble
                })

        for i, match in enumerate(matches):
            chapter_id = match.group(1)
            chapter_title = match.group(2).strip()

            try:
                chapter_num = cls._roman_to_int(chapter_id) if chapter_id.isalpha() else int(chapter_id)
            except (ValueError, KeyError):
                chapter_num = i + 1

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            chapters.append({
                "chapter_number": chapter_num,
                "chapter_title": chapter_title.strip().rstrip('.').rstrip(':'),
                "content": content
            })

        return chapters

    @classmethod
    def _try_toc_based_detection(cls, text: str) -> list:
        """
        Parse table of contents, then find chapter titles in body text
        as standalone lines (without number prefix).
        """
        # Find the TOC section
        toc_start = -1
        for marker in ["CONTENTS", "Contents", "TABLE OF CONTENTS", "Table of Contents"]:
            idx = text.find(marker)
            if idx >= 0:
                toc_start = idx
                break

        if toc_start < 0:
            return []

        # Extract TOC entries
        toc_region = text[toc_start:toc_start + 5000]
        toc_matches = list(cls.TOC_ENTRY_RE.finditer(toc_region))

        if len(toc_matches) < 3:
            return []

        # Extract chapter titles from TOC
        toc_titles = []
        for m in toc_matches:
            num = int(m.group(1))
            title = m.group(2).strip().rstrip('.')
            # Remove page numbers at the end
            title = re.sub(r'\s+\d+\s*[-–—]\s*\d+\s*$', '', title)
            title = re.sub(r'\s+\d+\s*$', '', title)
            toc_titles.append((num, title))

        # Find body start: skip past TOC and any front matter markers
        last_toc_entry = toc_matches[-1]
        body_search_start = toc_start + last_toc_entry.end()

        # Look for the first chapter title in the body text (after the TOC listing)
        # Skip at least 500 chars past the TOC to avoid matching TOC entries themselves
        body_text_start = body_search_start + 200

        # Also check for Foreword/Preface/Prologue as pre-chapter content
        preamble_markers = ["Author's Foreword", "Foreword", "Preface", "Prologue",
                            "Introduction", "Advance Praise"]
        preamble_start = None
        for marker in preamble_markers:
            idx = text.find(marker, body_text_start)
            if idx >= 0 and idx < body_text_start + 10000:
                if preamble_start is None or idx < preamble_start:
                    preamble_start = idx

        # Now find each chapter title in the body as a standalone line
        chapter_positions = []

        for num, title in toc_titles:
            # Search for the title as a standalone line in the body
            # Must be after the TOC section
            search_start = body_text_start

            # Try multiple matching strategies
            found = False

            # Strategy A: Exact title on its own line (or near-own line)
            escaped_title = re.escape(title)
            patterns = [
                re.compile(r'\n\s*' + escaped_title + r'\s*\n'),
                re.compile(r'\n\s*' + escaped_title + r'\s*\n', re.IGNORECASE),
            ]

            for pattern in patterns:
                match = pattern.search(text, pos=search_start)
                if match:
                    # Verify this isn't in the Notes/Bibliography section
                    # (crude heuristic: not in the last 20% of the document)
                    if match.start() < len(text) * 0.85:
                        chapter_positions.append((num, title, match.start()))
                        found = True
                        break

            if not found:
                # Strategy B: First significant words of title
                words = title.split()
                if len(words) >= 2:
                    short = re.escape(' '.join(words[:3]))
                    pattern = re.compile(r'\n\s*' + short, re.IGNORECASE)
                    match = pattern.search(text, pos=search_start)
                    if match and match.start() < len(text) * 0.85:
                        chapter_positions.append((num, title, match.start()))

        if len(chapter_positions) < max(3, len(toc_titles) // 2):
            return []

        # Sort by position
        chapter_positions.sort(key=lambda x: x[2])

        # Build chapters
        chapters = []

        # Add front matter / preamble
        first_ch_pos = chapter_positions[0][2]
        if preamble_start and preamble_start < first_ch_pos:
            preamble_content = text[preamble_start:first_ch_pos].strip()
        elif first_ch_pos > body_text_start + 100:
            preamble_content = text[body_text_start:first_ch_pos].strip()
        else:
            preamble_content = ""

        if len(preamble_content) > 100:
            chapters.append({
                "chapter_number": 0,
                "chapter_title": "Preamble / Front Matter",
                "content": preamble_content
            })

        for i, (num, title, pos) in enumerate(chapter_positions):
            # Content starts after the title line
            title_end = text.find('\n', pos + 1)
            if title_end < 0:
                title_end = pos + len(title)
            content_start = title_end + 1

            # End at next chapter or end of text (but exclude notes/bibliography)
            if i + 1 < len(chapter_positions):
                content_end = chapter_positions[i + 1][2]
            else:
                # For the last chapter, try to exclude notes/bibliography
                notes_markers = ["\nNotes\n", "\nBibliography\n", "\nAppendix",
                                 "\nAPPENDIX", "\nACKNOWLEDGEMENTS", "\nAcknowledgements"]
                content_end = len(text)
                for marker in notes_markers:
                    idx = text.find(marker, content_start)
                    if idx >= 0 and idx < content_end:
                        content_end = idx

            content = text[content_start:content_end].strip()

            chapters.append({
                "chapter_number": num,
                "chapter_title": title,
                "content": content
            })

        return chapters

    @classmethod
    def _try_section_headings(cls, text: str) -> list:
        """
        Try to find numbered section headings in the body text.
        Pattern: standalone lines like "1. Title Text" with substantial content between them.
        """
        pattern = re.compile(r'^\s*(\d{1,2})\.\s+([A-Z][A-Za-z\s,\-\'\"&;:]+?)\s*$', re.MULTILINE)
        matches = list(pattern.finditer(text))

        if len(matches) < 3:
            return []

        # Filter: only keep matches where there's substantial content between them
        good_matches = []
        for i, match in enumerate(matches):
            if i + 1 < len(matches):
                gap = matches[i + 1].start() - match.end()
                if gap > 500:
                    good_matches.append(match)
            else:
                remaining = len(text) - match.end()
                if remaining > 500:
                    good_matches.append(match)

        if len(good_matches) < 3:
            return []

        chapters = []
        first_start = good_matches[0].start()
        if first_start > 100:
            preamble = text[:first_start].strip()
            if len(preamble) > 50:
                chapters.append({
                    "chapter_number": 0,
                    "chapter_title": "Preamble / Front Matter",
                    "content": preamble
                })

        for i, match in enumerate(good_matches):
            chapter_id = match.group(1)
            chapter_title = match.group(2).strip()
            try:
                chapter_num = int(chapter_id)
            except ValueError:
                chapter_num = i + 1

            start = match.end()
            end = good_matches[i + 1].start() if i + 1 < len(good_matches) else len(text)
            content = text[start:end].strip()

            chapters.append({
                "chapter_number": chapter_num,
                "chapter_title": chapter_title.strip().rstrip('.').rstrip(':'),
                "content": content
            })

        return chapters

    @staticmethod
    def _roman_to_int(s: str) -> int:
        """Convert Roman numeral to integer."""
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        s = s.upper()
        result = 0
        prev = 0
        for c in reversed(s):
            val = roman_values.get(c, 0)
            if val < prev:
                result -= val
            else:
                result += val
            prev = val
        return result


# ============================================================
# DUPLICATE DETECTION
# ============================================================

class DuplicateDetector:
    """Detects duplicate/overlapping content across documents."""

    @staticmethod
    def compute_text_fingerprints(text: str, chunk_size: int = 200) -> set:
        """
        Create a set of fingerprints for text chunks.
        Uses rolling hash of normalized text chunks.
        """
        # Normalize: lowercase, collapse whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        fingerprints = set()

        words = normalized.split()
        for i in range(0, len(words) - chunk_size, chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            fp = hashlib.md5(chunk.encode()).hexdigest()
            fingerprints.add(fp)

        return fingerprints

    @classmethod
    def check_pairwise_overlap(cls, documents: dict) -> list:
        """
        Check all pairs of documents for content overlap.
        Returns list of overlap reports.
        """
        print("\n" + "=" * 60)
        print("DUPLICATE / OVERLAP DETECTION")
        print("=" * 60)

        doc_fingerprints = {}
        for name, text in documents.items():
            doc_fingerprints[name] = cls.compute_text_fingerprints(text)
            print(f"  Fingerprinted: {name} ({len(doc_fingerprints[name])} chunks)")

        overlaps = []
        names = list(documents.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a, name_b = names[i], names[j]
                fp_a, fp_b = doc_fingerprints[name_a], doc_fingerprints[name_b]

                if not fp_a or not fp_b:
                    continue

                common = fp_a & fp_b
                total = min(len(fp_a), len(fp_b))
                overlap_pct = (len(common) / total * 100) if total > 0 else 0

                report = {
                    "document_a": name_a,
                    "document_b": name_b,
                    "common_chunks": len(common),
                    "overlap_percentage": round(overlap_pct, 2),
                    "verdict": "UNIQUE" if overlap_pct < 5 else ("LOW OVERLAP" if overlap_pct < 15 else "SIGNIFICANT OVERLAP")
                }
                overlaps.append(report)

                icon = "✅" if overlap_pct < 5 else ("⚠️" if overlap_pct < 15 else "❌")
                print(f"\n  {icon} {name_a}")
                print(f"     vs {name_b}")
                print(f"     Overlap: {overlap_pct:.1f}% ({len(common)} common chunks) -> {report['verdict']}")

        return overlaps


# ============================================================
# PDF TO JSON CONVERTER
# ============================================================

class PDFToJSONConverter:
    """Main converter class."""

    def __init__(self, pdf_dir: Path, output_dir: Path):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract raw text from PDF using PyMuPDF."""
        print(f"\n  Extracting text from: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                text_parts.append(text)

        page_count = len(doc)
        doc.close()
        full_text = '\n'.join(text_parts)
        print(f"    -> Extracted {len(full_text):,} characters from {page_count} pages")
        return full_text

    def process_single_pdf(self, pdf_filename: str) -> dict:
        """Process a single PDF into structured JSON."""
        pdf_path = self.pdf_dir / pdf_filename
        config = PDF_CONFIGS.get(pdf_filename, {})

        print(f"\n{'='*60}")
        print(f"PROCESSING: {pdf_filename}")
        print(f"{'='*60}")

        # Step 1: Extract raw text
        raw_text = self.extract_text_from_pdf(pdf_path)
        raw_char_count = len(raw_text)

        # Step 2: Preprocess/clean text
        print(f"  Preprocessing (OCR quality: {config.get('ocr_quality', 'unknown')})...")
        ocr_quality = config.get("ocr_quality", "good")
        cleaned_text = TextPreprocessor.clean_text(raw_text, ocr_quality)
        cleaned_char_count = len(cleaned_text)
        reduction = ((raw_char_count - cleaned_char_count) / raw_char_count * 100) if raw_char_count > 0 else 0
        print(f"    -> Cleaned: {raw_char_count:,} -> {cleaned_char_count:,} chars ({reduction:.1f}% reduction)")

        # Step 3: Detect chapters
        print(f"  Detecting chapters...")
        category = config.get("category", "general")

        # Determine title corrections for books with OCR issues
        title_corrections = None
        if "dhananjay keer" in pdf_filename.lower():
            title_corrections = KEER_CHAPTER_TITLES

        chapters = ChapterDetector.detect_chapters(
            cleaned_text, category,
            title_corrections=title_corrections,
            source_file=pdf_filename
        )
        print(f"    -> Found {len(chapters)} chapters/sections")

        # Step 4: Compute statistics
        word_count = len(cleaned_text.split())
        chapter_summaries = []
        for ch in chapters:
            ch_words = len(ch["content"].split())
            chapter_summaries.append({
                "chapter_number": ch["chapter_number"],
                "chapter_title": ch["chapter_title"],
                "word_count": ch_words,
                "char_count": len(ch["content"]),
            })

        # Step 5: Split chapter content into paragraphs for better searchability
        chapters_with_paragraphs = []
        for ch in chapters:
            # Split content into paragraphs (on double newlines or single newlines with indent)
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', ch["content"]) if p.strip()]
            # Further split any remaining very long paragraphs (>2000 chars) on single newlines
            final_paragraphs = []
            for p in paragraphs:
                if len(p) > 2000:
                    sub_parts = [s.strip() for s in p.split('\n') if s.strip()]
                    final_paragraphs.extend(sub_parts)
                else:
                    final_paragraphs.append(p)

            chapters_with_paragraphs.append({
                "chapter_number": ch["chapter_number"],
                "chapter_title": ch["chapter_title"],
                "word_count": len(ch["content"].split()),
                "paragraphs": final_paragraphs,
            })

        # Split full_text into paragraphs too
        full_text_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', cleaned_text) if p.strip()]

        # Step 6: Build structured JSON
        result = {
            "metadata": {
                "title": config.get("title", pdf_filename),
                "author": config.get("author", "Unknown"),
                "translator": config.get("translator"),
                "publication_year": config.get("year"),
                "language": config.get("language", "English"),
                "category": config.get("category", "unknown"),
                "description": config.get("description", ""),
                "source_file": pdf_filename,
                "total_word_count": word_count,
                "total_char_count": cleaned_char_count,
                "total_chapters": len(chapters),
                "total_paragraphs": len(full_text_paragraphs),
                "ocr_quality": ocr_quality,
                "preprocessing_applied": [
                    "page_marker_removal",
                    "whitespace_normalization",
                    "broken_word_rejoining",
                ] + (["character_spacing_fix", "ocr_error_correction"] if ocr_quality in ("poor", "moderate") else []),
            },
            "table_of_contents": chapter_summaries,
            "chapters": chapters_with_paragraphs,
            "full_text_paragraphs": full_text_paragraphs,
        }

        return result

    def save_json(self, data: dict, output_filename: str):
        """Save structured data to JSON file."""
        output_path = self.output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path} ({size_mb:.2f} MB)")

    def process_all(self, skip_duplicates: bool = False):
        """Process all PDFs and optionally check for duplicates."""
        print("=" * 60)
        print("PDF TO STRUCTURED JSON CONVERTER")
        print("Savarkar GPT Project")
        print("=" * 60)

        all_texts = {}
        all_results = {}

        # Process each PDF
        for pdf_filename in PDF_CONFIGS:
            pdf_path = self.pdf_dir / pdf_filename
            if not pdf_path.exists():
                print(f"\n  ⚠️  SKIPPING (not found): {pdf_filename}")
                continue

            result = self.process_single_pdf(pdf_filename)

            # Generate clean output filename
            safe_name = re.sub(r'[^\w\-]', '_', Path(pdf_filename).stem)
            output_filename = f"{safe_name}.json"

            self.save_json(result, output_filename)

            all_texts[pdf_filename] = '\n\n'.join(result.get("full_text_paragraphs", []))
            all_results[pdf_filename] = result

        # Duplicate detection
        if not skip_duplicates and len(all_texts) > 1:
            overlaps = DuplicateDetector.check_pairwise_overlap(all_texts)

            # Save overlap report
            overlap_report = {
                "summary": "Cross-document overlap analysis",
                "total_documents": len(all_texts),
                "pairwise_comparisons": overlaps,
                "conclusion": self._generate_overlap_conclusion(overlaps)
            }
            self.save_json(overlap_report, "_overlap_report.json")

        # Save master index
        master_index = {
            "project": "Savarkar GPT",
            "total_documents": len(all_results),
            "documents": []
        }
        for pdf_filename, result in all_results.items():
            meta = result["metadata"]
            safe_name = re.sub(r'[^\w\-]', '_', Path(pdf_filename).stem)
            master_index["documents"].append({
                "json_file": f"{safe_name}.json",
                "title": meta["title"],
                "author": meta["author"],
                "category": meta["category"],
                "word_count": meta["total_word_count"],
                "chapters": meta["total_chapters"],
            })
        self.save_json(master_index, "_master_index.json")

        # Print summary
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"\n  Output directory: {self.output_dir}")
        print(f"  Documents processed: {len(all_results)}")
        total_words = sum(r["metadata"]["total_word_count"] for r in all_results.values())
        print(f"  Total words across all documents: {total_words:,}")
        print(f"\n  Files generated:")
        for f in sorted(self.output_dir.iterdir()):
            size = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name} ({size:.2f} MB)")

    @staticmethod
    def _generate_overlap_conclusion(overlaps: list) -> str:
        """Generate a human-readable conclusion about overlaps."""
        significant = [o for o in overlaps if o["overlap_percentage"] >= 15]
        low = [o for o in overlaps if 5 <= o["overlap_percentage"] < 15]

        if not significant and not low:
            return "All documents contain unique content. No significant overlap detected."
        elif not significant:
            return f"Minor overlaps found in {len(low)} pair(s), likely due to shared quotations or references. No significant duplication."
        else:
            pairs = [f"{o['document_a']} <-> {o['document_b']} ({o['overlap_percentage']}%)" for o in significant]
            return f"Significant overlap found in {len(significant)} pair(s): {'; '.join(pairs)}. Review recommended."


# ============================================================
# MAIN
# ============================================================

def main():
    skip_duplicates = "--skip-duplicates" in sys.argv
    check_only = "--check-duplicates" in sys.argv

    converter = PDFToJSONConverter(PDF_DIR, OUTPUT_DIR)

    if check_only:
        # Only run duplicate detection on existing JSONs
        print("Loading existing JSON files for duplicate check...")
        texts = {}
        for json_file in OUTPUT_DIR.glob("*.json"):
            if json_file.name.startswith("_"):
                continue
            with open(json_file, 'r') as f:
                data = json.load(f)
            texts[data["metadata"]["source_file"]] = data["full_text"]
        DuplicateDetector.check_pairwise_overlap(texts)
    else:
        converter.process_all(skip_duplicates=skip_duplicates)


if __name__ == "__main__":
    main()
