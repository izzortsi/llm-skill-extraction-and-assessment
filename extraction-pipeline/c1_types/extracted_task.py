"""
extracted_task.py

Shared data types and I/O for extracted tasks.

ExtractedTask is the data contract between the text-extraction-pipeline
(which produces tasks) and the task-skill-extraction-pipeline (which
consumes them). Both pipelines import from here instead of from each other.

Also contains text loading/chunking utilities used by both pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


VALID_QUERY_TYPES = ("YES_NO", "YES_NO_VERIFICATION", "SINGLE_WORD", "RANKING", "FREE_FORM")


@dataclass
class ExtractedTask:
    """A task generated from a text artifact.

    Canonical query structure fields per standard-query-type-taxonomy.txt:
        question:   the query itself (maps to legacy "challenge" field)
        input:      the corpus/passage to evaluate against (maps to legacy "passage" field)
        output:     the expected answer (maps to acceptance_criteria.correct_conclusion)
        query_type: one of YES_NO, YES_NO_VERIFICATION, SINGLE_WORD, RANKING, FREE_FORM
    """

    task_uid: str
    title: str
    domain: str
    source_artifact: str
    source_document_uid: str
    question: str
    input: str
    output: str
    difficulty: str
    acceptance_criteria: Dict[str, Any]
    query_type: str = "FREE_FORM"
    extraction_method: str = ""

    @property
    def passage(self) -> str:
        """Backward-compatible alias for input field."""
        return self.input

    @property
    def challenge(self) -> str:
        """Backward-compatible alias for question field."""
        return self.question


def validate_free_form_single_answer(task: ExtractedTask) -> None:
    """Validate that a FREE_FORM task has exactly one expected answer.

    FREE_FORM queries must have exactly one correct_conclusion in
    acceptance_criteria. The must_identify list contains supporting
    evidence items, not alternative answers.

    Args:
        task: ExtractedTask to validate.

    Raises:
        ValueError: If the task is FREE_FORM and has multiple
                    correct_conclusion values or no output.
    """
    if task.query_type != "FREE_FORM":
        return

    if not task.output.strip():
        raise ValueError(
            f"FREE_FORM task {task.task_uid} has empty output "
            f"(acceptance_criteria.correct_conclusion is missing or empty)"
        )

    conclusion = task.output.strip()
    separators = [" OR ", " / ", " | "]
    for sep in separators:
        if sep in conclusion:
            raise ValueError(
                f"FREE_FORM task {task.task_uid} has multiple answers separated "
                f"by '{sep.strip()}' in output: '{conclusion[:100]}'. "
                f"FREE_FORM tasks must have exactly ONE answer."
            )


def save_extracted_tasks(tasks: List[ExtractedTask], output_path: Path) -> None:
    """Save extracted tasks to a JSON file.

    Args:
        tasks: List of ExtractedTask objects
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for task in tasks:
        data.append({
            "task_uid": task.task_uid,
            "title": task.title,
            "domain": task.domain,
            "source_artifact": task.source_artifact,
            "source_document_uid": task.source_document_uid,
            "question": task.question,
            "input": task.input,
            "output": task.output,
            "difficulty": task.difficulty,
            "query_type": task.query_type,
            "extraction_method": task.extraction_method,
            "acceptance_criteria": task.acceptance_criteria,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_extracted_tasks(filepath: Path) -> List[ExtractedTask]:
    """Load extracted tasks from a JSON file.

    Args:
        filepath: Path to extracted tasks JSON

    Returns:
        List of ExtractedTask objects
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for entry in data:
        task_uid = entry.get("task_uid", entry.get("task_id", ""))
        ac = entry.get("acceptance_criteria", {})
        tasks.append(ExtractedTask(
            task_uid=task_uid,
            title=entry["title"],
            domain=entry["domain"],
            source_artifact=entry.get("source_artifact", ""),
            source_document_uid=entry.get("source_document_uid", ""),
            question=entry.get("question", entry.get("challenge", "")),
            input=entry.get("input", entry.get("passage", "")),
            output=entry.get("output", ac.get("correct_conclusion", "")),
            difficulty=entry.get("difficulty", "intermediate"),
            acceptance_criteria=ac,
            query_type=entry.get("query_type", "FREE_FORM"),
            extraction_method=entry.get("extraction_method", ""),
        ))

    for task in tasks:
        validate_free_form_single_answer(task)

    return tasks


def load_text_from_file(filepath: Path) -> str:
    """Load text from a file, auto-detecting format.

    For PDFs, uses Marker extraction. For other formats, reads directly.

    Args:
        filepath: Path to text file (.txt, .md, .xml) or PDF (.pdf)

    Returns:
        Plain text content of the file
    """
    ext = filepath.suffix.lower()
    if ext == ".pdf":
        from c1_tools.text_extractor import convert_pdf
        doc = convert_pdf(str(filepath))
        return doc.plain_text
    else:
        return filepath.read_text(encoding="utf-8")


def _split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks, breaking at paragraph boundaries.

    Splits on double-newline (paragraph break) and accumulates paragraphs
    until the chunk reaches chunk_size characters. If a single paragraph
    exceeds chunk_size, it becomes its own chunk.

    Args:
        text: Full text to split
        chunk_size: Target chunk size in characters

    Returns:
        List of text chunks
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk_parts = []
    current_chunk_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_length = len(paragraph)

        if current_chunk_length > 0 and current_chunk_length + paragraph_length + 2 > chunk_size:
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = []
            current_chunk_length = 0

        current_chunk_parts.append(paragraph)
        current_chunk_length += paragraph_length + 2

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    min_chunk_length = 200
    chunks = [c for c in chunks if len(c) >= min_chunk_length]

    return chunks


def _detect_text_column(ds) -> str:
    """Auto-detect the text column in a HuggingFace dataset.

    Args:
        ds: HuggingFace Dataset instance

    Returns:
        Column name string

    Raises:
        ValueError: If no text column can be detected
    """
    priority_names = ["text", "content", "passage", "article", "sentence",
                      "document", "paragraph", "body"]
    columns = ds.column_names

    for name in priority_names:
        if name in columns:
            return name

    for col_name in columns:
        sample_value = ds[0][col_name]
        if isinstance(sample_value, str):
            return col_name

    raise ValueError(
        f"Cannot auto-detect text column. Columns: {columns}. "
        f"Use --text-column to specify."
    )


def load_text_chunks_from_dataset(
    dataset_name: str,
    subset: str = "",
    split: str = "train",
    text_column: str = "",
    chunk_size: int = 6000,
    max_chunks: int = 0,
    verbose: bool = False,
) -> List[str]:
    """Load text chunks from a HuggingFace dataset.

    Downloads the dataset via the datasets library, concatenates all text
    entries, then splits into chunks of chunk_size characters.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "wikitext")
        subset: Dataset subset/config (e.g., "wikitext-2-raw-v1")
        split: Dataset split to load (default: "train")
        text_column: Name of the text column. If empty, auto-detected.
        chunk_size: Target size of each text chunk in characters.
        max_chunks: Maximum number of chunks to return. 0 means no limit.
        verbose: Print progress information

    Returns:
        List of text chunks, each chunk_size characters or fewer
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for --dataset mode: "
            "pip install datasets"
        )

    if verbose:
        dataset_label = f"{dataset_name}/{subset}" if subset else dataset_name
        print(f"Loading dataset: {dataset_label} (split={split})...")

    load_kwargs = {"path": dataset_name, "split": split}
    if subset:
        load_kwargs["name"] = subset

    ds = load_dataset(**load_kwargs)

    resolved_column = text_column
    if not resolved_column:
        resolved_column = _detect_text_column(ds)
    if resolved_column not in ds.column_names:
        raise ValueError(
            f"Column '{resolved_column}' not found in dataset. "
            f"Available columns: {ds.column_names}. "
            f"Use --text-column to specify."
        )

    if verbose:
        print(f"  Using column: '{resolved_column}' ({len(ds)} rows)")

    target_chars = chunk_size * max_chunks * 3 if max_chunks > 0 else 0
    raw_parts = []
    collected_chars = 0
    for row in ds:
        text_value = row[resolved_column]
        if isinstance(text_value, str) and text_value.strip():
            part = text_value.strip()
            raw_parts.append(part)
            collected_chars += len(part)
            if target_chars > 0 and collected_chars >= target_chars:
                break

    full_text = "\n\n".join(raw_parts)

    if verbose:
        print(f"  Total text: {len(full_text)} chars from {len(raw_parts)} rows")

    chunks = _split_into_chunks(full_text, chunk_size)

    if max_chunks > 0 and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    if verbose:
        print(f"  Produced {len(chunks)} chunks (target size: {chunk_size} chars)")

    return chunks
