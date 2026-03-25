"""
task_extractor.py

Extract evaluation tasks from text artifacts using Opus 4.6.
Three input modes:
  1. Text artifact (txt, md, xml) - extract tasks directly
  2. PDF artifact - extract text first, then extract tasks
  3. HuggingFace dataset - load dataset, chunk text, extract tasks from each chunk

Usage:
    python -m c2_skill_extraction.task_extractor --input artifact.txt --domain science -o tasks.json
    python -m c2_skill_extraction.task_extractor --input document.pdf --domain history -o tasks.json
    python -m c2_skill_extraction.task_extractor --dataset wikitext --subset wikitext-2-raw-v1 --domain language -o tasks.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from c0_utils.uid import generate_uid
from c0_utils.text_utils import strip_markdown_fences


@dataclass
class NormalizedDocument:
    """A normalized text document ready for task extraction.

    This is the validated intermediate format between raw data ingestion
    (file read, HuggingFace download) and task extraction. All text must
    pass validation before extraction proceeds.
    """

    source_type: str            # "file", "dataset", or "pdf"
    source_name: str            # file path or dataset identifier
    text: str                   # normalized plain text content
    chunk_index: int = 0        # chunk index (0 for single-file mode)
    total_chunks: int = 1       # total chunks from this source


NORMALIZED_DOCUMENT_MIN_TEXT_LENGTH = 200
NORMALIZED_DOCUMENT_MAX_TEXT_LENGTH = 100000


def validate_normalized_document(doc: "NormalizedDocument") -> None:
    """Validate a NormalizedDocument against the expected schema.

    Checks all required fields, types, and constraints. Raises ValueError
    with a specific message if validation fails.

    Args:
        doc: NormalizedDocument to validate.

    Raises:
        ValueError: If any field fails validation.
    """
    if not isinstance(doc.source_type, str) or doc.source_type not in ("file", "dataset", "pdf"):
        raise ValueError(
            f"NormalizedDocument.source_type must be 'file', 'dataset', or 'pdf', "
            f"got: '{doc.source_type}'"
        )
    if not isinstance(doc.source_name, str) or not doc.source_name.strip():
        raise ValueError("NormalizedDocument.source_name must be a non-empty string")
    if not isinstance(doc.text, str):
        raise ValueError("NormalizedDocument.text must be a string")
    if len(doc.text.strip()) < NORMALIZED_DOCUMENT_MIN_TEXT_LENGTH:
        raise ValueError(
            f"NormalizedDocument.text is too short ({len(doc.text.strip())} chars). "
            f"Minimum: {NORMALIZED_DOCUMENT_MIN_TEXT_LENGTH} chars"
        )
    if len(doc.text) > NORMALIZED_DOCUMENT_MAX_TEXT_LENGTH:
        raise ValueError(
            f"NormalizedDocument.text exceeds maximum length ({len(doc.text)} chars). "
            f"Maximum: {NORMALIZED_DOCUMENT_MAX_TEXT_LENGTH} chars"
        )
    if not isinstance(doc.chunk_index, int) or doc.chunk_index < 0:
        raise ValueError(f"NormalizedDocument.chunk_index must be a non-negative int, got: {doc.chunk_index}")
    if not isinstance(doc.total_chunks, int) or doc.total_chunks < 1:
        raise ValueError(f"NormalizedDocument.total_chunks must be >= 1, got: {doc.total_chunks}")
    if doc.chunk_index >= doc.total_chunks:
        raise ValueError(
            f"NormalizedDocument.chunk_index ({doc.chunk_index}) must be < "
            f"total_chunks ({doc.total_chunks})"
        )


VALID_QUERY_TYPES = ("YES_NO", "YES_NO_VERIFICATION", "SINGLE_WORD", "RANKING", "FREE_FORM")


def validate_free_form_single_answer(task: "ExtractedTask") -> None:
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


EXTRACTION_PROMPT = """You are a task designer for an LLM evaluation benchmark.

Given the following text artifact, generate {num_tasks} diverse tasks that test different reasoning capabilities. Each task should:

1. Select a relevant passage excerpt (100-300 words) from the artifact
2. Create a challenge question that requires careful reasoning about the passage
3. Tag the difficulty (basic, intermediate, advanced)
4. Classify the query type (see below)
5. Define structured acceptance criteria

Domain: {domain}

## Query Types

Each task MUST have one of these query_type values:
  - YES_NO: binary yes/no answer. Output must be "yes" or "no".
  - YES_NO_VERIFICATION: a question AND a proposed answer are given, model judges correctness. Output must be "correct" or "incorrect".
  - SINGLE_WORD: one-word answer expected. Output is a single word.
  - RANKING: given criteria and two documents A and B, rank them. Output must be "A" or "B".
  - FREE_FORM: open-ended response requiring reasoning. Must have exactly ONE correct conclusion.

Generate a MIX of query types across the {num_tasks} tasks.

## Text Artifact

{artifact_text}

## Output Format

Return ONLY valid JSON (no markdown, no explanation) as a list of objects:

[
  {{
    "title": "<short descriptive title>",
    "passage": "<relevant excerpt from the artifact>",
    "challenge": "<the challenge question>",
    "difficulty": "<basic|intermediate|advanced>",
    "query_type": "<YES_NO|YES_NO_VERIFICATION|SINGLE_WORD|RANKING|FREE_FORM>",
    "acceptance_criteria": {{
      "must_identify": ["<item 1>", "<item 2>", "<item 3>"],
      "correct_conclusion": "<the key conclusion a correct response must reach>"
    }}
  }}
]

Generate exactly {num_tasks} tasks with varying difficulty and different reasoning requirements."""


def _generate_task_uid(domain: str, title: str, index: int) -> str:
    """Generate a deterministic task UID in company standard format."""
    return generate_uid(f"{domain}|{title}|{index}")


def _generate_document_uid(source_artifact: str) -> str:
    """Generate a deterministic source document UID from the artifact identifier."""
    return generate_uid(f"document|{source_artifact}")


def _extract_response_text(result) -> str:
    """Extract text content from a provider chat result."""
    if isinstance(result.message.get("content"), str):
        return result.message["content"]
    if isinstance(result.message.get("content"), list):
        parts = []
        for block in result.message["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


_strip_markdown_fences = strip_markdown_fences


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
    entries, then splits into chunks of chunk_size characters. Each chunk
    becomes a separate text artifact for task extraction.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "wikitext")
        subset: Dataset subset/config (e.g., "wikitext-2-raw-v1")
        split: Dataset split to load (default: "train")
        text_column: Name of the text column. If empty, auto-detected
                     by scanning column names for "text", "content",
                     "passage", "article", "sentence", or the first
                     string column found.
        chunk_size: Target size of each text chunk in characters.
                    Default 6000 chars (~1500 tokens). Chunks break
                    at paragraph boundaries when possible.
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

    # Auto-detect text column
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

    # Concatenate text entries with paragraph breaks.
    # When max_chunks is set, stop reading rows once enough text is collected
    # to produce the requested number of chunks (avoids iterating the full
    # dataset for large sources like Wikipedia with 6M+ rows).
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

    # Split into chunks at paragraph boundaries
    chunks = _split_into_chunks(full_text, chunk_size)

    if max_chunks > 0 and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    if verbose:
        print(f"  Produced {len(chunks)} chunks (target size: {chunk_size} chars)")

    return chunks


def _detect_text_column(ds) -> str:
    """Auto-detect the text column in a HuggingFace dataset.

    Checks column names against a priority list of common text column
    names, then falls back to the first string-typed column.

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

    # Fallback: first column whose values are strings
    for col_name in columns:
        sample_value = ds[0][col_name]
        if isinstance(sample_value, str):
            return col_name

    raise ValueError(
        f"Cannot auto-detect text column. Columns: {columns}. "
        f"Use --text-column to specify."
    )


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

        # If adding this paragraph would exceed chunk_size, finalize current chunk
        if current_chunk_length > 0 and current_chunk_length + paragraph_length + 2 > chunk_size:
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = []
            current_chunk_length = 0

        current_chunk_parts.append(paragraph)
        current_chunk_length += paragraph_length + 2  # +2 for "\n\n" separator

    # Finalize last chunk
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    # Filter out chunks that are too short to produce meaningful tasks
    min_chunk_length = 200
    chunks = [c for c in chunks if len(c) >= min_chunk_length]

    return chunks


def extract_tasks_from_dataset(
    dataset_name: str,
    domain: str,
    provider,
    subset: str = "",
    split: str = "train",
    text_column: str = "",
    chunk_size: int = 6000,
    max_chunks: int = 0,
    tasks_per_chunk: int = 3,
    verbose: bool = False,
) -> List[ExtractedTask]:
    """Extract tasks from a HuggingFace dataset.

    Loads the dataset, splits into text chunks, and extracts tasks from
    each chunk independently. Each chunk is sent to the LLM as a separate
    text artifact.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "wikitext")
        domain: Domain label for generated tasks
        provider: LLM provider with chat() method (Opus 4.6)
        subset: Dataset subset/config (e.g., "wikitext-2-raw-v1")
        split: Dataset split (default: "train")
        text_column: Text column name (auto-detected if empty)
        chunk_size: Target chunk size in characters (default: 6000)
        max_chunks: Max chunks to process (0 = no limit)
        tasks_per_chunk: Number of tasks to extract per chunk (default: 3)
        verbose: Print progress

    Returns:
        List of ExtractedTask objects from all chunks
    """
    chunks = load_text_chunks_from_dataset(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        text_column=text_column,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        verbose=verbose,
    )

    dataset_label = f"{dataset_name}/{subset}" if subset else dataset_name
    all_tasks = []

    for chunk_index, chunk_text in enumerate(chunks):
        if verbose:
            print(f"\nChunk {chunk_index + 1}/{len(chunks)} ({len(chunk_text)} chars)")

        source_artifact = f"dataset:{dataset_label}:chunk-{chunk_index}"

        doc = NormalizedDocument(
            source_type="dataset",
            source_name=source_artifact,
            text=chunk_text,
            chunk_index=chunk_index,
            total_chunks=len(chunks),
        )
        try:
            validate_normalized_document(doc)
        except ValueError as validation_error:
            if verbose:
                print(f"  SKIP chunk {chunk_index}: {validation_error}")
            continue

        chunk_tasks = extract_tasks_from_artifact(
            artifact_text=chunk_text,
            domain=domain,
            provider=provider,
            num_tasks=tasks_per_chunk,
            source_artifact=source_artifact,
            verbose=verbose,
        )
        all_tasks.extend(chunk_tasks)

    if verbose:
        print(f"\nTotal: {len(all_tasks)} tasks from {len(chunks)} chunks")

    return all_tasks


def extract_tasks_from_artifact(
    artifact_text: str,
    domain: str,
    provider,
    num_tasks: int = 5,
    source_artifact: str = "",
    verbose: bool = False,
) -> List[ExtractedTask]:
    """Extract tasks from a text artifact using an LLM provider.

    Validates the input text as a NormalizedDocument before extraction.

    Args:
        artifact_text: The full text of the artifact
        domain: Domain label (e.g., "science", "history", "technology")
        provider: LLM provider with chat() method
        num_tasks: Number of tasks to generate
        source_artifact: Path or identifier of the source text
        verbose: Enable verbose logging

    Returns:
        List of ExtractedTask objects

    Raises:
        ValueError: If the input text fails normalization validation.
    """
    is_dataset = source_artifact.startswith("dataset:")
    source_type = "dataset" if is_dataset else "file"
    if source_artifact.endswith(".pdf"):
        source_type = "pdf"

    doc = NormalizedDocument(
        source_type=source_type,
        source_name=source_artifact,
        text=artifact_text,
    )
    validate_normalized_document(doc)

    prompt = EXTRACTION_PROMPT.format(
        artifact_text=artifact_text,
        domain=domain,
        num_tasks=num_tasks,
    )

    messages = [{"role": "user", "content": prompt}]

    if verbose:
        print(f"Extracting {num_tasks} tasks from artifact ({len(artifact_text)} chars)...")

    result = provider.chat(messages)
    response_text = _extract_response_text(result)
    json_text = _strip_markdown_fences(response_text)

    try:
        tasks_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"  ERROR: Failed to parse JSON: {e}")
            print(f"  Raw response: {response_text[:500]}")
        return []

    if not isinstance(tasks_data, list):
        tasks_data = [tasks_data]

    document_uid = _generate_document_uid(source_artifact)
    model_name = getattr(provider, "model_name", getattr(provider, "model", "unknown"))
    # TODO: import from llm-skills.llm-providers
    import re as _re
    short_model = _re.sub(r"^claude-", "", model_name)
    short_model = _re.sub(r"[:/]", "-", short_model)
    extraction_method = f"{short_model}-task-extraction-v1"

    tasks = []
    for i, entry in enumerate(tasks_data):
        task_uid = _generate_task_uid(domain, entry.get("title", f"task-{i}"), i)
        raw_query_type = entry.get("query_type", "FREE_FORM").upper()
        query_type = raw_query_type if raw_query_type in VALID_QUERY_TYPES else "FREE_FORM"

        ac = entry.get("acceptance_criteria", {})
        task = ExtractedTask(
            task_uid=task_uid,
            title=entry.get("title", f"Untitled task {i}"),
            domain=domain,
            source_artifact=source_artifact,
            source_document_uid=document_uid,
            question=entry.get("challenge", entry.get("question", "")),
            input=entry.get("passage", entry.get("input", "")),
            output=ac.get("correct_conclusion", entry.get("output", "")),
            difficulty=entry.get("difficulty", "intermediate"),
            acceptance_criteria=ac,
            query_type=query_type,
            extraction_method=extraction_method,
        )
        try:
            validate_free_form_single_answer(task)
        except ValueError as e:
            if verbose:
                print(f"  WARNING: {e}")

        tasks.append(task)
        if verbose:
            print(f"  -> Extracted: {task.task_uid} {task.title} ({task.difficulty}, {task.query_type})")

    return tasks


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
    # TODO: import from llm-skills.llm-providers
    # from c1_providers.schema_validator import validate_tasks_json

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # validate_tasks_json(data)  # TODO: import from llm-skills.llm-providers

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract tasks from text artifacts or HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a text file
  python -m c2_skill_extraction.task_extractor -i article.txt -d science -o tasks.json

  # From a PDF
  python -m c2_skill_extraction.task_extractor -i textbook.pdf -d psychology -o tasks.json

  # From wikitext2 (HuggingFace)
  python -m c2_skill_extraction.task_extractor \\
    --dataset wikitext --subset wikitext-2-raw-v1 -d language -o tasks.json

  # From wikitext2, limit to 10 chunks of 4000 chars each
  python -m c2_skill_extraction.task_extractor \\
    --dataset wikitext --subset wikitext-2-raw-v1 -d language \\
    --max-chunks 10 --chunk-size 4000 --tasks-per-chunk 5 -o tasks.json
""",
    )

    # Input source: --input XOR --dataset XOR --passages (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input", "-i", type=Path,
        help="Input text artifact (txt, md, xml, pdf)",
    )
    source_group.add_argument(
        "--dataset", type=str,
        help="HuggingFace dataset name (e.g., 'wikitext', 'wikipedia')",
    )
    source_group.add_argument(
        "--passages", type=Path,
        help="Pre-extracted passages JSON (from passage_extractor.py)",
    )

    # Common arguments
    parser.add_argument("--domain", "-d", type=str, required=True, help="Domain label")
    parser.add_argument("--output", "-o", type=Path, default=Path("stage1-task-extraction/tasks.json"), help="Output JSON path")
    parser.add_argument("--provider", type=str, default="anthropic", help="LLM provider (anthropic, openai, mock)")
    parser.add_argument("--model", type=str, default="claude-opus-4-6", help="Model to use")
    parser.add_argument("--verbose", "-v", action="store_true")

    # File mode arguments
    parser.add_argument("--num-tasks", "-n", type=int, default=5, help="Number of tasks (file mode)")

    # Dataset mode arguments
    parser.add_argument("--subset", type=str, default="", help="Dataset subset/config (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--text-column", type=str, default="", help="Text column name (auto-detected if empty)")
    parser.add_argument("--chunk-size", type=int, default=6000, help="Chunk size in chars for dataset mode (default: 6000)")
    parser.add_argument("--max-chunks", type=int, default=0, help="Max chunks to process, 0=no limit (default: 0)")
    parser.add_argument("--tasks-per-chunk", type=int, default=3, help="Tasks to extract per chunk (default: 3)")

    args = parser.parse_args()

    # TODO: import from llm-skills.llm-providers
    from c1_providers.providers import create_provider  # noqa: requires llm-skills.llm-providers on sys.path
    provider = create_provider(args.provider, args.model)

    if args.passages:
        # Passages mode: consume pre-extracted passages
        from c2_extraction.passage_extractor import load_passages
        if not args.passages.exists():
            raise FileNotFoundError(f"Passages file not found: {args.passages}")
        passages = load_passages(args.passages)
        tasks = []
        for pi, passage in enumerate(passages):
            if args.verbose:
                print(f"\nPassage {pi + 1}/{len(passages)} ({len(passage.text)} chars)")
            chunk_tasks = extract_tasks_from_artifact(
                artifact_text=passage.text,
                domain=args.domain,
                provider=provider,
                num_tasks=args.tasks_per_chunk,
                source_artifact=passage.source_artifact,
                verbose=args.verbose,
            )
            tasks.extend(chunk_tasks)
        if args.verbose:
            print(f"\nTotal: {len(tasks)} tasks from {len(passages)} passages")
    elif args.input:
        # File mode
        if not args.input.exists():
            raise FileNotFoundError(f"Input not found: {args.input}")

        artifact_text = load_text_from_file(args.input)
        tasks = extract_tasks_from_artifact(
            artifact_text, args.domain, provider,
            num_tasks=args.num_tasks, source_artifact=str(args.input),
            verbose=args.verbose,
        )
    else:
        # Dataset mode
        tasks = extract_tasks_from_dataset(
            dataset_name=args.dataset,
            domain=args.domain,
            provider=provider,
            subset=args.subset,
            split=args.split,
            text_column=args.text_column,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            tasks_per_chunk=args.tasks_per_chunk,
            verbose=args.verbose,
        )

    save_extracted_tasks(tasks, args.output)
    print(f"Saved {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()
