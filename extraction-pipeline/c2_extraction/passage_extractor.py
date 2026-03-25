"""
passage_extractor.py

Extract passages from source documents as an independent pipeline stage.

Pipeline flow:
    source document (text/PDF/HuggingFace) -> ExtractedPassage list -> JSON

Each passage links back to its source document UID and records the extraction
method used. Task extraction (task_extractor.py) consumes passages as input
instead of raw document text.

Usage:
    python -m c2_skill_extraction.passage_extractor \
        --input article.txt --domain science -o stage1-task-extraction/passages.json

    python -m c2_skill_extraction.passage_extractor \
        --dataset wikimedia/wikipedia --subset 20231101.en --domain language \
        --max-chunks 5 --chunk-size 4000 -o stage1-task-extraction/passages.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from c0_utils.uid import generate_uid


@dataclass
class ExtractedPassage:
    """A passage extracted from a source document.

    Each passage is a contiguous text excerpt linked to its source document
    by source_document_uid. Passages serve as the input to task extraction.
    """

    passage_uid: str
    source_document_uid: str
    source_artifact: str
    text: str
    chunk_index: int
    total_chunks: int
    extraction_method: str = ""


def _generate_passage_uid(document_uid: str, chunk_index: int, text: str) -> str:
    """Generate a deterministic passage UID from document UID, index, and text hash."""
    import hashlib
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return generate_uid(f"{document_uid}|{chunk_index}|{text_hash}")


def _generate_document_uid(source_artifact: str) -> str:
    """Generate a deterministic source document UID from the artifact identifier."""
    return generate_uid(f"document|{source_artifact}")


def extract_passages_from_file(
    filepath: Path,
    chunk_size: int = 6000,
) -> List[ExtractedPassage]:
    """Extract passages from a local text or PDF file.

    Args:
        filepath: Path to text file (.txt, .md, .xml) or PDF (.pdf).
        chunk_size: Target chunk size in characters.

    Returns:
        List of ExtractedPassage objects.
    """
    from c2_extraction.task_extractor import load_text_from_file, _split_into_chunks

    text = load_text_from_file(filepath)
    chunks = _split_into_chunks(text, chunk_size)

    source_artifact = str(filepath)
    document_uid = _generate_document_uid(source_artifact)

    passages = []
    for i, chunk_text in enumerate(chunks):
        passage_uid = _generate_passage_uid(document_uid, i, chunk_text)
        passages.append(ExtractedPassage(
            passage_uid=passage_uid,
            source_document_uid=document_uid,
            source_artifact=source_artifact,
            text=chunk_text,
            chunk_index=i,
            total_chunks=len(chunks),
        ))

    return passages


def extract_passages_from_dataset(
    dataset_name: str,
    subset: str = "",
    split: str = "train",
    text_column: str = "",
    chunk_size: int = 6000,
    max_chunks: int = 0,
    verbose: bool = False,
) -> List[ExtractedPassage]:
    """Extract passages from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier.
        subset: Dataset subset/config.
        split: Dataset split (default: "train").
        text_column: Text column name (auto-detected if empty).
        chunk_size: Target chunk size in characters.
        max_chunks: Max chunks to return (0 = no limit).
        verbose: Print progress.

    Returns:
        List of ExtractedPassage objects.
    """
    from c2_extraction.task_extractor import load_text_chunks_from_dataset

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
    source_artifact = f"dataset:{dataset_label}"
    document_uid = _generate_document_uid(source_artifact)

    passages = []
    for i, chunk_text in enumerate(chunks):
        passage_uid = _generate_passage_uid(document_uid, i, chunk_text)
        passages.append(ExtractedPassage(
            passage_uid=passage_uid,
            source_document_uid=document_uid,
            source_artifact=source_artifact,
            text=chunk_text,
            chunk_index=i,
            total_chunks=len(chunks),
        ))

    if verbose:
        print(f"Extracted {len(passages)} passages from {dataset_label}")

    return passages


def save_passages(passages: List[ExtractedPassage], output_path: Path) -> None:
    """Save extracted passages to a JSON file.

    Args:
        passages: List of ExtractedPassage objects.
        output_path: Path to output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for p in passages:
        data.append({
            "passage_uid": p.passage_uid,
            "source_document_uid": p.source_document_uid,
            "source_artifact": p.source_artifact,
            "text": p.text,
            "chunk_index": p.chunk_index,
            "total_chunks": p.total_chunks,
            "extraction_method": p.extraction_method,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_passages(filepath: Path) -> List[ExtractedPassage]:
    """Load extracted passages from a JSON file.

    Args:
        filepath: Path to passages JSON file.

    Returns:
        List of ExtractedPassage objects.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    passages = []
    for entry in data:
        passages.append(ExtractedPassage(
            passage_uid=entry["passage_uid"],
            source_document_uid=entry.get("source_document_uid", ""),
            source_artifact=entry.get("source_artifact", ""),
            text=entry["text"],
            chunk_index=entry.get("chunk_index", 0),
            total_chunks=entry.get("total_chunks", 1),
            extraction_method=entry.get("extraction_method", ""),
        ))

    return passages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract passages from source documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a text file
  python -m c2_skill_extraction.passage_extractor -i article.txt -o passages.json

  # From a HuggingFace dataset
  python -m c2_skill_extraction.passage_extractor \\
    --dataset wikimedia/wikipedia --subset 20231101.en \\
    --max-chunks 5 --chunk-size 4000 -o passages.json
""",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", "-i", type=Path, help="Input text artifact (txt, md, xml, pdf)")
    source_group.add_argument("--dataset", type=str, help="HuggingFace dataset name")

    parser.add_argument("--output", "-o", type=Path,
                        default=Path("stage1-task-extraction/passages.json"),
                        help="Output JSON path")
    parser.add_argument("--chunk-size", type=int, default=6000, help="Chunk size in chars (default: 6000)")
    parser.add_argument("--subset", type=str, default="", help="Dataset subset/config")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--text-column", type=str, default="", help="Text column name (auto-detected if empty)")
    parser.add_argument("--max-chunks", type=int, default=0, help="Max chunks, 0=no limit (default: 0)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.input:
        if not args.input.exists():
            raise FileNotFoundError(f"Input not found: {args.input}")
        passages = extract_passages_from_file(args.input, chunk_size=args.chunk_size)
    else:
        passages = extract_passages_from_dataset(
            dataset_name=args.dataset,
            subset=args.subset,
            split=args.split,
            text_column=args.text_column,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            verbose=args.verbose,
        )

    save_passages(passages, args.output)

    if args.verbose:
        print(f"Saved {len(passages)} passages to {args.output}")


if __name__ == "__main__":
    main()
