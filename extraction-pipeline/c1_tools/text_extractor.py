"""
text_extractor.py

Extract text from PDF files and post-process extraction artifacts.
Combines PDF conversion (via Marker) with text cleanup transforms.

Usage:
    python -m c1_providers.text_extractor extract input.pdf -o output/
    python -m c1_providers.text_extractor clean input.txt -o output/
    python -m c1_providers.text_extractor pipeline input.pdf -o output/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from c0_utils.data_structures import Block, DocumentExport, TransformConfig, ProcessingResult
from c0_utils.text_utils import (
    parse_markdown_blocks,
    markdown_to_plain_text,
    collapse_lettertracks,
    rejoin_hyphens,
    normalize_block_spacing,
    normalize_whitespace,
    clean_markdown_artifacts,
    apply_transforms,
    COMPOUND_PREFIXES,
)

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".xml"}


# ======================================================================
# PDF extraction (requires marker-pdf)
# ======================================================================

def convert_pdf(pdf_path: str) -> DocumentExport:
    """Run Marker conversion and structure the output."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    logger.info("Loading models...")
    models = create_model_dict()

    logger.info("Converting PDF...")
    converter = PdfConverter(artifact_dict=models)
    rendered = converter(str(pdf_path))

    md_text, metadata, images = text_from_rendered(rendered)
    blocks = parse_markdown_blocks(md_text)

    return DocumentExport(
        markdown=md_text,
        plain_text=markdown_to_plain_text(md_text),
        blocks=blocks,
        images=images,
        metadata=metadata,
    )


# ======================================================================
# Export formats
# ======================================================================

def blocks_to_dict(blocks: list) -> list:
    """Convert Block objects to dicts for JSON serialization."""
    return [{"block_type": b.block_type, "level": b.level, "text": b.text} for b in blocks]


def export_json(doc: DocumentExport, output_path: Path) -> None:
    """Serialize document blocks and metadata to JSON."""
    meta = doc.metadata if isinstance(doc.metadata, dict) else {"info": doc.metadata}
    data = {"metadata": meta, "blocks": blocks_to_dict(doc.blocks), "image_names": list(doc.images.keys())}
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def export_xml(doc: DocumentExport, output_path: Path) -> None:
    """Serialize document blocks and metadata to XML."""
    root = ET.Element("document")
    meta_elem = ET.SubElement(root, "metadata")
    if isinstance(doc.metadata, dict):
        for key, value in doc.metadata.items():
            m = ET.SubElement(meta_elem, "meta", name=str(key))
            m.text = str(value) if value else ""
    else:
        m = ET.SubElement(meta_elem, "meta", name="info")
        m.text = str(doc.metadata) if doc.metadata else ""

    content = ET.SubElement(root, "content")
    for b in doc.blocks:
        attrs = {"type": b.block_type}
        if b.level is not None:
            attrs["level"] = str(b.level)
        elem = ET.SubElement(content, "block", **attrs)
        if b.text:
            text_elem = ET.SubElement(elem, "text")
            text_elem.text = b.text

    xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")
    output_path.write_text(xml_str, encoding="utf-8")


def export_html(doc: DocumentExport, output_path: Path) -> None:
    """Export Markdown as rendered HTML."""
    import markdown as md_lib
    html_body = md_lib.markdown(doc.markdown, extensions=['tables', 'fenced_code', 'toc'])
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Document Export</title>
<style>body{{max-width:800px;margin:2rem auto;padding:0 1rem;font-family:sans-serif;line-height:1.6}}
img{{max-width:100%}}pre{{background:#f4f4f4;padding:1rem}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:.5rem}}</style></head>
<body>{html_body}</body></html>"""
    output_path.write_text(html_doc, encoding="utf-8")


def export_all(doc: DocumentExport, output_dir: Path, stem: str) -> None:
    """Export to all formats (.md, .txt, .json, .xml, .html)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for name, img in doc.images.items():
        img.save(images_dir / name)

    (output_dir / f"{stem}.md").write_text(doc.markdown, encoding="utf-8")
    (output_dir / f"{stem}.txt").write_text(doc.plain_text, encoding="utf-8")
    export_json(doc, output_dir / f"{stem}.json")
    export_xml(doc, output_dir / f"{stem}.xml")
    export_html(doc, output_dir / f"{stem}.html")


# ======================================================================
# Format handlers for post-processing
# ======================================================================

def process_txt(content: str, config: TransformConfig) -> str:
    return apply_transforms(content, config)


def process_md(content: str, config: TransformConfig) -> str:
    md_config = TransformConfig(
        should_rejoin_hyphens=config.should_rejoin_hyphens,
        should_collapse_lettertracks=config.should_collapse_lettertracks,
        should_normalize_block_spacing=config.should_normalize_block_spacing,
        should_clean_markdown_artifacts=config.should_clean_markdown_artifacts,
        should_normalize_whitespace=config.should_normalize_whitespace,
        is_markdown=True,
        should_use_dictionary=config.should_use_dictionary,
        dictionary_path=config.dictionary_path,
        min_lettertrack_length=config.min_lettertrack_length,
    )
    return apply_transforms(content, md_config)


def process_json_content(content: str, config: TransformConfig) -> str:
    """Process JSON files from PDF extraction."""
    data = json.loads(content)
    if "pages" in data:
        for page in data.get("pages", []):
            for block in page.get("blocks", []):
                for line_item in block.get("lines", []):
                    for span in line_item.get("spans", []):
                        if "text" in span and span["text"]:
                            span["text"] = apply_transforms(span["text"], config)
    elif "blocks" in data:
        for block in data.get("blocks", []):
            if "text" in block and block["text"]:
                block["text"] = apply_transforms(block["text"], config)
    else:
        _process_json_recursive(data, config)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _process_json_recursive(data, config: TransformConfig) -> None:
    """Fallback: recursively process 'text' fields."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "text" and isinstance(value, str) and value:
                data[key] = apply_transforms(value, config)
            else:
                _process_json_recursive(value, config)
    elif isinstance(data, list):
        for item in data:
            _process_json_recursive(item, config)


def process_xml_content(content: str, config: TransformConfig) -> str:
    """Process XML files from PDF extraction."""
    root = ET.fromstring(content)
    if root.tag == "pages" or root.find(".//pages") is not None:
        for span in root.iter("span"):
            if span.text:
                span.text = apply_transforms(span.text, config)
    elif root.tag == "content" or root.find(".//content") is not None:
        for text_elem in root.iter("text"):
            if text_elem.text:
                text_elem.text = apply_transforms(text_elem.text, config)
    else:
        for elem in root.iter():
            if elem.text and elem.text.strip():
                elem.text = apply_transforms(elem.text, config)
    return minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")


FORMAT_HANDLERS = {
    ".txt": process_txt,
    ".md": process_md,
    ".json": process_json_content,
    ".xml": process_xml_content,
}


def process_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[TransformConfig] = None,
    suffix: str = "_cleaned",
) -> ProcessingResult:
    """Process a single extracted file (cleanup transforms)."""
    if config is None:
        config = TransformConfig()

    ext = input_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ProcessingResult(input_path=input_path, output_path=input_path, format=ext, is_success=False, errors=[f"Unsupported format: {ext}"])

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}{suffix}{ext}"

    handler = FORMAT_HANDLERS[ext]
    content = input_path.read_text(encoding="utf-8")
    result_text = handler(content, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_text, encoding="utf-8")

    return ProcessingResult(input_path=input_path, output_path=output_path, format=ext)


def process_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    config: Optional[TransformConfig] = None,
    suffix: str = "_cleaned",
    is_recursive: bool = False,
) -> list:
    """Process all supported files in a directory."""
    if config is None:
        config = TransformConfig()

    pattern = "**/*" if is_recursive else "*"
    files = sorted(p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)

    results = []
    for f in files:
        if output_dir:
            rel = f.relative_to(input_dir)
            out = output_dir / rel
        else:
            out = None
        result = process_file(f, out, config, suffix)
        results.append(result)
    return results


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Text extraction and cleanup pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # extract: PDF -> multi-format
    extract_parser = subparsers.add_parser("extract", help="Extract text from PDF via Marker")
    extract_parser.add_argument("pdf_path", type=Path, help="Path to input PDF")
    extract_parser.add_argument("--output-dir", "-o", type=Path, default=Path("./output"), help="Output directory")

    # clean: post-process extracted text
    clean_parser = subparsers.add_parser("clean", help="Post-process extracted text files")
    clean_parser.add_argument("input", type=Path, nargs="+", help="Input file(s) or directory")
    clean_parser.add_argument("--output-dir", "-o", type=Path, default=None)
    clean_parser.add_argument("--suffix", type=str, default="_cleaned")
    clean_parser.add_argument("--recursive", "-r", action="store_true")
    clean_parser.add_argument("--no-hyphens", action="store_true")
    clean_parser.add_argument("--no-lettertracks", action="store_true")
    clean_parser.add_argument("--no-block-spacing", action="store_true")
    clean_parser.add_argument("--use-system-dictionary", action="store_true")

    # pipeline: PDF -> extract -> clean
    pipe_parser = subparsers.add_parser("pipeline", help="Full pipeline: PDF -> extract -> clean")
    pipe_parser.add_argument("pdf_path", type=Path, help="Path to input PDF")
    pipe_parser.add_argument("--output-dir", "-o", type=Path, default=Path("./output"))

    args = parser.parse_args()

    if args.command == "extract":
        if not args.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {args.pdf_path}")
        doc = convert_pdf(str(args.pdf_path))
        export_all(doc, args.output_dir, args.pdf_path.stem)
        print(f"Extracted to {args.output_dir}/")

    elif args.command == "clean":
        config = TransformConfig(
            should_rejoin_hyphens=not args.no_hyphens,
            should_collapse_lettertracks=not args.no_lettertracks,
            should_normalize_block_spacing=not args.no_block_spacing,
            should_use_dictionary=args.use_system_dictionary,
        )
        all_results = []
        for input_path in args.input:
            if not input_path.exists():
                print(f"  Error: {input_path} not found, skipping")
                continue
            if input_path.is_dir():
                results = process_directory(input_path, output_dir=args.output_dir, config=config, suffix=args.suffix, is_recursive=args.recursive)
                all_results.extend(results)
            else:
                out = (args.output_dir / input_path.name) if args.output_dir else None
                result = process_file(input_path, out, config, args.suffix)
                all_results.append(result)
        ok = sum(1 for r in all_results if r.is_success)
        fail = sum(1 for r in all_results if not r.is_success)
        for r in all_results:
            if r.is_success:
                print(f"  Processed: {r.input_path} -> {r.output_path}")
            else:
                print(f"  Failed:    {r.input_path} ({', '.join(r.errors)})")
        print(f"\nDone! {ok} processed, {fail} failed.")

    elif args.command == "pipeline":
        if not args.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {args.pdf_path}")
        doc = convert_pdf(str(args.pdf_path))
        export_all(doc, args.output_dir, args.pdf_path.stem)
        print(f"Extracted to {args.output_dir}/")
        # Now post-process the .txt and .md files
        config = TransformConfig()
        for ext in [".txt", ".md"]:
            fpath = args.output_dir / f"{args.pdf_path.stem}{ext}"
            if fpath.exists():
                process_file(fpath, fpath, config, suffix="")
                print(f"  Cleaned: {fpath}")
        print("Pipeline complete!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
