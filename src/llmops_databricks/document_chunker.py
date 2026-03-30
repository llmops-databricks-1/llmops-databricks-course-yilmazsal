from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    pass

MAX_CHUNK_CHARS = 2000
OVERLAP_PARAGRAPHS = 1
NOISE_ROLES = frozenset({"pageHeader", "pageFooter", "pageNumber"})


class DocumentChunk(BaseModel):
    chunk_id: str
    source_file: str
    document_title: str
    section_heading: str
    page_numbers: list[int]
    content: str
    chunk_index: int
    char_count: int = 0
    paragraph_count: int = 0

    @model_validator(mode="after")
    def _derive_counts(self) -> DocumentChunk:
        self.char_count = len(self.content)
        return self


def _is_noise(paragraph) -> bool:
    role = getattr(paragraph, "role", None)
    return role in NOISE_ROLES


def _get_page_numbers(paragraph) -> list[int]:
    pages: set[int] = set()
    regions = getattr(paragraph, "bounding_regions", None) or []
    for region in regions:
        page = getattr(region, "page_number", None)
        if page is not None:
            pages.add(page)
    return sorted(pages)


def _extract_document_title(paragraphs) -> str:
    for p in paragraphs:
        if getattr(p, "role", None) == "title":
            print(p)
            return (p.content or "").strip()
    return ""


def _split_paragraph_buffer(
    paragraphs: list,
    heading: str,
    source_file: str,
    document_title: str,
    chunk_index_start: int,
    max_chars: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Greedy accumulation with paragraph-overlap on overflow."""
    chunks: list[DocumentChunk] = []
    chunk_index = chunk_index_start
    i = 0

    while i < len(paragraphs):
        group: list = []
        char_total = 0

        while i < len(paragraphs):
            p = paragraphs[i]
            text = (p.content or "").strip()
            if not group or char_total + len(text) + 1 <= max_chars:
                group.append(p)
                char_total += len(text) + 1
                i += 1
            else:
                break

        if not group:
            # Single paragraph exceeds max_chars; include it anyway to avoid infinite loop
            group = [paragraphs[i]]
            i += 1

        content = "\n".join((p.content or "").strip() for p in group).strip()
        pages: list[int] = []
        for p in group:
            for pg in _get_page_numbers(p):
                if pg not in pages:
                    pages.append(pg)

        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                source_file=source_file,
                document_title=document_title,
                section_heading=heading,
                page_numbers=sorted(pages),
                content=content,
                chunk_index=chunk_index,
                paragraph_count=len(group),
            )
        )
        chunk_index += 1

        # Step back by `overlap` paragraphs for the next window
        if i < len(paragraphs) and overlap > 0:
            i = max(i - overlap, chunk_index_start + len(group) - overlap)

    return chunks


def chunk_analyze_result(
    result,
    source_file: str,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    overlap_paragraphs: int = OVERLAP_PARAGRAPHS,
) -> list[DocumentChunk]:
    """Walk AnalyzeResult paragraphs, group by section heading, split on overflow."""
    all_paragraphs = list(result.paragraphs or [])
    document_title = _extract_document_title(all_paragraphs)

    current_heading = "Preamble"
    buffer: list = []
    chunks: list[DocumentChunk] = []

    def _flush(heading: str, buf: list) -> None:
        if not buf:
            return
        new_chunks = _split_paragraph_buffer(
            paragraphs=buf,
            heading=heading,
            source_file=source_file,
            document_title=document_title,
            chunk_index_start=len(chunks),
            max_chars=max_chunk_chars,
            overlap=overlap_paragraphs,
        )
        chunks.extend(new_chunks)

    for paragraph in all_paragraphs:
        if _is_noise(paragraph):
            continue

        role = getattr(paragraph, "role", None)
        if role == "sectionHeading":
            _flush(current_heading, buffer)
            buffer = []
            current_heading = (paragraph.content or "").strip()
        else:
            buffer.append(paragraph)

    _flush(current_heading, buffer)
    return chunks
