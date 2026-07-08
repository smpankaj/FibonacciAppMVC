

from google import genai

client = genai.Client(api_key="")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in Dutch in one short sentence."
)

print(response.text)

from pydantic import BaseModel
from google import genai
from google.genai import types
import os
from getpass import getpass


class RewrittenBlock(BaseModel):
    id: str
    revised_text: str

class BlocksResponse(BaseModel):
    blocks: list[RewrittenBlock]


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="""
Return JSON only.

Rewrite this block into simple Dutch:
{
  "blocks": [
    {
      "id": "s_1",
      "text": "Hello, this is a difficult internal company document."
    }
  ]
}
""",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BlocksResponse,
    ),
)

print(response.text)
print(BlocksResponse.model_validate_json(response.text))

from __future__ import annotations

import json
import re
import time
import uuid
from copy import copy
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from google.genai import types
from pydantic import BaseModel


# =============================================================================
# Assumption
# =============================================================================
# This code assumes you already have a working Gemini client variable named:
#
#     client
#
# For example, you already tested something like:
#
#     response = client.models.generate_content(...)
#
# Do NOT recreate the client in this code.
# =============================================================================


# =============================================================================
# Config
# =============================================================================

DEFAULT_MODEL = "gemini-2.5-flash"

STAGE1_MAX_CHARS_PER_BATCH = 60_000
STAGE2_MAX_CHARS_PER_BATCH = 18_000

# Tags whose direct text should be editable.
# The code is also able to process direct text inside structural tags like div.
TEXT_CONTAINER_TAGS = {
    "p", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "td", "th",
    "blockquote", "figcaption",
    "label", "caption",
}

# Structural tags are preserved and recursively walked.
STRUCTURAL_TAGS = {
    "div", "section", "article", "main", "aside",
    "ul", "ol",
    "table", "thead", "tbody", "tfoot", "tr",
    "figure",
    "dl", "dt", "dd",
}

# Inline tags whose position may move with the rewritten text.
INLINE_TAGS = {
    "b", "strong", "i", "em", "u", "a", "span",
    "sup", "sub", "mark", "small", "code",
    "abbr", "cite", "q", "s", "del", "ins",
}

# Display/void tags that may appear inside text.
# If they appear inside a sentence, they become empty placeholder tokens.
VOID_OR_DISPLAY_TAGS = {
    "img", "br", "hr",
}

# Tags to ignore entirely for rewriting.
# Based on your constraints, script/style should not appear, but this is defensive.
NEVER_EDIT_TAGS = {
    "script", "style", "noscript", "template",
}

TOKEN_RE = re.compile(r"⟦/?(\d+)⟧")
_OPEN_RE = re.compile(r"⟦(\d+)⟧")
_CLOSE_RE = re.compile(r"⟦/(\d+)⟧")


# =============================================================================
# Gemini structured output models
# =============================================================================

class RewrittenBlock(BaseModel):
    id: str
    revised_text: str


class BlocksResponse(BaseModel):
    blocks: list[RewrittenBlock]


# =============================================================================
# Internal data structure
# =============================================================================

@dataclass
class Segment:
    id: str
    marker: Comment
    original_text: str
    token_map: dict[str, Tag]
    original_nodes: list[Any]

    stage1_text: Optional[str] = None
    revised_text: Optional[str] = None
    issues: list[str] = field(default_factory=list)


# =============================================================================
# Gemini call helper
# =============================================================================

def call_gemini_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float,
    max_retries: int = 3,
) -> BlocksResponse:
    """
    Calls Gemini using the existing global `client` variable.

    Expects JSON shaped like:
    {
      "blocks": [
        {"id": "...", "revised_text": "..."}
      ]
    }
    """
    full_prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=BlocksResponse,
                ),
            )

            if hasattr(response, "parsed") and response.parsed is not None:
                if isinstance(response.parsed, BlocksResponse):
                    return response.parsed
                return BlocksResponse.model_validate(response.parsed)

            return BlocksResponse.model_validate_json(response.text)

        except Exception as exc:
            last_error = exc
            sleep_seconds = 2 ** attempt
            print(f"Gemini call failed. Retrying in {sleep_seconds}s. Error: {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_error}")


# =============================================================================
# HTML tokenization helpers
# =============================================================================

def clone_tag_template(tag: Tag) -> Tag:
    """
    Store only the tag name and attributes.
    Children are reconstructed separately.
    """
    factory = BeautifulSoup("", "html.parser")
    return factory.new_tag(tag.name, **dict(tag.attrs))


def make_node_copy(node: Any) -> Any:
    """
    Store a safe copy of an original node so we can restore it if validation fails.
    """
    return copy(node)


def is_editable_direct_child(child: Any) -> bool:
    """
    A direct child is editable if it is text, an inline tag, or a display/void tag
    that appears inside a text run.
    """
    if isinstance(child, NavigableString) and not isinstance(child, Comment):
        return True

    if isinstance(child, Tag):
        name = (child.name or "").lower()
        return name in INLINE_TAGS or name in VOID_OR_DISPLAY_TAGS

    return False


def tokenized_plain_text_has_content(tokenized_text: str) -> bool:
    """
    A segment should be sent to the LLM only if it contains actual text after
    placeholder tokens are removed.

    This avoids creating rewrite segments for standalone images.
    """
    return bool(TOKEN_RE.sub("", tokenized_text).strip())


def tokenize_node_list(nodes: list[Any]) -> tuple[str, dict[str, Tag]]:
    """
    Tokenize a list of direct nodes from the DOM.

    Inline tags become tokens:
        <strong>Hello</strong>
        -> ⟦1⟧Hello⟦/1⟧

    Display/void tags inside text become empty tokens:
        <img src="x.png">
        -> ⟦1⟧⟦/1⟧
    """
    token_map: dict[str, Tag] = {}
    counter = [0]

    def next_token_id() -> str:
        counter[0] += 1
        return str(counter[0])

    def walk_node(node: Any) -> str:
        if isinstance(node, NavigableString) and not isinstance(node, Comment):
            return str(node)

        if not isinstance(node, Tag):
            return ""

        name = (node.name or "").lower()

        if name in INLINE_TAGS:
            tid = next_token_id()
            token_map[tid] = clone_tag_template(node)
            inner = "".join(walk_node(child) for child in node.children)
            return f"⟦{tid}⟧{inner}⟦/{tid}⟧"

        if name in VOID_OR_DISPLAY_TAGS:
            tid = next_token_id()
            token_map[tid] = clone_tag_template(node)
            return f"⟦{tid}⟧⟦/{tid}⟧"

        # Defensive fallback:
        # If an unexpected tag appears inside a run, preserve its visible text
        # rather than dropping it.
        return node.get_text()

    text = "".join(walk_node(node) for node in nodes)
    return text, token_map


# =============================================================================
# Recursive, nested-safe extraction
# =============================================================================

def extract_segments(soup: BeautifulSoup) -> list[Segment]:
    """
    Extract editable text runs from the DOM.

    Important design:
    - The code does NOT process only top-level blocks.
    - It walks recursively.
    - It splits direct text around nested structures.

    Example:
        <li>
          Main text
          <ol><li>Nested text</li></ol>
          trailing text
        </li>

    becomes separate editable segments:
        "Main text"
        "Nested text"
        "trailing text"

    This avoids dropping or collapsing nested bullets/headings/tables.
    """
    segments: list[Segment] = []

    def should_process_direct_text_in(node: Tag) -> bool:
        name = (node.name or "").lower()

        if name in NEVER_EDIT_TAGS:
            return False

        # Process standard text containers.
        if name in TEXT_CONTAINER_TAGS:
            return True

        # Also process direct text inside structural containers if present.
        # This helps with fragments like:
        #   <div>Some intro text <p>Paragraph</p></div>
        if name in STRUCTURAL_TAGS:
            return True

        return False

    def flush_run(parent: Tag, run_nodes: list[Any]) -> None:
        if not run_nodes:
            return

        tokenized_text, token_map = tokenize_node_list(run_nodes)

        if not tokenized_plain_text_has_content(tokenized_text):
            return

        segment_id = f"s_{uuid.uuid4().hex[:8]}"
        marker = Comment(f"REWRITE_SEGMENT:{segment_id}")

        original_nodes = [make_node_copy(node) for node in run_nodes]

        # Insert marker at the original location.
        run_nodes[0].insert_before(marker)

        # Remove the original editable run from the DOM.
        for node in run_nodes:
            node.extract()

        segments.append(
            Segment(
                id=segment_id,
                marker=marker,
                original_text=tokenized_text,
                token_map=token_map,
                original_nodes=original_nodes,
            )
        )

    def walk(node: Tag) -> None:
        if not isinstance(node, Tag):
            return

        name = (node.name or "").lower()

        if name in NEVER_EDIT_TAGS:
            return

        children_snapshot = list(node.children)

        if should_process_direct_text_in(node):
            current_run: list[Any] = []

            for child in children_snapshot:
                if is_editable_direct_child(child):
                    current_run.append(child)
                    continue

                flush_run(node, current_run)
                current_run = []

                if isinstance(child, Tag):
                    walk(child)

            flush_run(node, current_run)

        else:
            for child in children_snapshot:
                if isinstance(child, Tag):
                    walk(child)

    for child in list(soup.children):
        if isinstance(child, Tag):
            walk(child)

    return segments


# =============================================================================
# Chunking
# =============================================================================

def plain_text(segment: Segment) -> str:
    return TOKEN_RE.sub("", segment.original_text)


def stage2_size_text(segment: Segment) -> str:
    return segment.original_text + (segment.stage1_text or "")


def chunk_segments_by_chars(
    segments: list[Segment],
    max_chars: int,
    text_fn: Callable[[Segment], str],
) -> list[list[Segment]]:
    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    current_chars = 0

    for segment in segments:
        length = len(text_fn(segment))

        if length > max_chars:
            if current:
                chunks.append(current)
                current = []
                current_chars = 0

            chunks.append([segment])
            continue

        if current and current_chars + length > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(segment)
        current_chars += length

    if current:
        chunks.append(current)

    return chunks


# =============================================================================
# Prompt builders
# =============================================================================

def build_stage1_system_prompt(transformation_rules: str) -> str:
    return f"""
You are a careful document editor.

You will receive text segments extracted from an HTML document.

Your task:
- Apply the transformation rules provided by the user.
- Preserve the original meaning.
- Preserve names, numbers, dates, product names, technical terms, and references.
- Preserve the source language of each segment.
- If a segment is in English, return English.
- If a segment is in Dutch, return Dutch.
- If the document contains mixed English and Dutch, preserve the language per segment.
- Do not translate the document unless the transformation rules explicitly ask for translation.
- Do not add new facts.
- Do not remove important information.
- Return one revised_text for every id.

Transformation rules:
{transformation_rules}

Return only valid JSON matching the requested schema.
"""


def build_stage2_system_prompt() -> str:
    return """
You receive text segments from an HTML document.

For each segment you get:

1. text:
The original text with placeholder tokens like ⟦1⟧...⟦/1⟧.
These tokens represent original inline HTML tags such as links, bold text, spans,
line breaks, or images.

2. reference:
The already transformed plain-text version of the same segment.

Your task:
- Use the reference text as the final wording.
- Place every placeholder token from text into the correct position in reference.
- Do not output HTML.
- Do not invent, remove, rename, or duplicate tokens.
- Every opening token and closing token from text must appear exactly once.
- Preserve the language of the reference.
- Do not translate.
- Do not rewrite the reference except for the smallest possible adjustment needed
  to place a token naturally.
- If a token represented a link or emphasized phrase, place it around the closest
  matching phrase in the reference.
- If a token represented an image, br, or hr, keep it as an empty token pair
  near its original semantic position.

Return only valid JSON matching the requested schema.
"""


# =============================================================================
# Stage 1: transform plain text
# =============================================================================

def rewrite_stage1_batch(
    segments: list[Segment],
    *,
    transformation_rules: str,
    model: str,
    max_retries: int,
) -> None:
    payload = [
        {
            "id": segment.id,
            "text": plain_text(segment),
        }
        for segment in segments
    ]

    user_prompt = (
        "Transform the following text segments.\n\n"
        + json.dumps({"blocks": payload}, ensure_ascii=False, indent=2)
    )

    result = call_gemini_json(
        build_stage1_system_prompt(transformation_rules),
        user_prompt,
        model=model,
        temperature=0.2,
        max_retries=max_retries,
    )

    by_id = {item.id: item.revised_text for item in result.blocks}

    for segment in segments:
        revised = by_id.get(segment.id)

        if not revised or not revised.strip():
            segment.issues.append("stage1_missing_or_empty")
            segment.stage1_text = plain_text(segment)
        else:
            segment.stage1_text = revised


def run_stage1(
    segments: list[Segment],
    *,
    transformation_rules: str,
    model: str,
    max_retries: int,
    max_chars_per_batch: int,
) -> None:
    chunks = chunk_segments_by_chars(
        segments,
        max_chars_per_batch,
        plain_text,
    )

    for index, chunk in enumerate(chunks, start=1):
        print(f"Stage 1 batch {index}/{len(chunks)}: {len(chunk)} segments")

        rewrite_stage1_batch(
            chunk,
            transformation_rules=transformation_rules,
            model=model,
            max_retries=max_retries,
        )


# =============================================================================
# Stage 2: place inline tokens into transformed text
# =============================================================================

def rewrite_stage2_batch(
    segments: list[Segment],
    *,
    model: str,
    max_retries: int,
) -> None:
    payload = [
        {
            "id": segment.id,
            "text": segment.original_text,
            "reference": segment.stage1_text or plain_text(segment),
        }
        for segment in segments
    ]

    user_prompt = (
        "Place the placeholder tokens from 'text' into the matching 'reference' "
        "for each segment.\n\n"
        + json.dumps({"blocks": payload}, ensure_ascii=False, indent=2)
    )

    result = call_gemini_json(
        build_stage2_system_prompt(),
        user_prompt,
        model=model,
        temperature=0.1,
        max_retries=max_retries,
    )

    by_id = {item.id: item.revised_text for item in result.blocks}

    for segment in segments:
        revised = by_id.get(segment.id)

        if not revised or not revised.strip():
            segment.issues.append("stage2_missing_or_empty")
            segment.revised_text = None
        else:
            segment.revised_text = revised


def run_stage2(
    segments: list[Segment],
    *,
    model: str,
    max_retries: int,
    max_chars_per_batch: int,
) -> None:
    chunks = chunk_segments_by_chars(
        segments,
        max_chars_per_batch,
        stage2_size_text,
    )

    for index, chunk in enumerate(chunks, start=1):
        print(f"Stage 2 batch {index}/{len(chunks)}: {len(chunk)} segments")

        rewrite_stage2_batch(
            chunk,
            model=model,
            max_retries=max_retries,
        )


# =============================================================================
# Strict token validation
# =============================================================================

def validate_token_structure(segment: Segment) -> bool:
    """
    Validate that Stage 2 preserved all placeholder tokens safely.
    """
    text = segment.revised_text or ""

    if not text.strip():
        segment.issues.append("empty_revision")
        return False

    expected_ids = set(segment.token_map.keys())
    found_ids = set(TOKEN_RE.findall(text))

    missing = expected_ids - found_ids
    unexpected = found_ids - expected_ids

    if missing:
        segment.issues.append(f"token_missing:{sorted(missing)}")

    if unexpected:
        segment.issues.append(f"token_unexpected:{sorted(unexpected)}")

    stack: list[str] = []
    opens: dict[str, int] = {}
    closes: dict[str, int] = {}

    index = 0

    while index < len(text):
        open_match = _OPEN_RE.match(text, index)

        if open_match:
            token_id = open_match.group(1)
            opens[token_id] = opens.get(token_id, 0) + 1
            stack.append(token_id)
            index = open_match.end()
            continue

        close_match = _CLOSE_RE.match(text, index)

        if close_match:
            token_id = close_match.group(1)
            closes[token_id] = closes.get(token_id, 0) + 1

            if not stack:
                segment.issues.append(f"token_stray_close:{token_id}")
            elif stack[-1] != token_id:
                segment.issues.append(
                    f"token_misnested:expected_close_for_{stack[-1]}_got_{token_id}"
                )

                # Defensive recovery so we can continue checking the rest.
                if token_id in stack:
                    stack.remove(token_id)
            else:
                stack.pop()

            index = close_match.end()
            continue

        index += 1

    if stack:
        segment.issues.append(f"token_unclosed:{stack}")

    for token_id in expected_ids:
        if opens.get(token_id, 0) != 1:
            segment.issues.append(
                f"token_open_count_invalid:{token_id}:{opens.get(token_id, 0)}"
            )

        if closes.get(token_id, 0) != 1:
            segment.issues.append(
                f"token_close_count_invalid:{token_id}:{closes.get(token_id, 0)}"
            )

    return not any(
        issue.startswith("token_") or issue == "empty_revision"
        for issue in segment.issues
    )


# =============================================================================
# Detokenization / restoration
# =============================================================================

def detokenize_text(segment: Segment) -> list[Any]:
    """
    Convert placeholder-token text back into BeautifulSoup nodes.
    """
    factory = BeautifulSoup("", "html.parser")
    text = segment.revised_text or ""

    def build(source: str, index: int, into: list[Any], stop_id: Optional[str]) -> int:
        while index < len(source):
            close_match = _CLOSE_RE.match(source, index)

            if close_match:
                if close_match.group(1) == stop_id:
                    return close_match.end()

                # Stray close. Skip defensively.
                index = close_match.end()
                continue

            open_match = _OPEN_RE.match(source, index)

            if open_match:
                token_id = open_match.group(1)
                template = segment.token_map.get(token_id)

                if template is None:
                    index = open_match.end()
                    continue

                tag = factory.new_tag(template.name, **dict(template.attrs))
                inner: list[Any] = []

                index = build(source, open_match.end(), inner, token_id)

                for child in inner:
                    tag.append(child)

                into.append(tag)
                continue

            next_token = source.find("⟦", index)

            if next_token == -1:
                if source[index:]:
                    into.append(NavigableString(source[index:]))
                return len(source)

            if next_token > index:
                into.append(NavigableString(source[index:next_token]))

            index = next_token

        return index

    output: list[Any] = []
    build(text, 0, output, None)
    return output


def replace_marker_with_nodes(marker: Comment, nodes: list[Any]) -> None:
    """
    Replace a marker comment with the provided nodes.
    """
    if marker.parent is None:
        return

    for node in nodes:
        marker.insert_before(node)

    marker.extract()


def restore_original_segment(segment: Segment) -> None:
    """
    Safe fallback if Stage 2 fails validation.
    """
    original_copies = [make_node_copy(node) for node in segment.original_nodes]
    replace_marker_with_nodes(segment.marker, original_copies)


def apply_rewritten_segment(segment: Segment) -> None:
    """
    Replace marker with the valid transformed content.
    """
    new_nodes = detokenize_text(segment)
    replace_marker_with_nodes(segment.marker, new_nodes)


# =============================================================================
# Public function
# =============================================================================

def transform_document(
    html_document: str,
    transformation_rules: str,
    *,
    model: str = DEFAULT_MODEL,
    stage1_max_chars_per_batch: int = STAGE1_MAX_CHARS_PER_BATCH,
    stage2_max_chars_per_batch: int = STAGE2_MAX_CHARS_PER_BATCH,
    max_retries: int = 3,
    return_report: bool = True,
) -> tuple[str, dict] | str:
    """
    Transform an HTML document string using the global Gemini `client`.

    Parameters
    ----------
    html_document:
        Stringified HTML document or HTML fragment.

    transformation_rules:
        Plain-language rules for how the text should be transformed.
        Example:
            "Rewrite in clear B1-level language. Keep the same language as the source."

    model:
        Gemini model name.

    return_report:
        If True, returns:
            (rewritten_html, report)

        If False, returns:
            rewritten_html

    Important behavior
    ------------------
    - The model never rewrites raw HTML.
    - HTML structure is handled by Python.
    - Inline tags are represented as placeholder tokens.
    - Nested lists, nested headings, and nested table content are handled recursively.
    - If a segment fails token validation, the original segment is restored.
    - The model is told to preserve the source language:
        English stays English.
        Dutch stays Dutch.
    """
    if not isinstance(html_document, str):
        raise TypeError("html_document must be a string.")

    if not isinstance(transformation_rules, str):
        raise TypeError("transformation_rules must be a string.")

    soup = BeautifulSoup(html_document, "html.parser")

    segments = extract_segments(soup)

    report = {
        "model": model,
        "total_segments": len(segments),
        "stage1_batches": 0,
        "stage2_batches": 0,
        "updated_segments": 0,
        "fallback_segments": 0,
        "failed_segments": [],
        "warnings": [],
    }

    stage1_chunks = chunk_segments_by_chars(
        segments,
        stage1_max_chars_per_batch,
        plain_text,
    )

    stage2_chunks = chunk_segments_by_chars(
        segments,
        stage2_max_chars_per_batch,
        stage2_size_text,
    )

    report["stage1_batches"] = len(stage1_chunks)
    report["stage2_batches"] = len(stage2_chunks)

    print(f"Extracted {len(segments)} editable segments")
    print(f"Stage 1 batches: {len(stage1_chunks)}")

    for index, chunk in enumerate(stage1_chunks, start=1):
        print(f"Stage 1 batch {index}/{len(stage1_chunks)}: {len(chunk)} segments")

        rewrite_stage1_batch(
            chunk,
            transformation_rules=transformation_rules,
            model=model,
            max_retries=max_retries,
        )

    # Recompute Stage 2 chunks after Stage 1, because stage1_text is now filled.
    stage2_chunks = chunk_segments_by_chars(
        segments,
        stage2_max_chars_per_batch,
        stage2_size_text,
    )

    report["stage2_batches"] = len(stage2_chunks)
    print(f"Stage 2 batches: {len(stage2_chunks)}")

    for index, chunk in enumerate(stage2_chunks, start=1):
        print(f"Stage 2 batch {index}/{len(stage2_chunks)}: {len(chunk)} segments")

        rewrite_stage2_batch(
            chunk,
            model=model,
            max_retries=max_retries,
        )

    for segment in segments:
        valid = validate_token_structure(segment)

        if valid:
            apply_rewritten_segment(segment)
            report["updated_segments"] += 1

            if segment.issues:
                report["warnings"].append(
                    {
                        "id": segment.id,
                        "issues": segment.issues,
                    }
                )
        else:
            restore_original_segment(segment)
            report["fallback_segments"] += 1

            report["failed_segments"].append(
                {
                    "id": segment.id,
                    "issues": segment.issues,
                    "original_text": segment.original_text,
                    "stage1_text": segment.stage1_text,
                    "revised_text": segment.revised_text,
                }
            )

    rewritten_html = str(soup)

    if return_report:
        return rewritten_html, report

    return rewritten_html

rules = """
Make the text clearer and more professional.
Keep the same language as the source text.
Do not translate.
Do not change names, links, dates, numbers, or technical terms.
Do not change existing facts, numbers, or dates.
Do not create new information.
Do not delete existing information.
Avoid compound sentences.
Use short and active statements.
Use B1 language level.
Use correct grammar and spelling.
Use inclusive language.
"""

html_text = """
<div class="document">
  <h1>Internal Process for Customer Request Handling</h1>

  <p>
    This document is written for employees that need to know what they must do when a customer request is coming in and it is important that the request is handled in the correct way because otherwise the customer can receive different answers from different teams.
  </p>

  <p>
    The process started on <strong>12 March 2024</strong> and it applies to all requests that are received after this date. The owner of the process is <span class="person-name">Mike Johnson</span>. Mike Johnson is responsible for making sure the process is followed by the teams.
  </p>

  <h2>Purpose of the process</h2>

  <p>
    The purpose of this process is that we want to make sure that every customer receives an answer that is clear, complete and sent at the right moment, and that the same information is not asked many times by different people in the company.
  </p>

  <p>
    Employees must use the <a href="https://example.com/customer-request-form">Customer Request Form</a> before they start working on the request. The form has <strong>7 required fields</strong>. These fields may not be skipped.
  </p>

  <h2>What employees must do</h2>

  <ul>
    <li>
      Read the full request before you send an answer, because sometimes the most important information is written at the end of the request.
    </li>
    <li>
      Check if the customer already has an open ticket.
      <ol>
        <li>
          Search by customer name, email address and contract number.
        </li>
        <li>
          If there is already an open ticket, add the new information to that ticket and do not create a second ticket.
        </li>
        <li>
          If there is no open ticket, create a new ticket and add the request date.
        </li>
      </ol>
    </li>
    <li>
      Make sure that the status is changed to <span class="status">In progress</span> when somebody starts working on the request.
    </li>
    <li>
      Do not close the request before the customer has received the final answer.
    </li>
  </ul>

  <h3>Important exception</h3>

  <p>
    If the request is about a payment problem, the employee must send the request to the Finance team within <strong>2 working days</strong>. This rule was approved on <span class="date">4 June 2024</span>.
  </p>

  <blockquote>
    Requests about invoices, payment delays and refund questions must always be checked carefully before an answer is sent to the customer.
  </blockquote>

  <h2>Information that must not be changed</h2>

  <p>
    The following information must stay exactly the same in every document and message: customer name, contract number, request date, payment amount and ticket number.
  </p>

  <table>
    <thead>
      <tr>
        <th>Field</th>
        <th>Required action</th>
        <th>Deadline</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Customer name</td>
        <td>The name must be copied from the original request and it must not be shortened or changed.</td>
        <td>Same day</td>
      </tr>
      <tr>
        <td>Contract number</td>
        <td>The number must be checked in the system before the ticket is created.</td>
        <td>Same day</td>
      </tr>
      <tr>
        <td>Payment issue</td>
        <td>The issue must be sent to Finance when the amount is higher than €500.</td>
        <td>2 working days</td>
      </tr>
    </tbody>
  </table>

  <h2>Nested instructions</h2>

  <ul>
    <li>
      When the customer asks for an update, the employee must first check the current ticket status.
      <ul>
        <li>
          If the status is <strong>Waiting for customer</strong>, ask the customer for the missing information again.
        </li>
        <li>
          If the status is <strong>Waiting for internal team</strong>, contact the team owner.
          <ol>
            <li>
              Sent one reminder after <span>3 working dayss</span>.
            </li>
            <li>
              Send a second reminded after <span>5 working working days</span>.
            </li>
            <li>
              Escalate the ticket to <strong>Mike Johnson</strong> after <span>7 working days</span>.
            </li>
          </ol>
        </li>
      </ul>
    </li>
    <li>
      When the customer says that the answer is not clear, the employee must rewrite the answer and avoid long explanations that are difficult to understand.
    </li>
  </ul>

  <figure>
    <img src="customer-request-flow.png" alt="Customer request flow diagram">
    <figcaption>
      This image shows the customer request flow from the moment the request is received until the final answer is sent.
    </figcaption>
  </figure>

  <h2>Final check before closing</h2>

  <p>
    Before the ticket is closed, the employee must check if all required information is present, if the customer received the answer, if the ticket status is correct and if no duplicate ticket was created.
  </p>

  <p>
    The ticket may be closed only when the final answer was sent and the customer does not need to send more information. The closing note must include the date, the employee name and a short explanation of the answer that was sent.
  </p>
</div>
"""

html_text

new_html, report = transform_document(
    html_document=html_text,
    transformation_rules=rules,
)

new_html
