from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from bs4 import BeautifulSoup, NavigableString, Tag, Comment


# =============================================================================
# Assumption
# =============================================================================
# This code assumes you already have a working OpenAI client variable named:
#
#     client
#
# Example setup, NOT included here:
#
#     from openai import OpenAI
#     client = OpenAI(api_key="...")
# =============================================================================


# =============================================================================
# Config
# =============================================================================

DEFAULT_MODEL = "gpt-5.1"

PREVIOUS_CONTEXT_SEGMENTS = 2
NEXT_CONTEXT_SEGMENTS = 1
MAX_CONTEXT_CHARS = 1_200

TEXT_CONTAINER_TAGS = {
    "p", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "td", "th",
    "blockquote", "figcaption",
    "label", "caption",
}

STRUCTURAL_TAGS = {
    "div", "section", "article", "main", "aside",
    "ul", "ol",
    "table", "thead", "tbody", "tfoot", "tr",
    "figure",
    "dl", "dt", "dd",
}

INLINE_TAGS = {
    "b", "strong", "i", "em", "u", "a", "span",
    "sup", "sub", "mark", "small", "code",
    "abbr", "cite", "q", "s", "del", "ins",
}

VOID_OR_DISPLAY_TAGS = {
    "img", "br", "hr",
}

NEVER_EDIT_TAGS = {
    "script", "style", "noscript", "template",
}

TOKEN_RE = re.compile(r"⟦/?(\d+)⟧")
_OPEN_RE = re.compile(r"⟦(\d+)⟧")
_CLOSE_RE = re.compile(r"⟦/(\d+)⟧")


# =============================================================================
# Structured output schema
# =============================================================================

BLOCKS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "revised_text": {"type": "string"},
                },
                "required": ["id", "revised_text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


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
    parent_tag: str

    section_heading: str = ""
    previous_context: str = ""
    next_context: str = ""

    revised_text: Optional[str] = None
    issues: list[str] = field(default_factory=list)


# =============================================================================
# OpenAI call helper
# =============================================================================

def call_openai_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: Optional[float] = 0.1,
    max_retries: int = 3,
) -> dict:
    """
    Calls OpenAI Chat Completions using the existing global `client`.

    Expected response:
    {
      "blocks": [
        {"id": "...", "revised_text": "..."}
      ]
    }
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt.strip(),
                    },
                    {
                        "role": "user",
                        "content": user_prompt.strip(),
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rewrite_blocks_response",
                        "strict": True,
                        "schema": BLOCKS_RESPONSE_SCHEMA,
                    },
                },
            }

            if temperature is not None:
                kwargs["temperature"] = temperature

            response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content

            if not content:
                raise ValueError("OpenAI returned an empty response.")

            result = json.loads(content)

            if "blocks" not in result or not isinstance(result["blocks"], list):
                raise ValueError(f"Invalid response shape: {result}")

            return result

        except Exception as exc:
            last_error = exc
            sleep_seconds = 2 ** attempt
            print(f"OpenAI call failed. Retrying in {sleep_seconds}s. Error: {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_error}")


# =============================================================================
# HTML copy/tokenization helpers
# =============================================================================

def clone_tag_template(tag: Tag) -> Tag:
    """
    Store only tag name and attributes.
    Children are reconstructed from tokens later.
    """
    factory = BeautifulSoup("", "html.parser")
    return factory.new_tag(tag.name, attrs=dict(tag.attrs))


def clone_original_node(node: Any) -> Any:
    """
    Deep-copy an original node so a failed segment can be restored safely.
    """
    if isinstance(node, NavigableString) and not isinstance(node, Comment):
        return NavigableString(str(node))

    if isinstance(node, Comment):
        return Comment(str(node))

    if isinstance(node, Tag):
        parsed = BeautifulSoup(str(node), "html.parser")
        if parsed.contents:
            return parsed.contents[0]
        return NavigableString("")

    return NavigableString(str(node))


def is_editable_direct_child(child: Any) -> bool:
    """
    A direct child is editable if it is text, an inline tag, or a display/void tag.
    """
    if isinstance(child, NavigableString) and not isinstance(child, Comment):
        return True

    if isinstance(child, Tag):
        name = (child.name or "").lower()
        return name in INLINE_TAGS or name in VOID_OR_DISPLAY_TAGS

    return False


def tokenized_plain_text_has_content(tokenized_text: str) -> bool:
    """
    Only send a segment to the model if it has real text after removing tokens.
    """
    return bool(TOKEN_RE.sub("", tokenized_text).strip())


def plain_text_from_tokenized(tokenized_text: str) -> str:
    """
    Remove placeholder tokens and normalize whitespace.
    Used for context.
    """
    text = TOKEN_RE.sub("", tokenized_text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_node_list(nodes: list[Any]) -> tuple[str, dict[str, Tag]]:
    """
    Convert direct DOM nodes into tokenized text.

    Example:
        <a href="/form">Customer Request Form</a>

    becomes:
        ⟦1⟧Customer Request Form⟦/1⟧
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
            token_id = next_token_id()
            token_map[token_id] = clone_tag_template(node)
            inner = "".join(walk_node(child) for child in node.children)
            return f"⟦{token_id}⟧{inner}⟦/{token_id}⟧"

        if name in VOID_OR_DISPLAY_TAGS:
            token_id = next_token_id()
            token_map[token_id] = clone_tag_template(node)
            return f"⟦{token_id}⟧⟦/{token_id}⟧"

        # Defensive fallback:
        # If an unexpected tag appears inside editable text,
        # preserve visible text rather than dropping it.
        return node.get_text()

    text = "".join(walk_node(node) for node in nodes)
    return text, token_map


# =============================================================================
# Recursive, nested-safe extraction
# =============================================================================

def extract_segments(soup: BeautifulSoup) -> list[Segment]:
    """
    Extract editable text runs from the DOM.

    Structural tags stay in the DOM.
    Only editable text and inline tags are replaced by marker comments.
    """
    segments: list[Segment] = []

    def should_process_direct_text_in(node: Tag) -> bool:
        name = (node.name or "").lower()

        if name in NEVER_EDIT_TAGS:
            return False

        if name in TEXT_CONTAINER_TAGS:
            return True

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

        original_nodes = [clone_original_node(node) for node in run_nodes]

        # Put a marker where this text used to be.
        run_nodes[0].insert_before(marker)

        # Remove the original editable nodes.
        for node in run_nodes:
            node.extract()

        segments.append(
            Segment(
                id=segment_id,
                marker=marker,
                original_text=tokenized_text,
                token_map=token_map,
                original_nodes=original_nodes,
                parent_tag=(parent.name or "").lower(),
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
# Context building
# =============================================================================

def segment_plain_text(segment: Segment) -> str:
    return plain_text_from_tokenized(segment.original_text)


def trim_context(text: str, max_chars: int) -> str:
    """
    Keep context compact.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    return text[-max_chars:].strip()


def add_context_to_segments(
    segments: list[Segment],
    *,
    previous_count: int = PREVIOUS_CONTEXT_SEGMENTS,
    next_count: int = NEXT_CONTEXT_SEGMENTS,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> None:
    """
    Add read-only context to each segment.

    Context is built from original text, not rewritten text.
    """
    current_heading = ""

    # First pass: nearest previous heading.
    for segment in segments:
        segment.section_heading = current_heading

        if segment.parent_tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading_text = segment_plain_text(segment)
            if heading_text:
                current_heading = heading_text

    # Second pass: previous and next nearby text.
    for index, segment in enumerate(segments):
        previous_parts = []
        next_parts = []

        previous_start = max(0, index - previous_count)
        previous_segments = segments[previous_start:index]

        for previous_segment in previous_segments:
            text = segment_plain_text(previous_segment)
            if text:
                previous_parts.append(text)

        next_end = min(len(segments), index + 1 + next_count)
        next_segments = segments[index + 1:next_end]

        for next_segment in next_segments:
            text = segment_plain_text(next_segment)
            if text:
                next_parts.append(text)

        segment.previous_context = trim_context(
            " ".join(previous_parts),
            max_context_chars,
        )

        segment.next_context = trim_context(
            " ".join(next_parts),
            max_context_chars,
        )


# =============================================================================
# Prompt
# =============================================================================

def build_onepass_system_prompt(transformation_rules: str) -> str:
    return f"""
You are a careful document editor.

You will receive text segments extracted from an HTML document.

Each item may contain:
- id: unique segment id.
- parent_tag: the HTML tag that contained this text, such as p, li, h2, td.
- section_heading: the nearest previous heading.
- previous_context: nearby text before this segment.
- next_context: nearby text after this segment.
- text: the only text you must rewrite.

Important:
- previous_context, next_context, section_heading, and parent_tag are read-only.
- Use context only to improve flow, avoid repetition, and choose clear pronouns.
- Do not include context text in revised_text.
- Rewrite only the text field.
- The text may contain placeholder tokens like ⟦1⟧...⟦/1⟧.
- These tokens represent original inline HTML tags such as links, bold text,
  spans, line breaks, or images.
- You must not output raw HTML.

Your task:
- Rewrite each text field directly according to the transformation rules.
- Fix grammar and spelling.
- Improve clarity and flow.
- Avoid unnecessary repetition across nearby sentences.
- Use pronouns when the reference is clear.
- Do not use pronouns when they could be ambiguous.
- Preserve the source language of each segment.
- If a segment is in English, return English.
- If a segment is in Dutch, return Dutch.
- If the document contains mixed English and Dutch, preserve the language per segment.
- Do not translate unless the transformation rules explicitly ask for translation.
- Preserve the original meaning.
- Preserve names, numbers, dates, product names, technical terms, and references.
- Do not change existing facts.
- Do not create new information.
- Do not delete existing information.
- Return one revised_text for every id.

Token rules:
- Keep every token exactly.
- Do not rename tokens.
- Do not remove tokens.
- Do not duplicate tokens.
- Every opening token must appear exactly once.
- Every closing token must appear exactly once.
- Tokens may move only when needed so the original formatting still wraps the
  closest matching words in the rewritten text.
- Do not output HTML.
- Do not explain your answer.

Transformation rules:
{transformation_rules}

Return only valid JSON matching the requested schema.
"""


# =============================================================================
# One single OpenAI rewrite call
# =============================================================================

def rewrite_all_segments_once(
    segments: list[Segment],
    *,
    transformation_rules: str,
    model: str,
    temperature: Optional[float],
    max_retries: int,
) -> None:
    """
    Send all extracted segments to OpenAI in one single API call.
    No chunking.
    """
    payload = [
        {
            "id": segment.id,
            "parent_tag": segment.parent_tag,
            "section_heading": segment.section_heading,
            "previous_context": segment.previous_context,
            "next_context": segment.next_context,
            "text": segment.original_text,
        }
        for segment in segments
    ]

    user_prompt = (
        "Rewrite the following tokenized text segments. "
        "Use the context fields only as read-only context. "
        "Return only the rewritten version of each text field.\n\n"
        + json.dumps({"blocks": payload}, ensure_ascii=False, indent=2)
    )

    result = call_openai_json(
        build_onepass_system_prompt(transformation_rules),
        user_prompt,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

    by_id = {
        item.get("id"): item.get("revised_text", "")
        for item in result.get("blocks", [])
    }

    for segment in segments:
        revised = by_id.get(segment.id)

        if not revised or not revised.strip():
            segment.issues.append("onepass_missing_or_empty")
            segment.revised_text = None
        else:
            segment.revised_text = revised


# =============================================================================
# Strict token validation
# =============================================================================

def validate_token_structure(segment: Segment) -> bool:
    """
    Validate that the model preserved all placeholder tokens safely.
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
        issue.startswith("token_")
        or issue in {"empty_revision", "onepass_missing_or_empty"}
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

                index = close_match.end()
                continue

            open_match = _OPEN_RE.match(source, index)

            if open_match:
                token_id = open_match.group(1)
                template = segment.token_map.get(token_id)

                if template is None:
                    index = open_match.end()
                    continue

                tag = factory.new_tag(template.name, attrs=dict(template.attrs))
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
    Replace a marker comment with new nodes.
    """
    if marker.parent is None:
        return

    for node in nodes:
        marker.insert_before(node)

    marker.extract()


def restore_original_segment(segment: Segment) -> None:
    """
    Fallback: restore original text if the model output is invalid.
    """
    original_copies = [clone_original_node(node) for node in segment.original_nodes]
    replace_marker_with_nodes(segment.marker, original_copies)


def apply_rewritten_segment(segment: Segment) -> None:
    """
    Apply valid rewritten text.
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
    temperature: Optional[float] = 0.1,
    max_retries: int = 3,
    previous_context_segments: int = PREVIOUS_CONTEXT_SEGMENTS,
    next_context_segments: int = NEXT_CONTEXT_SEGMENTS,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    return_report: bool = False,
) -> str | tuple[str, dict]:
    """
    Transform an HTML document string in one OpenAI API call.

    No chunking.

    Parameters
    ----------
    html_document:
        Stringified HTML document or HTML fragment.

    transformation_rules:
        Rules/prompt telling the model how to transform the text.

    return_report:
        If False:
            returns transformed HTML string.

        If True:
            returns (transformed HTML string, report dict).
    """
    if not isinstance(html_document, str):
        raise TypeError("html_document must be a string.")

    if not isinstance(transformation_rules, str):
        raise TypeError("transformation_rules must be a string.")

    soup = BeautifulSoup(html_document, "html.parser")

    segments = extract_segments(soup)

    add_context_to_segments(
        segments,
        previous_count=previous_context_segments,
        next_count=next_context_segments,
        max_context_chars=max_context_chars,
    )

    report = {
        "model": model,
        "architecture": "one_pass_no_chunking_openai_chat_completions",
        "total_segments": len(segments),
        "updated_segments": 0,
        "fallback_segments": 0,
        "failed_segments": [],
        "warnings": [],
    }

    print(f"Extracted {len(segments)} editable segments")
    print("Sending all segments in one OpenAI call. No chunking is used.")

    if segments:
        rewrite_all_segments_once(
            segments,
            transformation_rules=transformation_rules,
            model=model,
            temperature=temperature,
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
                    "parent_tag": segment.parent_tag,
                    "section_heading": segment.section_heading,
                    "previous_context": segment.previous_context,
                    "next_context": segment.next_context,
                    "original_text": segment.original_text,
                    "revised_text": segment.revised_text,
                }
            )

    rewritten_html = str(soup)

    if return_report:
        return rewritten_html, report

    return rewritten_html


# Optional alias for the typo you mentioned: ransform_document
def ransform_document(*args, **kwargs):
    return transform_document(*args, **kwargs)


rules = """
Make the text clearer and easier to understand.
Keep the same language as the source text.
Do not translate.
Do not change names, links, dates, numbers, or technical terms.
Do not change existing facts, numbers, or dates.
Do not create new information.
Do not delete existing information.
Use simple B1 language.
Prefer common words over formal business words.
Avoid compound sentences.
Use short and active statements.
Use correct grammar and spelling.
Avoid unnecessary repetition.
Use pronouns when the reference is clear.
"""

new_html = transform_document(html_text, rules)

print(new_html)