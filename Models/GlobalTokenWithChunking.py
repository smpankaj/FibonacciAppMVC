from __future__ import annotations

import html
import re
import time
from collections import Counter
from typing import Any, Optional

from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from bs4.element import Doctype, ProcessingInstruction, Declaration


# =============================================================================
# Assumption
# =============================================================================
# This code assumes you already have a working OpenAI client variable named:
#
#     client
#
# Example setup is not included:
#
#     from openai import OpenAI
#     client = OpenAI(api_key="...")
# =============================================================================


DEFAULT_MODEL = "gpt-5.1"

# Main setting requested by you.
# If len(tokenized_html) <= DOC_CHARACTER_COUNT, no chunking is used.
# If len(tokenized_html) > DOC_CHARACTER_COUNT, the document is chunked.
DOC_CHARACTER_COUNT = 100_000

# Reserved room for labels/instructions inside each user message.
# This helps ensure previous + current + next chunk still fit under doc_character_count.
CHUNK_PROMPT_RESERVED_CHARS = 2_000

TAG_TOKEN_RE = re.compile(r"⟦HTML_TAG_\d{6}⟧")

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img",
    "input", "link", "meta", "param", "source", "track", "wbr",
}


# =============================================================================
# Global token management
# =============================================================================

class GlobalTagTokenizer:
    """
    Creates globally unique placeholder tokens for HTML tags.

    Example:
        ⟦HTML_TAG_000001⟧
        ⟦HTML_TAG_000002⟧
        ⟦HTML_TAG_000003⟧
    """

    def __init__(self) -> None:
        self.counter = 0
        self.token_map: dict[str, str] = {}
        self.token_sequence: list[str] = []

    def new_token(self, original_html: str) -> str:
        self.counter += 1
        token = f"⟦HTML_TAG_{self.counter:06d}⟧"
        self.token_map[token] = original_html
        self.token_sequence.append(token)
        return token


# =============================================================================
# BeautifulSoup rendering helpers
# =============================================================================

def render_attrs(attrs: dict[str, Any]) -> str:
    """
    Render BeautifulSoup attributes back into HTML.

    Example:
        {"href": "/x", "class": ["btn", "primary"]}

    becomes:
        href="/x" class="btn primary"
    """
    if not attrs:
        return ""

    rendered = []

    for name, value in attrs.items():
        if value is None or value is True:
            rendered.append(str(name))
            continue

        if value is False:
            continue

        if isinstance(value, (list, tuple, set)):
            value = " ".join(str(item) for item in value)

        escaped_value = html.escape(str(value), quote=True)
        rendered.append(f'{name}="{escaped_value}"')

    if not rendered:
        return ""

    return " " + " ".join(rendered)


def render_opening_tag(tag: Tag) -> str:
    name = (tag.name or "").lower()
    return f"<{name}{render_attrs(tag.attrs)}>"


def render_closing_tag(tag: Tag) -> str:
    name = (tag.name or "").lower()
    return f"</{name}>"


def render_void_tag(tag: Tag) -> str:
    name = (tag.name or "").lower()
    return f"<{name}{render_attrs(tag.attrs)}>"


# =============================================================================
# BeautifulSoup global-token serialization
# =============================================================================

def serialize_node_with_global_tokens(
    node: Any,
    tokenizer: GlobalTagTokenizer,
) -> str:
    """
    Convert a BeautifulSoup node into text where every HTML tag is replaced
    by a globally unique placeholder token.

    Text remains editable.
    Tags become protected tokens.
    """

    if isinstance(node, Comment):
        return tokenizer.new_token(f"<!--{str(node)}-->")

    if isinstance(node, Doctype):
        return tokenizer.new_token(f"<!DOCTYPE {str(node)}>")

    if isinstance(node, Declaration):
        return tokenizer.new_token(f"<!{str(node)}>")

    if isinstance(node, ProcessingInstruction):
        return tokenizer.new_token(f"<?{str(node)}?>")

    if isinstance(node, NavigableString):
        return str(node)

    if isinstance(node, Tag):
        name = (node.name or "").lower()

        if name in VOID_TAGS:
            return tokenizer.new_token(render_void_tag(node))

        opening = tokenizer.new_token(render_opening_tag(node))

        inner = "".join(
            serialize_node_with_global_tokens(child, tokenizer)
            for child in node.contents
        )

        closing = tokenizer.new_token(render_closing_tag(node))

        return opening + inner + closing

    return str(node)


def tokenize_html_with_beautifulsoup(
    html_text: str,
) -> tuple[str, dict[str, str], list[str]]:
    """
    Parse HTML with BeautifulSoup and replace every tag with a globally unique token.
    """
    if TAG_TOKEN_RE.search(html_text):
        raise ValueError(
            "Input HTML already contains strings that look like placeholder tokens. "
            "Use a different token format before processing this document."
        )

    soup = BeautifulSoup(html_text, "html.parser")
    tokenizer = GlobalTagTokenizer()

    tokenized_html = "".join(
        serialize_node_with_global_tokens(child, tokenizer)
        for child in soup.contents
    )

    return tokenized_html, tokenizer.token_map, tokenizer.token_sequence


def restore_html_from_global_tokens(
    tokenized_text: str,
    token_map: dict[str, str],
) -> str:
    """
    Replace global placeholder tokens with their original HTML tags.
    """

    def replace_token(match: re.Match) -> str:
        token = match.group(0)

        if token not in token_map:
            raise ValueError(f"Unexpected token during restore: {token}")

        return token_map[token]

    return TAG_TOKEN_RE.sub(replace_token, tokenized_text)


# =============================================================================
# Token validation
# =============================================================================

def extract_tokens(text: str) -> list[str]:
    return TAG_TOKEN_RE.findall(text)


def validate_token_sequence(
    expected_token_sequence: list[str],
    revised_text: str,
) -> list[str]:
    """
    Validate that a rewritten string preserved exactly the expected tokens.

    Checks:
    - same tokens
    - same counts
    - same order
    """
    issues = []

    revised_token_sequence = extract_tokens(revised_text)

    expected_counts = Counter(expected_token_sequence)
    revised_counts = Counter(revised_token_sequence)

    missing = sorted(set(expected_counts) - set(revised_counts))
    unexpected = sorted(set(revised_counts) - set(expected_counts))

    if missing:
        issues.append(f"missing_tokens:{missing[:20]}")

    if unexpected:
        issues.append(f"unexpected_tokens:{unexpected[:20]}")

    if expected_counts != revised_counts:
        issues.append("token_counts_changed")

    if revised_token_sequence != expected_token_sequence:
        issues.append("token_order_changed")

    return issues


# =============================================================================
# Chunking helpers
# =============================================================================

def split_plain_text_piece(piece: str, max_chars: int) -> list[str]:
    """
    Split a long non-token text piece into smaller pieces.
    Prefer whitespace/newline boundaries when possible.
    """
    if len(piece) <= max_chars:
        return [piece]

    parts = []
    remaining = piece

    while len(remaining) > max_chars:
        newline_cut = remaining.rfind("\n", 0, max_chars + 1)
        space_cut = remaining.rfind(" ", 0, max_chars + 1)
        punctuation_cut = max(
            remaining.rfind(".", 0, max_chars + 1),
            remaining.rfind(";", 0, max_chars + 1),
            remaining.rfind(",", 0, max_chars + 1),
        )

        cut = max(newline_cut, space_cut, punctuation_cut)

        # Avoid tiny chunks. If no good boundary exists, hard split.
        if cut < int(max_chars * 0.5):
            cut = max_chars
        else:
            cut += 1

        parts.append(remaining[:cut])
        remaining = remaining[cut:]

    if remaining:
        parts.append(remaining)

    return parts


def iter_tokenized_atoms(tokenized_text: str, max_text_piece_chars: int):
    """
    Yield tokens and text pieces without splitting placeholder tokens.
    """
    index = 0

    for match in TAG_TOKEN_RE.finditer(tokenized_text):
        if match.start() > index:
            text_piece = tokenized_text[index:match.start()]
            for part in split_plain_text_piece(text_piece, max_text_piece_chars):
                if part:
                    yield part

        yield match.group(0)
        index = match.end()

    if index < len(tokenized_text):
        text_piece = tokenized_text[index:]
        for part in split_plain_text_piece(text_piece, max_text_piece_chars):
            if part:
                yield part


def split_tokenized_html_into_chunks(
    tokenized_html: str,
    max_chunk_chars: int,
) -> list[str]:
    """
    Split tokenized HTML into chunks without splitting placeholder tokens.
    """
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero.")

    chunks = []
    current_parts = []
    current_length = 0

    for atom in iter_tokenized_atoms(tokenized_html, max_chunk_chars):
        atom_length = len(atom)

        if atom_length > max_chunk_chars:
            if current_parts:
                chunks.append("".join(current_parts))
                current_parts = []
                current_length = 0

            chunks.append(atom)
            continue

        if current_parts and current_length + atom_length > max_chunk_chars:
            chunks.append("".join(current_parts))
            current_parts = [atom]
            current_length = atom_length
            continue

        current_parts.append(atom)
        current_length += atom_length

    if current_parts:
        chunks.append("".join(current_parts))

    return chunks


def derive_rewrite_chunk_character_count(
    doc_character_count: int,
    reserved_chars: int = CHUNK_PROMPT_RESERVED_CHARS,
) -> int:
    """
    Derive the current chunk size.

    We need room for:
    - previous chunk as context
    - current chunk to rewrite
    - next chunk as context

    So the current chunk is about one third of the budget.
    """
    if doc_character_count <= reserved_chars + 3_000:
        raise ValueError(
            "doc_character_count is too small for chunking with previous/current/next context. "
            "Increase doc_character_count."
        )

    return max(1, (doc_character_count - reserved_chars) // 3)


def trim_previous_context(text: str, max_chars: int) -> str:
    """
    For previous context, keep the end because it is closest to the current chunk.
    """
    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    return text[-max_chars:]


def trim_next_context(text: str, max_chars: int) -> str:
    """
    For next context, keep the beginning because it is closest to the current chunk.
    """
    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    return text[:max_chars]


# =============================================================================
# Prompt building
# =============================================================================

def build_system_prompt(transformation_rules: str) -> str:
    return f"""
You are a careful document editor.

You will receive an HTML document where every HTML tag has been replaced by a globally unique placeholder token.

The placeholder tokens look like this:
⟦HTML_TAG_000001⟧
⟦HTML_TAG_000002⟧
⟦HTML_TAG_000003⟧

Each placeholder token represents one original HTML tag, comment, doctype, or similar protected HTML node.

Important:
- Rewrite human-readable text only.
- Do not rewrite placeholder tokens.
- Do not remove placeholder tokens.
- Do not add new placeholder tokens.
- Do not duplicate placeholder tokens.
- Keep placeholder tokens in the same order within the text you rewrite.
- Do not output raw HTML tags.
- Do not wrap your answer in Markdown.
- Do not explain your answer.

Writing rules:
- Follow the transformation rules exactly.
- Preserve the source language of the text.
- Do not translate unless the transformation rules explicitly ask for translation.
- Preserve the original meaning.
- Preserve names, numbers, dates, product names, technical terms, and references.
- Do not change existing facts.
- Do not create new information.
- Do not delete existing information.

Transformation rules:
{transformation_rules}
"""


def build_whole_document_user_prompt(tokenized_html: str) -> str:
    return (
        "Rewrite the following tokenized HTML document.\n"
        "Return only the rewritten tokenized document.\n\n"
        f"{tokenized_html}"
    )


def build_chunk_user_prompt(
    *,
    previous_context: str,
    current_chunk: str,
    next_context: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    return f"""
Rewrite only CURRENT_CHUNK_TO_REWRITE.

The PREVIOUS_CHUNK_CONTEXT and NEXT_CHUNK_CONTEXT are read-only context.
Use them only to improve flow and avoid repetition.
Do not include previous context or next context in your answer.
Return only the rewritten CURRENT_CHUNK_TO_REWRITE.
Do not include labels, explanations, Markdown, or code fences.

Chunk {chunk_index} of {total_chunks}.

PREVIOUS_CHUNK_CONTEXT:
{previous_context}

CURRENT_CHUNK_TO_REWRITE:
{current_chunk}

NEXT_CHUNK_CONTEXT:
{next_context}
"""


def fit_contexts_to_doc_character_count(
    *,
    previous_chunk: str,
    current_chunk: str,
    next_chunk: str,
    doc_character_count: int,
    chunk_index: int,
    total_chunks: int,
) -> tuple[str, str, str]:
    """
    Build previous/current/next context so the complete user prompt stays
    under doc_character_count.

    The current chunk is never trimmed here. If it is too large, raise an error.
    Previous and next contexts may be trimmed.
    """

    base_prompt = build_chunk_user_prompt(
        previous_context="",
        current_chunk=current_chunk,
        next_context="",
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )

    available_for_context = doc_character_count - len(base_prompt)

    if available_for_context < 0:
        raise ValueError(
            "Current chunk plus prompt labels exceed doc_character_count. "
            "Use a smaller rewrite_chunk_character_count or a larger doc_character_count."
        )

    if previous_chunk and next_chunk:
        previous_budget = available_for_context // 2
        next_budget = available_for_context - previous_budget
    elif previous_chunk:
        previous_budget = available_for_context
        next_budget = 0
    elif next_chunk:
        previous_budget = 0
        next_budget = available_for_context
    else:
        previous_budget = 0
        next_budget = 0

    previous_context = trim_previous_context(previous_chunk, previous_budget)
    next_context = trim_next_context(next_chunk, next_budget)

    # Final safety loop in case label lengths or edge cases push us over.
    while True:
        prompt = build_chunk_user_prompt(
            previous_context=previous_context,
            current_chunk=current_chunk,
            next_context=next_context,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )

        if len(prompt) <= doc_character_count:
            return previous_context, current_chunk, next_context

        if len(next_context) > 0:
            next_context = next_context[:-100]

        elif len(previous_context) > 0:
            previous_context = previous_context[100:]

        else:
            raise ValueError(
                "Could not fit chunk prompt under doc_character_count."
            )


# =============================================================================
# OpenAI call
# =============================================================================

def clean_model_output(text: str) -> str:
    """
    Remove a surrounding Markdown code fence if the model accidentally adds one.
    """
    stripped = text.strip()

    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()

        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1])

    return text


def call_openai_text(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: Optional[float] = 0.1,
    max_retries: int = 3,
) -> str:
    """
    Call OpenAI Chat Completions and return plain text.
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
                        "content": user_prompt,
                    },
                ],
            }

            if temperature is not None:
                kwargs["temperature"] = temperature

            response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content

            if not content:
                raise ValueError("OpenAI returned an empty response.")

            return clean_model_output(content)

        except Exception as exc:
            last_error = exc
            sleep_seconds = 2 ** attempt
            print(f"OpenAI call failed. Retrying in {sleep_seconds}s. Error: {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_error}")


# =============================================================================
# Rewrite functions
# =============================================================================

def rewrite_whole_document_once(
    *,
    tokenized_html: str,
    transformation_rules: str,
    model: str,
    temperature: Optional[float],
    max_retries: int,
) -> str:
    """
    Rewrite the full tokenized document in one model call.
    Used when len(tokenized_html) <= doc_character_count.
    """
    system_prompt = build_system_prompt(transformation_rules)
    user_prompt = build_whole_document_user_prompt(tokenized_html)

    return call_openai_text(
        system_prompt,
        user_prompt,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )


def rewrite_chunked_document(
    *,
    tokenized_html: str,
    transformation_rules: str,
    doc_character_count: int,
    model: str,
    temperature: Optional[float],
    max_retries: int,
    fallback_to_original_chunk: bool,
) -> tuple[str, dict]:
    """
    Rewrite a long tokenized document chunk by chunk.

    Each request includes:
    - previous chunk as read-only context
    - current chunk to rewrite
    - next chunk as read-only context

    The full user prompt is kept under doc_character_count.
    """

    rewrite_chunk_character_count = derive_rewrite_chunk_character_count(
        doc_character_count
    )

    chunks = split_tokenized_html_into_chunks(
        tokenized_html,
        max_chunk_chars=rewrite_chunk_character_count,
    )

    system_prompt = build_system_prompt(transformation_rules)

    rewritten_chunks = []
    failed_chunks = []

    max_user_prompt_length = 0

    for index, current_chunk in enumerate(chunks):
        previous_chunk = chunks[index - 1] if index > 0 else ""
        next_chunk = chunks[index + 1] if index + 1 < len(chunks) else ""

        previous_context, current_chunk, next_context = fit_contexts_to_doc_character_count(
            previous_chunk=previous_chunk,
            current_chunk=current_chunk,
            next_chunk=next_chunk,
            doc_character_count=doc_character_count,
            chunk_index=index + 1,
            total_chunks=len(chunks),
        )

        user_prompt = build_chunk_user_prompt(
            previous_context=previous_context,
            current_chunk=current_chunk,
            next_context=next_context,
            chunk_index=index + 1,
            total_chunks=len(chunks),
        )

        max_user_prompt_length = max(max_user_prompt_length, len(user_prompt))

        print(
            f"Rewriting chunk {index + 1}/{len(chunks)} "
            f"current={len(current_chunk)} chars, "
            f"previous_context={len(previous_context)} chars, "
            f"next_context={len(next_context)} chars, "
            f"user_prompt={len(user_prompt)} chars"
        )

        revised_chunk = call_openai_text(
            system_prompt,
            user_prompt,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )

        expected_tokens = extract_tokens(current_chunk)

        chunk_issues = validate_token_sequence(
            expected_tokens,
            revised_chunk,
        )

        if chunk_issues:
            failed_chunks.append(
                {
                    "chunk_index": index + 1,
                    "issues": chunk_issues,
                    "current_chunk_chars": len(current_chunk),
                    "previous_context_chars": len(previous_context),
                    "next_context_chars": len(next_context),
                }
            )

            if fallback_to_original_chunk:
                rewritten_chunks.append(current_chunk)
                continue

            raise ValueError(
                f"Chunk {index + 1} failed token validation: {chunk_issues}"
            )

        rewritten_chunks.append(revised_chunk)

    rewritten_tokenized_html = "".join(rewritten_chunks)

    report = {
        "chunked": True,
        "doc_character_count": doc_character_count,
        "rewrite_chunk_character_count": rewrite_chunk_character_count,
        "total_chunks": len(chunks),
        "failed_chunks": failed_chunks,
        "fallback_chunks": len(failed_chunks) if fallback_to_original_chunk else 0,
        "max_user_prompt_length": max_user_prompt_length,
    }

    return rewritten_tokenized_html, report


# =============================================================================
# Public function
# =============================================================================

def transform_document(
    html_text: str,
    transformation_rules: str,
    *,
    model: str = DEFAULT_MODEL,
    doc_character_count: int = DOC_CHARACTER_COUNT,
    temperature: Optional[float] = 0.1,
    max_retries: int = 3,
    fallback_to_original_chunk: bool = True,
    return_report: bool = False,
) -> str | tuple[str, dict]:
    """
    Transform an HTML string using BeautifulSoup global tag placeholders.

    Behavior
    --------
    1. Parse HTML with BeautifulSoup.
    2. Replace every HTML tag with a globally unique placeholder token.
    3. If tokenized HTML length <= doc_character_count:
       - send the whole tokenized document in one OpenAI call.
    4. If tokenized HTML length > doc_character_count:
       - split into chunks.
       - for each chunk, send previous and next chunks as context.
       - keep the user prompt under doc_character_count.
    5. Validate placeholder tokens.
    6. Restore HTML tags.
    7. Return transformed HTML.

    Important
    ---------
    doc_character_count is a character-budget setting, not a true token-budget.
    It controls Python string length.
    """

    if not isinstance(html_text, str):
        raise TypeError("html_text must be a string.")

    if not isinstance(transformation_rules, str):
        raise TypeError("transformation_rules must be a string.")

    if doc_character_count <= 0:
        raise ValueError("doc_character_count must be greater than zero.")

    tokenized_html, token_map, original_token_sequence = tokenize_html_with_beautifulsoup(
        html_text
    )

    report = {
        "model": model,
        "architecture": "beautifulsoup_global_tokens_with_optional_chunking",
        "doc_character_count": doc_character_count,
        "original_html_chars": len(html_text),
        "tokenized_html_chars": len(tokenized_html),
        "tag_token_count": len(original_token_sequence),
        "chunked": False,
        "status": "started",
        "issues": [],
    }

    if len(tokenized_html) <= doc_character_count:
        print(
            f"No chunking needed. tokenized_html={len(tokenized_html)} "
            f"<= doc_character_count={doc_character_count}"
        )

        revised_tokenized_html = rewrite_whole_document_once(
            tokenized_html=tokenized_html,
            transformation_rules=transformation_rules,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )

    else:
        print(
            f"Chunking needed. tokenized_html={len(tokenized_html)} "
            f"> doc_character_count={doc_character_count}"
        )

        revised_tokenized_html, chunk_report = rewrite_chunked_document(
            tokenized_html=tokenized_html,
            transformation_rules=transformation_rules,
            doc_character_count=doc_character_count,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            fallback_to_original_chunk=fallback_to_original_chunk,
        )

        report.update(chunk_report)

    # Final whole-document validation.
    final_issues = validate_token_sequence(
        original_token_sequence,
        revised_tokenized_html,
    )

    if final_issues:
        report["status"] = "failed_final_token_validation"
        report["issues"] = final_issues

        if return_report:
            return html_text, report

        raise ValueError(
            "Final transformed document failed token validation. "
            f"Issues: {final_issues}"
        )

    transformed_html = restore_html_from_global_tokens(
        revised_tokenized_html,
        token_map,
    )

    report["status"] = "success"

    if return_report:
        return transformed_html, report

    return transformed_html

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

new_html, report = transform_document(
    html_text,
    rules,
    doc_character_count=100_000,
    return_report=True,
)

print(new_html)
print(report)


def derive_rewrite_chunk_character_count11(
    *,
    doc_character_count: int,
    system_prompt: str,
    reserved_user_prompt_chars: int = 1_500,
) -> int:
    """
    Derive the current rewrite chunk size.

    The total request should fit under doc_character_count:

        system prompt
      + user prompt labels
      + previous context
      + current chunk
      + next context

    We divide the remaining budget by 3 because each request may include:
    previous + current + next.
    """

    available = doc_character_count - len(system_prompt) - reserved_user_prompt_chars

    if available <= 3_000:
        raise ValueError(
            "doc_character_count is too small for this prompt and chunking strategy. "
            f"doc_character_count={doc_character_count}, "
            f"system_prompt_chars={len(system_prompt)}, "
            f"reserved_user_prompt_chars={reserved_user_prompt_chars}."
        )

    return max(1, available // 3)