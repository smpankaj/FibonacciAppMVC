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
# Example setup is not included here:
#
#     from openai import OpenAI
#     client = OpenAI(api_key="...")
# =============================================================================


DEFAULT_MODEL = "gpt-5.1"

TAG_TOKEN_RE = re.compile(r"⟦HTML_TAG_\d{6}⟧")

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img",
    "input", "link", "meta", "param", "source", "track", "wbr",
}


# =============================================================================
# Token management
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
# HTML rendering helpers
# =============================================================================

def render_attrs(attrs: dict[str, Any]) -> str:
    """
    Render BeautifulSoup attributes back into HTML text.

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
    """
    Render only the opening tag.

    Example:
        <a href="/form">
    """
    name = (tag.name or "").lower()
    return f"<{name}{render_attrs(tag.attrs)}>"


def render_closing_tag(tag: Tag) -> str:
    """
    Render the closing tag.

    Example:
        </a>
    """
    name = (tag.name or "").lower()
    return f"</{name}>"


def render_void_tag(tag: Tag) -> str:
    """
    Render a standalone/void tag.

    Example:
        <img src="x.png" alt="Example">
    """
    name = (tag.name or "").lower()
    return f"<{name}{render_attrs(tag.attrs)}>"


# =============================================================================
# BeautifulSoup-based global tokenization
# =============================================================================

def serialize_node_with_global_tokens(
    node: Any,
    tokenizer: GlobalTagTokenizer,
) -> str:
    """
    Convert a BeautifulSoup node into text where every tag is replaced by a
    globally unique placeholder token.

    Text stays as text.
    Tags become tokens.
    """

    # Comments are not normal tags, but we protect them from the model too.
    if isinstance(node, Comment):
        return tokenizer.new_token(f"<!--{str(node)}-->")

    # Doctype, declarations and processing instructions are also protected.
    if isinstance(node, Doctype):
        return tokenizer.new_token(f"<!DOCTYPE {str(node)}>")

    if isinstance(node, Declaration):
        return tokenizer.new_token(f"<!{str(node)}>")

    if isinstance(node, ProcessingInstruction):
        return tokenizer.new_token(f"<?{str(node)}?>")

    # Normal text is what the model may rewrite.
    if isinstance(node, NavigableString):
        return str(node)

    # Normal HTML tags.
    if isinstance(node, Tag):
        name = (node.name or "").lower()

        if name in VOID_TAGS:
            original_tag = render_void_tag(node)
            return tokenizer.new_token(original_tag)

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

    Example input:
        <p>Use the <strong>form</strong>.</p>

    Example output:
        ⟦HTML_TAG_000001⟧Use the ⟦HTML_TAG_000002⟧form⟦HTML_TAG_000003⟧.⟦HTML_TAG_000004⟧

    token_map:
        {
          "⟦HTML_TAG_000001⟧": "<p>",
          "⟦HTML_TAG_000002⟧": "<strong>",
          "⟦HTML_TAG_000003⟧": "</strong>",
          "⟦HTML_TAG_000004⟧": "</p>"
        }
    """
    if TAG_TOKEN_RE.search(html_text):
        raise ValueError(
            "Input HTML already contains strings that look like placeholder tokens. "
            "Choose a different token format before processing this document."
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
# Validation
# =============================================================================

def extract_tokens(text: str) -> list[str]:
    return TAG_TOKEN_RE.findall(text)


def validate_global_tokens(
    original_token_sequence: list[str],
    revised_tokenized_text: str,
) -> list[str]:
    """
    Validate that the model preserved all global HTML tokens.

    We require:
    - same tokens
    - same counts
    - same order

    Same order is important because the tokens represent opening and closing tags.
    If the order changes, the reconstructed HTML can break.
    """
    issues = []

    revised_token_sequence = extract_tokens(revised_tokenized_text)

    original_counts = Counter(original_token_sequence)
    revised_counts = Counter(revised_token_sequence)

    missing = sorted(set(original_counts) - set(revised_counts))
    unexpected = sorted(set(revised_counts) - set(original_counts))

    if missing:
        issues.append(f"missing_tokens:{missing[:20]}")

    if unexpected:
        issues.append(f"unexpected_tokens:{unexpected[:20]}")

    if original_counts != revised_counts:
        issues.append("token_counts_changed")

    if revised_token_sequence != original_token_sequence:
        issues.append("token_order_changed")

    return issues


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
# Prompt
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
- Rewrite the human-readable text only.
- Do not rewrite placeholder tokens.
- Do not remove placeholder tokens.
- Do not add new placeholder tokens.
- Do not duplicate placeholder tokens.
- Keep all placeholder tokens in the exact same order.
- Do not output raw HTML tags.
- Do not wrap your answer in Markdown.
- Do not explain your answer.
- Return only the rewritten tokenized document.

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


# =============================================================================
# Public function
# =============================================================================

def transform_document(
    html_text: str,
    transformation_rules: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = 0.1,
    max_retries: int = 3,
    return_report: bool = False,
) -> str | tuple[str, dict]:
    """
    Transform an HTML string using BeautifulSoup-based global tag placeholders.

    Parameters
    ----------
    html_text:
        Original HTML string or HTML fragment.

    transformation_rules:
        Prompt/rules that tell the model how to rewrite the text.

    return_report:
        If False:
            returns transformed HTML string.

        If True:
            returns (transformed HTML string, report dict).

    Important
    ---------
    This sends the whole tokenized document in one API call.
    If the document is too large for the model context window, the API call may fail.
    """
    if not isinstance(html_text, str):
        raise TypeError("html_text must be a string.")

    if not isinstance(transformation_rules, str):
        raise TypeError("transformation_rules must be a string.")

    tokenized_html, token_map, original_token_sequence = tokenize_html_with_beautifulsoup(
        html_text
    )

    system_prompt = build_system_prompt(transformation_rules)

    user_prompt = (
        "Rewrite the following tokenized HTML document. "
        "Return only the rewritten tokenized document.\n\n"
        f"{tokenized_html}"
    )

    revised_tokenized_html = call_openai_text(
        system_prompt,
        user_prompt,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

    issues = validate_global_tokens(
        original_token_sequence,
        revised_tokenized_html,
    )

    report = {
        "model": model,
        "architecture": "beautifulsoup_global_html_tag_placeholders",
        "tag_token_count": len(original_token_sequence),
        "status": "success" if not issues else "failed_token_validation",
        "issues": issues,
    }

    if issues:
        if return_report:
            return html_text, report

        raise ValueError(
            "Model output failed HTML token validation. "
            f"Issues: {issues}"
        )

    transformed_html = restore_html_from_global_tokens(
        revised_tokenized_html,
        token_map,
    )

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
    return_report=True,
)

print(new_html)
print(report)