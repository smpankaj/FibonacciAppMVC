"""
Dutch document language-rewriter pipeline for HTML fragments.

This version deliberately has no local NLP dependency and no automated
language-quality checker. It keeps the pipeline simple and focused on
getting reliable end-to-end rewriting working, including nested lists and headings:

  HTML fragment
    -> parse with BeautifulSoup
    -> extract editable text segments without swallowing nested lists/headings
    -> tokenize inline/image tags into placeholders
    -> STAGE 1: document-wide plain-text rewrite, chunked by char budget
    -> basic Stage 1 completeness repair for missing/empty model outputs
    -> STAGE 2: reinsert formatting/image placeholders into the Stage 1 wording
    -> strict placeholder-token validation
    -> detokenize and reinsert revised text into the ORIGINAL DOM locations
    -> serialize back to HTML

Design principle:
The LLM never receives or generates raw HTML. It only sees plain text with
lightweight placeholders such as ⟦1⟧...⟦/1⟧. BeautifulSoup handles the HTML
parse/serialize steps, and this code reconstructs the tags deterministically
from the original DOM.
"""

from __future__ import annotations

import json
import re
import time
import uuid
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

try:
    from openai import OpenAI
except ImportError:  # lets tokenization/validation tests run without OpenAI installed
    OpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Block-level tags whose text content is sent to the LLM as one semantic unit.
BLOCK_TAGS = {
    "p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "td", "th", "blockquote", "figcaption",
}

# Inline tags whose text content may be rewritten while the tag itself must stay.
INLINE_TAGS = {"b", "strong", "i", "em", "u", "a", "span", "sup", "sub", "mark"}

# Display/void tags that should be preserved as opaque placeholders.
# Add more here if your source fragments use them.
OPAQUE_TAGS = {"img", "br", "hr", "picture", "source"}

# Character budgets per LLM call. Character count is a simple proxy for token
# count. You can replace this later with tiktoken if you want tighter budgeting.
STAGE1_MAX_CHARS_PER_BATCH = 80_000
STAGE2_MAX_CHARS_PER_BATCH = 20_000

MODEL = "gpt-4.1"  # use whichever model you have validated for Dutch rewriting
MAX_API_RETRIES = 3
API_TIMEOUT_SECONDS = 120
STAGE1_EMPTY_REPAIR_PASSES = 1

client: Optional[OpenAI] = None


TOKEN_RE = re.compile(r"⟦/?(\d+)⟧")
TOKEN_PART_RE = re.compile(r"⟦(/?)(\d+)⟧")
_OPEN_RE = re.compile(r"⟦(\d+)⟧")
_CLOSE_RE = re.compile(r"⟦/(\d+)⟧")


def get_openai_client() -> OpenAI:
    """Create the OpenAI client lazily so tests can import this module without an API key."""
    global client
    if OpenAI is None:
        raise RuntimeError("Install the openai package to run rewriting: pip install openai")
    if client is None:
        client = OpenAI()  # reads OPENAI_API_KEY from environment
    return client


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATIVE_RULES = """
Je herschrijft interne bedrijfsdocumenten naar helder Nederlands.

Taalniveau en toon:
- Schrijf op B1-taalniveau: veelgebruikte woorden, korte zinnen, directe stijl.
- Gebruik actieve zinnen waar mogelijk.
- Wees direct en concreet; vermijd vaag jargon en bureaucratische taal.
- Behoud de oorspronkelijke betekenis en feitelijke inhoud volledig.
- Verander geen cijfers, namen, data of technische termen die niet dubbelzinnig zijn.
- Gebruik consistente terminologie: als een begrip eerder in het document op
  een bepaalde manier is vertaald/geschreven, gebruik dezelfde formulering
  overal in het document.

Zinsbouw:
- Houd zinnen overzichtelijk.
- Eén gedachte per zin waar dat natuurlijk kan.
"""

STAGE1_SYSTEM_PROMPT = f"""Je bent een gespecialiseerde tekstredacteur voor interne bedrijfsdocumenten in het Nederlands.

{GENERATIVE_RULES}

Je krijgt een lijst tekstblokken uit hetzelfde document, elk met een uniek id,
in de oorspronkelijke volgorde. Herschrijf elk blok volgens de regels hierboven.
Omdat deze blokken uit hetzelfde document komen: gebruik overal dezelfde
terminologie, formuleringen voor herhaalde begrippen, en een consistente toon.

Geef uitsluitend geldige JSON terug volgens het gevraagde schema, zonder uitleg.
"""

STAGE1_REPAIR_SYSTEM_PROMPT = f"""Je bent een gespecialiseerde tekstredacteur voor interne bedrijfsdocumenten in het Nederlands.

{GENERATIVE_RULES}

Je krijgt tekstblokken waarvoor de vorige Stage-1 output ontbrak of leeg was.
Herschrijf elk blok opnieuw op basis van "original_text".

Regels:
- Geef alleen platte tekst terug, zonder HTML en zonder opmaaktokens.
- Behoud betekenis, cijfers, namen, data en technische termen volledig.
- Verwijder geen inhoud.

Geef uitsluitend geldige JSON terug volgens het gevraagde schema, zonder uitleg.
"""

STAGE2_SYSTEM_PROMPT = """Je krijgt per tekstblok twee versies:
1. "text": de oorspronkelijke tekst met opmaaktokens zoals ⟦1⟧...⟦/1⟧ voor inline tags, links, afbeeldingen of andere bewaarde elementen.
2. "reference": de goedgekeurde, herschreven versie van diezelfde tekst zonder opmaaktokens.

Jouw taak is UITSLUITEND: neem de "reference"-tekst over, en plaats de
opmaaktokens uit "text" op de juiste plek in die reference-tekst, rond of nabij
het tekstfragment waar ze inhoudelijk bij horen.

Regels:
- Herschrijf de reference-tekst verder NIET; gebruik hem zo letterlijk mogelijk.
- Alle tokens uit "text" MOETEN exact terugkomen: zelfde nummer, zelfde openings- en sluitingsvorm.
- Verzin geen nieuwe tokens en verwijder er geen.
- Als een token geen duidelijke tekstuele tegenhanger heeft, bijvoorbeeld een afbeelding of regelafbreking,
  plaats het token op de meest logische nabije plek in de tekst.

Geef uitsluitend geldige JSON terug volgens het gevraagde schema, zonder uitleg.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Block:
    id: str
    parent: Tag                       # parent whose direct children will be replaced
    nodes: list                       # contiguous direct child nodes represented by this block
    original_text: str                # tokenized plain text sent to the LLM in Stage 2
    token_map: dict[str, Tag]         # placeholder id -> original Tag template
    stage1_text: Optional[str] = None # plain-text wording from Stage 1
    revised_text: Optional[str] = None # final tokenized text from Stage 2
    validation_issues: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def location(self) -> str:
        name = getattr(self.parent, "name", "document") or "document"
        return f"<{name}>"


# ---------------------------------------------------------------------------
# Step 1: Extraction & tokenization
# ---------------------------------------------------------------------------

# Structural/container tags are boundaries. Their own direct text can be edited,
# but their nested block/list/table/header children are processed separately.
# This prevents nested lists/headings from being converted into empty placeholders.
CONTAINER_TAGS = {
    "div", "section", "article", "main", "aside", "header", "footer", "nav",
    "ul", "ol", "dl", "dt", "dd", "table", "thead", "tbody", "tfoot", "tr",
    "figure", "details", "summary",
}
STRUCTURAL_TAGS = BLOCK_TAGS | CONTAINER_TAGS


def _is_boundary_tag(tag: Tag) -> bool:
    """
    Return True when a tag should be handled as its own DOM area instead of being
    swallowed into the current text segment.

    A normal inline tag such as <strong> is not a boundary. A list, heading,
    table row, paragraph, or any wrapper that contains such a tag is a boundary.
    """
    if tag.name in OPAQUE_TAGS:
        return False
    if tag.name in STRUCTURAL_TAGS:
        return True
    return tag.find(list(STRUCTURAL_TAGS)) is not None


def tokenize_nodes(nodes: list) -> tuple[str, dict[str, Tag]]:
    """
    Tokenize a contiguous sequence of direct DOM children.

    Inline tags are represented as paired placeholders and their inner text is
    still editable. Opaque display tags such as img/br/hr are represented as
    paired placeholders too, but detokenization restores the original full tag.
    """
    token_map: dict[str, Tag] = {}
    counter = [0]

    def next_token_id(tag: Tag) -> str:
        counter[0] += 1
        tid = str(counter[0])
        token_map[tid] = tag
        return tid

    def walk_items(items) -> str:
        out: list[str] = []
        for child in items:
            if isinstance(child, NavigableString):
                out.append(str(child))
                continue

            if not isinstance(child, Tag):
                continue

            tid = next_token_id(child)

            if child.name in OPAQUE_TAGS:
                # Marker for an image, break, etc. The tag is copied back later.
                out.append(f"⟦{tid}⟧⟦/{tid}⟧")
            else:
                # Treat any non-boundary tag as inline/preserved formatting.
                inner = walk_items(child.children)
                out.append(f"⟦{tid}⟧{inner}⟦/{tid}⟧")

        return "".join(out)

    return walk_items(nodes), token_map


# Backward-compatible helper for tests that tokenize a single tag.
def tokenize_block(node: Tag) -> tuple[str, dict[str, Tag]]:
    return tokenize_nodes(list(node.contents))


def _flush_segment(blocks: list[Block], parent: Tag, segment_nodes: list) -> None:
    if not segment_nodes:
        return

    text, token_map = tokenize_nodes(segment_nodes)

    # Do not send pure whitespace or image-only segments to the model. They stay
    # exactly where they are in the DOM.
    if not TOKEN_RE.sub("", text).strip():
        return

    blocks.append(
        Block(
            id=f"b_{uuid.uuid4().hex[:8]}",
            parent=parent,
            nodes=list(segment_nodes),
            original_text=text,
            token_map=token_map,
        )
    )


def _extract_from_parent(parent: Tag, blocks: list[Block]) -> None:
    """
    Walk direct children in order. Consecutive non-boundary children become one
    editable block. Boundary children are recursively processed in their own area.
    """
    segment_nodes: list = []

    for child in list(parent.contents):
        if isinstance(child, Tag) and _is_boundary_tag(child):
            _flush_segment(blocks, parent, segment_nodes)
            segment_nodes = []
            _extract_from_parent(child, blocks)
            continue

        segment_nodes.append(child)

    _flush_segment(blocks, parent, segment_nodes)


def extract_blocks(soup: BeautifulSoup) -> list[Block]:
    """
    Extract editable text segments while preserving nested structures.

    Unlike the original approach, this does not skip a nested <li>, <p>, <h2>,
    etc. It also does not turn nested lists/headings into empty opaque tokens.
    """
    blocks: list[Block] = []
    _extract_from_parent(soup, blocks)
    return blocks


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def chunk_blocks_by_chars(
    blocks: list[Block],
    max_chars: int,
    text_fn: Callable[[Block], str],
) -> list[list[Block]]:
    """Greedily group blocks so each chunk stays under max_chars where possible."""
    chunks: list[list[Block]] = []
    current: list[Block] = []
    current_chars = 0

    for block in blocks:
        length = len(text_fn(block))

        if length > max_chars:
            if current:
                chunks.append(current)
                current = []
                current_chars = 0
            chunks.append([block])
            continue

        if current and current_chars + length > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(block)
        current_chars += length

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# OpenAI helper and response schemas
# ---------------------------------------------------------------------------

BLOCKS_RESPONSE_SCHEMA = {
    "name": "rewritten_blocks",
    "schema": {
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
            }
        },
        "required": ["blocks"],
        "additionalProperties": False,
    },
    "strict": True,
}


def call_openai_json(*, system_prompt: str, user_prompt: str, response_schema: dict, temperature: float) -> dict:
    """Call OpenAI with simple retry/backoff and return parsed JSON."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = get_openai_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_schema", "json_schema": response_schema},
                temperature=temperature,
                timeout=API_TIMEOUT_SECONDS,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty_openai_response")

            return json.loads(content)

        except Exception as exc:
            last_error = exc
            if attempt >= MAX_API_RETRIES:
                raise RuntimeError(f"OpenAI call failed after {MAX_API_RETRIES} attempts: {exc}") from exc
            time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(f"OpenAI call failed: {last_error}")


# ---------------------------------------------------------------------------
# Step 2a: Stage 1 - document-wide plain-text rewrite
# ---------------------------------------------------------------------------

def _plain_text(block: Block) -> str:
    """Block text with placeholder tokens stripped. This is what Stage 1 sees."""
    return TOKEN_RE.sub("", block.original_text)


def _valid_stage1_text(text: Optional[str]) -> bool:
    return bool(text and text.strip())


def rewrite_stage1_chunk(blocks: list[Block]) -> None:
    """Send one char-budgeted chunk of plain-text blocks to the LLM."""
    payload = [{"id": block.id, "text": _plain_text(block)} for block in blocks]

    user_prompt = (
        "Herschrijf onderstaande blokken uit hetzelfde document:\n\n"
        + json.dumps({"blocks": payload}, ensure_ascii=False, indent=2)
    )

    result = call_openai_json(
        system_prompt=STAGE1_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=BLOCKS_RESPONSE_SCHEMA,
        temperature=0.2,
    )

    by_id = {item["id"]: item["revised_text"] for item in result.get("blocks", [])}

    for block in blocks:
        revised = by_id.get(block.id)
        if revised is None:
            block.issues.append("stage1_missing_from_response")
            block.stage1_text = _plain_text(block)  # safe fallback
        elif not revised.strip():
            block.issues.append("stage1_empty_response")
            block.stage1_text = _plain_text(block)  # safe fallback
        else:
            block.stage1_text = revised


def run_stage1(blocks: list[Block]) -> None:
    for chunk in chunk_blocks_by_chars(blocks, STAGE1_MAX_CHARS_PER_BATCH, _plain_text):
        rewrite_stage1_chunk(chunk)


def _stage1_needs_repair(block: Block) -> bool:
    return (
        not _valid_stage1_text(block.stage1_text)
        or "stage1_missing_from_response" in block.issues
        or "stage1_empty_response" in block.issues
    )


def _stage1_repair_payload_text(block: Block) -> str:
    return "\n".join([_plain_text(block), block.stage1_text or ""])


def repair_stage1_batch(blocks: list[Block]) -> None:
    """Repair only missing/empty Stage 1 outputs. This is not a language-rule checker."""
    payload = [
        {
            "id": block.id,
            "original_text": _plain_text(block),
            "current_rewrite": block.stage1_text or "",
        }
        for block in blocks
    ]

    user_prompt = (
        "Herschrijf onderstaande blokken opnieuw omdat de vorige output ontbrak of leeg was:\n\n"
        + json.dumps({"blocks": payload}, ensure_ascii=False, indent=2)
    )

    result = call_openai_json(
        system_prompt=STAGE1_REPAIR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=BLOCKS_RESPONSE_SCHEMA,
        temperature=0.1,
    )

    by_id = {item["id"]: item["revised_text"] for item in result.get("blocks", [])}

    for block in blocks:
        revised = by_id.get(block.id)
        if revised and revised.strip():
            block.stage1_text = revised
            block.issues = [
                issue for issue in block.issues
                if issue not in {"stage1_missing_from_response", "stage1_empty_response"}
            ]
        else:
            block.issues.append("stage1_repair_failed")


def run_stage1_empty_repairs(blocks: list[Block], max_passes: int = STAGE1_EMPTY_REPAIR_PASSES) -> None:
    """
    Basic Stage 1 repair loop for operational failures only.

    This intentionally does not check B1 quality or any other language rule.
    Those checks can be added later once the pipeline works.
    """
    for _ in range(max_passes):
        candidates = [block for block in blocks if _stage1_needs_repair(block)]
        if not candidates:
            return

        for chunk in chunk_blocks_by_chars(candidates, STAGE1_MAX_CHARS_PER_BATCH, _stage1_repair_payload_text):
            repair_stage1_batch(chunk)


# ---------------------------------------------------------------------------
# Step 2b: Stage 2 - markup/image placeholder reinsertion
# ---------------------------------------------------------------------------

def _stage2_payload_text(block: Block) -> str:
    return block.original_text + (block.stage1_text or "")


def rewrite_stage2_batch(blocks: list[Block]) -> None:
    """Send tokenized originals plus Stage 1 references to the LLM."""
    payload = [
        {
            "id": block.id,
            "text": block.original_text,
            "reference": block.stage1_text or _plain_text(block),
        }
        for block in blocks
    ]

    user_prompt = (
        "Plaats de opmaaktokens uit 'text' in de bijbehorende 'reference'-tekst "
        "voor elk van onderstaande blokken:\n\n"
        + json.dumps({"blocks": payload}, ensure_ascii=False, indent=2)
    )

    result = call_openai_json(
        system_prompt=STAGE2_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=BLOCKS_RESPONSE_SCHEMA,
        temperature=0.1,
    )

    by_id = {item["id"]: item["revised_text"] for item in result.get("blocks", [])}

    for block in blocks:
        revised = by_id.get(block.id)
        if revised is None:
            block.issues.append("stage2_missing_from_response")
            continue
        if not revised.strip():
            block.issues.append("stage2_empty_response")
            continue
        block.revised_text = revised


def run_stage2(blocks: list[Block]) -> None:
    for chunk in chunk_blocks_by_chars(blocks, STAGE2_MAX_CHARS_PER_BATCH, _stage2_payload_text):
        rewrite_stage2_batch(chunk)


# ---------------------------------------------------------------------------
# Step 3: Structural validation for placeholder tokens
# ---------------------------------------------------------------------------

def _token_structure_issues(text: str, expected_ids: set[str], token_map: dict[str, Tag]) -> list[str]:
    """
    Validate token balance and nesting without requiring the exact original order.

    Stage 2 may move formatting to a better semantic location, so the sequence
    can change. But every token must appear exactly once as an opening token and
    once as a closing token, and the result must be well-nested.
    """
    issues: list[str] = []
    open_counts = {tid: 0 for tid in expected_ids}
    close_counts = {tid: 0 for tid in expected_ids}
    stack: list[str] = []

    for match in TOKEN_PART_RE.finditer(text):
        slash, tid = match.groups()

        if tid not in expected_ids:
            issues.append(f"unexpected_token_id: {tid}")
            continue

        if tid not in token_map:
            issues.append(f"token_missing_from_map: {tid}")
            continue

        if not slash:
            open_counts[tid] += 1
            if open_counts[tid] > 1:
                issues.append(f"duplicate_open_token: {tid}")
            stack.append(tid)
            continue

        close_counts[tid] += 1
        if close_counts[tid] > 1:
            issues.append(f"duplicate_close_token: {tid}")

        if not stack:
            issues.append(f"stray_close_token: {tid}")
            continue

        if stack[-1] != tid:
            issues.append(f"misnested_close_token: expected_close_for_{stack[-1]}, got_{tid}")
            # Recover enough to keep reporting further issues.
            if tid in stack:
                while stack and stack[-1] != tid:
                    stack.pop()
                if stack:
                    stack.pop()
            continue

        stack.pop()

    for tid in sorted(expected_ids, key=int):
        if open_counts[tid] != 1:
            issues.append(f"open_token_count_{tid}: expected_1_got_{open_counts[tid]}")
        if close_counts[tid] != 1:
            issues.append(f"close_token_count_{tid}: expected_1_got_{close_counts[tid]}")

    if stack:
        issues.append(f"unclosed_tokens: {stack}")

    return issues


def validate_tokens(block: Block) -> bool:
    """Validate that Stage 2 preserved a structurally valid tokenized text."""
    block.validation_issues = []
    revised = block.revised_text or ""

    if not revised.strip():
        block.validation_issues.append("empty_revision")
        block.issues.extend(block.validation_issues)
        return False

    original_ids = set(TOKEN_RE.findall(block.original_text))
    revised_ids = set(TOKEN_RE.findall(revised))

    if original_ids != revised_ids:
        block.validation_issues.append(
            f"token_id_mismatch: expected {sorted(original_ids)}, got {sorted(revised_ids)}"
        )

    block.validation_issues.extend(_token_structure_issues(revised, original_ids, block.token_map))

    if block.validation_issues:
        block.issues.extend(block.validation_issues)
        return False

    return True


# ---------------------------------------------------------------------------
# Step 4: Detokenize and reinsert into the original DOM location
# ---------------------------------------------------------------------------

def detokenize_into_node(block: Block) -> None:
    """
    Parse revised placeholder text back into real tags and replace only the
    contiguous direct child nodes represented by this block. Nested lists,
    headings, tables, and other boundary tags outside this block stay untouched.
    """
    factory = BeautifulSoup("", "html.parser")

    def build(s: str, i: int, into: list, stop_id: Optional[str]) -> int:
        while i < len(s):
            close_match = _CLOSE_RE.match(s, i)
            if close_match:
                if close_match.group(1) == stop_id:
                    return close_match.end()
                i = close_match.end()  # stray/unmatched close: skip defensively
                continue

            open_match = _OPEN_RE.match(s, i)
            if open_match:
                tid = open_match.group(1)
                template = block.token_map.get(tid)

                if template is None:
                    i = open_match.end()
                    continue

                inner: list = []
                i = build(s, open_match.end(), inner, tid)

                if template.name in OPAQUE_TAGS:
                    # Images/breaks/etc. cannot contain rewritten text. Preserve
                    # the original tag/subtree, and keep any accidental inner text
                    # adjacent instead of dropping it.
                    into.append(copy.deepcopy(template))
                    into.extend(inner)
                    continue

                tag = factory.new_tag(template.name, **template.attrs)
                for child in inner:
                    tag.append(child)

                into.append(tag)
                continue

            next_token = s.find("⟦", i)
            if next_token == -1:
                if s[i:]:
                    into.append(NavigableString(s[i:]))
                return len(s)

            if next_token > i:
                into.append(NavigableString(s[i:next_token]))

            i = next_token

        return i

    if not block.nodes:
        return

    new_contents: list = []
    build(block.revised_text or "", 0, new_contents, None)

    first = block.nodes[0]
    for item in new_contents:
        first.insert_before(item)

    for old_node in block.nodes:
        old_node.extract()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_document(input_path: str, output_path: str) -> dict:
    html = Path(input_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    blocks = extract_blocks(soup)

    # Stage 1: document-wide plain-text rewrite.
    run_stage1(blocks)

    # Basic operational repair only: missing/empty Stage 1 responses.
    # No local language-rule checks are used.
    run_stage1_empty_repairs(blocks)

    # Stage 2: place formatting/image placeholders into the Stage 1 wording.
    run_stage2(blocks)

    # Structural validation: tokens must be present, balanced, and nested.
    for block in blocks:
        validate_tokens(block)

    report = {
        "total": len(blocks),
        "rewritten": 0,
        "failed": [],
        "warnings": [],
    }

    for block in blocks:
        failed = bool(
            block.validation_issues
            or "stage2_missing_from_response" in block.issues
            or "stage2_empty_response" in block.issues
        )

        if failed:
            report["failed"].append(
                {
                    "id": block.id,
                    "issues": block.issues,
                    "original_text": block.original_text,
                    "stage1_text": block.stage1_text,
                    "revised_text": block.revised_text,
                    "location": block.location,
                }
            )
            continue  # leave block.node untouched -> original content remains

        detokenize_into_node(block)
        report["rewritten"] += 1

        if block.issues:
            report["warnings"].append({"id": block.id, "issues": block.issues})

    Path(output_path).write_text(str(soup), encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python html_rewriter_nested_safe.py <input.html> <output.html>")
        sys.exit(1)

    result = process_document(sys.argv[1], sys.argv[2])
    print(json.dumps(result, ensure_ascii=False, indent=2))
