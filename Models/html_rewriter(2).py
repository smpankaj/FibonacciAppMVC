"""
Dutch document language-rewriter pipeline.

Two-stage design for long documents (50-100+ pages):

  Stage 1 - DOCUMENT-WIDE LANGUAGE PASS (plain text, no markup)
    All blocks' plain text (tags stripped) is sent to the LLM in
    character-budgeted chunks covering the *whole* document, each block
    tagged with its id. This produces one consistent, rule-compliant
    version of the wording for every block, chunk after chunk. Because
    there's no markup overhead per block, far more of the document fits
    per call than in stage 2, so coherence (terminology, tone) holds over
    a much larger span of the document than a single-stage approach would
    allow.

  Stage 2 - MARKUP RE-INSERTION (per block, tokenized)
    For each block, the LLM is given both the original tokenized text
    (with ⟦n⟧...⟦/n⟧ placeholders for inline tags) and the stage-1
    "approved" wording for that same block. Its job is narrower here: fit
    the placeholder tokens into the approved wording, not invent new
    wording. Blocks are still batched together (char-budgeted) so the
    model has local context, but the heavy lifting on language already
    happened in stage 1.

Full pipeline:
  HTML file
    -> parse with BeautifulSoup
    -> extract block-level elements, tokenize inline tags into placeholders
    -> STAGE 1: document-wide plain-text rewrite, chunked by char budget
    -> STAGE 2: batch blocks (tokenized + stage-1 reference) into OpenAI
       calls with structured JSON output, chunked by char budget
    -> validate structural integrity (placeholder tokens preserved)
    -> run automated rule checks (sentence length, passive voice, etc.)
    -> re-prompt (stage 2 only) the blocks that fail automated checks
    -> detokenize and reinsert revised text into the ORIGINAL DOM nodes
    -> serialize back to HTML

Design principle: the LLM only ever sees/produces plain text with lightweight
placeholder tokens for inline markup. It never generates raw HTML. All markup
is handled deterministically by this code, so a mangled tag is structurally
impossible rather than merely "hopefully avoided by the prompt".
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from bs4 import BeautifulSoup, NavigableString, Tag
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Block-level tags whose entire text content is sent to the LLM as one unit.
# Adjust to match the semantic units in your documents (add "th" etc if needed).
BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "blockquote", "figcaption"}

# Inline tags that must be preserved but whose exact position within the
# sentence may shift as the LLM rewrites (bold, links, emphasis, etc).
INLINE_TAGS = {"b", "strong", "i", "em", "u", "a", "span", "sup", "sub", "mark"}

# Character budgets per LLM call. These are conservative defaults for a
# large-context model; tune based on your model's actual context window,
# desired cost per call, and rate limits. Character count is a cheap proxy
# for token count (~4 chars/token for Dutch is a reasonable rule of thumb);
# swap in a real tokenizer (e.g. tiktoken) if you want tighter budgeting.
#
# Stage 1 has no markup overhead per block, so its budget can be much
# larger than stage 2's, which carries tokenized text + reference text +
# JSON structure for every block.
STAGE1_MAX_CHARS_PER_BATCH = 80_000   # ~20k tokens of plain text per call
STAGE2_MAX_CHARS_PER_BATCH = 20_000   # ~5k tokens; markup + reference doubles the payload

MODEL = "gpt-4.1"  # use whichever model you've validated for Dutch B1 rewriting

client = OpenAI()  # reads OPENAI_API_KEY from environment


# ---------------------------------------------------------------------------
# Style guide - consolidate your 80 rules here.
# Split rules into GENERATIVE (best done by the LLM's judgement) and
# CHECKABLE (objectively verifiable -> enforce with code, not prompting).
# ---------------------------------------------------------------------------

GENERATIVE_RULES = """
Je herschrijft interne bedrijfsdocumenten naar helder Nederlands.

Taalniveau en toon:
- Schrijf op B1-taalniveau: veelgebruikte woorden, korte zinnen, directe stijl.
- Gebruik actieve zinnen, vermijd de lijdende vorm waar mogelijk.
- Wees direct en concreet; vermijd vaag jargon en bureaucratische taal.
- Behoud de oorspronkelijke betekenis en feitelijke inhoud volledig.
- Verander geen cijfers, namen, data of technische termen die niet dubbelzinnig zijn.
- Gebruik consistente terminologie: als een begrip eerder in het document op
  een bepaalde manier is vertaald/geschreven, gebruik dezelfde formulering
  overal in het document.

Zinsbouw:
- Gebruik korte, enkelvoudige zinnen.
- Vermijd samengestelde zinnen met meer dan één bijzin.
- Eén gedachte per zin.
"""

# Rules enforced programmatically after the LLM pass (see check_rules()).
# Keep the LLM-facing prompt free of these numeric thresholds if you want the
# model reasoning qualitatively; or include them too as a first line of defense.
CHECKABLE_RULES_DESCRIPTION = """
- Maximaal 20 woorden per zin.
- Geen zinnen met meer dan 1 bijzin (komma + voegwoord constructies).
- Geen lijdende vorm ("wordt gedaan door", "werd gemaakt").
"""

STAGE1_SYSTEM_PROMPT = f"""Je bent een gespecialiseerde tekstredacteur voor interne bedrijfsdocumenten in het Nederlands.

{GENERATIVE_RULES}

Extra regels die automatisch gecontroleerd worden (houd hier ook rekening mee):
{CHECKABLE_RULES_DESCRIPTION}

Je krijgt een lijst tekstblokken uit hetzelfde document, elk met een uniek id,
in de oorspronkelijke volgorde. Herschrijf elk blok volgens de regels hierboven.
Omdat deze blokken uit hetzelfde document komen: gebruik overal dezelfde
terminologie, formuleringen voor herhaalde begrippen, en een consistente toon.

Geef uitsluitend geldige JSON terug volgens het gevraagde schema, zonder uitleg.
"""

STAGE2_SYSTEM_PROMPT = """Je krijgt per tekstblok twee versies:
1. "text": de oorspronkelijke tekst met opmaaktokens zoals ⟦1⟧...⟦/1⟧ (vet, cursief, links, etc).
2. "reference": de al goedgekeurde, herschreven versie van diezelfde tekst (zonder opmaaktokens).

Jouw taak is UITSLUITEND: neem de "reference"-tekst exact over, en plaats de
opmaaktokens uit "text" op de juiste plek in die reference-tekst, rond het
tekstfragment waar ze inhoudelijk bij horen.

Regels:
- Herschrijf de reference-tekst verder NIET; gebruik hem woordelijk.
- Pas de tekst alleen minimaal aan (bijv. een klein woordje) als dat strikt
  nodig is om een token natuurlijk te laten passen - dit zou zelden nodig
  moeten zijn.
- Alle tokens uit "text" MOETEN exact terugkomen: zelfde nummer, zelfde
  openings- en sluitingsvorm. Verzin geen nieuwe tokens en verwijder er geen.
- Als een tokenblok geen duidelijke tegenhanger heeft in de reference-tekst
  (bijvoorbeeld omdat het om een cijfer, naam of link gaat), plaats het token
  rond het dichtstbijzijnde overeenkomstige fragment.

Geef uitsluitend geldige JSON terug volgens het gevraagde schema, zonder uitleg.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Block:
    id: str
    node: Tag                       # reference to the live DOM node
    original_text: str              # tokenized plain text sent to the LLM
    token_map: dict[str, Tag]       # placeholder id -> original inline Tag (a copy, for attrs)
    stage1_text: Optional[str] = None   # approved plain-text wording from stage 1
    revised_text: Optional[str] = None  # final tokenized text, filled in stage 2
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1: Extraction & tokenization
# ---------------------------------------------------------------------------

def tokenize_block(node: Tag) -> tuple[str, dict[str, Tag]]:
    """
    Walk a block-level node's contents, replacing inline tags with
    placeholder tokens like ⟦1⟧...⟦/1⟧ and returning the plain text plus
    a mapping from token id -> a template Tag (name + attrs) to reconstruct it.
    """
    token_map: dict[str, Tag] = {}
    counter = [0]

    def walk(node: Tag) -> str:
        out = []
        for child in node.children:
            if isinstance(child, NavigableString):
                out.append(str(child))
            elif isinstance(child, Tag):
                if child.name in INLINE_TAGS:
                    counter[0] += 1
                    tid = str(counter[0])
                    token_map[tid] = child  # keep original tag as template (name+attrs)
                    inner = walk(child)
                    out.append(f"⟦{tid}⟧{inner}⟦/{tid}⟧")
                else:
                    # Unexpected nested block/void tag inside a block element -
                    # keep it as an opaque, non-editable token so nothing breaks.
                    counter[0] += 1
                    tid = str(counter[0])
                    token_map[tid] = child
                    out.append(f"⟦{tid}⟧⟦/{tid}⟧")
        return "".join(out)

    text = walk(node)
    return text, token_map


def extract_blocks(soup: BeautifulSoup) -> list[Block]:
    blocks = []
    for node in soup.find_all(list(BLOCK_TAGS)):
        # Skip blocks nested inside other blocks we already capture at a
        # higher level (e.g. avoid double-processing <li><p>...) - adjust
        # this guard if your documents nest block tags intentionally.
        if node.find_parent(list(BLOCK_TAGS)) is not None:
            continue
        text, token_map = tokenize_block(node)
        if not text.strip():
            continue
        blocks.append(Block(id=f"b_{uuid.uuid4().hex[:8]}", node=node,
                             original_text=text, token_map=token_map))
    return blocks


# ---------------------------------------------------------------------------
# Chunking helper - character-budgeted, not fixed block count.
# Used for both stage 1 and stage 2 so long documents (50-100+ pages) are
# split into as few calls as the budget allows, rather than a fixed and
# possibly-too-large or too-small number of blocks per call.
# ---------------------------------------------------------------------------

def chunk_blocks_by_chars(
    blocks: list[Block],
    max_chars: int,
    text_fn: Callable[[Block], str],
) -> list[list[Block]]:
    """
    Greedily group blocks into chunks so the total character count of
    text_fn(block) per chunk stays under max_chars. A single block whose
    own text exceeds max_chars is placed alone in its own chunk (it will
    still be sent - LLM calls can usually absorb a single oversized block
    even if the budget is nominally exceeded - but flagged for awareness).
    """
    chunks: list[list[Block]] = []
    current: list[Block] = []
    current_chars = 0

    for b in blocks:
        length = len(text_fn(b))
        if length > max_chars:
            if current:
                chunks.append(current)
                current, current_chars = [], 0
            chunks.append([b])
            continue
        if current and current_chars + length > max_chars:
            chunks.append(current)
            current, current_chars = [], 0
        current.append(b)
        current_chars += length

    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Step 2a: STAGE 1 - document-wide plain-text language pass
# ---------------------------------------------------------------------------

STAGE1_RESPONSE_SCHEMA = {
    "name": "stage1_blocks",
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


def _plain_text(block: Block) -> str:
    """Block text with placeholder tokens stripped - what stage 1 sees."""
    return TOKEN_RE.sub("", block.original_text)


def rewrite_stage1_chunk(blocks: list[Block]) -> None:
    """Send one char-budgeted chunk of blocks (plain text) to the LLM and
    fill in block.stage1_text in place."""
    payload = [{"id": b.id, "text": _plain_text(b)} for b in blocks]

    user_prompt = (
        "Herschrijf onderstaande blokken uit hetzelfde document:\n\n"
        + json.dumps({"blocks": payload}, ensure_ascii=False, indent=2)
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": STAGE1_RESPONSE_SCHEMA},
        temperature=0.2,
    )

    result = json.loads(resp.choices[0].message.content)
    by_id = {r["id"]: r["revised_text"] for r in result["blocks"]}

    for b in blocks:
        if b.id not in by_id:
            b.issues.append("stage1_missing_from_response")
            b.stage1_text = _plain_text(b)  # fall back to original wording
            continue
        b.stage1_text = by_id[b.id]


def run_stage1(blocks: list[Block]) -> None:
    """
    Run the document-wide language pass over ALL blocks, chunked by
    character budget so long documents (50-100+ pages) are covered in as
    few calls as possible while staying within a safe per-call size.
    """
    for chunk_blocks in chunk_blocks_by_chars(blocks, STAGE1_MAX_CHARS_PER_BATCH, _plain_text):
        rewrite_stage1_chunk(chunk_blocks)


# ---------------------------------------------------------------------------
# Step 2b: STAGE 2 - markup re-insertion, batched with structured output
# ---------------------------------------------------------------------------

STAGE2_RESPONSE_SCHEMA = {
    "name": "stage2_blocks",
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


def _stage2_payload_text(block: Block) -> str:
    """Combined size proxy for chunking stage 2 (tokenized text + reference)."""
    return block.original_text + (block.stage1_text or "")


def rewrite_stage2_batch(blocks: list[Block]) -> None:
    """
    Send a batch of blocks to the LLM with both the tokenized original text
    and the stage-1 approved wording, and fill in revised_text in place.
    The model's job here is markup placement, not fresh language rewriting.
    """
    payload = [
        {"id": b.id, "text": b.original_text, "reference": b.stage1_text or _plain_text(b)}
        for b in blocks
    ]

    user_prompt = (
        "Plaats de opmaaktokens uit 'text' in de bijbehorende 'reference'-tekst "
        "voor elk van onderstaande blokken:\n\n"
        + json.dumps({"blocks": payload}, ensure_ascii=False, indent=2)
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": STAGE2_RESPONSE_SCHEMA},
        temperature=0.1,
    )

    result = json.loads(resp.choices[0].message.content)
    by_id = {r["id"]: r["revised_text"] for r in result["blocks"]}

    for b in blocks:
        if b.id not in by_id:
            b.issues.append("missing_from_response")
            continue
        b.revised_text = by_id[b.id]


def run_stage2(blocks: list[Block]) -> None:
    """Run markup re-insertion over all blocks, chunked by character budget."""
    for chunk_blocks in chunk_blocks_by_chars(blocks, STAGE2_MAX_CHARS_PER_BATCH, _stage2_payload_text):
        rewrite_stage2_batch(chunk_blocks)


# ---------------------------------------------------------------------------
# Step 3: Structural validation (tokens intact)
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"⟦/?(\d+)⟧")
_OPEN_RE = re.compile(r"⟦(\d+)⟧")
_CLOSE_RE = re.compile(r"⟦/(\d+)⟧")


def validate_tokens(block: Block) -> bool:
    original_ids = sorted(set(TOKEN_RE.findall(block.original_text)))
    revised_ids = sorted(set(TOKEN_RE.findall(block.revised_text or "")))
    if original_ids != revised_ids:
        block.issues.append(f"token_mismatch: expected {original_ids}, got {revised_ids}")
        return False
    if not (block.revised_text or "").strip():
        block.issues.append("empty_revision")
        return False
    return True


# ---------------------------------------------------------------------------
# Step 4: Automated rule checks (the "checkable" rules from your 80)
# Requires: pip install spacy && python -m spacy download nl_core_news_lg
# ---------------------------------------------------------------------------

def check_rules(text: str, nlp) -> list[str]:
    """Return a list of human-readable rule violations for one plain-text block."""
    clean = TOKEN_RE.sub("", text)
    issues = []
    doc = nlp(clean)
    for sent in doc.sents:
        words = [t for t in sent if not t.is_punct]
        if len(words) > 20:
            issues.append(f"zin te lang ({len(words)} woorden): '{sent.text.strip()}'")
        # crude passive-voice heuristic for Dutch: "worden/wordt/werd(en)" + past participle
        if re.search(r"\b(wordt|worden|werd|werden|is|zijn)\b.*\b\w+(d|t)\b", sent.text, re.I) \
                and re.search(r"\bdoor\b", sent.text, re.I):
            issues.append(f"mogelijk lijdende vorm: '{sent.text.strip()}'")
    return issues


# ---------------------------------------------------------------------------
# Step 5: Detokenize & reinsert into the original DOM
# ---------------------------------------------------------------------------

def detokenize_into_node(block: Block) -> None:
    """
    Parse the revised placeholder text back into real tags (using the
    original tag name + attributes from token_map) and replace the node's
    children in-place. The block's own tag (p, li, h2, ...) is untouched -
    only its inner content is replaced.

    Recursive-descent parse of ⟦n⟧...⟦/n⟧ spans into real bs4 Tag /
    NavigableString objects, so nesting and reordering by the LLM are both
    handled correctly.
    """
    factory = BeautifulSoup("", "html.parser")

    def build(s, i, into, stop_id):
        while i < len(s):
            cm = _CLOSE_RE.match(s, i)
            if cm:
                if cm.group(1) == stop_id:
                    return cm.end()
                i = cm.end()  # stray/unmatched close: skip defensively
                continue
            om = _OPEN_RE.match(s, i)
            if om:
                tid = om.group(1)
                template = block.token_map.get(tid)
                tag = factory.new_tag(template.name, **(template.attrs if template else {}))
                inner = []
                i = build(s, om.end(), inner, tid)
                for child in inner:
                    tag.append(child)
                into.append(tag)
                continue
            nxt = s.find("⟦", i)
            if nxt == -1:
                if s[i:]:
                    into.append(NavigableString(s[i:]))
                return len(s)
            if nxt > i:
                into.append(NavigableString(s[i:nxt]))
            i = nxt
        return i

    new_contents = []
    build(block.revised_text, 0, new_contents, None)

    block.node.clear()
    for item in new_contents:
        block.node.append(item)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_document(input_path: str, output_path: str) -> dict:
    import spacy
    nlp = spacy.load("nl_core_news_lg")

    html = Path(input_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    blocks = extract_blocks(soup)

    # --- stage 1: document-wide plain-text language pass, chunked by chars ---
    run_stage1(blocks)

    # --- stage 2: markup re-insertion, chunked by chars ---
    run_stage2(blocks)

    for b in blocks:
        validate_tokens(b)

    # --- automated rule check + targeted re-prompt (stage 2 only - the
    # approved wording from stage 1 doesn't change, we just re-run markup
    # placement in case the first attempt introduced a rule violation) ---
    retry_candidates = []
    for b in blocks:
        if "token_mismatch" in " ".join(b.issues) or "empty_revision" in b.issues:
            continue  # handled separately below
        rule_issues = check_rules(b.revised_text, nlp)
        if rule_issues:
            b.issues.extend(rule_issues)
            retry_candidates.append(b)

    if retry_candidates:
        for chunk_blocks in chunk_blocks_by_chars(
            retry_candidates, STAGE2_MAX_CHARS_PER_BATCH, _stage2_payload_text
        ):
            rewrite_stage2_batch(chunk_blocks)
            for b in chunk_blocks:
                validate_tokens(b)

    # --- blocks that still failed structural validation: fall back to original ---
    report = {"total": len(blocks), "failed": [], "rule_flagged": []}
    for b in blocks:
        if any("token_mismatch" in i or i == "empty_revision" or i == "missing_from_response"
               for i in b.issues):
            report["failed"].append({"id": b.id, "issues": b.issues, "text": b.original_text})
            continue  # leave block.node untouched -> original text stays
        detokenize_into_node(b)
        if b.issues:
            report["rule_flagged"].append({"id": b.id, "issues": b.issues})

    Path(output_path).write_text(str(soup), encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python html_rewriter.py <input.html> <output.html>")
        sys.exit(1)

    report = process_document(sys.argv[1], sys.argv[2])
    print(json.dumps(report, ensure_ascii=False, indent=2))
