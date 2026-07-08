"""
Document language-rewriter pipeline.

Two-stage design for long documents (50-100+ pages):

  Stage 1 - DOCUMENT-WIDE LANGUAGE PASS (plain text, no markup)
    All blocks' plain text (tags stripped) is sent to the LLM in
    token-budgeted chunks covering the *whole* document, each block
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
    wording. Blocks are still batched together (token-budgeted) so the
    model has local context, but the heavy lifting on language already
    happened in stage 1.

Full pipeline:
  HTML text (string in, string out - no file I/O inside the pipeline)
    -> parse with BeautifulSoup
    -> extract block-level elements, tokenize inline tags into placeholders
    -> STAGE 1: document-wide plain-text rewrite, chunked by token budget
    -> STAGE 2: batch blocks (tokenized + stage-1 reference) into OpenAI
       calls with structured JSON output, chunked by token budget
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

# Hard cap on INPUT tokens per OpenAI API call (system prompt + user message
# combined). Does NOT include the model's output tokens - if you also need to
# bound total request+response tokens (e.g. for a strict rate-limit budget),
# reduce this further to leave headroom for the response.
MAX_INPUT_TOKENS_PER_CALL = 10_000

# Fixed per-call overhead that isn't part of any single block: the JSON
# wrapper text ('{"blocks": [...]}'), the short instruction line before the
# JSON payload, and a small safety margin for tokenizer estimation error.
# This is subtracted from MAX_INPUT_TOKENS_PER_CALL - after also subtracting
# the system prompt's own token count - to get the real per-batch budget
# available for block payloads.
CALL_OVERHEAD_TOKENS = 150

MODEL = "gpt-5.1"

client = OpenAI()  # reads OPENAI_API_KEY from environment


# ---------------------------------------------------------------------------
# Token counting
#
# Uses tiktoken (OpenAI's real tokenizer) when available, so batches are
# sized against the ACTUAL token count the API will see - not an estimate.
# Falls back to a conservative chars/4 approximation if tiktoken isn't
# installed, so the pipeline still runs, but tiktoken is strongly
# recommended: `pip install tiktoken`.
# ---------------------------------------------------------------------------

try:
    import tiktoken
    try:
        _ENCODING = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        # tiktoken may not have a direct mapping for a very new model name
        # yet (e.g. "gpt-5.1") - cl100k_base is a safe, widely-used default
        # and gives a close enough count for budgeting purposes. Swap this
        # for the exact encoding once tiktoken ships official support.
        _ENCODING = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))

except ImportError:
    import warnings
    warnings.warn(
        "tiktoken not installed - falling back to an approximate "
        "chars/4 token estimate. Install tiktoken for accurate token "
        "budgeting: pip install tiktoken",
        stacklevel=2,
    )

    def count_tokens(text: str) -> int:
        return len(text) // 4 + 1


# ---------------------------------------------------------------------------
# Style guide - consolidate your 80 rules here.
# Split rules into GENERATIVE (best done by the LLM's judgement) and
# CHECKABLE (objectively verifiable -> enforce with code, not prompting).
# ---------------------------------------------------------------------------

GENERATIVE_RULES = """
You are rewriting internal company documents into clear, plain language.

Language level and tone:
- Write at a B1 language level: common words, short sentences, a direct style.
- Use active sentences; avoid the passive voice where possible.
- Be direct and concrete; avoid vague jargon and bureaucratic language.
- Preserve the original meaning and factual content completely.
- Do not change numbers, names, dates, or technical terms that are unambiguous.
- Use consistent terminology: if a term has already been phrased a certain
  way earlier in the document, use the same phrasing everywhere else in the
  document.

Sentence structure:
- Use short, simple sentences.
- Avoid compound sentences with more than one subordinate clause.
- One idea per sentence.
"""

# Rules enforced programmatically after the LLM pass (see check_rules()).
# Keep the LLM-facing prompt free of these numeric thresholds if you want the
# model reasoning qualitatively; or include them too as a first line of defense.
CHECKABLE_RULES_DESCRIPTION = """
- Maximum 20 words per sentence.
- No sentences with more than 1 subordinate clause (comma + conjunction constructions).
- No passive voice ("is done by", "was made by").
"""

STAGE1_SYSTEM_PROMPT = f"""You are a specialized text editor for internal company documents.

{GENERATIVE_RULES}

Additional rules that are checked automatically (keep these in mind too):
{CHECKABLE_RULES_DESCRIPTION}

You will receive a list of text blocks from the same document, each with a
unique id, in their original order. Rewrite each block according to the
rules above. Because these blocks come from the same document: use the same
terminology, phrasing for recurring concepts, and a consistent tone
throughout.

Return ONLY valid JSON matching the requested schema, with no explanation.
"""

STAGE2_SYSTEM_PROMPT = """For each text block you will receive two versions:
1. "text": the original text with markup tokens such as ⟦1⟧...⟦/1⟧ (bold, italics, links, etc).
2. "reference": the already-approved, rewritten version of that same text (without markup tokens).

Your task is ONLY the following: take the "reference" text exactly as given,
and place the markup tokens from "text" at the correct position within that
reference text, around the text fragment they belong to in meaning.

Rules:
- Do NOT further rewrite the reference text; use it verbatim.
- Only adjust the text minimally (e.g. a small word) if strictly necessary
  to make a token fit naturally - this should rarely be needed.
- All tokens from "text" MUST come back exactly: same number, same opening
  and closing form. Never invent new tokens and never drop existing ones.
- If a token span has no clear counterpart in the reference text (for
  example because it wraps a number, a name, or a link), place the token
  around the closest corresponding fragment.

Return ONLY valid JSON matching the requested schema, with no explanation.
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
# Token budgeting helpers
# ---------------------------------------------------------------------------

def _available_payload_tokens(system_prompt: str) -> int:
    """
    Tokens left over for the block payload once the system prompt and a
    fixed per-call overhead (JSON wrapper text, instruction line, safety
    margin) are subtracted from the hard MAX_INPUT_TOKENS_PER_CALL cap.
    Computed from the system prompt's REAL token count, not an estimate -
    so if you grow your 80 rules, this automatically shrinks the batch
    size to compensate and keep every call under the cap.
    """
    system_tokens = count_tokens(system_prompt)
    budget = MAX_INPUT_TOKENS_PER_CALL - system_tokens - CALL_OVERHEAD_TOKENS
    if budget <= 0:
        raise ValueError(
            f"MAX_INPUT_TOKENS_PER_CALL ({MAX_INPUT_TOKENS_PER_CALL}) is too small to fit "
            f"the system prompt ({system_tokens} tokens) plus overhead "
            f"({CALL_OVERHEAD_TOKENS} tokens) - there'd be no room left for any block. "
            f"Increase MAX_INPUT_TOKENS_PER_CALL or shorten the system prompt."
        )
    return budget


# ---------------------------------------------------------------------------
# Chunking helper - TOKEN-budgeted (via count_tokens), not fixed block count
# and not a character approximation. Used for both stage 1 and stage 2 so
# every API call - system prompt + user payload combined - stays under
# MAX_INPUT_TOKENS_PER_CALL, regardless of document length.
# ---------------------------------------------------------------------------

def chunk_blocks_by_tokens(
    blocks: list[Block],
    max_tokens: int,
    text_fn: Callable[[Block], str],
) -> list[list[Block]]:
    """
    Greedily group blocks into chunks so the total token count of
    text_fn(block) per chunk stays under max_tokens. A single block whose
    own text exceeds max_tokens is placed alone in its own chunk and its
    call will exceed the nominal budget - there's no way to shrink a single
    block further without splitting its sentences, which risks breaking
    placeholder token integrity, so this is flagged rather than silently
    truncated.
    """
    chunks: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0

    for b in blocks:
        length = count_tokens(text_fn(b))
        if length > max_tokens:
            if current:
                chunks.append(current)
                current, current_tokens = [], 0
            b.issues.append(
                f"oversized_block: {length} tokens exceeds the {max_tokens}-token "
                f"batch budget on its own; sent alone, call will exceed the target cap"
            )
            chunks.append([b])
            continue
        if current and current_tokens + length > max_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(b)
        current_tokens += length

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


def _stage1_payload_json(block: Block) -> str:
    """The exact JSON snippet that will be sent for this block in stage 1 -
    used to size chunks against real token counts, not an approximation."""
    return json.dumps({"id": block.id, "text": _plain_text(block)}, ensure_ascii=False)


def rewrite_stage1_chunk(blocks: list[Block]) -> None:
    """Send one token-budgeted chunk of blocks (plain text) to the LLM and
    fill in block.stage1_text in place."""
    payload = [{"id": b.id, "text": _plain_text(b)} for b in blocks]

    user_prompt = (
        "Rewrite the following blocks from the same document:\n\n"
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
    Run the document-wide language pass over ALL blocks, chunked so that
    system prompt + user payload together stay under MAX_INPUT_TOKENS_PER_CALL
    for every call, however long the document is.
    """
    budget = _available_payload_tokens(STAGE1_SYSTEM_PROMPT)
    for chunk_blocks in chunk_blocks_by_tokens(blocks, budget, _stage1_payload_json):
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


def _stage2_payload_json(block: Block) -> str:
    """The exact JSON snippet that will be sent for this block in stage 2 -
    used to size chunks against real token counts, not an approximation."""
    return json.dumps(
        {"id": block.id, "text": block.original_text, "reference": block.stage1_text or _plain_text(block)},
        ensure_ascii=False,
    )


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
        "Place the markup tokens from 'text' into the corresponding "
        "'reference' text for each of the following blocks:\n\n"
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
    """Run markup re-insertion over all blocks, chunked so system prompt +
    user payload together stay under MAX_INPUT_TOKENS_PER_CALL."""
    budget = _available_payload_tokens(STAGE2_SYSTEM_PROMPT)
    for chunk_blocks in chunk_blocks_by_tokens(blocks, budget, _stage2_payload_json):
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
#
# NOTE: the passive-voice heuristic below is Dutch-specific (it matches
# Dutch auxiliary verbs "wordt/worden/werd/werden" + "door"). If you run
# this against English/Spanish/Portuguese documents, this check will not
# fire correctly for those languages - it needs a per-language pattern (or
# a dependency-parse-based check) before it's meaningful outside Dutch.
# ---------------------------------------------------------------------------

def check_rules(text: str, nlp) -> list[str]:
    """Return a list of human-readable rule violations for one plain-text block."""
    clean = TOKEN_RE.sub("", text)
    issues = []
    doc = nlp(clean)
    for sent in doc.sents:
        words = [t for t in sent if not t.is_punct]
        if len(words) > 20:
            issues.append(f"sentence too long ({len(words)} words): '{sent.text.strip()}'")
        # crude passive-voice heuristic for Dutch: "worden/wordt/werd(en)" + past participle
        if re.search(r"\b(wordt|worden|werd|werden|is|zijn)\b.*\b\w+(d|t)\b", sent.text, re.I) \
                and re.search(r"\bdoor\b", sent.text, re.I):
            issues.append(f"possible passive voice: '{sent.text.strip()}'")
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

def process_document(html_text: str) -> tuple[str, dict]:
    """
    Rewrite an HTML document's text content according to the style rules,
    while preserving markup structure exactly.

    Pure function: takes the HTML as a string, returns the rewritten HTML
    as a string (plus a report dict) - no file paths, no disk I/O. Call
    this directly from your own code:

        output_html, report = process_document(input_html)

    Args:
        html_text: the full HTML document as a string.

    Returns:
        (output_html, report) where:
          - output_html is the rewritten document as a string. Blocks that
            failed structural validation are left with their ORIGINAL text
            untouched, so a bad LLM response degrades to "this one
            paragraph didn't get rewritten" rather than corrupting anything.
          - report is a dict: {"total": int, "failed": [...], "rule_flagged": [...]}
            summarizing which blocks failed structural validation (and were
            left unchanged) vs. which were rewritten but still have an
            unresolved rule-check flag.
    """
    import spacy
    nlp = spacy.load("nl_core_news_lg")

    soup = BeautifulSoup(html_text, "html.parser")

    blocks = extract_blocks(soup)

    # --- stage 1: document-wide plain-text language pass, chunked by tokens ---
    run_stage1(blocks)

    # --- stage 2: markup re-insertion, chunked by tokens ---
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
        retry_budget = _available_payload_tokens(STAGE2_SYSTEM_PROMPT)
        for chunk_blocks in chunk_blocks_by_tokens(
            retry_candidates, retry_budget, _stage2_payload_json
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

    return str(soup), report


if __name__ == "__main__":
    # Simple CLI wrapper for manual testing only - process_document() itself
    # takes and returns strings, with no file I/O of its own.
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python html_rewriter.py <input.html>   (rewritten HTML goes to stdout, report to stderr)")
        sys.exit(1)

    input_html = Path(sys.argv[1]).read_text(encoding="utf-8")
    output_html, report = process_document(input_html)
    print(output_html)
    print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
