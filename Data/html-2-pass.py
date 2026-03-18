"""
Two-pass HTML transformer for high-accuracy knowledge-base enrichment.

Strategy:
1) Pass 1 (parallel): Rewrite one block at a time (p/li), with read-only context from
   neighboring original blocks and nearest heading.
2) Pass 2 (targeted): Polish only selected transitions between adjacent rewritten blocks.
3) Hard safety: per-block and global invariant checks (numbers, dates, currencies, URLs,
   IDs, percentages, product names). Any mismatch falls back safely.

No Delta writes. Reads one HTML document (via read_html()) and saves transformed HTML to
OUTPUT_PATH plus an audit JSON sidecar.
"""

import os
import re
import json
import time
import difflib
import random
import asyncio
from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Comment
from bs4.element import Tag
from openai import AsyncOpenAI

try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception:
    pass


# ---------------------------
# 1) Configuration
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1500
REQUEST_TIMEOUT_SECONDS = 60

# Output path requested by user; can be overridden with env OUTPUT_PATH.
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output path")

# Optional product name allowlist (pipe-separated).
# Example: "Premier Savings Account|Gold Rewards Card"
PRODUCT_NAME_ALLOWLIST = [
    p.strip()
    for p in os.getenv("PRODUCT_NAME_ALLOWLIST", "").split("|")
    if p.strip()
]

# Concurrency and scale controls
MAX_REQUEST_CONCURRENCY = int(os.getenv("MAX_REQUEST_CONCURRENCY", "12"))
PASS1_BATCH_SIZE = int(os.getenv("PASS1_BATCH_SIZE", "300"))
PASS2_BATCH_SIZE = int(os.getenv("PASS2_BATCH_SIZE", "150"))
MAX_PASS2_CALLS = int(os.getenv("MAX_PASS2_CALLS", "400"))

# Context and block limits
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1200"))
MAX_BLOCK_CHARS = int(os.getenv("MAX_BLOCK_CHARS", "3500"))
MIN_BLOCK_CHARS = int(os.getenv("MIN_BLOCK_CHARS", "20"))
MAX_DIFF_CHARS = int(os.getenv("MAX_DIFF_CHARS", "120000"))

# Retry
MAX_RETRIES = 4
RETRY_BASE_SECONDS = 1.5

# Scope controls
APPROVED_BLOCK_TAGS = {"p", "li"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
NON_EDITABLE_TAGS = {
    "head",
    "meta",
    "link",
    "script",
    "style",
    "code",
    "pre",
    "title",
    "noscript",
    "textarea",
    "svg",
    "math",
}
HIGH_RISK_BLOCK_TAGS = {"table", "code", "pre"}
FREEZE_TAGS = {"blockquote", "caption", "th", "td"}


# ---------------------------
# 2) Regex and invariants
# ---------------------------
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_CURRENCY_RE = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\b(?:USD|EUR|GBP)\s?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_ID_RE = re.compile(
    r"\b(?:[A-Z]{2}\d{2}[A-Z0-9]{10,30}|[A-Z]{1,5}-\d{3,}|"
    r"(?:ID|ACCOUNT|ACCT|REF|REFERENCE|TICKET|CASE|CUSIP|ISIN|IBAN|SWIFT)"
    r"[:#\s-]*[A-Z0-9-]{3,})\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_PRODUCT_NAME_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&/-]+(?:\s+[A-Z][A-Za-z0-9&/-]+){0,4}\s+"
    r"(?:Account|Card|Loan|Mortgage|Fund|Plan|Policy|ETF|IRA|CD|Bond|Note))\b"
)
_DISCLOSURE_RE = re.compile(
    r"\b(?:disclaimer|not\s+investment\s+advice|for\s+informational\s+purposes\s+only|"
    r"past\s+performance|subject\s+to\s+change|terms\s+and\s+conditions)\b",
    re.IGNORECASE,
)
_RATE_KEYWORD_RE = re.compile(
    r"\b(?:apr|apy|interest\s+rate|fixed\s+rate|variable\s+rate|basis\s+points?|bps)\b",
    re.IGNORECASE,
)
_AMOUNT_CONTEXT_RE = re.compile(
    r"\b(?:amount|balance|principal|payment|fee|charge|limit|minimum|maximum)\b",
    re.IGNORECASE,
)


class LLMCallError(Exception):
    pass


@dataclass
class Block:
    block_id: int
    node: Tag
    tag_name: str
    heading: str
    original_text: str
    prev_text: str = ""
    next_text: str = ""
    rewritten_text: str = ""
    status: str = "pending"
    skip_reason: str = ""
    pass1_error: str = ""
    pass2_error: str = ""
    invariants_ok_pass1: bool = True
    invariants_ok_pass2: bool = True
    transition_polished: bool = False


def normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", s or "").strip()


def trim_context(s: str, max_chars: int) -> str:
    t = normalize_ws(s)
    if len(t) <= max_chars:
        return t
    return f"{t[:max_chars]} ..."


def to_counter_dict(values: List[str]) -> Dict[str, int]:
    cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    return dict(sorted(Counter(cleaned).items()))


def extract_product_name_candidates(text: str) -> List[str]:
    txt = text or ""
    out = list(_PRODUCT_NAME_RE.findall(txt))
    for name in PRODUCT_NAME_ALLOWLIST:
        out.extend(m.group(0) for m in re.finditer(re.escape(name), txt))
    return out


def extract_invariants(text: str) -> Dict[str, Dict[str, int]]:
    txt = text or ""
    return {
        "numbers": to_counter_dict(_NUMBER_RE.findall(txt)),
        "currencies": to_counter_dict(_CURRENCY_RE.findall(txt)),
        "percentages": to_counter_dict(_PERCENT_RE.findall(txt)),
        "dates": to_counter_dict(_DATE_RE.findall(txt)),
        "urls": to_counter_dict(_URL_RE.findall(txt)),
        "ids": to_counter_dict(_ID_RE.findall(txt)),
        "product_names": to_counter_dict(extract_product_name_candidates(txt)),
    }


def capped_dict(d: Dict[str, int], max_items: int = 50) -> Dict[str, int]:
    if len(d) <= max_items:
        return d
    items = list(d.items())[:max_items]
    out = dict(items)
    out["__truncated__"] = len(d) - max_items
    return out


def invariant_mismatches(
    original_inv: Dict[str, Dict[str, int]],
    rewritten_inv: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    mismatches: Dict[str, Dict[str, Dict[str, int]]] = {}
    keys = sorted(set(original_inv.keys()) | set(rewritten_inv.keys()))
    for k in keys:
        before = original_inv.get(k, {})
        after = rewritten_inv.get(k, {})
        if before != after:
            mismatches[k] = {
                "original": capped_dict(before),
                "rewritten": capped_dict(after),
            }
    return mismatches


def has_explicit_body(html_text: str) -> bool:
    return bool(re.search(r"<body\b", html_text or "", flags=re.IGNORECASE))


def get_processing_root(soup: BeautifulSoup, source_html: str) -> Tuple[Tag, str]:
    explicit_body = has_explicit_body(source_html)
    if soup.body is not None:
        return soup.body, ("body" if explicit_body else "fragment")
    return soup, "fragment"


def serialize_html(soup: BeautifulSoup, scope_mode: str) -> str:
    if scope_mode == "fragment" and soup.body is not None:
        return "".join(str(c) for c in soup.body.contents)
    return str(soup)


def has_ancestor(tag: Tag, names: set) -> bool:
    p = tag.parent
    while p is not None and isinstance(p, Tag):
        nm = p.name.lower() if p.name else ""
        if nm in names:
            return True
        p = p.parent
    return False


def has_inline_children(tag: Tag) -> bool:
    # We skip blocks with inline markup to preserve structure safely.
    for child in tag.contents:
        if isinstance(child, Tag):
            return True
    return False


def contains_high_risk_content(text: str) -> bool:
    t = text or ""
    return bool(
        _DISCLOSURE_RE.search(t)
        or _RATE_KEYWORD_RE.search(t)
        or _PERCENT_RE.search(t)
        or _CURRENCY_RE.search(t)
        or _DATE_RE.search(t)
        or _ID_RE.search(t)
        or (_AMOUNT_CONTEXT_RE.search(t) and _NUMBER_RE.search(t))
    )


def sanitize_model_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:text)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip().strip('"')
    return t


def parse_blocks(html_text: str) -> Tuple[BeautifulSoup, List[Block], str]:
    soup = BeautifulSoup(html_text or "", "lxml")
    root, scope_mode = get_processing_root(soup, html_text or "")

    blocks: List[Block] = []
    current_heading = ""
    block_id = 0

    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        nm = el.name.lower() if el.name else ""

        if nm in HEADING_TAGS:
            heading_txt = normalize_ws(el.get_text(" ", strip=True))
            if heading_txt:
                current_heading = heading_txt
            continue

        if nm not in APPROVED_BLOCK_TAGS:
            continue

        txt = normalize_ws(el.get_text(" ", strip=True))
        if not txt:
            continue

        blocks.append(
            Block(
                block_id=block_id,
                node=el,
                tag_name=nm,
                heading=current_heading,
                original_text=txt,
            )
        )
        block_id += 1

    for i, b in enumerate(blocks):
        b.prev_text = blocks[i - 1].original_text if i > 0 else ""
        b.next_text = blocks[i + 1].original_text if i + 1 < len(blocks) else ""

    return soup, blocks, scope_mode


def freeze_reason_for_block(block: Block) -> str:
    if len(block.original_text) < MIN_BLOCK_CHARS:
        return "too_short"
    if len(block.original_text) > MAX_BLOCK_CHARS:
        return "too_long"
    if has_ancestor(block.node, NON_EDITABLE_TAGS):
        return "non_editable_ancestor"
    if has_ancestor(block.node, HIGH_RISK_BLOCK_TAGS):
        return "high_risk_block_ancestor"
    if has_ancestor(block.node, FREEZE_TAGS):
        return "frozen_container_ancestor"
    if has_inline_children(block.node):
        return "inline_markup_present"
    if contains_high_risk_content(block.original_text):
        return "high_risk_content"
    return ""


def transition_needs_polish(prev_text: str, curr_text: str) -> bool:
    prev = normalize_ws(prev_text)
    curr = normalize_ws(curr_text)
    if not prev or not curr:
        return False

    if len(curr) <= 120:
        return True

    curr_start = curr[:120].lower()
    if re.match(r"^(however|therefore|moreover|additionally|furthermore|in addition)\b", curr_start):
        return True

    first_word = curr.split(" ", 1)[0].lower()
    if first_word in {"this", "these", "it", "they", "such"}:
        return True

    first_four = " ".join(curr.lower().split()[:4])
    if first_four and first_four in prev.lower():
        return True

    return False


def compute_html_diff(original_html: str, transformed_html: str) -> str:
    diff_lines = difflib.unified_diff(
        (original_html or "").splitlines(),
        (transformed_html or "").splitlines(),
        fromfile="original_html",
        tofile="transformed_html",
        lineterm="",
        n=2,
    )
    txt = "\n".join(diff_lines)
    if len(txt) <= MAX_DIFF_CHARS:
        return txt
    return f"{txt[:MAX_DIFF_CHARS]}\n...<diff_truncated>"


def extract_text_for_global_invariants(html_text: str) -> str:
    soup = BeautifulSoup(html_text or "", "lxml")
    root, _ = get_processing_root(soup, html_text or "")
    return normalize_ws(root.get_text(" ", strip=True))


def openai_output_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    parts: List[str] = []
    for item in (getattr(resp, "output", None) or []):
        for content in (getattr(item, "content", None) or []):
            t = getattr(content, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
                continue
            if isinstance(content, dict):
                dt = content.get("text") or content.get("output_text")
                if isinstance(dt, str) and dt.strip():
                    parts.append(dt.strip())
    return "\n".join(parts).strip()


async def generate_with_retry_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    system_instruction: str,
    user_prompt: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                resp = await asyncio.wait_for(
                    client.responses.create(
                        model=MODEL,
                        input=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_output_tokens=max_output_tokens,
                    ),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            out = sanitize_model_text(openai_output_text(resp))
            if out:
                return out
            raise LLMCallError("Empty model output")
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES:
                break
            sleep_s = RETRY_BASE_SECONDS * (2 ** (attempt - 1)) + random.random()
            await asyncio.sleep(sleep_s)
    raise LLMCallError(str(last_err)) from last_err


def build_pass1_prompt(block: Block) -> Tuple[str, str]:
    system_instruction = (
        "You are an enterprise financial content editor.\n"
        "Task: rewrite ONLY the target paragraph/list item for clarity and professional tone.\n"
        "Hard constraints:\n"
        "- Preserve meaning and facts exactly.\n"
        "- Do NOT change numbers, percentages, currencies, dates, URLs, IDs, product names, legal/disclosure content, or deadlines.\n"
        "- Do NOT merge/split paragraphs or list items.\n"
        "- Return plain text for TARGET only. No bullets, no labels, no markdown."
    )
    user_prompt = f"""
Heading (context only):
{trim_context(block.heading, MAX_CONTEXT_CHARS)}

Previous block (context only):
{trim_context(block.prev_text, MAX_CONTEXT_CHARS)}

Target block (rewrite this only):
{block.original_text}

Next block (context only):
{trim_context(block.next_text, MAX_CONTEXT_CHARS)}
""".strip()
    return system_instruction, user_prompt


def build_pass2_prompt(prev_text: str, target_text: str, next_text: str, heading: str) -> Tuple[str, str]:
    system_instruction = (
        "You are polishing local transitions in enterprise financial text.\n"
        "Task: revise ONLY the target block to improve flow from previous block.\n"
        "Hard constraints:\n"
        "- Preserve all facts and meaning.\n"
        "- Do NOT change numbers, percentages, currencies, dates, URLs, IDs, product names, legal/disclosure content, or deadlines.\n"
        "- Do NOT merge/split/reorder blocks.\n"
        "- Return plain text for TARGET only."
    )
    user_prompt = f"""
Heading (context only):
{trim_context(heading, MAX_CONTEXT_CHARS)}

Previous rewritten block (context):
{trim_context(prev_text, MAX_CONTEXT_CHARS)}

Target rewritten block (revise this only):
{target_text}

Next rewritten block (context only):
{trim_context(next_text, MAX_CONTEXT_CHARS)}
""".strip()
    return system_instruction, user_prompt


async def gather_in_batches(coros: List, batch_size: int) -> List:
    out: List = []
    for i in range(0, len(coros), batch_size):
        out.extend(await asyncio.gather(*coros[i : i + batch_size]))
    return out


async def rewrite_block_pass1(
    block: Block,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> Block:
    reason = freeze_reason_for_block(block)
    if reason:
        block.rewritten_text = block.original_text
        block.status = "skipped"
        block.skip_reason = reason
        return block

    original_inv = extract_invariants(block.original_text)
    system_instruction, user_prompt = build_pass1_prompt(block)

    try:
        candidate = await generate_with_retry_async(client, semaphore, system_instruction, user_prompt)
        candidate = normalize_ws(candidate)
        if not candidate:
            block.rewritten_text = block.original_text
            block.status = "pass1_fallback"
            block.pass1_error = "empty_candidate"
            return block

        candidate_inv = extract_invariants(candidate)
        mismatches = invariant_mismatches(original_inv, candidate_inv)
        if mismatches:
            block.rewritten_text = block.original_text
            block.status = "pass1_invariant_fallback"
            block.invariants_ok_pass1 = False
            block.pass1_error = f"invariant_mismatch:{','.join(sorted(mismatches.keys()))}"
            return block

        block.rewritten_text = candidate
        block.status = "rewritten_pass1"
        return block
    except Exception as e:
        block.rewritten_text = block.original_text
        block.status = "pass1_error_fallback"
        block.pass1_error = str(e)
        return block


async def polish_transition_pass2(
    blocks: List[Block],
    idx: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, str, str, bool]:
    # returns: (block_index, final_text, error_msg, polished)
    curr = blocks[idx]
    prev = blocks[idx - 1]
    nxt = blocks[idx + 1] if idx + 1 < len(blocks) else None

    prev_text = prev.rewritten_text or prev.original_text
    curr_text = curr.rewritten_text or curr.original_text
    next_text = (nxt.rewritten_text if nxt else "") or (nxt.original_text if nxt else "")

    if not transition_needs_polish(prev_text, curr_text):
        return idx, curr_text, "", False

    original_inv = extract_invariants(curr.original_text)
    system_instruction, user_prompt = build_pass2_prompt(prev_text, curr_text, next_text, curr.heading)

    try:
        candidate = await generate_with_retry_async(client, semaphore, system_instruction, user_prompt)
        candidate = normalize_ws(candidate)
        if not candidate:
            return idx, curr_text, "empty_candidate", False

        candidate_inv = extract_invariants(candidate)
        mismatches = invariant_mismatches(original_inv, candidate_inv)
        if mismatches:
            return idx, curr_text, f"invariant_mismatch:{','.join(sorted(mismatches.keys()))}", False

        if candidate == curr_text:
            return idx, curr_text, "", False

        return idx, candidate, "", True
    except Exception as e:
        return idx, curr_text, str(e), False


def apply_block_text(block: Block, new_text: str) -> None:
    # Safe because we only rewrite blocks without inline child tags.
    block.node.clear()
    block.node.append(new_text)


async def transform_single_html_document(
    html_text: str,
    client: AsyncOpenAI,
) -> Tuple[str, Dict]:
    t0 = time.perf_counter()
    original_html = html_text or ""

    soup, blocks, scope_mode = parse_blocks(original_html)

    request_semaphore = asyncio.Semaphore(MAX_REQUEST_CONCURRENCY)

    # Pass 1: parallel block rewrites with immutable original neighbor context.
    pass1_coros = [rewrite_block_pass1(b, client, request_semaphore) for b in blocks]
    blocks = await gather_in_batches(pass1_coros, batch_size=PASS1_BATCH_SIZE)

    # Apply pass-1 rewrites into DOM.
    for b in blocks:
        final_block_text = b.rewritten_text or b.original_text
        apply_block_text(b, final_block_text)

    # Refresh references for pass 2 context.
    # (Block objects still point to same tags; only text changed.)
    transition_idxs: List[int] = []
    for i in range(1, len(blocks)):
        if blocks[i].status.startswith("rewritten_pass1"):
            transition_idxs.append(i)

    if len(transition_idxs) > MAX_PASS2_CALLS:
        transition_idxs = transition_idxs[:MAX_PASS2_CALLS]

    pass2_coros = [
        polish_transition_pass2(blocks, i, client, request_semaphore)
        for i in transition_idxs
    ]
    pass2_results = await gather_in_batches(pass2_coros, batch_size=PASS2_BATCH_SIZE)

    for idx, new_text, err, polished in pass2_results:
        block = blocks[idx]
        if polished:
            block.rewritten_text = new_text
            block.transition_polished = True
            block.status = "rewritten_pass1_pass2"
            apply_block_text(block, new_text)
        elif err:
            block.pass2_error = err
            if err.startswith("invariant_mismatch"):
                block.invariants_ok_pass2 = False

    candidate_html = serialize_html(soup, scope_mode)

    # Global invariants: hard safety gate.
    original_scope_text = extract_text_for_global_invariants(original_html)
    candidate_scope_text = extract_text_for_global_invariants(candidate_html)
    original_inv = extract_invariants(original_scope_text)
    candidate_inv = extract_invariants(candidate_scope_text)
    global_mismatch = invariant_mismatches(original_inv, candidate_inv)

    global_invariants_ok = len(global_mismatch) == 0
    if global_invariants_ok:
        final_html = candidate_html
        global_status = "ok"
    else:
        final_html = original_html
        global_status = "global_invariant_fallback"

    t1 = time.perf_counter()

    skip_counter = Counter(b.skip_reason for b in blocks if b.skip_reason)
    status_counter = Counter(b.status for b in blocks)
    pass1_rewritten = sum(1 for b in blocks if b.status in {"rewritten_pass1", "rewritten_pass1_pass2"})
    pass2_polished = sum(1 for b in blocks if b.transition_polished)

    audit = {
        "status": global_status,
        "scope_mode": scope_mode,
        "elapsed_sec": round(t1 - t0, 4),
        "total_blocks": len(blocks),
        "pass1_rewritten_blocks": pass1_rewritten,
        "pass2_polished_blocks": pass2_polished,
        "status_counts": dict(sorted(status_counter.items())),
        "skip_reason_counts": dict(sorted(skip_counter.items())),
        "global_invariants_ok": global_invariants_ok,
        "global_invariant_mismatch": global_mismatch,
        "html_diff": compute_html_diff(original_html, final_html),
        "blocks": [
            {
                "block_id": b.block_id,
                "tag": b.tag_name,
                "status": b.status,
                "skip_reason": b.skip_reason,
                "pass1_error": b.pass1_error,
                "pass2_error": b.pass2_error,
                "transition_polished": b.transition_polished,
                "invariants_ok_pass1": b.invariants_ok_pass1,
                "invariants_ok_pass2": b.invariants_ok_pass2,
                "heading": b.heading,
                "original_text": b.original_text,
                "rewritten_text": b.rewritten_text or b.original_text,
            }
            for b in blocks
        ],
    }

    return final_html, audit


# ---------------------------
# 3) I/O helpers
# ---------------------------
def read_html() -> str:
    """
    Replace this with your API call if needed.
    For quick local usage, set INPUT_HTML_PATH env var to a source HTML file.
    """
    input_path = os.getenv("INPUT_HTML_PATH", "").strip()
    if input_path:
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(
        "read_html() is not implemented for API retrieval in this script. "
        "Either implement read_html() or set INPUT_HTML_PATH."
    )


def write_text(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


def write_json(path: str, payload: Dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def main_async() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("Set OPENAI_API_KEY first.")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    html_text = read_html()

    transformed_html, audit = await transform_single_html_document(html_text, client)

    write_text(OUTPUT_PATH, transformed_html)
    audit_path = f"{OUTPUT_PATH}.audit.json"
    write_json(audit_path, audit)

    print(f"Saved transformed HTML to: {OUTPUT_PATH}")
    print(f"Saved audit JSON to     : {audit_path}")
    print(f"Status                  : {audit['status']}")
    print(f"Total blocks            : {audit['total_blocks']}")
    print(f"Pass1 rewritten blocks  : {audit['pass1_rewritten_blocks']}")
    print(f"Pass2 polished blocks   : {audit['pass2_polished_blocks']}")


if __name__ == "__main__":
    run_async(main_async())
