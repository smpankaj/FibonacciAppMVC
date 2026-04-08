from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from bs4 import BeautifulSoup
from openai import OpenAI


logger = logging.getLogger(__name__)


# ============================================================
# ENUMS / CONSTANTS
# ============================================================

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClaimComparisonResult(str, Enum):
    PRESERVED = "preserved"
    PRESERVED_BUT_REPHRASED = "preserved_but_rephrased"
    WEAKENED = "weakened"
    STRENGTHENED = "strengthened"
    CONTRADICTED = "contradicted"
    MISSING = "missing"
    UNCERTAIN = "uncertain"


class Decision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


MODALITY_RANK = {
    None: 0,
    "none": 0,
    "optional": 1,
    "may": 2,
    "should": 3,
    "required": 4,
    "must": 5,
    "prohibited": 5,
}


DEFAULT_NUMERIC_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "currency_amount",
        re.compile(
            r"""
            (?<!\w)
            [€$£¥]
            \s?
            [+-]?
            (?:
                \d{1,3}(?:,\d{3})+|\d+
            )
            (?:\.\d+)?
            (?!\w)
            """,
            re.VERBOSE,
        ),
    ),
    (
        "percentage",
        re.compile(
            r"""
            (?<![\w/.-])
            [+-]?
            (?:
                (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
                |
                \.\d+
            )
            %
            (?![\w.-])
            """,
            re.VERBOSE,
        ),
    ),
    (
        "range",
        re.compile(
            r"""
            (?<![\w/.-])
            [+-]?
            (?:
                (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
                |
                \.\d+
            )
            \s?(?:to|-|–|—)\s?
            [+-]?
            (?:
                (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
                |
                \.\d+
            )
            %?
            (?![\w.-])
            """,
            re.VERBOSE,
        ),
    ),
    (
        "duration_or_limit",
        re.compile(
            r"""
            (?<![\w/.-])
            [+-]?
            (?:
                (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
                |
                \.\d+
            )
            \s+
            (?:
                day|days|business\ day|business\ days|
                week|weeks|month|months|year|years|
                hour|hours|minute|minutes
            )
            (?![\w.-])
            """,
            re.VERBOSE | re.IGNORECASE,
        ),
    ),
    (
        "plain_number",
        re.compile(
            r"""
            (?<![\w/.-])
            [+-]?
            (?:
                (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
                |
                \.\d+
            )
            (?![\w.-])
            """,
            re.VERBOSE,
        ),
    ),
]


# ============================================================
# JSON SCHEMAS
# ============================================================

CLAIM_EXTRACTION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "claim_extraction_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim_type": {"type": ["string", "null"]},
                        "subject": {"type": ["string", "null"]},
                        "actor": {"type": ["string", "null"]},
                        "action": {"type": ["string", "null"]},
                        "target": {"type": ["string", "null"]},
                        "modality": {"type": ["string", "null"]},
                        "negated": {"type": ["boolean", "null"]},
                        "condition": {"type": ["string", "null"]},
                        "timing": {"type": ["string", "null"]},
                        "approval_required": {"type": ["boolean", "null"]},
                        "approver": {"type": ["string", "null"]},
                        "value": {"type": ["string", "null"]},
                        "unit": {"type": ["string", "null"]},
                        "exclusive": {"type": ["boolean", "null"]},
                        "severity": {
                            "type": ["string", "null"],
                            "enum": ["critical", "high", "medium", "low", None],
                        },
                        "source_text": {"type": ["string", "null"]},
                    },
                    "required": [
                        "claim_type",
                        "subject",
                        "actor",
                        "action",
                        "target",
                        "modality",
                        "negated",
                        "condition",
                        "timing",
                        "approval_required",
                        "approver",
                        "value",
                        "unit",
                        "exclusive",
                        "severity",
                        "source_text",
                    ],
                },
            }
        },
        "required": ["claims"],
    },
}

CLAIM_COMPARISON_JSON_SCHEMA: Dict[str, Any] = {
    "name": "claim_comparison_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "result": {
                "type": "string",
                "enum": [
                    "preserved",
                    "preserved_but_rephrased",
                    "weakened",
                    "strengthened",
                    "contradicted",
                    "missing",
                    "uncertain",
                ],
            },
            "best_match_claim_id": {"type": ["string", "null"]},
            "explanation": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low"],
            },
        },
        "required": [
            "result",
            "best_match_claim_id",
            "explanation",
            "severity",
        ],
    },
}

FINAL_JUDGE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "final_judge_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_equivalent": {"type": "boolean"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                        },
                        "original": {"type": "string"},
                        "transformed": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "type",
                        "severity",
                        "original",
                        "transformed",
                        "explanation",
                    ],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["semantic_equivalent", "issues", "summary"],
    },
}


# ============================================================
# CONFIG
# ============================================================

@dataclass(slots=True)
class ValidatorConfig:
    max_chars_per_chunk: int = 3500
    overlap_paragraphs: int = 1
    candidate_top_k: int = 5
    use_llm_for_ambiguous_claims: bool = True
    use_final_llm_judge: bool = True
    require_same_protected_term_count: bool = True
    case_sensitive_protected_terms: bool = False
    min_candidate_score: float = 0.5

    fail_on_numeric_changes: bool = True
    fail_on_protected_term_changes: bool = True
    fail_on_missing_critical_claim: bool = True
    fail_on_contradicted_claim: bool = True
    review_on_uncertain_claim: bool = True
    review_on_unsupported_new_claims: bool = True


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(slots=True)
class NumericItem:
    raw: str
    normalized: str
    kind: str
    start: int
    end: int
    context: str


@dataclass(slots=True)
class Claim:
    claim_id: str
    claim_type: str
    subject: Optional[str] = None
    actor: Optional[str] = None
    action: Optional[str] = None
    target: Optional[str] = None
    modality: Optional[str] = None
    negated: Optional[bool] = None
    condition: Optional[str] = None
    timing: Optional[str] = None
    approval_required: Optional[bool] = None
    approver: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    exclusive: Optional[bool] = None
    severity: str = Severity.MEDIUM.value
    source_text: str = ""
    source_chunk_id: str = ""


@dataclass(slots=True)
class ClaimMatchResult:
    original_claim_id: str
    result: str
    matched_transformed_claim_id: Optional[str]
    issues: List[Dict[str, Any]]
    method: str
    original_claim: Dict[str, Any]


@dataclass(slots=True)
class NumericCheckReport:
    passed: bool
    missing_items: Dict[str, int]
    added_items: Dict[str, int]
    original_counts: Dict[str, int]
    transformed_counts: Dict[str, int]
    original_matches: List[Dict[str, Any]]
    transformed_matches: List[Dict[str, Any]]


@dataclass(slots=True)
class ProtectedTermsReport:
    passed: bool
    missing_terms: Dict[str, int]
    added_terms: Dict[str, int]
    original_counts: Dict[str, int]
    transformed_counts: Dict[str, int]


@dataclass(slots=True)
class ClaimComparisonReport:
    summary: Dict[str, int]
    claim_results: List[ClaimMatchResult]
    unsupported_new_claims: List[Dict[str, Any]]


@dataclass(slots=True)
class FinalJudgeReport:
    semantic_equivalent: bool
    issues: List[Dict[str, Any]]
    summary: str


@dataclass(slots=True)
class DocumentEvaluationReport:
    decision: str
    reasons: List[str]
    original_claim_count: int
    transformed_claim_count: int
    numeric_check: NumericCheckReport
    protected_terms_check: ProtectedTermsReport
    claim_comparison: ClaimComparisonReport
    final_judge: Optional[FinalJudgeReport] = None


# ============================================================
# LLM CLIENT
# ============================================================

class LLMClient(Protocol):
    def extract_claims(self, prompt: str) -> Dict[str, Any]:
        ...
    def compare_claims(self, prompt: str) -> Dict[str, Any]:
        ...
    def judge_semantics(self, prompt: str) -> Dict[str, Any]:
        ...


class OpenAIChatLLMClient:
    """
    Uses chat.completions.create with json_schema structured outputs.
    """

    def __init__(
        self,
        *,
        api_key: str = "123test",
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_completion_tokens: int = 4000,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def _complete_with_schema(self, prompt: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "developer",
                    "content": (
                        "You are a precise validation assistant. "
                        "Return only JSON that matches the provided schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned empty content.")
        return json.loads(content)

    def extract_claims(self, prompt: str) -> Dict[str, Any]:
        return self._complete_with_schema(prompt, CLAIM_EXTRACTION_JSON_SCHEMA)

    def compare_claims(self, prompt: str) -> Dict[str, Any]:
        return self._complete_with_schema(prompt, CLAIM_COMPARISON_JSON_SCHEMA)

    def judge_semantics(self, prompt: str) -> Dict[str, Any]:
        return self._complete_with_schema(prompt, FINAL_JUDGE_JSON_SCHEMA)


# ============================================================
# HTML / TEXT
# ============================================================

def html_to_visible_text(html_doc: str) -> str:
    if not isinstance(html_doc, str):
        raise TypeError("html_doc must be a string")

    soup = BeautifulSoup(html_doc, "html.parser")

    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = html.unescape(text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_text_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def chunk_paragraphs(
    paragraphs: Sequence[str],
    *,
    max_chars: int,
    overlap_paragraphs: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    start = 0

    while start < len(paragraphs):
        current: List[str] = []
        total = 0
        end = start

        while end < len(paragraphs):
            p = paragraphs[end]
            projected = total + len(p) + (1 if current else 0)
            if current and projected > max_chars:
                break
            current.append(p)
            total = projected
            end += 1

        chunks.append(
            {
                "chunk_id": f"chunk_{len(chunks) + 1}",
                "paragraph_start": start,
                "paragraph_end": end - 1,
                "text": "\n".join(current).strip(),
            }
        )

        if end >= len(paragraphs):
            break

        start = max(start + 1, end - overlap_paragraphs)

    return chunks


# ============================================================
# NUMERIC CHECK
# ============================================================

def _normalize_numeric_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"\s+", " ", token)
    token = token.replace("–", "-").replace("—", "-")
    token = re.sub(r"([€$£¥])\s+", r"\1", token)

    def _remove_commas(match: re.Match[str]) -> str:
        return match.group(0).replace(",", "")

    token = re.sub(r"\d[\d,]*\.?\d*", _remove_commas, token)
    return token.lower()


def _context_window(text: str, start: int, end: int, width: int = 35) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right]


def extract_numeric_items_from_html(html_doc: str) -> Dict[str, Any]:
    text = html_to_visible_text(html_doc)
    consumed = [False] * len(text)
    matches: List[NumericItem] = []

    for kind, pattern in DEFAULT_NUMERIC_PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(consumed[i] for i in range(start, end)):
                continue

            raw = m.group(0)
            matches.append(
                NumericItem(
                    raw=raw,
                    normalized=_normalize_numeric_token(raw),
                    kind=kind,
                    start=start,
                    end=end,
                    context=_context_window(text, start, end),
                )
            )
            for i in range(start, end):
                consumed[i] = True

    counts = Counter(item.normalized for item in matches)

    return {
        "visible_text": text,
        "matches": [asdict(m) for m in matches],
        "counts": dict(counts),
    }


def compare_numeric_integrity(original_html: str, transformed_html: str) -> NumericCheckReport:
    original = extract_numeric_items_from_html(original_html)
    transformed = extract_numeric_items_from_html(transformed_html)

    orig_counter = Counter(original["counts"])
    new_counter = Counter(transformed["counts"])

    missing: Dict[str, int] = {}
    added: Dict[str, int] = {}

    for token, count in orig_counter.items():
        diff = count - new_counter.get(token, 0)
        if diff > 0:
            missing[token] = diff

    for token, count in new_counter.items():
        diff = count - orig_counter.get(token, 0)
        if diff > 0:
            added[token] = diff

    return NumericCheckReport(
        passed=not missing and not added,
        missing_items=missing,
        added_items=added,
        original_counts=dict(orig_counter),
        transformed_counts=dict(new_counter),
        original_matches=original["matches"],
        transformed_matches=transformed["matches"],
    )


# ============================================================
# PROTECTED TERMS CHECK
# ============================================================

def _build_term_pattern(term: str, *, case_sensitive: bool) -> re.Pattern[str]:
    escaped_parts = [re.escape(part) for part in re.split(r"\s+", term.strip()) if part]
    joined = r"\s+".join(escaped_parts)
    pattern = rf"(?<!\w){joined}(?!\w)"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern, flags)


def _extract_protected_term_counts(
    html_doc: str,
    protected_terms: Iterable[str],
    *,
    case_sensitive: bool,
) -> Dict[str, int]:
    text = html_to_visible_text(html_doc)
    term_counts: Counter[str] = Counter()

    for term in protected_terms:
        clean = term.strip()
        if not clean:
            continue
        pattern = _build_term_pattern(clean, case_sensitive=case_sensitive)
        matches = list(pattern.finditer(text))
        if matches:
            term_counts[clean] += len(matches)

    return dict(term_counts)


def compare_protected_terms(
    original_html: str,
    transformed_html: str,
    protected_terms: Iterable[str],
    *,
    case_sensitive: bool,
    require_same_count: bool,
) -> ProtectedTermsReport:
    original_counts = Counter(
        _extract_protected_term_counts(original_html, protected_terms, case_sensitive=case_sensitive)
    )
    transformed_counts = Counter(
        _extract_protected_term_counts(transformed_html, protected_terms, case_sensitive=case_sensitive)
    )

    all_terms = {t.strip() for t in protected_terms if t.strip()}
    missing_terms: Dict[str, int] = {}
    added_terms: Dict[str, int] = {}

    for term in all_terms:
        o = original_counts.get(term, 0)
        t = transformed_counts.get(term, 0)

        if require_same_count:
            if o > t:
                missing_terms[term] = o - t
            if t > o:
                added_terms[term] = t - o
        else:
            if o > 0 and t == 0:
                missing_terms[term] = o

    return ProtectedTermsReport(
        passed=not missing_terms and not added_terms,
        missing_terms=missing_terms,
        added_terms=added_terms,
        original_counts=dict(original_counts),
        transformed_counts=dict(transformed_counts),
    )


# ============================================================
# PROMPTS
# ============================================================

def build_claim_extraction_prompt(chunk_text: str) -> str:
    return f"""
You are extracting material policy and procedural claims from a banking/procedural document.

Extract ONLY material claims that matter for correctness, compliance, customer handling, or procedure.

Include claims such as:
- instructions
- prohibitions
- eligibility rules
- deadlines / timing constraints
- fee rules
- rate / percentage rules
- approval requirements
- disclosure requirements
- sequence requirements
- exceptions / conditions
- customer-impacting factual rules

Do NOT extract:
- style guidance
- rhetorical filler
- duplicate claims
- broad summaries not grounded in explicit text

Chunk:
\"\"\"
{chunk_text}
\"\"\"
""".strip()


def build_claim_comparison_prompt(
    original_claim: Dict[str, Any],
    candidate_claims: List[Dict[str, Any]],
) -> str:
    return f"""
You are validating whether a transformed document preserved the meaning of a material claim from the original document.

Guidance:
- preserved: same meaning
- preserved_but_rephrased: same meaning, different wording
- weakened: transformed is less strict or lost an important limitation
- strengthened: transformed is stricter than original
- contradicted: transformed says the opposite
- missing: original claim is absent
- uncertain: insufficient confidence

ORIGINAL CLAIM:
{json.dumps(original_claim, ensure_ascii=False, indent=2)}

CANDIDATE TRANSFORMED CLAIMS:
{json.dumps(candidate_claims, ensure_ascii=False, indent=2)}
""".strip()


def build_document_judge_prompt(
    original_excerpt: str,
    transformed_excerpt: str,
    structured_findings: Dict[str, Any],
) -> str:
    return f"""
You are a validator for a rewritten financial/procedural document.

Determine whether the transformed text preserves the business meaning, policy meaning, and procedural meaning of the original text.

Check especially:
- obligation level
- negation and prohibitions
- eligibility and exclusivity
- approvals required
- timing and sequence
- conditions and exceptions
- customer-impacting meaning
- compliance/legal meaning

Use the structured findings below, but do not rely only on them.

ORIGINAL TEXT:
\"\"\"
{original_excerpt}
\"\"\"

TRANSFORMED TEXT:
\"\"\"
{transformed_excerpt}
\"\"\"

STRUCTURED FINDINGS:
{json.dumps(structured_findings, ensure_ascii=False, indent=2)}
""".strip()


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def _make_claim_id(source_chunk_id: str, idx: int, source_text: str) -> str:
    digest = hashlib.sha1(f"{source_chunk_id}|{idx}|{source_text}".encode("utf-8")).hexdigest()[:12]
    return f"{source_chunk_id}_claim_{digest}"


def _normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _none_if_empty(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _normalize_modality(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip().lower()


def _normalize_severity(value: Any) -> str:
    lowered = _normalize_text_value(value)
    if lowered in {
        Severity.CRITICAL.value,
        Severity.HIGH.value,
        Severity.MEDIUM.value,
        Severity.LOW.value,
    }:
        return lowered
    return Severity.MEDIUM.value


def parse_claims_from_llm_response(
    llm_response: Dict[str, Any],
    source_chunk_id: str,
) -> List[Claim]:
    raw_claims = llm_response.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ValueError("LLM response 'claims' must be a list")

    parsed: List[Claim] = []
    for idx, raw in enumerate(raw_claims, start=1):
        if not isinstance(raw, dict):
            continue

        claim = Claim(
            claim_id=_make_claim_id(source_chunk_id, idx, str(raw.get("source_text", ""))),
            claim_type=str(raw.get("claim_type", "other")),
            subject=_none_if_empty(raw.get("subject")),
            actor=_none_if_empty(raw.get("actor")),
            action=_none_if_empty(raw.get("action")),
            target=_none_if_empty(raw.get("target")),
            modality=_normalize_modality(_none_if_empty(raw.get("modality"))),
            negated=_coerce_optional_bool(raw.get("negated")),
            condition=_none_if_empty(raw.get("condition")),
            timing=_none_if_empty(raw.get("timing")),
            approval_required=_coerce_optional_bool(raw.get("approval_required")),
            approver=_none_if_empty(raw.get("approver")),
            value=_none_if_empty(raw.get("value")),
            unit=_none_if_empty(raw.get("unit")),
            exclusive=_coerce_optional_bool(raw.get("exclusive")),
            severity=_normalize_severity(raw.get("severity")),
            source_text=str(raw.get("source_text", "")),
            source_chunk_id=source_chunk_id,
        )
        parsed.append(claim)

    return parsed


def claims_are_near_duplicates(a: Claim, b: Claim) -> bool:
    core_a = (
        _normalize_text_value(a.claim_type),
        _normalize_text_value(a.subject),
        _normalize_text_value(a.actor),
        _normalize_text_value(a.action),
        _normalize_text_value(a.target),
        _normalize_text_value(a.modality),
        _normalize_text_value(a.condition),
        _normalize_text_value(a.timing),
        _normalize_text_value(a.value),
        _normalize_text_value(a.unit),
        str(a.negated),
        str(a.approval_required),
        str(a.exclusive),
    )
    core_b = (
        _normalize_text_value(b.claim_type),
        _normalize_text_value(b.subject),
        _normalize_text_value(b.actor),
        _normalize_text_value(b.action),
        _normalize_text_value(b.target),
        _normalize_text_value(b.modality),
        _normalize_text_value(b.condition),
        _normalize_text_value(b.timing),
        _normalize_text_value(b.value),
        _normalize_text_value(b.unit),
        str(b.negated),
        str(b.approval_required),
        str(b.exclusive),
    )
    return core_a == core_b


def deduplicate_claims(claims: Sequence[Claim]) -> List[Claim]:
    deduped: List[Claim] = []
    for claim in claims:
        if not any(claims_are_near_duplicates(claim, existing) for existing in deduped):
            deduped.append(claim)
    return deduped


def extract_claims_from_chunks(
    chunks: Sequence[Dict[str, Any]],
    llm_client: LLMClient,
) -> List[Claim]:
    claims: List[Claim] = []

    for chunk in chunks:
        prompt = build_claim_extraction_prompt(chunk["text"])
        response = llm_client.extract_claims(prompt)
        claims.extend(parse_claims_from_llm_response(response, chunk["chunk_id"]))

    return claims


def build_claim_set_from_html(
    html_doc: str,
    llm_client: LLMClient,
    config: ValidatorConfig,
) -> Dict[str, Any]:
    text = html_to_visible_text(html_doc)
    paragraphs = split_text_into_paragraphs(text)
    chunks = chunk_paragraphs(
        paragraphs,
        max_chars=config.max_chars_per_chunk,
        overlap_paragraphs=config.overlap_paragraphs,
    )

    raw_claims = extract_claims_from_chunks(chunks, llm_client)
    deduped_claims = deduplicate_claims(raw_claims)

    return {
        "text": text,
        "paragraphs": paragraphs,
        "chunks": chunks,
        "raw_claims": raw_claims,
        "claims": deduped_claims,
    }


# ============================================================
# CLAIM MATCHING / COMPARISON
# ============================================================

def similarity_score(a: Optional[str], b: Optional[str]) -> float:
    a_norm = _normalize_text_value(a)
    b_norm = _normalize_text_value(b)

    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def candidate_match_score(original: Claim, transformed: Claim) -> float:
    score = 0.0

    if _normalize_text_value(original.claim_type) == _normalize_text_value(transformed.claim_type):
        score += 2.0

    score += 2.0 * similarity_score(original.action, transformed.action)
    score += 1.5 * similarity_score(original.subject, transformed.subject)
    score += 1.0 * similarity_score(original.actor, transformed.actor)
    score += 1.0 * similarity_score(original.target, transformed.target)

    if _normalize_text_value(original.value) and _normalize_text_value(original.value) == _normalize_text_value(transformed.value):
        score += 1.5

    if _normalize_text_value(original.timing) and _normalize_text_value(original.timing) == _normalize_text_value(transformed.timing):
        score += 1.0

    if original.approval_required is not None and original.approval_required == transformed.approval_required:
        score += 0.5

    if original.exclusive is not None and original.exclusive == transformed.exclusive:
        score += 0.5

    return score


def shortlist_candidate_claims(
    original_claim: Claim,
    transformed_claims: Sequence[Claim],
    *,
    top_k: int,
    min_score: float,
) -> List[Claim]:
    scored = [(candidate_match_score(original_claim, c), c) for c in transformed_claims]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:top_k] if score >= min_score]


def _severity_rank(value: str) -> int:
    order = {
        Severity.LOW.value: 1,
        Severity.MEDIUM.value: 2,
        Severity.HIGH.value: 3,
        Severity.CRITICAL.value: 4,
    }
    return order.get(value, 2)


def _severity_for_claim_issue(issue_type: str, claim: Claim) -> str:
    if issue_type in {"negation_change", "approval_change", "value_change"}:
        return Severity.CRITICAL.value
    if issue_type in {"modality_change", "condition_change", "timing_change", "exclusivity_change"}:
        return max(claim.severity, Severity.HIGH.value, key=_severity_rank)
    return claim.severity


def compare_claim_fields(original: Claim, transformed: Claim) -> Tuple[str, List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []

    if _normalize_text_value(original.modality) != _normalize_text_value(transformed.modality):
        issues.append(
            {
                "type": "modality_change",
                "severity": _severity_for_claim_issue("modality_change", original),
                "original": original.modality,
                "transformed": transformed.modality,
            }
        )

    if original.negated != transformed.negated:
        issues.append(
            {
                "type": "negation_change",
                "severity": _severity_for_claim_issue("negation_change", original),
                "original": original.negated,
                "transformed": transformed.negated,
            }
        )

    if original.approval_required != transformed.approval_required:
        issues.append(
            {
                "type": "approval_change",
                "severity": _severity_for_claim_issue("approval_change", original),
                "original": original.approval_required,
                "transformed": transformed.approval_required,
            }
        )

    if _normalize_text_value(original.condition) != _normalize_text_value(transformed.condition):
        if original.condition or transformed.condition:
            issues.append(
                {
                    "type": "condition_change",
                    "severity": _severity_for_claim_issue("condition_change", original),
                    "original": original.condition,
                    "transformed": transformed.condition,
                }
            )

    if _normalize_text_value(original.timing) != _normalize_text_value(transformed.timing):
        if original.timing or transformed.timing:
            issues.append(
                {
                    "type": "timing_change",
                    "severity": _severity_for_claim_issue("timing_change", original),
                    "original": original.timing,
                    "transformed": transformed.timing,
                }
            )

    if original.exclusive != transformed.exclusive:
        if original.exclusive is not None or transformed.exclusive is not None:
            issues.append(
                {
                    "type": "exclusivity_change",
                    "severity": _severity_for_claim_issue("exclusivity_change", original),
                    "original": original.exclusive,
                    "transformed": transformed.exclusive,
                }
            )

    if _normalize_text_value(original.value) != _normalize_text_value(transformed.value):
        if original.value or transformed.value:
            issues.append(
                {
                    "type": "value_change",
                    "severity": _severity_for_claim_issue("value_change", original),
                    "original": original.value,
                    "transformed": transformed.value,
                }
            )

    if not issues:
        if _normalize_text_value(original.source_text) != _normalize_text_value(transformed.source_text):
            return ClaimComparisonResult.PRESERVED_BUT_REPHRASED.value, []
        return ClaimComparisonResult.PRESERVED.value, []

    types = {issue["type"] for issue in issues}

    if "negation_change" in types:
        return ClaimComparisonResult.CONTRADICTED.value, issues

    if "approval_change" in types or "exclusivity_change" in types:
        return ClaimComparisonResult.WEAKENED.value, issues

    if "modality_change" in types:
        orig_rank = MODALITY_RANK.get(_normalize_text_value(original.modality), 0)
        new_rank = MODALITY_RANK.get(_normalize_text_value(transformed.modality), 0)
        if new_rank < orig_rank:
            return ClaimComparisonResult.WEAKENED.value, issues
        if new_rank > orig_rank:
            return ClaimComparisonResult.STRENGTHENED.value, issues

    return ClaimComparisonResult.UNCERTAIN.value, issues


def compare_claim_with_llm(
    original_claim: Claim,
    candidate_claims: Sequence[Claim],
    llm_client: LLMClient,
) -> Dict[str, Any]:
    prompt = build_claim_comparison_prompt(
        original_claim=asdict(original_claim),
        candidate_claims=[asdict(c) for c in candidate_claims],
    )
    return llm_client.compare_claims(prompt)


def compare_claim_sets(
    original_claims: Sequence[Claim],
    transformed_claims: Sequence[Claim],
    *,
    config: ValidatorConfig,
    llm_client: Optional[LLMClient] = None,
) -> ClaimComparisonReport:
    results: List[ClaimMatchResult] = []
    matched_transformed_ids: set[str] = set()

    for original_claim in original_claims:
        candidates = shortlist_candidate_claims(
            original_claim,
            transformed_claims,
            top_k=config.candidate_top_k,
            min_score=config.min_candidate_score,
        )

        if not candidates:
            results.append(
                ClaimMatchResult(
                    original_claim_id=original_claim.claim_id,
                    result=ClaimComparisonResult.MISSING.value,
                    matched_transformed_claim_id=None,
                    issues=[],
                    method="deterministic",
                    original_claim=asdict(original_claim),
                )
            )
            continue

        best_candidate = candidates[0]
        result, issues = compare_claim_fields(original_claim, best_candidate)
        method = "deterministic"
        best_match_id = best_candidate.claim_id

        if (
            result in {
                ClaimComparisonResult.UNCERTAIN.value,
                ClaimComparisonResult.WEAKENED.value,
                ClaimComparisonResult.STRENGTHENED.value,
            }
            and config.use_llm_for_ambiguous_claims
            and llm_client is not None
        ):
            llm_result = compare_claim_with_llm(original_claim, candidates, llm_client)
            result = llm_result.get("result", result)
            best_match_id = llm_result.get("best_match_claim_id", best_match_id)
            issues = [
                {
                    "type": "llm_assessment",
                    "severity": llm_result.get("severity", "medium"),
                    "explanation": llm_result.get("explanation", ""),
                },
                *issues,
            ]
            method = "llm_assisted"

        if best_match_id:
            matched_transformed_ids.add(best_match_id)

        results.append(
            ClaimMatchResult(
                original_claim_id=original_claim.claim_id,
                result=result,
                matched_transformed_claim_id=best_match_id,
                issues=issues,
                method=method,
                original_claim=asdict(original_claim),
            )
        )

    summary_counter = Counter(r.result for r in results)
    unsupported_new_claims = [
        asdict(c) for c in transformed_claims if c.claim_id not in matched_transformed_ids
    ]

    return ClaimComparisonReport(
        summary=dict(summary_counter),
        claim_results=results,
        unsupported_new_claims=unsupported_new_claims,
    )


# ============================================================
# FINAL JUDGE
# ============================================================

def judge_document_with_llm(
    original_text: str,
    transformed_text: str,
    structured_findings: Dict[str, Any],
    llm_client: LLMClient,
) -> FinalJudgeReport:
    prompt = build_document_judge_prompt(
        original_excerpt=original_text[:12000],
        transformed_excerpt=transformed_text[:12000],
        structured_findings=structured_findings,
    )
    result = llm_client.judge_semantics(prompt)

    return FinalJudgeReport(
        semantic_equivalent=bool(result.get("semantic_equivalent", False)),
        issues=result.get("issues", []) if isinstance(result.get("issues"), list) else [],
        summary=str(result.get("summary", "")),
    )


# ============================================================
# DECISION ENGINE
# ============================================================

def decide_evaluation(
    *,
    numeric_check: NumericCheckReport,
    protected_terms_check: ProtectedTermsReport,
    claim_report: ClaimComparisonReport,
    final_judge: Optional[FinalJudgeReport],
    config: ValidatorConfig,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if config.fail_on_numeric_changes and not numeric_check.passed:
        reasons.append("Numeric integrity check failed.")
        return Decision.FAIL.value, reasons

    if config.fail_on_protected_term_changes and not protected_terms_check.passed:
        reasons.append("Protected terminology/entity check failed.")
        return Decision.FAIL.value, reasons

    for result in claim_report.claim_results:
        claim_severity = result.original_claim.get("severity", Severity.MEDIUM.value)

        if (
            config.fail_on_missing_critical_claim
            and result.result == ClaimComparisonResult.MISSING.value
            and claim_severity == Severity.CRITICAL.value
        ):
            reasons.append(f"Critical claim missing: {result.original_claim_id}")
            return Decision.FAIL.value, reasons

        if config.fail_on_contradicted_claim and result.result == ClaimComparisonResult.CONTRADICTED.value:
            reasons.append(f"Contradicted claim detected: {result.original_claim_id}")
            return Decision.FAIL.value, reasons

        if (
            result.result == ClaimComparisonResult.WEAKENED.value
            and claim_severity in {Severity.CRITICAL.value, Severity.HIGH.value}
        ):
            reasons.append(f"High-risk claim weakened: {result.original_claim_id}")
            return Decision.FAIL.value, reasons

    if config.review_on_uncertain_claim:
        uncertain_count = claim_report.summary.get(ClaimComparisonResult.UNCERTAIN.value, 0)
        if uncertain_count > 0:
            reasons.append(f"{uncertain_count} claim(s) remain uncertain.")
            return Decision.REVIEW.value, reasons

    if config.review_on_unsupported_new_claims and claim_report.unsupported_new_claims:
        reasons.append("Unsupported new claims were introduced.")
        return Decision.REVIEW.value, reasons

    if final_judge is not None and not final_judge.semantic_equivalent:
        reasons.append("Final semantic judge found meaning drift.")
        return Decision.REVIEW.value, reasons

    reasons.append("All checks passed.")
    return Decision.PASS.value, reasons


# ============================================================
# ENTRY POINT
# ============================================================

def evaluate_document(
    *,
    original_html: str,
    transformed_html: str,
    llm_client: LLMClient,
    protected_terms: Optional[Iterable[str]] = None,
    config: Optional[ValidatorConfig] = None,
) -> DocumentEvaluationReport:
    cfg = config or ValidatorConfig()
    terms = list(protected_terms or [])

    numeric_report = compare_numeric_integrity(original_html, transformed_html)

    protected_terms_report = compare_protected_terms(
        original_html,
        transformed_html,
        terms,
        case_sensitive=cfg.case_sensitive_protected_terms,
        require_same_count=cfg.require_same_protected_term_count,
    )

    original_bundle = build_claim_set_from_html(original_html, llm_client, cfg)
    transformed_bundle = build_claim_set_from_html(transformed_html, llm_client, cfg)

    claim_report = compare_claim_sets(
        original_bundle["claims"],
        transformed_bundle["claims"],
        config=cfg,
        llm_client=llm_client,
    )

    final_judge: Optional[FinalJudgeReport] = None
    if cfg.use_final_llm_judge:
        structured_findings = {
            "numeric_check_passed": numeric_report.passed,
            "protected_terms_passed": protected_terms_report.passed,
            "claim_summary": claim_report.summary,
            "claim_results_sample": [asdict(x) for x in claim_report.claim_results[:50]],
            "unsupported_new_claims_sample": claim_report.unsupported_new_claims[:20],
        }
        final_judge = judge_document_with_llm(
            original_bundle["text"],
            transformed_bundle["text"],
            structured_findings,
            llm_client,
        )

    decision, reasons = decide_evaluation(
        numeric_check=numeric_report,
        protected_terms_check=protected_terms_report,
        claim_report=claim_report,
        final_judge=final_judge,
        config=cfg,
    )

    return DocumentEvaluationReport(
        decision=decision,
        reasons=reasons,
        original_claim_count=len(original_bundle["claims"]),
        transformed_claim_count=len(transformed_bundle["claims"]),
        numeric_check=numeric_report,
        protected_terms_check=protected_terms_report,
        claim_comparison=claim_report,
        final_judge=final_judge,
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    original_html = """
    <html>
      <body>
        <h1>Fee Waiver Procedure</h1>
        <p>Agents must not waive the monthly maintenance fee without manager approval.</p>
        <p>Only existing customers may request expedited review.</p>
        <p>The promotional interest rate is 3.5% for 12 months.</p>
      </body>
    </html>
    """

    transformed_html = """
    <html>
      <body>
        <h1>Fee Waiver Procedure</h1>
        <p>Agents may waive the monthly maintenance fee.</p>
        <p>Existing customers may request expedited review.</p>
        <p>The promotional interest rate is 3.5% for 12 months.</p>
      </body>
    </html>
    """

    llm_client = OpenAIChatLLMClient(
        api_key="123test",
        model="gpt-4.1",
        temperature=0.0,
        max_completion_tokens=3000,
    )

    report = evaluate_document(
        original_html=original_html,
        transformed_html=transformed_html,
        llm_client=llm_client,
        protected_terms=["Fee Waiver Procedure", "manager approval"],
        config=ValidatorConfig(
            use_final_llm_judge=True,
            use_llm_for_ambiguous_claims=True,
        ),
    )

    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
