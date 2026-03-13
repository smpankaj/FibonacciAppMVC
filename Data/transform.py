# Databricks notebook source
# MAGIC %md
# MAGIC # Safe Knowledge-Base Enrichment with Azure OpenAI
# MAGIC
# MAGIC This notebook is designed for large-scale HTML article enrichment in Databricks.
# MAGIC It is intentionally conservative for financial content:
# MAGIC - rewrite only editable text nodes
# MAGIC - never send bold text to the model
# MAGIC - freeze dates, numbers, amounts, percentages, URLs, emails, codes, and modal verbs
# MAGIC - validate the rewritten text before saving
# MAGIC - send risky outputs to human review instead of auto-publishing

# COMMAND ----------

# MAGIC %pip install openai beautifulsoup4 tenacity pydantic

# COMMAND ----------

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from delta.tables import DeltaTable
from openai import APIError, AzureOpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kb_safe_enrichment")

spark = SparkSession.builder.getOrCreate()


# COMMAND ----------


@dataclass(frozen=True)
class EditorialRule:
    rule_id: str
    instruction: str
    priority: int = 100
    active: bool = True
    categories: Tuple[str, ...] = ("all",)
    html_tags: Tuple[str, ...] = ("all",)
    deterministic: bool = False


@dataclass(frozen=True)
class DeterministicRuleConfig:
    convert_all_ul_to_ol: bool = False
    festival_names: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineConfig:
    azure_endpoint: str
    api_key: str
    api_version: str
    chat_deployment: str
    source_table: str
    result_table: str
    source_id_col: str = "article_id"
    source_category_col: str = "category"
    source_html_col: str = "html"
    batch_size: int = 50
    max_workers: int = 8
    prompt_version: str = "kb-safe-v1"
    secret_scope: Optional[str] = None
    secret_key: Optional[str] = None


@dataclass
class FrozenText:
    frozen_text: str
    token_map: Dict[str, str]
    token_types: Dict[str, str]


@dataclass
class NodeResult:
    original_text: str
    rewritten_text: str
    changed: bool
    needs_review: bool
    review_reasons: List[str] = field(default_factory=list)
    applied_rule_ids: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ArticleResult:
    article_id: str
    category: str
    original_html: str
    enriched_html: str
    status: str
    needs_review: bool
    review_reasons: List[str]
    rules_applied: List[str]
    changed_node_count: int
    reviewed_node_count: int
    processed_node_count: int
    skipped_node_count: int
    prompt_tokens: int
    completion_tokens: int
    source_html_sha256: str
    enriched_html_sha256: str
    prompt_version: str
    model_deployment: str
    processed_at_utc: str
    error: Optional[str] = None


class RewritePayload(BaseModel):
    rewritten_text: str
    applied_rule_ids: List[str] = Field(default_factory=list)
    needs_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)


REWRITE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_text": {"type": "string"},
        "applied_rule_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "needs_review": {"type": "boolean"},
        "review_reasons": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "rewritten_text",
        "applied_rule_ids",
        "needs_review",
        "review_reasons",
    ],
    "additionalProperties": False,
}


DEFAULT_RULES: List[EditorialRule] = [
    EditorialRule("R001", "Rewrite to CEFR B2 language level.", priority=1),
    EditorialRule("R002", "Use clear, direct sentences.", priority=2),
    EditorialRule("R003", "Keep the original meaning exactly the same.", priority=1),
    EditorialRule("R004", "Do not add any new facts or assumptions.", priority=1),
    EditorialRule(
        "R005",
        "Do not change dates, figures, numbers, percentages, amounts, names, codes, URLs, or emails.",
        priority=1,
    ),
    EditorialRule(
        "R006",
        "Do not change bold text, even indirectly. Bold text is excluded from rewriting.",
        priority=1,
    ),
    EditorialRule(
        "R007",
        "Keep modal verbs with legal or procedural force unchanged: must, shall, may, should, will, can.",
        priority=1,
    ),
    EditorialRule(
        "R008",
        "If a sentence is already compliant, leave it unchanged.",
        priority=2,
    ),
    EditorialRule(
        "R020",
        "Convert bulleted lists to numbered lists when the business rule requires it.",
        priority=5,
        deterministic=True,
    ),
]


DEFAULT_DETERMINISTIC_RULES = DeterministicRuleConfig(
    convert_all_ul_to_ol=False,
    festival_names=(),
)


PROTECTED_ANCESTOR_TAGS = {
    "b",
    "strong",
    "code",
    "pre",
    "kbd",
    "samp",
    "var",
    "a",
    "script",
    "style",
    "abbr",
    "acronym",
}

SKIP_PARENT_TAGS = {
    "[document]",
    "head",
    "html",
    "meta",
    "link",
    "script",
    "style",
    "title",
}

PROTECTED_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("url", re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    (
        "currency",
        re.compile(
            r"(?:USD|EUR|GBP|CHF|SEK|NOK|DKK)\s?\d[\d,.\s]*"
            r"|[$\u20AC\u00A3]\s?\d[\d,.\s]*",
            re.IGNORECASE,
        ),
    ),
    ("percent", re.compile(r"\b\d+(?:[.,]\d+)?\s?%", re.IGNORECASE)),
    ("date_iso", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("date_slash", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    (
        "date_text",
        re.compile(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b"
            r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ),
    ("code", re.compile(r"\b[A-Z]{2,}(?:[-_/][A-Z0-9]{2,})+\b")),
    (
        "modal",
        re.compile(
            r"\b(?:must not|shall not|may not|must|shall|may|should|will|can|cannot)\b",
            re.IGNORECASE,
        ),
    ),
    ("number", re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")),
]


# COMMAND ----------


def get_secret_or_env(secret_scope: Optional[str], secret_key: Optional[str], env_key: str) -> str:
    if secret_scope and secret_key:
        try:
            return dbutils.secrets.get(scope=secret_scope, key=secret_key)  # type: ignore[name-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read Databricks secret %s/%s: %s", secret_scope, secret_key, exc)
    value = os.getenv(env_key)
    if not value:
        raise ValueError(f"Missing required secret or env var: {env_key}")
    return value


def build_pipeline_config(
    source_table: str,
    result_table: str,
    source_id_col: str = "article_id",
    source_category_col: str = "category",
    source_html_col: str = "html",
    batch_size: int = 50,
    max_workers: int = 8,
    prompt_version: str = "kb-safe-v1",
    secret_scope: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> PipelineConfig:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    api_key = get_secret_or_env(secret_scope, secret_key, "AZURE_OPENAI_API_KEY")

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is required.")
    if not deployment:
        raise ValueError("AZURE_OPENAI_CHAT_DEPLOYMENT is required.")

    return PipelineConfig(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        chat_deployment=deployment,
        source_table=source_table,
        result_table=result_table,
        source_id_col=source_id_col,
        source_category_col=source_category_col,
        source_html_col=source_html_col,
        batch_size=batch_size,
        max_workers=max_workers,
        prompt_version=prompt_version,
        secret_scope=secret_scope,
        secret_key=secret_key,
    )


def load_rules_from_json(path: str) -> List[EditorialRule]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [EditorialRule(**row) for row in payload if row.get("active", True)]


def load_rules_from_delta(table_name: str) -> List[EditorialRule]:
    rows = spark.table(table_name).where(F.col("active") == F.lit(True)).collect()
    return [
        EditorialRule(
            rule_id=row["rule_id"],
            instruction=row["instruction"],
            priority=int(row["priority"]),
            active=bool(row["active"]),
            categories=tuple(row["categories"] or ["all"]),
            html_tags=tuple(row["html_tags"] or ["all"]),
            deterministic=bool(row["deterministic"]),
        )
        for row in rows
    ]


def applicable_rules(category: str, html_tag: str, rules: Sequence[EditorialRule]) -> List[EditorialRule]:
    selected = []
    for rule in rules:
        if not rule.active:
            continue
        category_match = "all" in rule.categories or category in rule.categories
        tag_match = "all" in rule.html_tags or html_tag in rule.html_tags
        if category_match and tag_match:
            selected.append(rule)
    return sorted(selected, key=lambda item: (item.priority, item.rule_id))


def build_prompt_version(rules: Sequence[EditorialRule], base_label: str) -> str:
    stable_payload = json.dumps(
        [
            {
                "rule_id": rule.rule_id,
                "instruction": rule.instruction,
                "priority": rule.priority,
                "categories": list(rule.categories),
                "html_tags": list(rule.html_tags),
                "deterministic": rule.deterministic,
            }
            for rule in sorted(rules, key=lambda item: item.rule_id)
        ],
        ensure_ascii=True,
        sort_keys=True,
    )
    suffix = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()[:12]
    return f"{base_label}-{suffix}"


# COMMAND ----------


def has_protected_ancestor(node: NavigableString) -> bool:
    for parent in node.parents:
        if isinstance(parent, Tag) and parent.name and parent.name.lower() in PROTECTED_ANCESTOR_TAGS:
            return True
    return False


def iter_editable_text_nodes(soup: BeautifulSoup) -> Iterable[NavigableString]:
    for node in soup.find_all(string=True):
        if isinstance(node, Comment):
            continue
        text = str(node)
        if not text or not text.strip():
            continue

        parent = node.parent
        parent_name = parent.name.lower() if isinstance(parent, Tag) and parent.name else "[document]"
        if parent_name in SKIP_PARENT_TAGS:
            continue
        if has_protected_ancestor(node):
            continue
        yield node


def apply_deterministic_html_rules(soup: BeautifulSoup, config: DeterministicRuleConfig) -> None:
    if config.convert_all_ul_to_ol:
        for ul_tag in soup.find_all("ul"):
            ul_tag.name = "ol"

    if config.festival_names:
        compiled = [
            (
                festival_name,
                re.compile(rf"\b{re.escape(festival_name)}\b", re.IGNORECASE),
            )
            for festival_name in config.festival_names
        ]
        for node in list(iter_editable_text_nodes(soup)):
            text = str(node)
            new_text = text
            for festival_name, pattern in compiled:
                canonical = festival_name[0].upper() + festival_name[1:]
                new_text = pattern.sub(canonical, new_text)
            if new_text != text:
                node.replace_with(NavigableString(new_text))


def extract_bold_texts(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    values = []
    for tag in soup.find_all(["b", "strong"]):
        values.append(tag.get_text(" ", strip=False))
    return values


def extract_signal_counter(text: str) -> Dict[str, Counter]:
    counters: Dict[str, Counter] = {}
    for name, pattern in PROTECTED_PATTERNS:
        counters[name] = Counter(match.group(0) for match in pattern.finditer(text))
    return counters


def find_protected_spans(text: str) -> List[Tuple[int, int, str, str]]:
    spans: List[Tuple[int, int, str, str]] = []
    for token_type, pattern in PROTECTED_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), token_type, match.group(0)))

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    chosen: List[Tuple[int, int, str, str]] = []
    last_end = -1
    for start, end, token_type, value in spans:
        if start < last_end:
            continue
        chosen.append((start, end, token_type, value))
        last_end = end
    return chosen


def freeze_text(text: str) -> FrozenText:
    spans = find_protected_spans(text)
    if not spans:
        return FrozenText(frozen_text=text, token_map={}, token_types={})

    pieces: List[str] = []
    token_map: Dict[str, str] = {}
    token_types: Dict[str, str] = {}
    cursor = 0

    for idx, (start, end, token_type, value) in enumerate(spans, start=1):
        token = f"[[IMM_{idx:03d}]]"
        token_map[token] = value
        token_types[token] = token_type
        pieces.append(text[cursor:start])
        pieces.append(token)
        cursor = end

    pieces.append(text[cursor:])
    return FrozenText(
        frozen_text="".join(pieces),
        token_map=token_map,
        token_types=token_types,
    )


def restore_text(text: str, token_map: Dict[str, str]) -> str:
    restored = text
    for token, original_value in token_map.items():
        restored = restored.replace(token, original_value)
    return restored


def validate_frozen_tokens(rewritten_text: str, frozen: FrozenText) -> List[str]:
    problems = []
    for token in frozen.token_map:
        count = rewritten_text.count(token)
        if count != 1:
            problems.append(f"Immutable token {token} expected once but found {count} times.")
    return problems


def validate_factual_integrity(original_text: str, rewritten_text: str) -> List[str]:
    original_signals = extract_signal_counter(original_text)
    rewritten_signals = extract_signal_counter(rewritten_text)

    failures = []
    for signal_name in original_signals:
        if original_signals[signal_name] != rewritten_signals[signal_name]:
            failures.append(f"{signal_name} values changed.")
    return failures


def validate_length_change(original_text: str, rewritten_text: str) -> List[str]:
    if not original_text.strip():
        return []
    ratio = len(rewritten_text.strip()) / max(len(original_text.strip()), 1)
    if ratio < 0.65 or ratio > 1.35:
        return [f"Length ratio {ratio:.2f} is outside the allowed safety range."]
    return []


def validate_bold_integrity(original_html: str, rewritten_html: str) -> List[str]:
    if extract_bold_texts(original_html) != extract_bold_texts(rewritten_html):
        return ["Bold text changed."]
    return []


def visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def validate_article_integrity(original_html: str, rewritten_html: str) -> List[str]:
    failures = []
    failures.extend(validate_bold_integrity(original_html, rewritten_html))
    failures.extend(
        validate_factual_integrity(
            visible_text_from_html(original_html),
            visible_text_from_html(rewritten_html),
        )
    )
    return failures


# COMMAND ----------


def build_system_prompt(category: str, html_tag: str, rules: Sequence[EditorialRule]) -> str:
    active_rules = applicable_rules(category, html_tag, rules)
    business_rules = "\n".join(
        f"- [{rule.rule_id}] {rule.instruction}"
        for rule in active_rules
        if not rule.deterministic
    )

    return f"""
You are rewriting text from a financial institution knowledge base.

You receive plain text only. The text was extracted from an HTML document.
Some protected values have been replaced by immutable tokens such as [[IMM_001]].

Apply the business rules below while preserving meaning exactly:
{business_rules}

Absolute constraints:
- Never change meaning.
- Never add facts, assumptions, examples, warnings, or explanations.
- Never remove facts that affect meaning.
- Never change any immutable token such as [[IMM_001]].
- Never change dates, figures, numbers, percentages, amounts, names, codes, URLs, emails, or modal verbs.
- If the text is already compliant, return it unchanged.
- If you are unsure, return the input unchanged and set needs_review to true.

Return JSON only with:
- rewritten_text: string
- applied_rule_ids: array of rule ids
- needs_review: boolean
- review_reasons: array of strings
""".strip()


def build_user_prompt(text: str, parent_tag: str) -> str:
    return json.dumps(
        {
            "html_parent_tag": parent_tag,
            "text": text,
        },
        ensure_ascii=False,
    )


class AzureStructuredRewriter:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.azure_endpoint,
        )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError, ValidationError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def rewrite(self, category: str, html_tag: str, text: str, rules: Sequence[EditorialRule]) -> Tuple[RewritePayload, int, int]:
        system_prompt = build_system_prompt(category, html_tag, rules)
        user_prompt = build_user_prompt(text, html_tag)

        try:
            response = self.client.responses.create(
                model=self.config.chat_deployment,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "kb_rewrite_payload",
                        "schema": REWRITE_JSON_SCHEMA,
                        "strict": True,
                    }
                },
            )
            payload = RewritePayload.model_validate_json(response.output_text)
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            return payload, prompt_tokens, completion_tokens
        except Exception as primary_exc:  # noqa: BLE001
            logger.info("Responses API path failed, using chat.completions fallback: %s", primary_exc)

        completion = self.client.chat.completions.create(
            model=self.config.chat_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        payload = RewritePayload.model_validate_json(raw)
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return payload, prompt_tokens, completion_tokens


# COMMAND ----------


class SafeArticleProcessor:
    def __init__(
        self,
        config: PipelineConfig,
        rules: Sequence[EditorialRule],
        deterministic_rules: DeterministicRuleConfig,
    ):
        self.config = config
        self.rules = list(rules)
        self.deterministic_rules = deterministic_rules
        self.rewriter = AzureStructuredRewriter(config)

    def rewrite_text_node(self, category: str, parent_tag: str, original_text: str) -> NodeResult:
        if len(original_text.strip()) < 3:
            return NodeResult(
                original_text=original_text,
                rewritten_text=original_text,
                changed=False,
                needs_review=False,
            )

        frozen = freeze_text(original_text)
        payload, prompt_tokens, completion_tokens = self.rewriter.rewrite(
            category=category,
            html_tag=parent_tag,
            text=frozen.frozen_text,
            rules=self.rules,
        )

        validation_failures = []
        validation_failures.extend(validate_frozen_tokens(payload.rewritten_text, frozen))

        restored = restore_text(payload.rewritten_text, frozen.token_map)
        validation_failures.extend(validate_factual_integrity(original_text, restored))
        validation_failures.extend(validate_length_change(original_text, restored))

        needs_review = payload.needs_review or bool(validation_failures)
        review_reasons = list(payload.review_reasons) + validation_failures

        return NodeResult(
            original_text=original_text,
            rewritten_text=original_text if needs_review else restored,
            changed=(not needs_review) and restored != original_text,
            needs_review=needs_review,
            review_reasons=review_reasons,
            applied_rule_ids=payload.applied_rule_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def process_article(self, article_id: str, category: str, html: str) -> ArticleResult:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        original_html = html or ""
        soup = BeautifulSoup(original_html, "html.parser")

        apply_deterministic_html_rules(soup, self.deterministic_rules)

        changed_node_count = 0
        reviewed_node_count = 0
        processed_node_count = 0
        skipped_node_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        applied_rule_ids = set()
        review_reasons: List[str] = []

        for node in list(iter_editable_text_nodes(soup)):
            parent = node.parent
            parent_tag = parent.name.lower() if isinstance(parent, Tag) and parent.name else "unknown"
            original_text = str(node)

            try:
                node_result = self.rewrite_text_node(category, parent_tag, original_text)
            except Exception as exc:  # noqa: BLE001
                reviewed_node_count += 1
                processed_node_count += 1
                review_reasons.append(f"Node rewrite failed under <{parent_tag}>: {exc}")
                continue

            processed_node_count += 1
            prompt_tokens += node_result.prompt_tokens
            completion_tokens += node_result.completion_tokens
            applied_rule_ids.update(node_result.applied_rule_ids)

            if node_result.needs_review:
                reviewed_node_count += 1
                review_reasons.extend(node_result.review_reasons)
                continue

            if node_result.changed:
                changed_node_count += 1
                node.replace_with(NavigableString(node_result.rewritten_text))
            else:
                skipped_node_count += 1

        enriched_html = str(soup)
        article_failures = validate_article_integrity(original_html, enriched_html)
        review_reasons.extend(article_failures)

        needs_review = bool(review_reasons)
        status = "review_required" if needs_review else "succeeded"

        return ArticleResult(
            article_id=article_id,
            category=category,
            original_html=original_html,
            enriched_html=enriched_html if not needs_review else original_html,
            status=status,
            needs_review=needs_review,
            review_reasons=sorted(set(review_reasons)),
            rules_applied=sorted(applied_rule_ids),
            changed_node_count=changed_node_count,
            reviewed_node_count=reviewed_node_count,
            processed_node_count=processed_node_count,
            skipped_node_count=skipped_node_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source_html_sha256=hashlib.sha256(original_html.encode("utf-8")).hexdigest(),
            enriched_html_sha256=hashlib.sha256(
                (enriched_html if not needs_review else original_html).encode("utf-8")
            ).hexdigest(),
            prompt_version=self.config.prompt_version,
            model_deployment=self.config.chat_deployment,
            processed_at_utc=started_at,
            error=None,
        )


# COMMAND ----------


RESULT_SCHEMA = T.StructType(
    [
        T.StructField("article_id", T.StringType(), False),
        T.StructField("category", T.StringType(), True),
        T.StructField("original_html", T.StringType(), True),
        T.StructField("enriched_html", T.StringType(), True),
        T.StructField("status", T.StringType(), True),
        T.StructField("needs_review", T.BooleanType(), True),
        T.StructField("review_reasons", T.ArrayType(T.StringType()), True),
        T.StructField("rules_applied", T.ArrayType(T.StringType()), True),
        T.StructField("changed_node_count", T.IntegerType(), True),
        T.StructField("reviewed_node_count", T.IntegerType(), True),
        T.StructField("processed_node_count", T.IntegerType(), True),
        T.StructField("skipped_node_count", T.IntegerType(), True),
        T.StructField("prompt_tokens", T.LongType(), True),
        T.StructField("completion_tokens", T.LongType(), True),
        T.StructField("source_html_sha256", T.StringType(), True),
        T.StructField("enriched_html_sha256", T.StringType(), True),
        T.StructField("prompt_version", T.StringType(), True),
        T.StructField("model_deployment", T.StringType(), True),
        T.StructField("processed_at_utc", T.StringType(), True),
        T.StructField("error", T.StringType(), True),
    ]
)


def ensure_result_table(table_name: str) -> None:
    if spark.catalog.tableExists(table_name):
        return

    empty_df = spark.createDataFrame([], RESULT_SCHEMA)
    empty_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    logger.info("Created result table %s", table_name)


def fetch_pending_articles(config: PipelineConfig) -> List[dict]:
    source_df = (
        spark.table(config.source_table)
        .where(F.col(config.source_html_col).isNotNull())
        .select(
            F.col(config.source_id_col).alias("article_id"),
            F.col(config.source_category_col).alias("category"),
            F.col(config.source_html_col).alias("html"),
        )
    )

    if spark.catalog.tableExists(config.result_table):
        processed_df = (
            spark.table(config.result_table)
            .where(F.col("prompt_version") == F.lit(config.prompt_version))
            .where(F.col("status").isin("succeeded", "review_required"))
            .select("article_id")
            .distinct()
        )
        pending_df = source_df.join(processed_df, on="article_id", how="left_anti")
    else:
        pending_df = source_df

    rows = pending_df.orderBy("article_id").limit(config.batch_size).collect()
    return [row.asDict() for row in rows]


_THREAD_LOCAL = threading.local()


def get_thread_processor(
    config: PipelineConfig,
    rules: Sequence[EditorialRule],
    deterministic_rules: DeterministicRuleConfig,
) -> SafeArticleProcessor:
    processor = getattr(_THREAD_LOCAL, "processor", None)
    if processor is None:
        processor = SafeArticleProcessor(config, rules, deterministic_rules)
        _THREAD_LOCAL.processor = processor
    return processor


def process_article_row(
    row: dict,
    config: PipelineConfig,
    rules: Sequence[EditorialRule],
    deterministic_rules: DeterministicRuleConfig,
) -> ArticleResult:
    processor = get_thread_processor(config, rules, deterministic_rules)
    return processor.process_article(
        article_id=str(row["article_id"]),
        category=str(row.get("category") or "all"),
        html=str(row.get("html") or ""),
    )


def process_batch(
    config: PipelineConfig,
    rules: Sequence[EditorialRule],
    deterministic_rules: DeterministicRuleConfig,
) -> List[ArticleResult]:
    rows = fetch_pending_articles(config)
    if not rows:
        logger.info("No pending rows found for prompt version %s", config.prompt_version)
        return []

    logger.info("Processing %s articles from %s", len(rows), config.source_table)
    results: List[ArticleResult] = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(process_article_row, row, config, rules, deterministic_rules): row["article_id"]
            for row in rows
        }
        for future in as_completed(futures):
            article_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.exception("Article %s failed.", article_id)
                results.append(
                    ArticleResult(
                        article_id=str(article_id),
                        category="unknown",
                        original_html="",
                        enriched_html="",
                        status="failed",
                        needs_review=True,
                        review_reasons=[],
                        rules_applied=[],
                        changed_node_count=0,
                        reviewed_node_count=0,
                        processed_node_count=0,
                        skipped_node_count=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        source_html_sha256="",
                        enriched_html_sha256="",
                        prompt_version=config.prompt_version,
                        model_deployment=config.chat_deployment,
                        processed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        error=str(exc),
                    )
                )

    return results


def merge_results(results: Sequence[ArticleResult], table_name: str) -> None:
    if not results:
        return

    rows = [
        {
            "article_id": item.article_id,
            "category": item.category,
            "original_html": item.original_html,
            "enriched_html": item.enriched_html,
            "status": item.status,
            "needs_review": item.needs_review,
            "review_reasons": item.review_reasons,
            "rules_applied": item.rules_applied,
            "changed_node_count": item.changed_node_count,
            "reviewed_node_count": item.reviewed_node_count,
            "processed_node_count": item.processed_node_count,
            "skipped_node_count": item.skipped_node_count,
            "prompt_tokens": item.prompt_tokens,
            "completion_tokens": item.completion_tokens,
            "source_html_sha256": item.source_html_sha256,
            "enriched_html_sha256": item.enriched_html_sha256,
            "prompt_version": item.prompt_version,
            "model_deployment": item.model_deployment,
            "processed_at_utc": item.processed_at_utc,
            "error": item.error,
        }
        for item in results
    ]
    batch_df = spark.createDataFrame(rows, schema=RESULT_SCHEMA)

    if not spark.catalog.tableExists(table_name):
        batch_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
        return

    delta_table = DeltaTable.forName(spark, table_name)
    (
        delta_table.alias("target")
        .merge(
            batch_df.alias("source"),
            "target.article_id = source.article_id "
            "AND target.prompt_version = source.prompt_version",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def run_job(
    config: PipelineConfig,
    rules: Optional[Sequence[EditorialRule]] = None,
    deterministic_rules: DeterministicRuleConfig = DEFAULT_DETERMINISTIC_RULES,
) -> List[ArticleResult]:
    resolved_rules = list(rules or DEFAULT_RULES)
    config = PipelineConfig(
        **{
            **config.__dict__,
            "prompt_version": build_prompt_version(resolved_rules, config.prompt_version),
        }
    )

    ensure_result_table(config.result_table)
    results = process_batch(config, resolved_rules, deterministic_rules)
    merge_results(results, config.result_table)
    logger.info("Wrote %s results into %s", len(results), config.result_table)
    return results


# COMMAND ----------

# Example setup:
#
# 1. Put your source HTML articles into a Delta table with at least:
#    - article_id
#    - category
#    - html
#
# 2. Configure Azure OpenAI via Databricks secrets or env vars:
#    - AZURE_OPENAI_ENDPOINT
#    - AZURE_OPENAI_API_KEY
#    - AZURE_OPENAI_API_VERSION
#    - AZURE_OPENAI_CHAT_DEPLOYMENT
#
# 3. Optionally load your full 100-rule set:
#    custom_rules = load_rules_from_json("/dbfs/FileStore/kb_enrichment/rules.json")
#    or:
#    custom_rules = load_rules_from_delta("main.ops.kb_rules")
#
# 4. Optionally enable deterministic HTML rules that are truly global:
#    deterministic_rules = DeterministicRuleConfig(
#        convert_all_ul_to_ol=True,
#        festival_names=("spring festival", "winter festival"),
#    )
#
# 5. Run:
#
# config = build_pipeline_config(
#     source_table="main.kb.raw_articles",
#     result_table="main.kb.enriched_articles",
#     source_id_col="article_id",
#     source_category_col="category",
#     source_html_col="content_html",
#     batch_size=50,
#     max_workers=8,
#     prompt_version="kb-safe-v1",
#     secret_scope="kb-enrichment",
#     secret_key="azure-openai-api-key",
# )
#
# results = run_job(
#     config,
#     rules=DEFAULT_RULES,
#     deterministic_rules=DEFAULT_DETERMINISTIC_RULES,
# )
# display(spark.table(config.result_table).orderBy(F.desc("processed_at_utc")).limit(20))
