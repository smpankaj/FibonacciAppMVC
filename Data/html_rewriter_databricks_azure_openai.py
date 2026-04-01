# Databricks notebook source
# MAGIC %md
# MAGIC # HTML document rewriter with Azure OpenAI (GPT-5.1)
# MAGIC
# MAGIC This Databricks notebook rewrites HTML **block by block** while preserving the original
# MAGIC document structure. It is designed for cases like:
# MAGIC
# MAGIC - spelling and grammar correction
# MAGIC - active voice
# MAGIC - B1/plain-language simplification
# MAGIC - concise wording
# MAGIC - preserving tags such as `a`, `span`, `em`, `li`, `ol`, `table`, `tr`, `td`, `th`
# MAGIC - preventing edits to text inside `b` and `strong`
# MAGIC
# MAGIC The notebook uses a **raw-HTML rewrite + validator** pattern:
# MAGIC
# MAGIC 1. Parse the fragment.
# MAGIC 2. Find leaf rewrite blocks such as `p`, `li`, `th`, `td`, and headings.
# MAGIC 3. Send each block's raw HTML to Azure OpenAI.
# MAGIC 4. Validate that tags, attributes, and protected bold text were preserved.
# MAGIC 5. Retry with a repair prompt if validation fails.
# MAGIC
# MAGIC This approach keeps more linguistic context than opaque placeholders while still giving you a
# MAGIC strong safety net before accepting the model output.

# COMMAND ----------
# MAGIC %pip install --quiet --upgrade openai beautifulsoup4 lxml

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Recommended in Databricks:
# MAGIC
# MAGIC - Store the Azure OpenAI key in a secret scope, or pass it in as a widget for testing.
# MAGIC - Set `deployment_name` to **your Azure deployment name** that points to GPT-5.1.
# MAGIC
# MAGIC You can fill the widgets in the notebook UI, or set environment variables instead.

# COMMAND ----------
import os

try:
    dbutils.widgets.text("azure_endpoint", "")
    dbutils.widgets.text("api_key", "")
    dbutils.widgets.text("api_version", "2024-10-21")
    dbutils.widgets.text("deployment_name", "")
except Exception:
    pass


def _widget(name: str, default: str = "") -> str:
    try:
        value = dbutils.widgets.get(name)
        return value if value else default
    except Exception:
        return default


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or _widget("azure_endpoint")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or _widget("api_key")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION") or _widget("api_version", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or _widget("deployment_name")

print({
    "azure_endpoint_set": bool(AZURE_OPENAI_ENDPOINT),
    "api_key_set": bool(AZURE_OPENAI_API_KEY),
    "api_version": OPENAI_API_VERSION,
    "deployment_name_set": bool(AZURE_OPENAI_DEPLOYMENT),
})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Imports and client

# COMMAND ----------
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from openai import AzureOpenAI


client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Rewrite rules
# MAGIC
# MAGIC Replace this list with your own full rulebook. The notebook passes the rules verbatim to the model.

# COMMAND ----------
STYLE_RULES = [
    "Fix spelling and grammar.",
    "Prefer active voice when it keeps the meaning accurate.",
    "Use B1 language level.",
    "Use concise wording.",
    "Preserve the original meaning.",
    "Do not modify any text inside <b> or <strong> tags.",
]

STYLE_RULES

# COMMAND ----------
# MAGIC %md
# MAGIC ## HTML utilities
# MAGIC
# MAGIC The validator below enforces the main invariants:
# MAGIC
# MAGIC - the rewritten block must still parse as HTML
# MAGIC - the root tag and its attributes must match
# MAGIC - all descendant tags and attributes must match exactly
# MAGIC - text inside `b` / `strong` must remain unchanged
# MAGIC - no unexpected tags may appear
# MAGIC - no original tags or attributes may disappear

# COMMAND ----------
BLOCK_TAGS = {
    "p", "li", "th", "td", "caption",
    "h1", "h2", "h3", "h4", "h5", "h6",
}
PROTECTED_TAGS = {"b", "strong"}


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]


@dataclass
class BlockRewriteResult:
    original_html: str
    rewritten_html: str
    attempts: int
    validation: ValidationResult


def parse_fragment(fragment: str) -> Tuple[BeautifulSoup, Tag]:
    soup = BeautifulSoup(f"<root>{fragment}</root>", "lxml")
    root = soup.find("root")
    if root is None:
        raise ValueError("Could not parse HTML fragment.")
    return soup, root


def fragment_to_string(root: Tag) -> str:
    return "".join(str(child) for child in root.contents)


def significant_children(node: Tag) -> List[object]:
    children: List[object] = []
    for child in node.contents:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            if str(child).strip() == "":
                continue
            children.append(child)
            continue
        if isinstance(child, Tag):
            children.append(child)
    return children


def extract_single_tag(fragment: str) -> Tag:
    _, root = parse_fragment(fragment)
    children = [c for c in significant_children(root) if isinstance(c, Tag)]
    non_tag_children = [c for c in significant_children(root) if not isinstance(c, Tag)]

    if non_tag_children:
        raise ValueError("Rewritten block contains significant text outside the root tag.")
    if len(children) != 1:
        raise ValueError(f"Expected exactly one root tag, found {len(children)}.")
    return children[0]


def normalize_attr_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    return str(value)


def normalize_attrs(tag: Tag) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted((key, normalize_attr_value(value)) for key, value in tag.attrs.items()))


def structure_signature(node: Tag) -> Tuple[object, ...]:
    signature: List[object] = []
    for child in significant_children(node):
        if isinstance(child, NavigableString):
            signature.append(("TEXT",))
        elif isinstance(child, Tag):
            signature.append((
                "TAG",
                child.name,
                normalize_attrs(child),
                structure_signature(child),
            ))
    return tuple(signature)


def protected_signature(node: Tag) -> Tuple[Tuple[str, Tuple[Tuple[str, object], ...], str], ...]:
    protected: List[Tuple[str, Tuple[Tuple[str, object], ...], str]] = []
    for descendant in node.descendants:
        if isinstance(descendant, Tag) and descendant.name in PROTECTED_TAGS:
            protected.append((
                descendant.name,
                normalize_attrs(descendant),
                descendant.decode_contents(),
            ))
    return tuple(protected)


def contains_descendant_block(tag: Tag) -> bool:
    for descendant in tag.find_all(BLOCK_TAGS):
        if descendant is not tag:
            return True
    return False


def find_leaf_rewrite_targets(root: Tag) -> List[Tag]:
    targets: List[Tag] = []
    for tag in root.find_all(BLOCK_TAGS):
        if contains_descendant_block(tag):
            continue
        targets.append(tag)
    return targets


def validate_block_rewrite(original_html: str, candidate_html: str) -> ValidationResult:
    errors: List[str] = []

    try:
        original_tag = extract_single_tag(original_html)
    except Exception as exc:
        return ValidationResult(False, [f"Original block could not be parsed: {exc}"])

    try:
        candidate_tag = extract_single_tag(candidate_html)
    except Exception as exc:
        return ValidationResult(False, [f"Candidate block could not be parsed: {exc}"])

    if original_tag.name != candidate_tag.name:
        errors.append(f"Root tag changed from <{original_tag.name}> to <{candidate_tag.name}>.")

    if normalize_attrs(original_tag) != normalize_attrs(candidate_tag):
        errors.append("Root tag attributes changed.")

    if structure_signature(original_tag) != structure_signature(candidate_tag):
        errors.append("Tag structure changed. A tag moved, disappeared, or a new tag was introduced.")

    if protected_signature(original_tag) != protected_signature(candidate_tag):
        errors.append("Protected text inside <b> or <strong> changed.")

    return ValidationResult(not errors, errors)


def strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    fence_match = re.match(r"^```(?:html)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text

# COMMAND ----------
# MAGIC %md
# MAGIC ## Model prompt and block rewrite functions
# MAGIC
# MAGIC The prompt intentionally asks the model to rewrite **raw HTML** for a single block, while the validator rejects any structural drift.

# COMMAND ----------
REWRITE_SYSTEM_PROMPT = """
You rewrite a single HTML block according to style rules.

Non-negotiable requirements:
- Return HTML only for the same single root block you received.
- Preserve every tag exactly as HTML tags, including tag names, nesting, and attributes.
- Do not add tags.
- Do not remove tags.
- Do not move tags to a different position in the tree.
- Text inside child tags flows as part of the parent sentence and may be rewritten only if the tag is not protected.
- Never modify text inside <b> or <strong> tags.
- Do not wrap the answer in markdown fences.
- Do not add explanations.
- Keep the output semantically equivalent to the input unless an explicit style rule requires simplification.
""".strip()


def format_rules(rules: Sequence[str]) -> str:
    return "\n".join(f"{idx + 1}. {rule}" for idx, rule in enumerate(rules))


def build_user_prompt(block_html: str, rules: Sequence[str], repair_errors: Optional[Sequence[str]] = None) -> str:
    base = f"""
Apply the following rules to this HTML block.

Rules:
{format_rules(rules)}

HTML block:
{block_html}
""".strip()

    if repair_errors:
        repair_text = "\n".join(f"- {err}" for err in repair_errors)
        base += f"""

Your previous answer failed validation for these reasons:
{repair_text}

Repair the HTML while keeping the same root tag, descendant tags, nesting, and attributes.
"""
    return base


def call_model_for_block(
    client: AzureOpenAI,
    deployment_name: str,
    block_html: str,
    rules: Sequence[str],
    repair_errors: Optional[Sequence[str]] = None,
) -> str:
    completion = client.chat.completions.create(
        model=deployment_name,
        temperature=0,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(block_html, rules, repair_errors)},
        ],
    )

    content = completion.choices[0].message.content or ""
    return strip_code_fences(content)


def rewrite_block_with_validation(
    client: AzureOpenAI,
    deployment_name: str,
    block_html: str,
    rules: Sequence[str],
    max_attempts: int = 3,
) -> BlockRewriteResult:
    last_candidate = block_html
    last_validation = ValidationResult(False, ["The model was not called."])
    repair_errors: Optional[Sequence[str]] = None

    for attempt in range(1, max_attempts + 1):
        candidate = call_model_for_block(
            client=client,
            deployment_name=deployment_name,
            block_html=block_html,
            rules=rules,
            repair_errors=repair_errors,
        )
        validation = validate_block_rewrite(block_html, candidate)
        last_candidate = candidate
        last_validation = validation

        if validation.is_valid:
            return BlockRewriteResult(
                original_html=block_html,
                rewritten_html=candidate,
                attempts=attempt,
                validation=validation,
            )

        repair_errors = validation.errors

    return BlockRewriteResult(
        original_html=block_html,
        rewritten_html=last_candidate,
        attempts=max_attempts,
        validation=last_validation,
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Whole-fragment transformation
# MAGIC
# MAGIC This function rewrites each **leaf block** independently.
# MAGIC
# MAGIC Typical units:
# MAGIC
# MAGIC - paragraphs: `p`
# MAGIC - list items: `li`
# MAGIC - table cells: `th`, `td`, `caption`
# MAGIC - headings: `h1` to `h6`
# MAGIC
# MAGIC Why leaf blocks? Because it prevents the model from merging list items, collapsing table rows, or rewriting entire container structures like `ol`, `ul`, and `table`.

# COMMAND ----------

def transform_html_fragment(
    fragment: str,
    client: AzureOpenAI,
    deployment_name: str,
    rules: Sequence[str],
    max_attempts_per_block: int = 3,
    fail_on_validation_error: bool = True,
) -> Tuple[str, List[Dict[str, object]]]:
    soup, root = parse_fragment(fragment)
    targets = list(find_leaf_rewrite_targets(root))
    report: List[Dict[str, object]] = []

    for index, tag in enumerate(targets, start=1):
        original_html = str(tag)
        result = rewrite_block_with_validation(
            client=client,
            deployment_name=deployment_name,
            block_html=original_html,
            rules=rules,
            max_attempts=max_attempts_per_block,
        )

        report_item = {
            "index": index,
            "tag": tag.name,
            "attempts": result.attempts,
            "is_valid": result.validation.is_valid,
            "errors": result.validation.errors,
            "original_html": result.original_html,
            "rewritten_html": result.rewritten_html,
        }
        report.append(report_item)

        if not result.validation.is_valid:
            if fail_on_validation_error:
                raise ValueError(
                    f"Validation failed for block {index} <{tag.name}>: {result.validation.errors}"
                )
            continue

        replacement_tag = extract_single_tag(result.rewritten_html)
        tag.replace_with(replacement_tag)

    return fragment_to_string(root), report

# COMMAND ----------
# MAGIC %md
# MAGIC ## Example input

# COMMAND ----------
example_html = """
<p>Today is Sunday. Sunday are a great day as they have a lot of <a href="/activities">activitties</a>.</p>
<ol>
  <li>We got park on Sunday and eat in a nice restaurants</li>
  <li>This is <strong>Early Years Plus</strong> for childrens on Sunday</li>
</ol>
<table>
  <tr>
    <th>Servce</th>
    <th>Descrption</th>
  </tr>
  <tr>
    <td><a href="/kids">Kids activitties</a></td>
    <td>This is <strong>fixed wording</strong> for Sunday programe.</td>
  </tr>
</table>
""".strip()

print(example_html)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run the transformation
# MAGIC
# MAGIC This will fail fast if any block breaks validation.

# COMMAND ----------
rewritten_html, rewrite_report = transform_html_fragment(
    fragment=example_html,
    client=client,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    rules=STYLE_RULES,
    max_attempts_per_block=3,
    fail_on_validation_error=True,
)

print(rewritten_html)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Inspect the per-block report

# COMMAND ----------
for item in rewrite_report:
    print(json.dumps(item, indent=2, ensure_ascii=False))
    print("-" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Notes for production use
# MAGIC
# MAGIC 1. Keep your rulebook in a versioned table or config file rather than hard-coding it.
# MAGIC 2. Process large documents section by section to control latency and retries.
# MAGIC 3. Log validation failures so you can refine prompts over time.
# MAGIC 4. For especially hard edge cases, add a fallback mode that introduces temporary control markers for selected inline spans only.
# MAGIC 5. If you need to preserve additional immutable tags, add them to `PROTECTED_TAGS`.
