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
# MAGIC This version of the notebook is intentionally written in a **beginner-friendly style**:
# MAGIC
# MAGIC - no list comprehensions
# MAGIC - step-by-step `for` loops
# MAGIC - detailed comments before and inside each function
# MAGIC - simpler control flow that is easier to modify

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


# This helper function reads a Databricks widget value.
#
# Overall purpose:
# - Try to get a value from a Databricks notebook widget.
# - If the widget does not exist, return a default value instead.
#
# Why this exists:
# - It lets the notebook run both inside Databricks and in simple local testing.
def _widget(name: str, default: str = "") -> str:
    """Return a widget value if it exists, otherwise return the default value."""

    try:
        # Try to read the widget from Databricks.
        value = dbutils.widgets.get(name)

        # If the widget exists but is empty, return the default.
        if value:
            return value
        else:
            return default
    except Exception:
        # If widgets are not available at all, return the default.
        return default


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or _widget("azure_endpoint")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or _widget("api_key")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION") or _widget("api_version", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or _widget("deployment_name")

print(
    {
        "azure_endpoint_set": bool(AZURE_OPENAI_ENDPOINT),
        "api_key_set": bool(AZURE_OPENAI_API_KEY),
        "api_version": OPENAI_API_VERSION,
        "deployment_name_set": bool(AZURE_OPENAI_DEPLOYMENT),
    }
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Imports and client

# COMMAND ----------
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


# These small data classes make the output easier to read.
#
# ValidationResult answers one question:
# - Did validation pass?
# - If not, what errors were found?
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]


# BlockRewriteResult stores the full story for one rewritten HTML block.
#
# It keeps:
# - the original block HTML
# - the rewritten block HTML
# - how many attempts were needed
# - the final validation result
@dataclass
class BlockRewriteResult:
    original_html: str
    rewritten_html: str
    attempts: int
    validation: ValidationResult


# This function parses an HTML fragment by wrapping it in a fake root tag.
#
# Overall purpose:
# - BeautifulSoup works more predictably when we give it one single outer element.
# - The fake <root> tag lets us parse fragments like multiple <p> tags together.
#
# Example:
# - Input:  "<p>One</p><p>Two</p>"
# - Parsed as: <root><p>One</p><p>Two</p></root>
def parse_fragment(fragment: str) -> Tuple[BeautifulSoup, Tag]:
    """Parse an HTML fragment and return the soup object plus the fake root tag."""

    # Add a fake outer wrapper so the fragment always has one top-level parent.
    wrapped_fragment = "<root>" + fragment + "</root>"

    # Parse the wrapped fragment using lxml.
    soup = BeautifulSoup(wrapped_fragment, "lxml")

    # Find the fake root tag we just created.
    root = soup.find("root")

    # If root is missing, parsing failed in an unexpected way.
    if root is None:
        raise ValueError("Could not parse HTML fragment.")

    return soup, root


# This function converts the children of the fake root tag back into a plain fragment string.
#
# Overall purpose:
# - The fake <root> tag is only for internal parsing.
# - We do not want to return it to the caller.
def fragment_to_string(root: Tag) -> str:
    """Return the inner HTML of the fake root tag as a normal fragment string."""

    parts: List[str] = []

    # Add each child node's string form to the output list.
    for child in root.contents:
        parts.append(str(child))

    # Join all children together into one HTML fragment.
    return "".join(parts)


# This function returns only the meaningful child nodes of a tag.
#
# Overall purpose:
# - Ignore comments.
# - Ignore whitespace-only text nodes.
# - Keep real text and real tags.
#
# Why this matters:
# - Validation should compare meaningful structure, not random whitespace.
def significant_children(node: Tag) -> List[object]:
    """Return child nodes that are important for structure checking."""

    children: List[object] = []

    # Look at every direct child of the node.
    for child in node.contents:
        # Ignore HTML comments.
        if isinstance(child, Comment):
            continue

        # For text nodes, ignore empty or whitespace-only text.
        if isinstance(child, NavigableString):
            child_text = str(child)
            if child_text.strip() == "":
                continue
            children.append(child)
            continue

        # Keep normal HTML tags.
        if isinstance(child, Tag):
            children.append(child)

    return children


# This function extracts exactly one real tag from a small HTML fragment.
#
# Overall purpose:
# - A rewritten block should return one root HTML element such as one <p> or one <td>.
# - If the model returns extra text outside that tag, or more than one tag, validation should fail.
def extract_single_tag(fragment: str) -> Tag:
    """Parse a fragment and return its single significant root tag."""

    parsed_soup, root = parse_fragment(fragment)

    # We do not actually use parsed_soup afterward, but keeping it in a named variable
    # makes the code easier for beginners to follow.
    _ = parsed_soup

    meaningful_children = significant_children(root)

    tag_children: List[Tag] = []
    non_tag_children: List[object] = []

    # Split meaningful children into two groups:
    # 1. HTML tags
    # 2. non-tag content, such as text outside the main tag
    for child in meaningful_children:
        if isinstance(child, Tag):
            tag_children.append(child)
        else:
            non_tag_children.append(child)

    # Text outside the root tag is not allowed.
    if len(non_tag_children) > 0:
        raise ValueError("Rewritten block contains significant text outside the root tag.")

    # We expect exactly one root tag, not zero and not two or more.
    if len(tag_children) != 1:
        message = "Expected exactly one root tag, found " + str(len(tag_children)) + "."
        raise ValueError(message)

    return tag_children[0]


# This function normalizes one HTML attribute value.
#
# Overall purpose:
# - BeautifulSoup may store attributes in slightly different Python types.
# - For example, a class list may be a Python list.
# - We convert values into a stable format so comparisons are reliable.
def normalize_attr_value(value: object) -> object:
    """Convert an attribute value into a stable, comparable format."""

    # Some attributes, like class, may be stored as a list.
    if isinstance(value, list):
        normalized_items: List[str] = []

        # Convert each item to a string and keep the order.
        for item in value:
            normalized_items.append(str(item))

        # Tuples are stable and easy to compare.
        return tuple(normalized_items)

    # Non-list values become plain strings.
    return str(value)


# This function normalizes all attributes on a tag.
#
# Overall purpose:
# - Create one consistent representation for a tag's attributes.
# - Sort attributes so that comparison does not depend on original order.
def normalize_attrs(tag: Tag) -> Tuple[Tuple[str, object], ...]:
    """Return a sorted tuple of normalized attribute key/value pairs."""

    normalized_pairs: List[Tuple[str, object]] = []

    # Turn each attribute into a normalized (name, value) pair.
    for key, value in tag.attrs.items():
        normalized_value = normalize_attr_value(value)
        pair = (key, normalized_value)
        normalized_pairs.append(pair)

    # Sort the list so attribute order differences do not affect validation.
    normalized_pairs.sort()

    # Convert the list to a tuple because tuples are immutable and easy to compare.
    return tuple(normalized_pairs)


# This function builds a structure signature for a tag and all of its children.
#
# Overall purpose:
# - Represent the HTML structure in a form that is easy to compare.
# - If a tag disappears, moves, or changes nesting, the signature will change.
#
# Important note:
# - Text content itself is not fully compared here.
# - The validator focuses on preserving structure and protected text.
def structure_signature(node: Tag) -> Tuple[object, ...]:
    """Return a recursive signature that describes the structure of a tag."""

    signature_parts: List[object] = []

    # Look at each meaningful child of the current node.
    for child in significant_children(node):
        # Text nodes are recorded simply as text placeholders.
        if isinstance(child, NavigableString):
            signature_parts.append(("TEXT",))

        # Tag nodes record more detail: tag name, attributes, and children.
        elif isinstance(child, Tag):
            child_name = child.name
            child_attrs = normalize_attrs(child)
            child_structure = structure_signature(child)

            part = (
                "TAG",
                child_name,
                child_attrs,
                child_structure,
            )
            signature_parts.append(part)

    return tuple(signature_parts)


# This function captures the exact protected bold text in a tag.
#
# Overall purpose:
# - The notebook promises that text inside <b> and <strong> does not change.
# - We therefore collect a signature that includes:
#   1. the protected tag name
#   2. its attributes
#   3. its exact inner HTML
#
# If any of those change, validation should fail.
def protected_signature(node: Tag) -> Tuple[Tuple[str, Tuple[Tuple[str, object], ...], str], ...]:
    """Return a signature for all protected tags inside a node."""

    protected_items: List[Tuple[str, Tuple[Tuple[str, object], ...], str]] = []

    # Walk through every descendant inside the node.
    for descendant in node.descendants:
        # We only care about actual HTML tags that are protected.
        if isinstance(descendant, Tag):
            if descendant.name in PROTECTED_TAGS:
                protected_tag_name = descendant.name
                protected_tag_attrs = normalize_attrs(descendant)
                protected_inner_html = descendant.decode_contents()

                protected_item = (
                    protected_tag_name,
                    protected_tag_attrs,
                    protected_inner_html,
                )
                protected_items.append(protected_item)

    return tuple(protected_items)


# This function checks whether a tag contains another rewrite block inside it.
#
# Overall purpose:
# - We only want to rewrite the smallest safe blocks.
# - Example: if a <li> contains another nested <li>, the outer <li> is not a leaf block.
def contains_descendant_block(tag: Tag) -> bool:
    """Return True if this tag contains another block-level rewrite target inside it."""

    # Search for descendant tags that are also block tags.
    for descendant in tag.find_all(BLOCK_TAGS):
        # Ignore the tag itself. We only care about deeper descendants.
        if descendant is not tag:
            return True

    return False


# This function finds all leaf rewrite targets in the document.
#
# Overall purpose:
# - Rewrite only the smallest blocks, such as individual paragraphs, list items, and table cells.
# - This helps prevent the model from merging list items or damaging tables.
def find_leaf_rewrite_targets(root: Tag) -> List[Tag]:
    """Return rewrite targets that do not contain smaller rewrite targets inside them."""

    targets: List[Tag] = []

    # Find every tag in the fragment that matches our block tag set.
    for tag in root.find_all(BLOCK_TAGS):
        # Skip non-leaf blocks. We only keep the smallest editable units.
        if contains_descendant_block(tag):
            continue

        targets.append(tag)

    return targets


# This function validates whether a rewritten block still respects the original HTML structure.
#
# Overall purpose:
# - Parse both the original and rewritten HTML.
# - Compare the root tag.
# - Compare attributes.
# - Compare the full tag structure.
# - Compare protected bold text.
#
# If anything important changed, return validation errors.
def validate_block_rewrite(original_html: str, candidate_html: str) -> ValidationResult:
    """Validate one rewritten HTML block against the original block."""

    errors: List[str] = []

    # Parse the original block.
    try:
        original_tag = extract_single_tag(original_html)
    except Exception as exc:
        error_message = "Original block could not be parsed: " + str(exc)
        return ValidationResult(False, [error_message])

    # Parse the candidate block produced by the model.
    try:
        candidate_tag = extract_single_tag(candidate_html)
    except Exception as exc:
        error_message = "Candidate block could not be parsed: " + str(exc)
        return ValidationResult(False, [error_message])

    # Check that the root tag name stayed the same, for example <p> remains <p>.
    if original_tag.name != candidate_tag.name:
        message = "Root tag changed from <" + original_tag.name + "> to <" + candidate_tag.name + ">."
        errors.append(message)

    # Check that root tag attributes are unchanged.
    if normalize_attrs(original_tag) != normalize_attrs(candidate_tag):
        errors.append("Root tag attributes changed.")

    # Check that child tags, nesting, and attributes still match exactly.
    if structure_signature(original_tag) != structure_signature(candidate_tag):
        errors.append("Tag structure changed. A tag moved, disappeared, or a new tag was introduced.")

    # Check that protected <b> or <strong> content stayed exactly the same.
    if protected_signature(original_tag) != protected_signature(candidate_tag):
        errors.append("Protected text inside <b> or <strong> changed.")

    # If there are no errors, validation passed.
    if len(errors) == 0:
        return ValidationResult(True, errors)
    else:
        return ValidationResult(False, errors)


# This function removes markdown code fences if the model wraps the answer in them.
#
# Overall purpose:
# - Sometimes models return ```html ... ``` even when told not to.
# - The validator expects plain HTML, so we strip those wrappers first.
def strip_code_fences(text: str) -> str:
    """Remove outer markdown code fences from a model response, if present."""

    # If the model returned nothing, return it unchanged.
    if not text:
        return text

    cleaned_text = text.strip()

    # This pattern matches a fenced code block such as:
    # ```html
    # <p>Hello</p>
    # ```
    fence_match = re.match(
        r"^```(?:html)?\s*(.*?)\s*```$",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # If a fenced block was found, return only its inner contents.
    if fence_match:
        return fence_match.group(1).strip()

    # Otherwise, return the text unchanged.
    return cleaned_text

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


# This function formats the rewrite rules as a numbered list for the prompt.
#
# Overall purpose:
# - LLM prompts are easier to follow when rules are clearly numbered.
def format_rules(rules: Sequence[str]) -> str:
    """Turn a list of rule strings into a numbered text block."""

    lines: List[str] = []
    rule_number = 1

    # Add each rule on its own numbered line.
    for rule in rules:
        line = str(rule_number) + ". " + rule
        lines.append(line)
        rule_number = rule_number + 1

    return "\n".join(lines)


# This function builds the user prompt for one HTML block.
#
# Overall purpose:
# - Include the rewrite rules.
# - Include the current HTML block.
# - Optionally include repair errors from a failed validation attempt.
def build_user_prompt(
    block_html: str,
    rules: Sequence[str],
    repair_errors: Optional[Sequence[str]] = None,
) -> str:
    """Build the user prompt sent to the model for one rewrite attempt."""

    formatted_rules = format_rules(rules)

    base = (
        "Apply the following rules to this HTML block.\n\n"
        "Rules:\n"
        + formatted_rules
        + "\n\n"
        "HTML block:\n"
        + block_html
    )

    # If validation failed earlier, explain the exact problems so the model can repair them.
    if repair_errors:
        repair_lines: List[str] = []

        for error in repair_errors:
            repair_line = "- " + str(error)
            repair_lines.append(repair_line)

        repair_text = "\n".join(repair_lines)

        base = (
            base
            + "\n\n"
            + "Your previous answer failed validation for these reasons:\n"
            + repair_text
            + "\n\n"
            + "Repair the HTML while keeping the same root tag, descendant tags, nesting, and attributes."
        )

    return base


# This function sends one HTML block to Azure OpenAI and returns the model's text.
#
# Overall purpose:
# - Call the chat completions API.
# - Use the deployment name that points to GPT-5.1 in Azure.
# - Return plain HTML text.
def call_model_for_block(
    client: AzureOpenAI,
    deployment_name: str,
    block_html: str,
    rules: Sequence[str],
    repair_errors: Optional[Sequence[str]] = None,
) -> str:
    """Call Azure OpenAI for a single HTML block rewrite."""

    user_prompt = build_user_prompt(block_html, rules, repair_errors)

    # Send the prompt to Azure OpenAI.
    completion = client.chat.completions.create(
        model=deployment_name,
        temperature=0,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Read the assistant's content. If content is missing, use an empty string.
    content = completion.choices[0].message.content or ""

    # Strip any accidental markdown code fences.
    cleaned_content = strip_code_fences(content)

    return cleaned_content


# This function rewrites one block and retries if validation fails.
#
# Overall purpose:
# - Make an initial rewrite attempt.
# - Validate the result.
# - If invalid, tell the model exactly what failed and retry.
# - Return the best final result.
def rewrite_block_with_validation(
    client: AzureOpenAI,
    deployment_name: str,
    block_html: str,
    rules: Sequence[str],
    max_attempts: int = 3,
) -> BlockRewriteResult:
    """Rewrite one HTML block and validate it, retrying if necessary."""

    # Keep track of the latest candidate and validation state.
    last_candidate = block_html
    last_validation = ValidationResult(False, ["The model was not called."])
    repair_errors: Optional[Sequence[str]] = None

    attempt = 1

    # Try up to max_attempts times.
    while attempt <= max_attempts:
        # Ask the model to rewrite the block.
        candidate = call_model_for_block(
            client=client,
            deployment_name=deployment_name,
            block_html=block_html,
            rules=rules,
            repair_errors=repair_errors,
        )

        # Validate the candidate result.
        validation = validate_block_rewrite(block_html, candidate)

        # Save the latest result in case all attempts fail.
        last_candidate = candidate
        last_validation = validation

        # If validation passed, return immediately.
        if validation.is_valid:
            return BlockRewriteResult(
                original_html=block_html,
                rewritten_html=candidate,
                attempts=attempt,
                validation=validation,
            )

        # If validation failed, use the errors to guide the next repair attempt.
        repair_errors = validation.errors
        attempt = attempt + 1

    # If all attempts failed, return the last candidate and its validation errors.
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

# This function is the main entry point for rewriting an HTML fragment.
#
# Overall purpose:
# - Parse the full fragment.
# - Find all safe leaf rewrite blocks.
# - Rewrite each block one by one.
# - Replace only blocks that passed validation.
# - Return the final HTML plus a detailed report.
def transform_html_fragment(
    fragment: str,
    client: AzureOpenAI,
    deployment_name: str,
    rules: Sequence[str],
    max_attempts_per_block: int = 3,
    fail_on_validation_error: bool = True,
) -> Tuple[str, List[Dict[str, object]]]:
    """Rewrite an HTML fragment block by block and return the rewritten HTML plus a report."""

    # Parse the input fragment into a BeautifulSoup tree.
    soup, root = parse_fragment(fragment)

    # We do not use soup later, but keeping it in a named variable makes the parsing step explicit.
    _ = soup

    # Find all leaf rewrite targets such as p, li, th, td, and headings.
    targets = find_leaf_rewrite_targets(root)

    # This report stores one dictionary per block so you can inspect what happened.
    report: List[Dict[str, object]] = []

    index = 1

    # Rewrite each target block separately.
    for tag in targets:
        original_html = str(tag)

        result = rewrite_block_with_validation(
            client=client,
            deployment_name=deployment_name,
            block_html=original_html,
            rules=rules,
            max_attempts=max_attempts_per_block,
        )

        # Build a readable report item for this block.
        report_item: Dict[str, object] = {}
        report_item["index"] = index
        report_item["tag"] = tag.name
        report_item["attempts"] = result.attempts
        report_item["is_valid"] = result.validation.is_valid
        report_item["errors"] = result.validation.errors
        report_item["original_html"] = result.original_html
        report_item["rewritten_html"] = result.rewritten_html
        report.append(report_item)

        # Decide what to do if validation failed.
        if not result.validation.is_valid:
            if fail_on_validation_error:
                message = (
                    "Validation failed for block "
                    + str(index)
                    + " <"
                    + str(tag.name)
                    + ">: "
                    + str(result.validation.errors)
                )
                raise ValueError(message)
            else:
                # If we do not fail fast, we simply leave the original block unchanged.
                index = index + 1
                continue

        # Parse the rewritten HTML and replace the original block in the tree.
        replacement_tag = extract_single_tag(result.rewritten_html)
        tag.replace_with(replacement_tag)

        index = index + 1

    # Convert the modified tree back to a plain HTML fragment string.
    rewritten_fragment = fragment_to_string(root)

    return rewritten_fragment, report

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
