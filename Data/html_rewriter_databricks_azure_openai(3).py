# Databricks notebook source
# MAGIC %md
# MAGIC # HTML document rewriter with Azure OpenAI (GPT-5.1) using hybrid markers
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
# MAGIC The notebook uses a **hybrid marker** pattern:
# MAGIC
# MAGIC 1. Parse the fragment.
# MAGIC 2. Find leaf rewrite blocks such as `p`, `li`, `th`, `td`, and headings.
# MAGIC 3. Convert inline HTML inside each block into visible text plus control markers.
# MAGIC 4. Send the marker text to Azure OpenAI.
# MAGIC 5. Validate that all markers are preserved and balanced.
# MAGIC 6. Rebuild the original HTML tags from the markers.
# MAGIC 7. Replace only blocks that pass validation.
# MAGIC
# MAGIC Why use markers instead of raw HTML?
# MAGIC
# MAGIC - The LLM still sees the visible text inside inline tags.
# MAGIC - Your Python code, not the LLM, is responsible for rebuilding the HTML tags.
# MAGIC - This is often more reliable for stronger rewriting rules such as B1 language,
# MAGIC   concision, and active voice.
# MAGIC
# MAGIC This notebook is intentionally written in a **beginner-friendly style**:
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
    "Do not modify any text inside bold tags, such as <b> and <strong>.",
]

STYLE_RULES

# COMMAND ----------
# MAGIC %md
# MAGIC ## HTML utilities
# MAGIC
# MAGIC The validator below enforces the main invariants:
# MAGIC
# MAGIC - the reconstructed block must still parse as HTML
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
VOID_TAGS = {"br", "img", "hr", "input", "source", "track", "wbr", "meta", "link"}
MARKER_PATTERN = re.compile(r"⟦[^⟧]+⟧")


# These small data classes make the output easier to read.
#
# ValidationResult answers one question:
# - Did validation pass?
# - If not, what errors were found?
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]


# MarkerInfo stores how one marker maps back to the original HTML.
#
# Overall purpose:
# - Keep the original tag name and attributes.
# - Remember whether the marker is protected.
# - Remember the exact marker tokens used in the prompt.
@dataclass
class MarkerInfo:
    marker_id: str
    tag_name: str
    attrs: Dict[str, object]
    is_protected: bool
    is_empty: bool
    single_token: str
    start_token: str
    end_token: str


# MarkerizedBlock stores everything we need for one rewrite unit.
#
# Overall purpose:
# - Keep the original HTML block.
# - Keep the markerized text sent to the model.
# - Keep the marker map used for reconstruction.
@dataclass
class MarkerizedBlock:
    original_html: str
    marker_text: str
    marker_map: Dict[str, MarkerInfo]


# BlockRewriteResult stores the full story for one rewritten HTML block.
#
# It keeps:
# - the original block HTML
# - the marker text sent to the model
# - the marker text returned by the model
# - the reconstructed HTML block
# - how many attempts were needed
# - the final validation result
@dataclass
class BlockRewriteResult:
    original_html: str
    input_marker_text: str
    output_marker_text: str
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
# - If the fragment returns extra text outside that tag, validation should fail.
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
        raise ValueError("Fragment contains significant text outside the root tag.")

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


# This helper function copies HTML attributes into a plain Python dictionary.
#
# Overall purpose:
# - Keep attribute values stable and easy to reuse later.
# - Avoid sharing the original BeautifulSoup attribute object directly.
def copy_attrs(attrs: Dict[str, object]) -> Dict[str, object]:
    """Return a plain copied dictionary of HTML attributes."""

    new_attrs: Dict[str, object] = {}

    # Copy one attribute at a time.
    for key, value in attrs.items():
        # If the value is a list, copy each item into a new list.
        if isinstance(value, list):
            copied_list: List[object] = []

            for item in value:
                copied_list.append(item)

            new_attrs[key] = copied_list
        else:
            new_attrs[key] = value

    return new_attrs


# This helper function creates a new empty tag with the same name and attributes.
#
# Overall purpose:
# - Rebuild HTML tags from markers without copying their old child nodes.
def clone_empty_tag(builder_soup: BeautifulSoup, tag_name: str, attrs: Dict[str, object]) -> Tag:
    """Create a new empty BeautifulSoup tag with copied attributes."""

    # Create the new tag first.
    new_tag = builder_soup.new_tag(tag_name)

    # Copy attributes one at a time.
    for key, value in attrs.items():
        if isinstance(value, list):
            copied_list: List[object] = []

            for item in value:
                copied_list.append(item)

            new_tag.attrs[key] = copied_list
        else:
            new_tag.attrs[key] = value

    return new_tag


# This helper function checks whether a tag should be treated as empty.
#
# Overall purpose:
# - Some HTML tags such as <br> and <img> do not wrap text.
# - We preserve them as single markers instead of start/end pairs.
def is_empty_inline_tag(tag: Tag) -> bool:
    """Return True if the tag should be represented by one atomic marker."""

    # HTML void tags are always treated as atomic markers.
    if tag.name in VOID_TAGS:
        return True

    # Tags with no children and no text are also treated as atomic markers.
    if len(tag.contents) == 0:
        return True

    return False


# These helper functions build marker strings.
#
# Overall purpose:
# - Use one consistent marker format everywhere in the notebook.
def build_start_token(marker_id: str) -> str:
    """Return a START marker token for one marker id."""

    return "⟦" + marker_id + "_START⟧"


def build_end_token(marker_id: str) -> str:
    """Return an END marker token for one marker id."""

    return "⟦" + marker_id + "_END⟧"


def build_single_token(marker_id: str) -> str:
    """Return a single atomic marker token for one marker id."""

    return "⟦" + marker_id + "⟧"


# This recursive function walks through a tag's children and converts inline tags into markers.
#
# Overall purpose:
# - Keep visible text visible to the LLM.
# - Replace the HTML boundaries with markers that Python can later rebuild.
# - Protected tags such as <b> and <strong> become KEEP markers.
# - Empty inline tags such as <br> become one atomic marker.
def markerize_children(
    parent_tag: Tag,
    output_parts: List[str],
    marker_map: Dict[str, MarkerInfo],
    state: Dict[str, int],
) -> None:
    """Convert the children of one tag into marker text and fill marker_map."""

    # Loop through every direct child of the parent tag.
    for child in parent_tag.contents:
        # Ignore HTML comments in the rewrite input.
        if isinstance(child, Comment):
            continue

        # Plain text stays as plain text.
        if isinstance(child, NavigableString):
            output_parts.append(str(child))
            continue

        # At this point we have a normal HTML tag.
        if isinstance(child, Tag):
            # Increase the shared counter so every marker id is unique.
            state["counter"] = state["counter"] + 1
            marker_number = state["counter"]

            # Decide which kind of marker we need.
            if child.name in PROTECTED_TAGS:
                marker_id = "KEEP_" + str(marker_number)
                start_token = build_start_token(marker_id)
                end_token = build_end_token(marker_id)
                single_token = ""

                info = MarkerInfo(
                    marker_id=marker_id,
                    tag_name=child.name,
                    attrs=copy_attrs(child.attrs),
                    is_protected=True,
                    is_empty=False,
                    single_token=single_token,
                    start_token=start_token,
                    end_token=end_token,
                )
                marker_map[marker_id] = info

                # Add the start marker, then recurse into the protected tag,
                # then add the end marker.
                output_parts.append(start_token)
                markerize_children(child, output_parts, marker_map, state)
                output_parts.append(end_token)
                continue

            if is_empty_inline_tag(child):
                marker_id = "EMPTY_" + str(marker_number)
                single_token = build_single_token(marker_id)
                start_token = ""
                end_token = ""

                info = MarkerInfo(
                    marker_id=marker_id,
                    tag_name=child.name,
                    attrs=copy_attrs(child.attrs),
                    is_protected=False,
                    is_empty=True,
                    single_token=single_token,
                    start_token=start_token,
                    end_token=end_token,
                )
                marker_map[marker_id] = info

                # Add one atomic marker to represent the empty tag.
                output_parts.append(single_token)
                continue

            # Normal editable inline tag.
            marker_id = "TAG_" + str(marker_number)
            start_token = build_start_token(marker_id)
            end_token = build_end_token(marker_id)
            single_token = ""

            info = MarkerInfo(
                marker_id=marker_id,
                tag_name=child.name,
                attrs=copy_attrs(child.attrs),
                is_protected=False,
                is_empty=False,
                single_token=single_token,
                start_token=start_token,
                end_token=end_token,
            )
            marker_map[marker_id] = info

            # Add the start marker, recurse into the tag's children, then add the end marker.
            output_parts.append(start_token)
            markerize_children(child, output_parts, marker_map, state)
            output_parts.append(end_token)


# This function creates the marker-based rewrite input for one block.
#
# Overall purpose:
# - Take one leaf block such as <p> or <td>.
# - Convert only its inner content into marker text.
# - Keep the outer block tag under Python's control.
def markerize_block(tag: Tag) -> MarkerizedBlock:
    """Return the original block HTML plus marker text and marker metadata."""

    marker_map: Dict[str, MarkerInfo] = {}
    output_parts: List[str] = []
    state: Dict[str, int] = {}
    state["counter"] = 0

    # Convert the block's children into marker text.
    markerize_children(tag, output_parts, marker_map, state)

    marker_text = "".join(output_parts)
    original_html = str(tag)

    return MarkerizedBlock(
        original_html=original_html,
        marker_text=marker_text,
        marker_map=marker_map,
    )


# This function splits text into alternating text segments and marker tokens.
#
# Overall purpose:
# - Make it easier to validate and rebuild marker text.
# - Keep markers and normal text separate.
def tokenize_marker_text(text: str) -> List[str]:
    """Return a list containing text segments and marker tokens in order."""

    parts: List[str] = []
    position = 0

    # Find every marker token such as ⟦TAG_1_START⟧.
    for match in MARKER_PATTERN.finditer(text):
        start_index = match.start()
        end_index = match.end()

        # Add the plain text before the marker, if any.
        if start_index > position:
            plain_text = text[position:start_index]
            parts.append(plain_text)

        # Add the marker token itself.
        marker_token = match.group(0)
        parts.append(marker_token)

        # Move the current position to the end of the marker.
        position = end_index

    # Add any remaining plain text after the last marker.
    if position < len(text):
        remaining_text = text[position:]
        parts.append(remaining_text)

    return parts


# This helper function checks whether a piece of text is a marker token.
#
# Overall purpose:
# - Distinguish marker tokens from ordinary plain text segments.
def is_marker_token(text: str) -> bool:
    """Return True if the text looks like a full marker token."""

    if text.startswith("⟦") and text.endswith("⟧"):
        return True
    else:
        return False


# This function builds a readable marker summary for the prompt.
#
# Overall purpose:
# - Show the model which markers exist in the current block.
# - Explain which markers are editable and which are protected.
def describe_markers(marker_map: Dict[str, MarkerInfo]) -> str:
    """Return a human-readable list of marker descriptions."""

    lines: List[str] = []

    # Sort marker ids numerically as strings for stable output.
    sorted_ids = list(marker_map.keys())
    sorted_ids.sort()

    for marker_id in sorted_ids:
        info = marker_map[marker_id]

        if info.is_empty:
            line = (
                "- "
                + info.single_token
                + " = empty <"
                + info.tag_name
                + "> tag. Keep it exactly as written."
            )
            lines.append(line)
            continue

        if info.is_protected:
            line = (
                "- "
                + info.start_token
                + " ... "
                + info.end_token
                + " = protected <"
                + info.tag_name
                + "> tag. Keep these markers and keep the text inside them exactly unchanged."
            )
            lines.append(line)
            continue

        line = (
            "- "
            + info.start_token
            + " ... "
            + info.end_token
            + " = editable <"
            + info.tag_name
            + "> tag. Keep both markers exactly."
        )
        lines.append(line)

    if len(lines) == 0:
        return "- This block contains no inline markers."

    return "\n".join(lines)


# This function validates the raw marker set.
#
# Overall purpose:
# - Check whether the model removed, duplicated, or invented markers.
# - This is the first line of defense before rebuilding any HTML.
def validate_marker_inventory(candidate_text: str, marker_map: Dict[str, MarkerInfo]) -> List[str]:
    """Return a list of inventory errors for the candidate marker text."""

    errors: List[str] = []
    expected_counts: Dict[str, int] = {}

    # Build the expected token counts from the marker map.
    for marker_id, info in marker_map.items():
        if info.is_empty:
            expected_counts[info.single_token] = 1
        else:
            expected_counts[info.start_token] = 1
            expected_counts[info.end_token] = 1

    actual_counts: Dict[str, int] = {}
    parts = tokenize_marker_text(candidate_text)

    # Count the actual marker tokens returned by the model.
    for part in parts:
        if is_marker_token(part):
            current_count = actual_counts.get(part, 0)
            actual_counts[part] = current_count + 1

    # Check for missing or duplicated expected markers.
    for token, expected_count in expected_counts.items():
        actual_count = actual_counts.get(token, 0)
        if actual_count != expected_count:
            message = (
                "Marker inventory problem for "
                + token
                + ": expected "
                + str(expected_count)
                + ", got "
                + str(actual_count)
                + "."
            )
            errors.append(message)

    # Check for completely unexpected markers.
    for token in actual_counts.keys():
        if token not in expected_counts:
            message = "Unexpected marker found: " + token
            errors.append(message)

    return errors


# This helper function returns a marker id from a marker token.
#
# Overall purpose:
# - Convert ⟦TAG_1_START⟧ into TAG_1
# - Convert ⟦TAG_1_END⟧ into TAG_1
# - Convert ⟦EMPTY_3⟧ into EMPTY_3
#
# This helps the balancing logic work with plain marker ids instead of full token strings.
def marker_id_from_token(token: str) -> str:
    """Return the marker id for a full marker token."""

    # Remove the outer marker brackets first.
    inner = token[1:-1]

    # For paired markers, drop the _START or _END suffix.
    if inner.endswith("_START"):
        return inner[:-6]
    if inner.endswith("_END"):
        return inner[:-4]

    # Atomic markers keep their whole inner text as the id.
    return inner


# This function checks that paired markers are balanced and nested correctly.
#
# Overall purpose:
# - Every START marker must have a matching END marker.
# - Markers must close in the correct order.
# - This prevents broken reconstruction.
def validate_marker_balance(candidate_text: str, marker_map: Dict[str, MarkerInfo]) -> List[str]:
    """Return a list of balance errors for the candidate marker text."""

    errors: List[str] = []
    open_stack: List[str] = []
    parts = tokenize_marker_text(candidate_text)

    # Build quick lookup sets for start and end tokens.
    start_tokens: Dict[str, str] = {}
    end_tokens: Dict[str, str] = {}

    for marker_id, info in marker_map.items():
        if not info.is_empty:
            start_tokens[info.start_token] = marker_id
            end_tokens[info.end_token] = marker_id

    # Walk through the token stream in order.
    for part in parts:
        if not is_marker_token(part):
            continue

        # Empty markers do not affect nesting.
        if part in start_tokens:
            open_stack.append(start_tokens[part])
            continue

        if part in end_tokens:
            closing_id = end_tokens[part]

            # If nothing is open, this end marker is invalid.
            if len(open_stack) == 0:
                errors.append("Closing marker appears without a matching start marker: " + part)
                continue

            last_open_id = open_stack[-1]

            # Markers must close in last-opened-first-closed order.
            if last_open_id != closing_id:
                message = (
                    "Markers are not balanced correctly. Tried to close "
                    + closing_id
                    + " while "
                    + last_open_id
                    + " was still open."
                )
                errors.append(message)
                continue

            # Remove the correctly closed marker.
            open_stack.pop()

    # Any markers left open at the end are an error.
    if len(open_stack) > 0:
        for marker_id in open_stack:
            errors.append("Marker was opened but not closed: " + marker_id)

    return errors


# This helper function extracts the exact inner segment between one START/END marker pair.
#
# Overall purpose:
# - Compare protected regions before and after rewriting.
# - We keep the entire inner segment as a string so nested markers are included too.
def extract_inner_segment(text: str, start_token: str, end_token: str) -> str:
    """Return the exact string between one marker pair."""

    parts = tokenize_marker_text(text)
    collecting = False
    collected_parts: List[str] = []

    for part in parts:
        if part == start_token:
            collecting = True
            continue

        if part == end_token:
            break

        if collecting:
            collected_parts.append(part)

    return "".join(collected_parts)


# This function checks that protected bold content stayed exactly unchanged.
#
# Overall purpose:
# - The text inside KEEP markers must not change.
# - This includes nested markers and plain text inside the protected region.
def validate_protected_content(
    original_marker_text: str,
    candidate_marker_text: str,
    marker_map: Dict[str, MarkerInfo],
) -> List[str]:
    """Return a list of errors if protected regions changed."""

    errors: List[str] = []

    # Check every protected marker pair one by one.
    for marker_id, info in marker_map.items():
        if not info.is_protected:
            continue

        original_segment = extract_inner_segment(
            original_marker_text,
            info.start_token,
            info.end_token,
        )
        candidate_segment = extract_inner_segment(
            candidate_marker_text,
            info.start_token,
            info.end_token,
        )

        if original_segment != candidate_segment:
            message = (
                "Protected text changed inside marker pair "
                + info.start_token
                + " ... "
                + info.end_token
                + "."
            )
            errors.append(message)

    return errors


# This function rebuilds one HTML block from marker text.
#
# Overall purpose:
# - Create a fresh copy of the original root tag.
# - Rebuild its child nodes from the marker text.
# - Use the marker map to restore the original tag names and attributes.
def rebuild_block_from_markers(
    original_html: str,
    candidate_marker_text: str,
    marker_map: Dict[str, MarkerInfo],
) -> str:
    """Rebuild one HTML block from validated marker text."""

    # Parse the original block so we can copy its root tag name and attributes.
    original_tag = extract_single_tag(original_html)

    # Create a small helper soup that can manufacture new tags.
    builder_soup = BeautifulSoup("", "lxml")

    # Create the new root tag.
    new_root = clone_empty_tag(builder_soup, original_tag.name, copy_attrs(original_tag.attrs))

    # Prepare token lookup dictionaries.
    start_lookup: Dict[str, MarkerInfo] = {}
    end_lookup: Dict[str, MarkerInfo] = {}
    single_lookup: Dict[str, MarkerInfo] = {}

    for marker_id, info in marker_map.items():
        if info.is_empty:
            single_lookup[info.single_token] = info
        else:
            start_lookup[info.start_token] = info
            end_lookup[info.end_token] = info

    # The stack tracks the currently open reconstructed tags.
    tag_stack: List[Tag] = [new_root]
    id_stack: List[str] = []

    parts = tokenize_marker_text(candidate_marker_text)

    for part in parts:
        # Plain text becomes a text node inside the current open tag.
        if not is_marker_token(part):
            if part != "":
                text_node = NavigableString(part)
                tag_stack[-1].append(text_node)
            continue

        # Empty markers create one empty tag and append it immediately.
        if part in single_lookup:
            info = single_lookup[part]
            empty_tag = clone_empty_tag(builder_soup, info.tag_name, copy_attrs(info.attrs))
            tag_stack[-1].append(empty_tag)
            continue

        # Start markers open a new tag and push it on the stack.
        if part in start_lookup:
            info = start_lookup[part]
            new_child_tag = clone_empty_tag(builder_soup, info.tag_name, copy_attrs(info.attrs))
            tag_stack[-1].append(new_child_tag)
            tag_stack.append(new_child_tag)
            id_stack.append(info.marker_id)
            continue

        # End markers close the most recent open marker tag.
        if part in end_lookup:
            info = end_lookup[part]

            if len(id_stack) == 0:
                raise ValueError("Tried to close a marker when no marker tag was open.")

            current_open_id = id_stack[-1]
            if current_open_id != info.marker_id:
                raise ValueError(
                    "Tried to close marker "
                    + info.marker_id
                    + " while marker "
                    + current_open_id
                    + " was still open."
                )

            id_stack.pop()
            tag_stack.pop()
            continue

        # Any other marker token is unexpected.
        raise ValueError("Unknown marker token found during reconstruction: " + part)

    # All paired markers should be closed by the end.
    if len(id_stack) > 0:
        raise ValueError("One or more markers were left open during reconstruction.")

    return str(new_root)


# This function validates whether a reconstructed block still respects the original HTML structure.
#
# Overall purpose:
# - Parse both the original and reconstructed HTML.
# - Compare the root tag.
# - Compare attributes.
# - Compare the full tag structure.
# - Compare protected bold text.
#
# If anything important changed, return validation errors.
def validate_reconstructed_block(original_html: str, candidate_html: str) -> ValidationResult:
    """Validate one reconstructed HTML block against the original block."""

    errors: List[str] = []

    # Parse the original block.
    try:
        original_tag = extract_single_tag(original_html)
    except Exception as exc:
        error_message = "Original block could not be parsed: " + str(exc)
        return ValidationResult(False, [error_message])

    # Parse the candidate block produced after reconstruction.
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


# This function validates one marker-based model output end to end.
#
# Overall purpose:
# - Check marker inventory.
# - Check marker balance.
# - Check protected content.
# - Rebuild the HTML block.
# - Check the reconstructed block structure.
#
# The returned tuple contains:
# - the validation result
# - the reconstructed HTML block if reconstruction succeeded
# - an empty string if reconstruction failed
#
def validate_marker_output(
    original_html: str,
    original_marker_text: str,
    candidate_marker_text: str,
    marker_map: Dict[str, MarkerInfo],
) -> Tuple[ValidationResult, str]:
    """Validate marker output and return the reconstructed block if valid."""

    errors: List[str] = []

    # Step 1: marker inventory.
    inventory_errors = validate_marker_inventory(candidate_marker_text, marker_map)
    for error in inventory_errors:
        errors.append(error)

    # Step 2: marker balance.
    balance_errors = validate_marker_balance(candidate_marker_text, marker_map)
    for error in balance_errors:
        errors.append(error)

    # Step 3: protected content.
    protected_errors = validate_protected_content(
        original_marker_text,
        candidate_marker_text,
        marker_map,
    )
    for error in protected_errors:
        errors.append(error)

    # If marker validation already failed, stop before reconstruction.
    if len(errors) > 0:
        return ValidationResult(False, errors), ""

    # Step 4: rebuild the HTML block.
    try:
        rebuilt_html = rebuild_block_from_markers(
            original_html,
            candidate_marker_text,
            marker_map,
        )
    except Exception as exc:
        error_message = "HTML reconstruction failed: " + str(exc)
        errors.append(error_message)
        return ValidationResult(False, errors), ""

    # Step 5: validate the rebuilt HTML block.
    reconstructed_validation = validate_reconstructed_block(original_html, rebuilt_html)

    if not reconstructed_validation.is_valid:
        for error in reconstructed_validation.errors:
            errors.append(error)
        return ValidationResult(False, errors), rebuilt_html

    return ValidationResult(True, []), rebuilt_html


# This function removes markdown code fences if the model wraps the answer in them.
#
# Overall purpose:
# - Sometimes models return ```text ... ``` even when told not to.
# - The validator expects plain marker text, so we strip those wrappers first.
def strip_code_fences(text: str) -> str:
    """Remove outer markdown code fences from a model response, if present."""

    # If the model returned nothing, return it unchanged.
    if not text:
        return text

    cleaned_text = text.strip()

    # This pattern matches a fenced code block such as:
    # ```text
    # Hello
    # ```
    fence_match = re.match(
        r"^```(?:text|html)?\s*(.*?)\s*```$",
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
# MAGIC The prompt asks the model to rewrite **marker text**, not raw HTML.
# MAGIC The model still sees the visible words from inline tags, but your Python code keeps control of the actual tags.

# COMMAND ----------
REWRITE_SYSTEM_PROMPT = """
You rewrite the inner content of one HTML block according to style rules.

You do NOT receive raw HTML. Instead, you receive plain text plus control markers.

Non-negotiable requirements:
- Return rewritten marker text only.
- Do not output HTML.
- Do not output explanations.
- Do not wrap the answer in markdown fences.
- Keep every marker token exactly as written.
- Do not add markers.
- Do not remove markers.
- Do not rename markers.
- Do not duplicate markers.
- Keep START and END markers balanced.
- Do not cross marker pairs.
- Text inside KEEP markers is protected and must stay exactly unchanged.
- Text inside normal TAG markers may be rewritten.
- Empty markers such as ⟦EMPTY_1⟧ must remain exactly unchanged.
- Preserve the original meaning unless an explicit style rule requires simplification.
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


# This function builds the user prompt for one markerized block.
#
# Overall purpose:
# - Include the rewrite rules.
# - Include the current markerized block text.
# - Include a marker summary.
# - Optionally include repair errors from a failed validation attempt.
def build_user_prompt(
    marker_text: str,
    marker_map: Dict[str, MarkerInfo],
    rules: Sequence[str],
    block_tag_name: str,
    repair_errors: Optional[Sequence[str]] = None,
) -> str:
    """Build the user prompt sent to the model for one rewrite attempt."""

    formatted_rules = format_rules(rules)
    marker_description = describe_markers(marker_map)

    base = (
        "Apply the following rules to the inner content of this <"
        + block_tag_name
        + "> block.\n\n"
        + "Rules:\n"
        + formatted_rules
        + "\n\n"
        + "Marker guide:\n"
        + marker_description
        + "\n\n"
        + "Marker text to rewrite:\n"
        + marker_text
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
            + "Repair the marker text. Keep all markers exact and balanced."
        )

    return base


# This function sends one markerized block to Azure OpenAI and returns the model's text.
#
# Overall purpose:
# - Call the chat completions API.
# - Use the deployment name that points to GPT-5.1 in Azure.
# - Return plain marker text.
def call_model_for_block(
    client: AzureOpenAI,
    deployment_name: str,
    marker_text: str,
    marker_map: Dict[str, MarkerInfo],
    rules: Sequence[str],
    block_tag_name: str,
    repair_errors: Optional[Sequence[str]] = None,
) -> str:
    """Call Azure OpenAI for a single markerized block rewrite."""

    user_prompt = build_user_prompt(
        marker_text=marker_text,
        marker_map=marker_map,
        rules=rules,
        block_tag_name=block_tag_name,
        repair_errors=repair_errors,
    )

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
# - Markerize the block.
# - Make an initial rewrite attempt.
# - Validate the marker output.
# - If invalid, tell the model exactly what failed and retry.
# - Return the best final result.
def rewrite_block_with_validation(
    client: AzureOpenAI,
    deployment_name: str,
    block_tag: Tag,
    rules: Sequence[str],
    max_attempts: int = 3,
) -> BlockRewriteResult:
    """Rewrite one HTML block using markers and validate it, retrying if necessary."""

    markerized_block = markerize_block(block_tag)

    # Keep track of the latest candidate and validation state.
    last_candidate_marker_text = markerized_block.marker_text
    last_rewritten_html = markerized_block.original_html
    last_validation = ValidationResult(False, ["The model was not called."])
    repair_errors: Optional[Sequence[str]] = None

    attempt = 1

    # Try up to max_attempts times.
    while attempt <= max_attempts:
        # Ask the model to rewrite the marker text.
        candidate_marker_text = call_model_for_block(
            client=client,
            deployment_name=deployment_name,
            marker_text=markerized_block.marker_text,
            marker_map=markerized_block.marker_map,
            rules=rules,
            block_tag_name=block_tag.name,
            repair_errors=repair_errors,
        )

        # Validate the candidate marker text and rebuild HTML if possible.
        validation, rebuilt_html = validate_marker_output(
            original_html=markerized_block.original_html,
            original_marker_text=markerized_block.marker_text,
            candidate_marker_text=candidate_marker_text,
            marker_map=markerized_block.marker_map,
        )

        # Save the latest result in case all attempts fail.
        last_candidate_marker_text = candidate_marker_text
        last_validation = validation
        if rebuilt_html:
            last_rewritten_html = rebuilt_html

        # If validation passed, return immediately.
        if validation.is_valid:
            return BlockRewriteResult(
                original_html=markerized_block.original_html,
                input_marker_text=markerized_block.marker_text,
                output_marker_text=candidate_marker_text,
                rewritten_html=rebuilt_html,
                attempts=attempt,
                validation=validation,
            )

        # If validation failed, use the errors to guide the next repair attempt.
        repair_errors = validation.errors
        attempt = attempt + 1

    # If all attempts failed, return the last candidate and its validation errors.
    return BlockRewriteResult(
        original_html=markerized_block.original_html,
        input_marker_text=markerized_block.marker_text,
        output_marker_text=last_candidate_marker_text,
        rewritten_html=last_rewritten_html,
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
        result = rewrite_block_with_validation(
            client=client,
            deployment_name=deployment_name,
            block_tag=tag,
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
        report_item["input_marker_text"] = result.input_marker_text
        report_item["output_marker_text"] = result.output_marker_text
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

        # Parse the reconstructed HTML and replace the original block in the tree.
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
# MAGIC ## Preview marker text for one block
# MAGIC
# MAGIC This cell helps you understand what the LLM actually receives.

# COMMAND ----------
preview_soup, preview_root = parse_fragment(example_html)
preview_targets = find_leaf_rewrite_targets(preview_root)
first_preview_target = preview_targets[0]
preview_markerized = markerize_block(first_preview_target)

print("Original block HTML:")
print(preview_markerized.original_html)
print()
print("Marker text sent to the model:")
print(preview_markerized.marker_text)
print()
print("Marker guide:")
print(describe_markers(preview_markerized.marker_map))

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
# MAGIC 4. If you want to preserve additional immutable tags, add them to `PROTECTED_TAGS`.
# MAGIC 5. If you want more block types, add them to `BLOCK_TAGS`.
# MAGIC 6. If you need custom handling for more empty inline tags, add them to `VOID_TAGS`.
