import os
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from docx import Document
from openai import OpenAI
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# Configuration
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5.4"

INPUT_DOCX = "input.docx"
RULES_FILE = "rules.txt"
OUTPUT_DOCX = "output_enriched.docx"

# First pass: paragraph rewriting
PARAGRAPH_BATCH_SIZE = 20
PARAGRAPH_MAX_WORKERS = 4

# Second pass: section cleanup
SECTION_TARGET_CHARS = 6000
SECTION_OVERLAP_CHARS = 500
SECTION_MAX_WORKERS = 3

# Retry behavior
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2


# =========================
# Data structures
# =========================
@dataclass
class ParagraphItem:
    index: int
    text: str
    is_empty: bool


@dataclass
class SectionItem:
    section_id: int
    start_index: int
    end_index: int
    text: str


# =========================
# Client
# =========================
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# File I/O
# =========================
def read_rules(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_docx_paragraphs(path: str) -> List[ParagraphItem]:
    doc = Document(path)
    items: List[ParagraphItem] = []

    for i, p in enumerate(doc.paragraphs):
        text = p.text
        items.append(
            ParagraphItem(
                index=i,
                text=text,
                is_empty=(text.strip() == "")
            )
        )
    return items

from pathlib import Path
from docx import Document
import shutil
import tempfile

def write_docx_from_paragraphs(paragraphs: list[str], output_path: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name

    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(tmp_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp_path, output_path)

def write_docx_from_paragraphs(paragraphs: List[str], output_path: str) -> None:
    out = Document()
    for p in paragraphs:
        out.add_paragraph(p)
    out.save(output_path)


# =========================
# Utility helpers
# =========================
def chunk_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Tries to recover a JSON object even if the model wraps it in extra text.
    """
    text = text.strip()

    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from model output.")


def call_with_retries(fn, *args, **kwargs):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                raise
            sleep_s = min(INITIAL_BACKOFF_SECONDS * (2 ** attempt), 30)
            time.sleep(sleep_s)
    raise last_error


# =========================
# First pass: paragraph-level rewrite
# =========================
def build_paragraph_batch_prompt(rules_text: str, batch: List[ParagraphItem]) -> str:
    payload = [
        {"index": item.index, "text": item.text}
        for item in batch
        if not item.is_empty
    ]

    return f"""
Apply the editorial rules to each paragraph independently.

RULES:
{rules_text}

INSTRUCTIONS:
- Rewrite each paragraph so it complies with the rules.
- Preserve meaning.
- Preserve names, dates, numbers, URLs, and citations unless a rule explicitly requires changes.
- Do not merge paragraphs.
- Do not split paragraphs.
- Return exactly one rewritten paragraph for each input paragraph.
- Return ONLY valid JSON.
- Do not include markdown fences.

Return JSON with this shape:
{{
  "items": [
    {{
      "index": 12,
      "rewritten": "Rewritten paragraph text"
    }}
  ]
}}

PARAGRAPHS:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def request_paragraph_batch(batch: List[ParagraphItem], rules_text: str) -> Dict[str, Any]:
    prompt = build_paragraph_batch_prompt(rules_text, batch)

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an expert editor. Rewrite each paragraph independently. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return extract_json_object(response.output_text)


def validate_paragraph_batch_result(
    batch: List[ParagraphItem], result: Dict[str, Any]
) -> Dict[int, str]:
    expected_indices = {item.index for item in batch if not item.is_empty}
    items = result.get("items")

    if not isinstance(items, list):
        raise ValueError("Result JSON missing 'items' list.")

    rewritten_map: Dict[int, str] = {}

    for obj in items:
        if not isinstance(obj, dict):
            raise ValueError("Each item must be an object.")
        idx = obj.get("index")
        rewritten = obj.get("rewritten")

        if idx not in expected_indices:
            raise ValueError(f"Unexpected paragraph index returned: {idx}")
        if not isinstance(rewritten, str):
            raise ValueError(f"Paragraph {idx} missing rewritten text.")
        rewritten_map[idx] = rewritten

    if set(rewritten_map.keys()) != expected_indices:
        missing = expected_indices - set(rewritten_map.keys())
        raise ValueError(f"Missing rewritten paragraphs for indices: {sorted(missing)}")

    return rewritten_map


def process_paragraph_batch(batch: List[ParagraphItem], rules_text: str) -> Dict[int, str]:
    # Empty paragraphs are preserved without sending to the model
    nonempty = [item for item in batch if not item.is_empty]
    rewritten_map: Dict[int, str] = {item.index: item.text for item in batch if item.is_empty}

    if not nonempty:
        return rewritten_map

    result = call_with_retries(request_paragraph_batch, nonempty, rules_text)
    validated = validate_paragraph_batch_result(nonempty, result)
    rewritten_map.update(validated)
    return rewritten_map


def run_first_pass(paragraphs: List[ParagraphItem], rules_text: str) -> List[str]:
    batches = chunk_list(paragraphs, PARAGRAPH_BATCH_SIZE)
    final_map: Dict[int, str] = {p.index: p.text for p in paragraphs}

    with ThreadPoolExecutor(max_workers=PARAGRAPH_MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_paragraph_batch, batch, rules_text): batch
            for batch in batches
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="First pass"):
            batch_result = future.result()
            final_map.update(batch_result)

    return [final_map[i] for i in range(len(paragraphs))]


# =========================
# Second pass: section-level cleanup
# =========================
def build_sections_from_paragraphs(paragraphs: List[str]) -> List[SectionItem]:
    """
    Build sections from already-rewritten paragraphs.
    We use RecursiveCharacterTextSplitter to group paragraphs into larger blocks
    while trying to preserve paragraph boundaries.
    """
    indexed_paragraphs = []
    for i, p in enumerate(paragraphs):
        indexed_paragraphs.append(f"[[P{i}]]\n{p}")

    full_text = "\n\n".join(indexed_paragraphs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SECTION_TARGET_CHARS,
        chunk_overlap=SECTION_OVERLAP_CHARS,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(full_text)

    sections: List[SectionItem] = []

    for section_id, chunk in enumerate(chunks):
        indices = [int(x) for x in re.findall(r"\[\[P(\d+)\]\]", chunk)]
        if not indices:
            continue

        start_index = min(indices)
        end_index = max(indices)

        # Reconstruct clean section text from paragraph range to avoid partial overlaps
        section_text_parts = []
        for i in range(start_index, end_index + 1):
            section_text_parts.append(f"[[P{i}]]\n{paragraphs[i]}")
        section_text = "\n\n".join(section_text_parts)

        sections.append(
            SectionItem(
                section_id=section_id,
                start_index=start_index,
                end_index=end_index,
                text=section_text,
            )
        )

    # De-duplicate fully repeated ranges caused by splitter overlap
    deduped = []
    seen = set()
    for s in sections:
        key = (s.start_index, s.end_index)
        if key not in seen:
            deduped.append(s)
            seen.add(key)

    return deduped


def build_section_prompt(rules_text: str, section: SectionItem) -> str:
    return f"""
You are performing a second-pass editorial cleanup on a section of a document.

RULES:
{rules_text}

GOAL:
Improve consistency across paragraphs in this section.

FOCUS ON:
- consistent terminology
- consistent tone and voice
- consistent tense where appropriate
- consistent rule application
- smoother transitions where useful
- remove contradictions created by local paragraph rewrites

HARD CONSTRAINTS:
- Keep the same paragraph count.
- Keep paragraph markers exactly unchanged.
- Do not merge paragraphs.
- Do not split paragraphs.
- Preserve meaning and factual content.
- Preserve names, dates, numbers, URLs, and citations unless a rule explicitly requires changes.
- Return ONLY the revised section text.
- Every paragraph must remain prefixed by its marker like [[P12]] on its own line.

SECTION TEXT:
{section.text}
""".strip()


def request_section_cleanup(section: SectionItem, rules_text: str) -> str:
    prompt = build_section_prompt(rules_text, section)

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an expert editor doing a section-level consistency pass. "
                    "Return only the revised section text."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return response.output_text.strip()


def parse_section_output(section_text: str) -> Dict[int, str]:
    """
    Parse output in the format:

    [[P10]]
    Paragraph text...

    [[P11]]
    Next paragraph...
    """
    matches = list(re.finditer(r"\[\[P(\d+)\]\]\s*\n", section_text))
    if not matches:
        raise ValueError("No paragraph markers found in section output.")

    result: Dict[int, str] = {}

    for i, match in enumerate(matches):
        idx = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        para_text = section_text[start:end].strip()
        result[idx] = para_text

    return result


def process_section(section: SectionItem, rules_text: str) -> Dict[int, str]:
    cleaned = call_with_retries(request_section_cleanup, section, rules_text)
    parsed = parse_section_output(cleaned)

    expected = set(range(section.start_index, section.end_index + 1))
    returned = set(parsed.keys())

    if expected != returned:
        raise ValueError(
            f"Section {section.section_id} returned wrong paragraph set. "
            f"Expected {sorted(expected)}, got {sorted(returned)}"
        )

    return parsed


def run_second_pass(paragraphs: List[str], rules_text: str) -> List[str]:
    sections = build_sections_from_paragraphs(paragraphs)

    # A paragraph can appear in multiple overlapping sections.
    # We keep the latest successful rewrite, but you could also
    # make this smarter with voting or confidence tracking.
    revised_map: Dict[int, str] = {i: p for i, p in enumerate(paragraphs)}

    with ThreadPoolExecutor(max_workers=SECTION_MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_section, section, rules_text): section
            for section in sections
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Second pass"):
            try:
                section_result = future.result()
                revised_map.update(section_result)
            except Exception as e:
                # Safe fallback: keep first-pass paragraphs for failed sections
                print(f"[WARN] Section cleanup failed: {e}")

    return [revised_map[i] for i in range(len(paragraphs))]


# =========================
# Main
# =========================
def main():
    rules_text = read_rules(RULES_FILE)
    paragraph_items = read_docx_paragraphs(INPUT_DOCX)

    print(f"Loaded {len(paragraph_items)} paragraphs from {INPUT_DOCX}")

    # First pass
    first_pass_paragraphs = run_first_pass(paragraph_items, rules_text)

    # Second pass
    final_paragraphs = run_second_pass(first_pass_paragraphs, rules_text)

    # Save
    write_docx_from_paragraphs(final_paragraphs, OUTPUT_DOCX)
    print(f"Done. Wrote {OUTPUT_DOCX}")


if __name__ == "__main__":

    main()
