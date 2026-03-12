# Databricks notebook source
# COMMAND ----------
# Run once if needed, then restart Python from Databricks UI if prompted.
# %pip install -q google-genai beautifulsoup4 lxml nest_asyncio pandas

# COMMAND ----------
import os
import re
import time
import uuid
import random
import asyncio
from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd
from bs4 import BeautifulSoup, Comment
from pyspark.sql import functions as F

try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

from google import genai
from google.genai import types

# COMMAND ----------
# ---------------------------
# 1) Configuration
# ---------------------------
TABLE_NAME = "default.kb_poc_docs"

ID_COL = "doc_id"          # if missing, notebook auto-generates one
TEXT_COL = "html_text"     # you said your text column name is html_text
CAT_COL = "category"       # optional; defaults to "internal" if missing

N_DOCS = 30                # number of docs to benchmark
MIN_TEXT_CHARS = 2000      # use longer docs to make chunk parallelism visible

# Model/API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# If env var is not set, paste directly below:
# GEMINI_API_KEY = "YOUR_GEMINI_KEY"

MODEL = "gemini-1.5-flash"   # fast + cheaper for benchmarking
TEMPERATURE = 0.1

# Protected tags: text inside these tags is never rewritten
PROTECTED_TAGS = {"b", "strong"}

# Chunking settings
MAX_SEGMENT_CHARS = 12000
CHUNK_TARGET_CHARS = 3500
CHUNK_HARD_LIMIT_CHARS = 5000
HARMONIZE_GROUP_CHARS = 8000

# Retry settings
MAX_RETRIES = 4
RETRY_BASE_SECONDS = 1.5

# Parallel benchmark variants
EXPERIMENTS = [
    {"variant_id": "A_seq", "max_doc_concurrency": 1, "max_request_concurrency": 1},
    {"variant_id": "B_chunk_parallel", "max_doc_concurrency": 1, "max_request_concurrency": 4},
    {"variant_id": "C_doc_and_chunk_parallel", "max_doc_concurrency": 3, "max_request_concurrency": 8},
]

assert GEMINI_API_KEY, "Set GEMINI_API_KEY first."
client = genai.Client(api_key=GEMINI_API_KEY)

# COMMAND ----------
# ---------------------------
# 2) Load sample docs
# ---------------------------
df = spark.table(TABLE_NAME)
cols = set(df.columns)

if TEXT_COL not in cols:
    raise ValueError(f"Column '{TEXT_COL}' not found in table {TABLE_NAME}. Available columns: {sorted(cols)}")

select_cols = []
if ID_COL in cols:
    select_cols.append(F.col(ID_COL).cast("string").alias("doc_id"))
else:
    select_cols.append(F.monotonically_increasing_id().cast("string").alias("doc_id"))

select_cols.append(F.col(TEXT_COL).cast("string").alias("html_text"))

if CAT_COL in cols:
    select_cols.append(F.col(CAT_COL).cast("string").alias("category"))
else:
    select_cols.append(F.lit("internal").alias("category"))

sample_df = (
    df.select(*select_cols)
      .filter(F.col("html_text").isNotNull())
      .withColumn("char_len", F.length("html_text"))
      .filter(F.col("char_len") >= MIN_TEXT_CHARS)
      .orderBy(F.rand(42))
      .limit(N_DOCS)
)

rows = sample_df.collect()
docs: List[Tuple[str, str, str]] = [
    (r["doc_id"], r["html_text"], r["category"] or "internal")
    for r in rows
]

if not docs:
    raise ValueError("No docs matched MIN_TEXT_CHARS filter. Lower MIN_TEXT_CHARS and rerun.")

print(f"Loaded {len(docs)} docs for benchmark.")
display(sample_df.select("doc_id", "category", "char_len"))

# COMMAND ----------
# ---------------------------
# 3) HTML + chunking helpers
# ---------------------------
class LLMCallError(Exception):
    pass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def gemini_text(resp) -> str:
    txt = getattr(resp, "text", None)
    if txt:
        return txt
    try:
        parts = []
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (getattr(content, "parts", None) or []):
                t = getattr(part, "text", None)
                if t:
                    parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""

def is_protected_node(node) -> bool:
    p = node.parent
    while p is not None and getattr(p, "name", None):
        if p.name and p.name.lower() in PROTECTED_TAGS:
            return True
        p = p.parent
    return False

def parse_segments(html_text: str):
    # Parses HTML and returns all non-empty text nodes as editable segments.
    soup = BeautifulSoup(html_text, "lxml")
    segments = []
    idx = 0
    for node in soup.find_all(string=True):
        if isinstance(node, Comment):
            continue
        txt = str(node)
        if not txt.strip():
            continue
        segments.append({
            "seg_id": idx,
            "node": node,
            "text": txt,
            "is_protected": is_protected_node(node),
        })
        idx += 1
    return soup, segments

def force_split_text(text: str, hard_limit: int) -> List[str]:
    parts = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + hard_limit, n)
        if end < n:
            cut = text.rfind(" ", i, end)
            if cut > i + hard_limit // 2:
                end = cut
        part = text[i:end].strip()
        if part:
            parts.append(part)
        i = end
        while i < n and text[i].isspace():
            i += 1
    return parts

def split_large_paragraph(paragraph: str, hard_limit: int) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    out = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > hard_limit:
            if current:
                out.append(current)
                current = ""
            out.extend(force_split_text(s, hard_limit))
            continue
        candidate = s if not current else f"{current} {s}"
        if len(candidate) <= hard_limit:
            current = candidate
        else:
            out.append(current)
            current = s
    if current:
        out.append(current)
    return out

def split_text_for_rewrite(text: str, target_chars: int, hard_limit: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Split by paragraph blocks first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    atomic_parts = []
    for p in paragraphs:
        if len(p) <= hard_limit:
            atomic_parts.append(p)
        else:
            atomic_parts.extend(split_large_paragraph(p, hard_limit))

    # Pack to near target size
    chunks = []
    current = ""
    for part in atomic_parts:
        sep = "\n\n" if current else ""
        candidate = f"{current}{sep}{part}"
        if len(candidate) <= target_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
                current = part
            else:
                chunks.append(part)
                current = ""
    if current:
        chunks.append(current)

    # Final hard safety
    safe = []
    for c in chunks:
        if len(c) <= hard_limit:
            safe.append(c)
        else:
            safe.extend(force_split_text(c, hard_limit))
    return safe

def pack_text_groups(texts: List[str], max_chars: int) -> List[List[str]]:
    groups = []
    current = []
    current_len = 0
    for t in texts:
        add_len = len(t) + (2 if current else 0)
        if current and (current_len + add_len > max_chars):
            groups.append(current)
            current = [t]
            current_len = len(t)
        else:
            current.append(t)
            current_len += add_len
    if current:
        groups.append(current)
    return groups

async def gather_in_batches(coros, batch_size: int = 200):
    results = []
    for i in range(0, len(coros), batch_size):
        results.extend(await asyncio.gather(*coros[i:i + batch_size]))
    return results

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def protected_texts(html_text: str) -> List[str]:
    soup = BeautifulSoup(html_text, "lxml")
    out = []
    for t in soup.find_all(PROTECTED_TAGS):
        for n in t.find_all(string=True):
            v = normalize_ws(str(n))
            if v:
                out.append(v)
    return out

# COMMAND ----------
# ---------------------------
# 4) Async LLM calls
# ---------------------------
async def generate_with_retry_async(
    user_prompt: str,
    system_instruction: str,
    semaphore: asyncio.Semaphore,
    max_output_tokens: int = 4096,
) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            cfg = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=TEMPERATURE,
                max_output_tokens=max_output_tokens,
            )
            async with semaphore:
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=MODEL,
                    contents=user_prompt,
                    config=cfg,
                )
            return (gemini_text(resp) or "").strip()
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES:
                break
            sleep_s = RETRY_BASE_SECONDS * (2 ** (attempt - 1)) + random.random()
            await asyncio.sleep(sleep_s)

    raise LLMCallError(str(last_err)) from last_err

async def rewrite_chunk_async(
    chunk_text: str,
    category: str,
    chunk_idx: int,
    chunk_total: int,
    semaphore: asyncio.Semaphore,
) -> str:
    system_instruction = (
        "You are an enterprise technical editor. Rewrite for clarity and professionalism. "
        "Do not change meaning or facts. Do not change numbers, dates, monetary amounts, URLs, IDs, or deadlines. "
        "Return plain text only."
    )
    user_prompt = f"""
Category: {category}
Chunk: {chunk_idx}/{chunk_total}

Rewrite this text:
{chunk_text}
""".strip()

    out = await generate_with_retry_async(user_prompt, system_instruction, semaphore)
    return out if out else chunk_text

async def harmonize_group_async(
    group_texts: List[str],
    category: str,
    semaphore: asyncio.Semaphore,
) -> str:
    joined = "\n\n".join(group_texts)
    system_instruction = (
        "You are an enterprise editor performing harmonization. "
        "Merge rewritten chunks into one coherent text with consistent tone and flow. "
        "Do not change meaning, facts, numbers, dates, URLs, IDs, or deadlines. "
        "Do not add new content. Return plain text only."
    )
    user_prompt = f"""
Category: {category}

Harmonize these rewritten chunks into one continuous text:
{joined}
""".strip()

    try:
        out = await generate_with_retry_async(user_prompt, system_instruction, semaphore)
        return out if out else joined
    except Exception:
        # Fail-open to already rewritten text
        return joined

# COMMAND ----------
# ---------------------------
# 5) Rewriter (segment -> html -> batch)
# ---------------------------
async def rewrite_segment_async(
    text: str,
    category: str,
    semaphore: asyncio.Semaphore,
) -> str:
    raw = text or ""
    if not raw.strip():
        return raw

    # Preserve surrounding whitespace
    leading_ws = raw[: len(raw) - len(raw.lstrip())]
    trailing_ws = raw[len(raw.rstrip()):]
    core = raw.strip()

    # Small segment => single request
    if len(core) <= MAX_SEGMENT_CHARS:
        rewritten = await rewrite_chunk_async(core, category, 1, 1, semaphore)
        return f"{leading_ws}{rewritten}{trailing_ws}"

    # Large segment => chunk + parallel rewrite + hierarchical harmonize
    chunks = split_text_for_rewrite(core, CHUNK_TARGET_CHARS, CHUNK_HARD_LIMIT_CHARS)
    if not chunks:
        return raw

    coros = [
        rewrite_chunk_async(ch, category, i + 1, len(chunks), semaphore)
        for i, ch in enumerate(chunks)
    ]
    rewritten_chunks = await gather_in_batches(coros, batch_size=100)

    current = rewritten_chunks
    while len(current) > 1:
        groups = pack_text_groups(current, HARMONIZE_GROUP_CHARS)
        coros = [harmonize_group_async(g, category, semaphore) for g in groups]
        current = await gather_in_batches(coros, batch_size=50)

    final_core = current[0] if current else core
    return f"{leading_ws}{final_core}{trailing_ws}"

async def rewrite_html_async(
    html_text: str,
    category: str,
    request_semaphore: asyncio.Semaphore,
) -> Tuple[str, Dict[str, Any]]:
    soup, segments = parse_segments(html_text)

    editable_idxs = []
    coros = []
    protected_count = 0

    for i, seg in enumerate(segments):
        if seg["is_protected"]:
            protected_count += 1
            continue
        editable_idxs.append(i)
        coros.append(rewrite_segment_async(seg["text"], category, request_semaphore))

    rewritten_texts = []
    if coros:
        rewritten_texts = await gather_in_batches(coros, batch_size=200)

    for i, new_text in zip(editable_idxs, rewritten_texts):
        segments[i]["node"].replace_with(new_text)

    stats = {
        "total_segments": len(segments),
        "editable_segments": len(editable_idxs),
        "protected_segments": protected_count,
    }
    return str(soup), stats

async def rewrite_document_task(
    doc_id: str,
    html_text: str,
    category: str,
    request_semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    original_protected = protected_texts(html_text)

    try:
        rewritten_html, seg_stats = await rewrite_html_async(html_text, category, request_semaphore)
        status = "ok"
        error = ""
    except Exception as e:
        rewritten_html = html_text
        seg_stats = {"total_segments": 0, "editable_segments": 0, "protected_segments": 0}
        status = "error"
        error = str(e)

    t1 = time.perf_counter()
    rewritten_protected = protected_texts(rewritten_html)

    return {
        "doc_id": doc_id,
        "category": category,
        "status": status,
        "error": error,
        "elapsed_sec": round(t1 - t0, 4),
        "input_chars": len(html_text or ""),
        "output_chars": len(rewritten_html or ""),
        "changed": rewritten_html != html_text,
        "protected_unchanged": original_protected == rewritten_protected,
        "total_segments": seg_stats["total_segments"],
        "editable_segments": seg_stats["editable_segments"],
        "protected_segments": seg_stats["protected_segments"],
        "rewritten_html": rewritten_html,
    }

async def rewrite_html_batch_async(
    docs: List[Tuple[str, str, str]],  # (doc_id, html_text, category)
    max_doc_concurrency: int,
    max_request_concurrency: int,
) -> List[Dict[str, Any]]:
    request_sem = asyncio.Semaphore(max_request_concurrency)
    doc_sem = asyncio.Semaphore(max_doc_concurrency)

    async def run_one(doc):
        doc_id, html_text, category = doc
        async with doc_sem:
            return await rewrite_document_task(
                doc_id=doc_id,
                html_text=html_text,
                category=category,
                request_semaphore=request_sem,
            )

    coros = [run_one(d) for d in docs]
    return await gather_in_batches(coros, batch_size=max_doc_concurrency * 10)

# COMMAND ----------
# ---------------------------
# 6) Benchmark runner
# ---------------------------
async def run_experiment(
    docs: List[Tuple[str, str, str]],
    variant_id: str,
    max_doc_concurrency: int,
    max_request_concurrency: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start = time.perf_counter()
    doc_rows = await rewrite_html_batch_async(
        docs,
        max_doc_concurrency=max_doc_concurrency,
        max_request_concurrency=max_request_concurrency,
    )
    wall = time.perf_counter() - start

    total = len(doc_rows)
    ok = sum(1 for r in doc_rows if r["status"] == "ok")
    err = total - ok
    changed = sum(1 for r in doc_rows if r["changed"])
    protected_ok = sum(1 for r in doc_rows if r["protected_unchanged"])
    total_chars = sum(r["input_chars"] for r in doc_rows)
    avg_doc_sec = sum(r["elapsed_sec"] for r in doc_rows) / max(total, 1)

    summary = {
        "variant_id": variant_id,
        "max_doc_concurrency": max_doc_concurrency,
        "max_request_concurrency": max_request_concurrency,
        "docs_total": total,
        "docs_ok": ok,
        "docs_error": err,
        "wall_time_sec": round(wall, 4),
        "avg_doc_elapsed_sec": round(avg_doc_sec, 4),
        "docs_per_min": round((ok / wall) * 60, 4) if wall > 0 else 0.0,
        "chars_per_sec": round(total_chars / wall, 4) if wall > 0 else 0.0,
        "changed_rate": round(changed / max(total, 1), 4),
        "protected_unchanged_rate": round(protected_ok / max(total, 1), 4),
    }

    for r in doc_rows:
        r["variant_id"] = variant_id

    return doc_rows, summary

async def run_all_experiments(docs, experiments):
    all_doc_rows = []
    summary_rows = []
    for exp in experiments:
        print(f"Running {exp['variant_id']} ...")
        doc_rows, summary = await run_experiment(
            docs=docs,
            variant_id=exp["variant_id"],
            max_doc_concurrency=exp["max_doc_concurrency"],
            max_request_concurrency=exp["max_request_concurrency"],
        )
        all_doc_rows.extend(doc_rows)
        summary_rows.append(summary)
    return all_doc_rows, summary_rows

# COMMAND ----------
# ---------------------------
# 7) Execute benchmark
# ---------------------------
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

all_doc_rows, summary_rows = run_async(run_all_experiments(docs, EXPERIMENTS))

summary_pdf = pd.DataFrame(summary_rows).sort_values("docs_per_min", ascending=False)
display(summary_pdf)

# COMMAND ----------
# ---------------------------
# 8) Save results to Delta tables
# ---------------------------
RUN_ID = f"parallel_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
RUN_TS_UTC = datetime.utcnow().isoformat()

for r in summary_rows:
    r["run_id"] = RUN_ID
    r["run_ts_utc"] = RUN_TS_UTC

for r in all_doc_rows:
    r["run_id"] = RUN_ID
    r["run_ts_utc"] = RUN_TS_UTC

summary_table = "default.kb_parallel_benchmark_summary"
detail_table = "default.kb_parallel_benchmark_detail"

spark.createDataFrame(summary_rows).write.mode("append").saveAsTable(summary_table)
spark.createDataFrame(all_doc_rows).write.mode("append").saveAsTable(detail_table)

print("Saved summary table:", summary_table)
print("Saved detail table :", detail_table)
print("RUN_ID:", RUN_ID)

# COMMAND ----------
# ---------------------------
# 9) Inspect results
# ---------------------------
display(
    spark.table(summary_table)
         .filter(F.col("run_id") == RUN_ID)
         .orderBy(F.col("docs_per_min").desc())
)

display(
    spark.table(detail_table)
         .filter(F.col("run_id") == RUN_ID)
         .select(
             "variant_id", "doc_id", "category", "status", "elapsed_sec",
             "input_chars", "output_chars", "changed", "protected_unchanged"
         )
         .orderBy("variant_id", "elapsed_sec")
)

# COMMAND ----------
# ---------------------------
# 10) Save rewritten HTML for best variant
# ---------------------------
best_variant = summary_pdf.iloc[0]["variant_id"]
best_rows = [r for r in all_doc_rows if r["variant_id"] == best_variant and r["status"] == "ok"]

best_out_table = "default.kb_poc_rewritten_best_variant"
spark.createDataFrame(best_rows).select(
    "doc_id", "category", "variant_id", "rewritten_html", "run_id", "run_ts_utc"
).write.mode("overwrite").saveAsTable(best_out_table)

print("Best variant:", best_variant)
print("Saved rewritten outputs to:", best_out_table)
