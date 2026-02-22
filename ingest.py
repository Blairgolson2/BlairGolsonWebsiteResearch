#!/usr/bin/env python3
"""
ingest.py — Process PDFs and populate the research library database.

Usage:
    python ingest.py                 # Process new PDFs only (skips already-ingested)
    python ingest.py --force         # Re-process all PDFs (regenerates summaries)
    python ingest.py --citations     # Refresh citation counts only (fast, no AI calls)

Requirements:
    pip install -r requirements.txt
    Copy .env.example to .env and set ANTHROPIC_API_KEY and PDF_DIR.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
import requests
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = Path(os.environ.get("PDF_DIR", "./pdfs"))
DB_PATH = Path("./data/library.db")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT    UNIQUE NOT NULL,
            title           TEXT    NOT NULL,
            authors         TEXT,          -- JSON array of strings
            year            INTEGER,
            journal         TEXT,
            summary         TEXT,
            citation_count  INTEGER,
            semantic_id     TEXT,
            last_updated    TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id  INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
            page_num    INTEGER NOT NULL,
            text        TEXT    NOT NULL
        );

        -- FTS5 full-text search index with stemming
        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
            text,
            content='pages',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        -- Keep FTS in sync with pages table
        CREATE TRIGGER IF NOT EXISTS pages_ai
            AFTER INSERT ON pages BEGIN
                INSERT INTO pages_fts(rowid, text) VALUES (new.id, new.text);
            END;

        CREATE TRIGGER IF NOT EXISTS pages_ad
            AFTER DELETE ON pages BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, text) VALUES ('delete', old.id, old.text);
            END;
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text from each page. Returns list of (page_num, text) tuples."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
    except Exception as e:
        print(f"  ⚠ PDF extraction error: {e}")
    return pages


# ---------------------------------------------------------------------------
# Claude Haiku — metadata + summary
# ---------------------------------------------------------------------------

def generate_metadata_and_summary(
    filename: str,
    pages: List[Tuple[int, str]],
    client: Anthropic,
) -> dict:
    """
    Call Claude Haiku to extract bibliographic metadata and write a summary.
    Returns a dict with keys: title, authors, year, journal, summary.
    """
    full_text = "\n\n".join(text for _, text in pages)
    # Keep within a comfortable token budget for Haiku
    text_for_claude = full_text[:60_000]

    prompt = f"""You are a research librarian analyzing an academic paper. Extract its metadata and write a reader-friendly summary.

Filename: {filename}

Paper text:
{text_for_claude}

Return a JSON object with exactly these fields:
- "title": The complete paper title (string)
- "authors": Array of author name strings, e.g. ["Jane Smith", "John Doe"]
- "year": Publication year as integer, or null if unknown
- "journal": Journal or publication name as string, or null if unknown
- "summary": A 3–4 paragraph summary covering:
    1. Research background and the question being addressed
    2. Study design and methodology
    3. Key findings and results
    4. Clinical or practical implications for readers

Return only the JSON object. No markdown fences, no extra commentary."""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if the model included them
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Drop first line (```json or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        raw = "\n".join(inner).rstrip("`").strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Semantic Scholar — citation count
# ---------------------------------------------------------------------------

def fetch_citations(title: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Look up a paper by title on Semantic Scholar.
    Returns (citation_count, paper_id) or (None, None) on failure.

    Semantic Scholar is free, has a proper REST API, and requires no API key
    for our volume (25 papers). It replaces Google Scholar, which has no
    public API and blocks automated access.
    """
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": title,
                "fields": "title,citationCount",
                "limit": 1,
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                paper = data["data"][0]
                return paper.get("citationCount"), paper.get("paperId")
        elif resp.status_code == 429:
            print("  ⚠ Semantic Scholar rate limit hit — waiting 60s...")
            time.sleep(60)
            return fetch_citations(title)
    except Exception as e:
        print(f"  ⚠ Semantic Scholar error: {e}")
    return None, None


# ---------------------------------------------------------------------------
# Per-PDF ingestion
# ---------------------------------------------------------------------------

def ingest_pdf(
    conn: sqlite3.Connection,
    pdf_path: Path,
    client: Anthropic,
    force: bool = False,
) -> None:
    filename = pdf_path.name

    existing = conn.execute(
        "SELECT id FROM articles WHERE filename = ?", (filename,)
    ).fetchone()

    if existing and not force:
        print("  ↩ Already processed — skipping (use --force to regenerate)")
        return

    print("  📄 Extracting text from PDF...")
    pages = extract_pdf_text(pdf_path)
    if not pages:
        print("  ✗ Could not extract text — skipping (may be a scanned/image PDF)")
        return

    print("  🤖 Calling Claude Haiku for metadata + summary...")
    try:
        meta = generate_metadata_and_summary(filename, pages, client)
    except json.JSONDecodeError as e:
        print(f"  ✗ Could not parse Claude response as JSON: {e}")
        return
    except Exception as e:
        print(f"  ✗ Claude error: {e}")
        return

    print("  📊 Looking up citation count on Semantic Scholar...")
    time.sleep(1.5)  # polite pause between API calls
    citation_count, semantic_id = fetch_citations(meta.get("title", filename))

    title   = meta.get("title") or filename
    authors = json.dumps(meta.get("authors") or [])
    year    = meta.get("year")
    journal = meta.get("journal")
    summary = meta.get("summary") or ""

    if existing:
        article_id = existing[0]
        conn.execute(
            """UPDATE articles
               SET title=?, authors=?, year=?, journal=?, summary=?,
                   citation_count=?, semantic_id=?, last_updated=datetime('now')
               WHERE id=?""",
            (title, authors, year, journal, summary, citation_count, semantic_id, article_id),
        )
        # Delete old pages so FTS triggers rebuild
        conn.execute("DELETE FROM pages WHERE article_id=?", (article_id,))
    else:
        cur = conn.execute(
            """INSERT INTO articles
               (filename, title, authors, year, journal, summary, citation_count, semantic_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, title, authors, year, journal, summary, citation_count, semantic_id),
        )
        article_id = cur.lastrowid

    for page_num, text in pages:
        conn.execute(
            "INSERT INTO pages (article_id, page_num, text) VALUES (?, ?, ?)",
            (article_id, page_num, text),
        )

    conn.commit()

    display_title = title[:70] + ("…" if len(title) > 70 else "")
    print(f"  ✓ {display_title}")
    if citation_count is not None:
        print(f"    Citations: {citation_count:,}")
    else:
        print("    Citations: not found on Semantic Scholar")


# ---------------------------------------------------------------------------
# Refresh citations only
# ---------------------------------------------------------------------------

def refresh_citations(conn: sqlite3.Connection) -> None:
    articles = conn.execute("SELECT id, title FROM articles ORDER BY title").fetchall()
    for i, article in enumerate(articles, 1):
        print(f"  [{i}/{len(articles)}] {article['title'][:60]}…")
        count, sid = fetch_citations(article["title"])
        if count is not None:
            conn.execute(
                """UPDATE articles
                   SET citation_count=?, semantic_id=?, last_updated=datetime('now')
                   WHERE id=?""",
                (count, sid, article["id"]),
            )
            print(f"    → {count:,} citations")
        else:
            print("    → Not found on Semantic Scholar")
        time.sleep(1.5)
    conn.commit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the research library database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process all PDFs (regenerate AI summaries)",
    )
    parser.add_argument(
        "--citations",
        action="store_true",
        help="Only refresh citation counts from Semantic Scholar (no AI calls)",
    )
    args = parser.parse_args()

    if not PDF_DIR.exists():
        print(f"✗ PDF directory not found: {PDF_DIR.resolve()}")
        print(f"  Create it, add your PDFs, then run this script again.")
        print(f"  Or set PDF_DIR=. in your .env to use the project root.")
        sys.exit(1)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)

    if args.citations:
        print("🔄 Refreshing citation counts from Semantic Scholar…\n")
        refresh_citations(conn)
        conn.close()
        print("\n✓ Citation counts updated.")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No .pdf files found in {PDF_DIR.resolve()}")
        print("  Add PDFs to that directory and re-run.")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    mode = "force-reprocessing all" if args.force else "processing new"
    print(f"Found {len(pdf_files)} PDF(s) in {PDF_DIR.resolve()}")
    print(f"Mode: {mode}\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        ingest_pdf(conn, pdf_path, client, force=args.force)

    conn.close()
    print("\n✓ Ingestion complete. Start the app with: flask run")


if __name__ == "__main__":
    main()
