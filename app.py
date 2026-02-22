"""
app.py — Research Library Flask application.

Routes:
    GET  /                       — Main library grid with search/ask interface
    GET  /article/<id>           — Article detail page with summary + PDF viewer
    GET  /api/search?q=<query>   — Full-text keyword search, returns JSON
    POST /api/ask                — RAG endpoint: SSE stream of AI answer + sources
    GET  /pdfs/<filename>        — Serve a PDF file

Start the app:
    flask run                              (development)
    gunicorn --worker-class gthread \
             --threads 4 app:app          (production — threads needed for SSE)
"""

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)
from markupsafe import Markup, escape

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-in-production")

PDF_DIR = Path(os.environ.get("PDF_DIR", "./pdfs"))
DB_PATH = Path("./data/library.db")


# ---------------------------------------------------------------------------
# Anthropic client (lazy singleton)
# ---------------------------------------------------------------------------

_anthropic_client: Optional[Anthropic] = None

def get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Template filters
# ---------------------------------------------------------------------------

@app.template_filter("fromjson")
def fromjson_filter(s):
    """Parse a JSON string into a Python object (for authors array)."""
    try:
        return json.loads(s) if s else []
    except Exception:
        return []


@app.template_filter("nl2p")
def nl2p_filter(text: str) -> Markup:
    """Convert newline-separated paragraphs to <p> tags."""
    if not text:
        return Markup("")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return Markup("".join(f"<p>{escape(p)}</p>" for p in paragraphs))


@app.template_filter("urlencode_path")
def urlencode_path_filter(s: str) -> str:
    """URL-encode a file path component (preserves nothing as safe)."""
    return quote(s, safe="")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if not DB_PATH.exists():
        return render_template("index.html", articles=[], db_empty=True)

    conn = get_db()
    articles = conn.execute(
        """SELECT id, filename, title, authors, year, journal, summary, citation_count
           FROM articles
           ORDER BY COALESCE(year, 0) DESC, title ASC"""
    ).fetchall()
    stats = conn.execute(
        """SELECT COUNT(*) AS total,
                  SUM(COALESCE(citation_count, 0)) AS total_citations
           FROM articles"""
    ).fetchone()
    conn.close()
    return render_template("index.html", articles=articles, stats=stats, db_empty=False)


@app.route("/article/<int:article_id>")
def article(article_id):
    if not DB_PATH.exists():
        abort(404)
    conn = get_db()
    art = conn.execute(
        "SELECT * FROM articles WHERE id = ?", (article_id,)
    ).fetchone()
    if not art:
        conn.close()
        abort(404)
    conn.close()
    return render_template("article.html", article=art)


@app.route("/api/search")
def search():
    """
    Full-text search over all PDF page text using SQLite FTS5.
    Returns up to 10 results, one per article (best snippet chosen).
    """
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])

    if not DB_PATH.exists():
        return jsonify([])

    conn = get_db()
    try:
        try:
            rows = conn.execute(
                """SELECT
                       a.id, a.title, a.authors, a.year, a.citation_count, a.filename,
                       snippet(pages_fts, 0, '<mark>', '</mark>', '…', 30) AS snippet,
                       p.page_num
                   FROM pages_fts
                   JOIN pages    p ON p.rowid  = pages_fts.rowid
                   JOIN articles a ON a.id     = p.article_id
                   WHERE pages_fts MATCH ?
                   ORDER BY pages_fts.rank
                   LIMIT 50""",
                (q,),
            ).fetchall()
        except Exception:
            # FTS5 syntax error (e.g. trailing operator) — fall back to LIKE
            rows = conn.execute(
                """SELECT
                       a.id, a.title, a.authors, a.year, a.citation_count, a.filename,
                       SUBSTR(p.text,
                              MAX(1, INSTR(LOWER(p.text), LOWER(?)) - 80),
                              200) AS snippet,
                       p.page_num
                   FROM pages p
                   JOIN articles a ON a.id = p.article_id
                   WHERE LOWER(p.text) LIKE LOWER(?)
                   LIMIT 50""",
                (q, f"%{q}%"),
            ).fetchall()

        # Keep only the best snippet per article
        seen: dict[int, dict] = {}
        for row in rows:
            aid = row["id"]
            if aid not in seen:
                seen[aid] = {
                    "id":             aid,
                    "title":          row["title"],
                    "authors":        json.loads(row["authors"] or "[]"),
                    "year":           row["year"],
                    "citation_count": row["citation_count"],
                    "filename":       row["filename"],
                    "snippet":        row["snippet"] or "",
                    "page_num":       row["page_num"],
                }
    finally:
        conn.close()

    return jsonify(list(seen.values())[:10])


@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    """Serve a PDF file from the PDF directory."""
    return send_from_directory(PDF_DIR.resolve(), filename)


# ---------------------------------------------------------------------------
# RAG — retrieval-augmented generation
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","not","if","then","than","yet","while","after","before","because",
    "since","although","though","when","where","who","which","that","this",
    "these","those","there","here","so","also","into","up","down","out","over",
    "under","through","about","what","how","why","i","we","you","he","she",
    "it","they","them","their","my","your","his","her","its","our","say","says",
    "said","just","more","some","any","all","one","two","three","can","cannot",
    "tell","get","know","use","used","using","show","shows","showed","shown",
    "find","found","make","made","take","took","come","came","see","seen",
    "look","looked","research","study","studies","paper","review","evidence",
})


def _extract_keywords(question: str) -> list:
    """Strip stopwords and return meaningful search terms (max 15)."""
    words = re.findall(r"[a-zA-Z']+", question.lower())
    seen: set = set()
    result = []
    for w in words:
        if w not in _STOPWORDS and len(w) > 2 and w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) == 15:
                break
    return result


def retrieve_passages(question: str, n_sources: int = 5) -> list:
    """
    Find the most relevant page for each of up to n_sources articles.
    Uses FTS5 with OR-joined keywords; falls back to LIKE on first keyword.
    Returns a list of passage dicts ordered by relevance.
    """
    keywords = _extract_keywords(question)
    if not keywords or not DB_PATH.exists():
        return []

    fts_query = " OR ".join(keywords)
    conn = get_db()
    try:
        try:
            rows = conn.execute(
                """SELECT
                       p.article_id,
                       p.page_num,
                       p.text,
                       a.title,
                       a.authors,
                       a.year,
                       a.journal,
                       a.citation_count,
                       pages_fts.rank
                   FROM pages_fts
                   JOIN pages    p ON p.rowid = pages_fts.rowid
                   JOIN articles a ON a.id    = p.article_id
                   WHERE pages_fts MATCH ?
                   ORDER BY pages_fts.rank
                   LIMIT 50""",
                (fts_query,),
            ).fetchall()
        except Exception:
            first_kw = keywords[0]
            rows = conn.execute(
                """SELECT
                       p.article_id,
                       p.page_num,
                       p.text,
                       a.title,
                       a.authors,
                       a.year,
                       a.journal,
                       a.citation_count,
                       0 AS rank
                   FROM pages p
                   JOIN articles a ON a.id = p.article_id
                   WHERE LOWER(p.text) LIKE LOWER(?)
                   LIMIT 50""",
                (f"%{first_kw}%",),
            ).fetchall()
    finally:
        conn.close()

    # One best-ranked page per article
    best: dict = {}
    for row in rows:
        aid = row["article_id"]
        if aid not in best:
            best[aid] = row

    passages = []
    for i, row in enumerate(list(best.values())[:n_sources], start=1):
        passages.append({
            "num":            i,
            "article_id":     row["article_id"],
            "title":          row["title"],
            "authors":        json.loads(row["authors"] or "[]"),
            "year":           row["year"],
            "journal":        row["journal"],
            "citation_count": row["citation_count"],
            "page_num":       row["page_num"],
            "text":           row["text"].strip()[:800],
        })
    return passages


def build_rag_prompt(question: str, passages: list, library_meta: list) -> tuple:
    """Return (system, user_message) strings for Claude."""
    system = (
        "You are a knowledgeable research assistant with expertise in chronic pain, "
        "placebo effects, and mind-body medicine. You have been given AI-generated "
        "summaries of every article in a curated library of peer-reviewed studies, "
        "along with citation counts and journal names for each. For questions requiring "
        "specific details, you also have relevant text passages pulled directly from the "
        "PDFs. Use all of this as your primary source. You may also draw freely on your "
        "own broad knowledge to provide context, explain significance, connect ideas to "
        "other fields, comment on author reputations, describe clinical applications, or "
        "answer anything the library alone doesn't cover — just be clear when you are "
        "doing so. Cite library articles inline as [1], [2], etc. when they are relevant. "
        "When listing or ranking items, use a numbered markdown list. For other answers, "
        "write in clear flowing paragraphs. Use **bold** for article titles and key terms. "
        "Do not use markdown headers (##). "
        "Draw confidently on your training knowledge. State what you know directly "
        "without adding disclaimers, caveats, or phrases like 'I want to be honest' "
        "or 'these are inferences' at the end of your response. Never refuse to engage "
        "with a question just because you lack complete information."
    )

    # Full summary block for every article in the library
    summary_sections = []
    for a in library_meta:
        authors = json.loads(a["authors"] or "[]")
        last = authors[0].split()[-1] if authors else "Unknown"
        year = a["year"] or "n.d."
        cites = f"{a['citation_count']:,}" if a["citation_count"] is not None else "N/A"
        journal = a["journal"] or "journal unknown"
        summary = (a["summary"] or "").strip()
        summary_sections.append(
            f"{last} et al. ({year}) — \"{a['title']}\" — {journal} — {cites} citations\n"
            f"{summary}"
        )
    library_block = (
        f"LIBRARY SUMMARIES ({len(library_meta)} articles):\n\n" +
        "\n\n---\n\n".join(summary_sections)
    )

    # Specific text passages (may be empty for world-knowledge questions)
    if passages:
        blocks = []
        for p in passages:
            authors = p["authors"]
            first_last = authors[0].split()[-1] if authors else "Unknown"
            year = p["year"] or "n.d."
            cites = f"{p['citation_count']:,}" if p["citation_count"] is not None else "N/A"
            blocks.append(
                f"[{p['num']}] {first_last} et al. ({year}), \"{p['title']}\" "
                f"({cites} citations) — page {p['page_num']}:\n{p['text']}"
            )
        passages_block = "RELEVANT TEXT PASSAGES:\n\n" + "\n\n".join(blocks)
    else:
        passages_block = ""

    user_message = (
        f"{library_block}\n\n"
        + (f"{passages_block}\n\n" if passages_block else "")
        + f"Question: {question}"
    )
    return system, user_message


@app.route("/api/ask", methods=["POST"])
def ask():
    """
    RAG question-answering endpoint. Streams a Server-Sent Events response.

    Events emitted:
      {"type": "sources", "sources": [...]}   — source list (before answer starts)
      {"type": "token",   "text":    "..."}   — one per streaming text delta
      {"type": "done"}                         — stream finished
      {"type": "error",   "message": "..."}   — on any failure

    For production, run gunicorn with --worker-class gthread --threads 4
    so concurrent SSE connections don't block each other.
    """
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    if len(question) < 5:
        return jsonify({"error": "Question too short"}), 400

    def generate():
        try:
            passages = retrieve_passages(question)

            # Fetch full summaries + metadata for all articles
            conn = get_db()
            library_meta = conn.execute(
                """SELECT title, authors, year, journal, citation_count, summary
                   FROM articles
                   ORDER BY COALESCE(citation_count, -1) DESC, title ASC"""
            ).fetchall()
            conn.close()

            # Emit sources first so cards render before the answer streams in
            sources_meta = [
                {
                    "num":        p["num"],
                    "article_id": p["article_id"],
                    "title":      p["title"],
                    "authors":    p["authors"],
                    "year":       p["year"],
                    "journal":    p["journal"],
                    "page_num":   p["page_num"],
                }
                for p in passages
            ]
            yield "data: " + json.dumps({"type": "sources", "sources": sources_meta}) + "\n\n"

            system_prompt, user_message = build_rag_prompt(question, passages, library_meta)
            client = get_anthropic_client()

            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1200,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text_delta in stream.text_stream:
                    yield "data: " + json.dumps({"type": "token", "text": text_delta}) + "\n\n"

            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

        except RuntimeError as exc:
            yield "data: " + json.dumps({"type": "error", "message": str(exc)}) + "\n\n"
        except Exception:
            app.logger.exception("RAG error")
            yield "data: " + json.dumps({
                "type": "error",
                "message": "An error occurred generating the answer. Please try again.",
            }) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
        },
    )


# ---------------------------------------------------------------------------
# Error pages
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


# ---------------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
