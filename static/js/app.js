/* =========================================================
   Research Library — Frontend JavaScript
   Handles: debounced full-text search with highlighted snippets
   ========================================================= */

(function () {
  "use strict";

  const searchInput        = document.getElementById("search-input");
  const searchResultsPanel = document.getElementById("search-results");
  const searchResultsInner = document.getElementById("search-results-inner");

  if (!searchInput) return; // Not on the index page

  let debounceTimer = null;
  let lastQuery = "";

  // ── Helpers ────────────────────────────────────────────

  function escapeHtml(str) {
    if (str == null) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /**
   * The snippet from FTS5 contains <mark>…</mark> only.
   * Everything else is plain text we need to escape.
   * Split on the tags, escape non-tag parts, reassemble.
   */
  function safeSnippet(raw) {
    if (!raw) return "";
    // Split on <mark> and </mark>, preserve delimiters
    const parts = raw.split(/(<mark>|<\/mark>)/);
    let inMark = false;
    let out = "";
    for (const part of parts) {
      if (part === "<mark>")       { inMark = true;  out += "<mark>"; }
      else if (part === "</mark>") { inMark = false; out += "</mark>"; }
      else                         { out += escapeHtml(part); }
    }
    return out;
  }

  function formatAuthors(authors) {
    if (!Array.isArray(authors) || !authors.length) return "";
    const visible = authors.slice(0, 3).map(escapeHtml).join(", ");
    return authors.length > 3 ? visible + ", et al." : visible;
  }

  function formatCitations(count) {
    if (count == null) return "";
    if (count === 0)   return "";
    return `<span style="color:var(--amber);font-weight:600">★ ${Number(count).toLocaleString()} citations</span>`;
  }

  // ── Render ─────────────────────────────────────────────

  function renderResults(results, query) {
    if (!results.length) {
      searchResultsInner.innerHTML = `
        <div class="search-no-results">
          <p>No results for <strong>${escapeHtml(query)}</strong>.</p>
          <p style="margin-top:6px;font-size:.84rem">Try different keywords or browse all articles below.</p>
        </div>`;
      return;
    }

    const items = results.map((r) => {
      const authors   = formatAuthors(r.authors);
      const citations = formatCitations(r.citation_count);
      const meta      = [authors, r.year ? escapeHtml(String(r.year)) : "", citations]
        .filter(Boolean).join(" &middot; ");

      return `
        <div class="search-result-item">
          <a href="/article/${r.id}" class="search-result-title">${escapeHtml(r.title)}</a>
          ${meta ? `<div class="search-result-meta">${meta}</div>` : ""}
          <div class="search-result-snippet">${safeSnippet(r.snippet)}</div>
          ${r.page_num ? `<div class="search-result-page">Page ${r.page_num}</div>` : ""}
          <a href="/article/${r.id}" class="search-result-link">View article →</a>
        </div>`;
    }).join("");

    searchResultsInner.innerHTML = `
      <div class="search-results-header">
        ${results.length} result${results.length !== 1 ? "s" : ""} for
        <strong>&ldquo;${escapeHtml(query)}&rdquo;</strong>
      </div>
      ${items}`;
  }

  function showLoading() {
    searchResultsInner.innerHTML =
      '<div class="search-loading">Searching…</div>';
    searchResultsPanel.classList.remove("hidden");
  }

  function hidePanel() {
    searchResultsPanel.classList.add("hidden");
    searchResultsInner.innerHTML = "";
  }

  // ── Fetch ──────────────────────────────────────────────

  async function performSearch(query) {
    if (!query || query.length < 2) { hidePanel(); return; }
    if (query === lastQuery) return;
    lastQuery = query;

    showLoading();

    try {
      const resp = await fetch(
        `/api/search?q=${encodeURIComponent(query)}`,
        { headers: { Accept: "application/json" } }
      );
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const results = await resp.json();
      // Guard: query may have changed while awaiting
      if (query !== searchInput.value.trim()) return;
      renderResults(results, query);
      searchResultsPanel.classList.remove("hidden");
    } catch (err) {
      console.error("Search error:", err);
      searchResultsInner.innerHTML =
        '<div class="search-no-results">Search unavailable. Please try again.</div>';
    }
  }

  // ── Event listeners ────────────────────────────────────

  searchInput.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const q = searchInput.value.trim();
    if (!q) { lastQuery = ""; hidePanel(); return; }
    debounceTimer = setTimeout(() => performSearch(q), 380);
  });

  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      searchInput.value = "";
      lastQuery = "";
      hidePanel();
      searchInput.blur();
    }
    if (e.key === "Enter") {
      clearTimeout(debounceTimer);
      performSearch(searchInput.value.trim());
    }
  });

  // Close panel when clicking outside
  document.addEventListener("click", (e) => {
    if (!searchResultsPanel.contains(e.target) && e.target !== searchInput) {
      hidePanel();
    }
  });

})();

/* =========================================================
   RAG — Ask AI mode
   ========================================================= */

(function () {
  "use strict";

  const searchInput  = document.getElementById("search-input");
  const searchBtn    = document.getElementById("search-btn");
  const btnSearch    = document.getElementById("mode-search");
  const btnAsk       = document.getElementById("mode-ask");

  if (!searchInput || !btnSearch || !btnAsk) return;

  const searchResultsPanel = document.getElementById("search-results");
  const askPanel       = document.getElementById("ask-panel");
  const askLoading     = document.getElementById("ask-loading");
  const askError       = document.getElementById("ask-error");
  const askErrorMsg    = document.getElementById("ask-error-msg");
  const askSources     = document.getElementById("ask-sources");
  const askSourcesList = document.getElementById("ask-sources-list");
  const askAnswerWrap  = document.getElementById("ask-answer-wrap");
  const askAnswerText  = document.getElementById("ask-answer-text");

  let currentMode = "search";
  let currentAbort = null;
  let answerBuffer = "";

  // ── Helpers ────────────────────────────────────────────

  function escHtml(str) {
    if (str == null) return "";
    return String(str)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // ── Panel state ────────────────────────────────────────

  function hideAskPanel() {
    askPanel.classList.add("hidden");
    askLoading.classList.remove("hidden");   // reset for next open
    askError.classList.add("hidden");
    askSources.classList.add("hidden");
    askAnswerWrap.classList.add("hidden");
    askAnswerText.textContent = "";
    askAnswerText.classList.remove("streaming");
    askSourcesList.innerHTML = "";
    answerBuffer = "";
  }

  function showLoading() {
    askPanel.classList.remove("hidden");
    askLoading.classList.remove("hidden");
    askError.classList.add("hidden");
    askSources.classList.add("hidden");
    askAnswerWrap.classList.add("hidden");
    askAnswerText.textContent = "";
    askAnswerText.classList.remove("streaming");
    askSourcesList.innerHTML = "";
    answerBuffer = "";
  }

  function showError(msg) {
    askLoading.classList.add("hidden");
    askErrorMsg.textContent = msg;
    askError.classList.remove("hidden");
    askAnswerText.classList.remove("streaming");
  }

  // ── Source cards ───────────────────────────────────────

  function renderSources(sources) {
    if (!sources || !sources.length) return;
    askSourcesList.innerHTML = sources.map((s) => {
      const authors = Array.isArray(s.authors) ? s.authors : [];
      const lastName = authors.length
        ? escHtml(authors[0].split(" ").pop())
        : "Unknown";
      const year = s.year ? escHtml(String(s.year)) : "n.d.";
      const journal = s.journal ? " · " + escHtml(s.journal) : "";
      return `
        <a href="/article/${s.article_id}" class="ask-source-card">
          <span class="ask-source-num">${s.num}</span>
          <div class="ask-source-body">
            <div class="ask-source-title">${escHtml(s.title)}</div>
            <div class="ask-source-meta">${lastName} et al. · ${year}${journal}</div>
          </div>
        </a>`;
    }).join("");
    askSources.classList.remove("hidden");
  }

  // ── Fetch + stream ─────────────────────────────────────

  async function performAsk(question) {
    if (!question || question.length < 5) return;

    if (currentAbort) { currentAbort.abort(); }
    currentAbort = new AbortController();

    showLoading();

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: currentAbort.signal,
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE data lines
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep any incomplete trailing fragment

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let evt;
          try { evt = JSON.parse(raw); } catch { continue; }

          if (evt.type === "sources") {
            askLoading.classList.add("hidden");
            renderSources(evt.sources);
            askAnswerText.classList.add("streaming");
            askAnswerWrap.classList.remove("hidden");

          } else if (evt.type === "token") {
            answerBuffer += evt.text;
            askAnswerText.textContent = answerBuffer;

          } else if (evt.type === "done") {
            askAnswerText.classList.remove("streaming");
            if (typeof marked !== "undefined") {
              askAnswerText.innerHTML = marked.parse(answerBuffer);
            }
            currentAbort = null;

          } else if (evt.type === "error") {
            showError(evt.message || "An error occurred. Please try again.");
            askAnswerWrap.classList.add("hidden");
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      console.error("Ask error:", err);
      showError("Connection error. Please try again.");
    }
  }

  // ── Mode switching ─────────────────────────────────────

  function setMode(mode) {
    currentMode = mode;

    if (currentAbort) { currentAbort.abort(); currentAbort = null; }

    btnSearch.classList.toggle("mode-btn--active", mode === "search");
    btnAsk.classList.toggle("mode-btn--active",    mode === "ask");
    btnSearch.setAttribute("aria-pressed", mode === "search");
    btnAsk.setAttribute("aria-pressed",    mode === "ask");

    if (mode === "search") {
      searchInput.placeholder = "Search across all studies… (try \u201cplacebo\u201d, \u201cRCT\u201d, \u201cfibromyalgia\u201d)";
      searchBtn.setAttribute("aria-label", "Search");
      hideAskPanel();
    } else {
      searchInput.placeholder = "Ask a question\u2026 (e.g. \u201cWhat is the nocebo effect?\u201d)";
      searchBtn.setAttribute("aria-label", "Ask AI");
      searchResultsPanel.classList.add("hidden");
    }

    searchInput.value = "";
    searchInput.focus();
  }

  // ── Event listeners ────────────────────────────────────

  btnSearch.addEventListener("click", () => setMode("search"));
  btnAsk.addEventListener("click",    () => setMode("ask"));

  // Button click submits ask in ask mode
  searchBtn.addEventListener("click", () => {
    if (currentMode === "ask") {
      performAsk(searchInput.value.trim());
    }
  });

  // Capture-phase: suppress existing IIFE's input/keydown in ask mode
  searchInput.addEventListener("input", (e) => {
    if (currentMode === "ask") e.stopImmediatePropagation();
  }, true);

  searchInput.addEventListener("keydown", (e) => {
    if (currentMode !== "ask") return;
    e.stopImmediatePropagation(); // prevent existing IIFE's Enter/Escape handlers
    if (e.key === "Enter") {
      performAsk(searchInput.value.trim());
    }
    if (e.key === "Escape") {
      hideAskPanel();
      searchInput.value = "";
      searchInput.blur();
    }
  }, true);

})();
