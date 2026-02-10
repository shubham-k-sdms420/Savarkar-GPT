/**
 * Savarkar GPT – Historical Explorer
 * Frontend application logic.
 *
 * Handles: chat messages, API calls, sample question chips,
 *          auto-resize textarea, sources toggle.
 */

(() => {
  "use strict";

  // ─── DOM Elements ───
  const chatArea      = document.getElementById("chatArea");
  const messagesEl    = document.getElementById("messages");
  const welcome       = document.getElementById("welcome");
  const input         = document.getElementById("questionInput");
  const sendBtn       = document.getElementById("sendBtn");
  const scrollAnchor  = document.getElementById("scrollAnchor");
  const sampleQs      = document.getElementById("sampleQuestions");
  const sampleBtns    = document.querySelectorAll(".chip[data-question]");

  const API_URL = "/api/query";

  let isLoading = false;

  // ─── Auto-resize Textarea ───
  function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  }
  input.addEventListener("input", autoResize);

  // ─── Scroll to Bottom ───
  function scrollToBottom() {
    requestAnimationFrame(() => {
      scrollAnchor.scrollIntoView({ behavior: "smooth", block: "end" });
    });
  }

  // ─── Hide Welcome & Sample Chips ───
  function hideWelcome() {
    if (!welcome.classList.contains("hidden")) {
      welcome.classList.add("hidden");
    }
    if (sampleQs && !sampleQs.classList.contains("hidden")) {
      sampleQs.classList.add("hidden");
    }
  }

  // ─── Render a User Message ───
  function addUserMessage(text) {
    const msgEl = document.createElement("div");
    msgEl.className = "message message--user";
    msgEl.innerHTML = `
      <div class="message-bubble">${escapeHtml(text)}</div>
      <div class="message-avatar">You</div>
    `;
    messagesEl.appendChild(msgEl);
    scrollToBottom();
  }

  // ─── Render Loading Indicator ───
  function addLoadingMessage() {
    const msgEl = document.createElement("div");
    msgEl.className = "message message--assistant";
    msgEl.id = "loadingMsg";
    msgEl.innerHTML = `
      <div class="message-avatar">SG</div>
      <div class="message-bubble">
        <div class="loading-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    messagesEl.appendChild(msgEl);
    scrollToBottom();
  }

  function removeLoadingMessage() {
    const el = document.getElementById("loadingMsg");
    if (el) el.remove();
  }

  // ─── Render Assistant Response ───
  function addAssistantMessage(answer, sources) {
    const msgEl = document.createElement("div");
    msgEl.className = "message message--assistant";

    // Convert plain-text paragraphs to <p> tags
    const paragraphs = answer
      .split(/\n{2,}/)
      .map(p => p.trim())
      .filter(Boolean)
      .map(p => `<p>${escapeHtml(p)}</p>`)
      .join("");

    // Build sources section
    let sourcesHtml = "";
    if (sources && sources.length > 0) {
      const items = sources.map((s, i) =>
        `<div class="source-item">
           <strong>${i + 1}.</strong> "${escapeHtml(s.title)}" by ${escapeHtml(s.author)}
           <span class="source-chapter">· ${escapeHtml(s.chapter)}</span>
         </div>`
      ).join("");

      sourcesHtml = `
        <button class="sources-toggle" onclick="toggleSources(this)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
          ${sources.length} source${sources.length > 1 ? "s" : ""} referenced
        </button>
        <div class="sources-list">${items}</div>
      `;
    }

    msgEl.innerHTML = `
      <div class="message-avatar">SG</div>
      <div class="message-bubble">
        ${paragraphs}
        ${sourcesHtml}
      </div>
    `;

    messagesEl.appendChild(msgEl);
    scrollToBottom();
  }

  // ─── Render Error Message ───
  function addErrorMessage(text) {
    const msgEl = document.createElement("div");
    msgEl.className = "message message--assistant";
    msgEl.innerHTML = `
      <div class="message-avatar">SG</div>
      <div class="message-bubble" style="border-color: rgba(239,68,68,0.2);">
        <p style="color:#F87171;">${escapeHtml(text)}</p>
      </div>
    `;
    messagesEl.appendChild(msgEl);
    scrollToBottom();
  }

  // ─── HTML Escape ───
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // ─── Sources Toggle (global, called from onclick) ───
  window.toggleSources = function(btn) {
    const list = btn.nextElementSibling;
    btn.classList.toggle("open");
    list.classList.toggle("visible");
  };

  // ─── Send Question ───
  async function sendQuestion(questionText) {
    const question = questionText.trim();
    if (!question || isLoading) return;

    isLoading = true;
    sendBtn.disabled = true;
    input.value = "";
    autoResize();

    hideWelcome();
    addUserMessage(question);
    addLoadingMessage();

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      removeLoadingMessage();

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      addAssistantMessage(data.answer, data.sources);

    } catch (err) {
      removeLoadingMessage();
      addErrorMessage(
        err.message || "Something went wrong. Please try again."
      );
    } finally {
      isLoading = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  // ─── Event Listeners ───

  // Send button click
  sendBtn.addEventListener("click", () => {
    sendQuestion(input.value);
  });

  // Enter to send, Shift+Enter for new line
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion(input.value);
    }
  });

  // Sample question chips
  sampleBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-question");
      if (q) sendQuestion(q);
    });
  });

  // Focus input on page load
  input.focus();

})();
