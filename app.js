/* ════════════════════════════════════════════════════════════
   app.js  —  AI Personal Tutor  |  All frontend logic
   ════════════════════════════════════════════════════════════ */

"use strict";

/* ── App State ──────────────────────────────────────────────── */
const STATE = {
  rawText:    "",
  summary:    "",
  notes:      [],
  defs:       [],
  flashcards: [],
  quiz:       [],
  chatHistory:[],
  quizHistory:[],
  qStats:     {},
  streak:     0,
  fc: {
    index:   0,
    flipped: false,
  },
  difficulty: "Medium",
  scoreChart: null,
};

/* ── DOM refs ─────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

/* ── Page navigation ────────────────────────────────────────── */
function showPage(name) {
  $$(".page").forEach(p => p.classList.remove("active"));
  $$(".nav-link").forEach(l => l.classList.remove("active"));
  $(`page-${name}`)?.classList.add("active");
  document.querySelector(`[data-page="${name}"]`)?.classList.add("active");
  // Close sidebar on mobile
  document.getElementById("sidebar").classList.remove("open");
}

$$(".nav-link").forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    showPage(link.dataset.page);
  });
});

/* ── Hamburger ──────────────────────────────────────────────── */
$("hamburger").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
});

/* ── Difficulty ─────────────────────────────────────────────── */
$$(".diff-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".diff-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    STATE.difficulty = btn.dataset.diff;
    $("quizDiffBadge").textContent = STATE.difficulty;
  });
});

/* ══════════════════════════════════════════════════════════════
   CONTENT PROCESSING
══════════════════════════════════════════════════════════════ */

async function processContent(formData) {
  showLoader(true, "Analysing your material…");
  try {
    const res  = await fetch("/api/process", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }

    STATE.rawText    = data.raw_text;
    STATE.summary    = data.summary;
    STATE.notes      = data.notes;
    STATE.defs       = data.defs;
    STATE.flashcards = data.flashcards;
    STATE.quiz       = data.quiz;

    renderAll();
    showPage("summary");
  } catch (err) {
    alert("Processing failed: " + err.message);
  } finally {
    showLoader(false);
  }
}

/* Text input */
$("btnGenerateText").addEventListener("click", () => {
  const txt = $("textInput").value.trim();
  if (!txt) { alert("Please paste some text first."); return; }
  const fd = new FormData();
  fd.append("text", txt);
  fd.append("difficulty", STATE.difficulty);
  processContent(fd);
});

/* PDF input */
$("btnGeneratePDF").addEventListener("click", () => {
  const file = $("pdfInput").files[0];
  if (!file) { alert("Please upload a PDF first."); return; }
  const fd = new FormData();
  fd.append("pdf", file);
  fd.append("difficulty", STATE.difficulty);
  processContent(fd);
});

/* Drop zone */
const dropZone = $("dropZone");
dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file?.type === "application/pdf") {
    $("pdfInput").files = e.dataTransfer.files;
    $("dropFilename").textContent = "📄 " + file.name;
  }
});
$("pdfInput").addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) $("dropFilename").textContent = "📄 " + file.name;
});
dropZone.addEventListener("click", () => $("pdfInput").click());

/* Loader */
function showLoader(show, msg = "Processing…") {
  $("loaderWrap").classList.toggle("hidden", !show);
  $("loaderText").textContent = msg;
}

/* ══════════════════════════════════════════════════════════════
   RENDER HELPERS
══════════════════════════════════════════════════════════════ */

function renderAll() {
  renderSummary();
  renderDefs();
  renderNotes();
  renderFlashcards();
  renderQuiz();
  updateDashboard();
}

/* Summary */
function renderSummary() {
  $("summaryText").textContent = STATE.summary || "No summary generated.";
}

/* Definitions */
function renderDefs() {
  const el = $("defsList");
  if (!STATE.defs.length) { el.innerHTML = '<p style="color:var(--muted)">No definitions found.</p>'; return; }
  el.innerHTML = STATE.defs.map(([term, meaning]) => `
    <div class="def-item">
      <div class="def-term">${esc(term)}</div>
      <div class="def-meaning">${esc(meaning)}</div>
    </div>`).join("");
}

/* Notes */
function renderNotes() {
  const el = $("notesList");
  if (!STATE.notes.length) { el.innerHTML = '<p style="color:var(--muted)">No notes generated.</p>'; return; }
  el.innerHTML = STATE.notes.map((note, i) => `
    <div class="note-item">
      <div class="note-num">${String(i+1).padStart(2,"0")}</div>
      <div class="note-text">${esc(note)}</div>
    </div>`).join("");
}

/* ── Flashcards ─────────────────────────────────────────────── */
function renderFlashcards() {
  const cards = STATE.flashcards;
  if (!cards.length) return;

  STATE.fc.index   = 0;
  STATE.fc.flipped = false;
  updateFlashcard();
}

function updateFlashcard() {
  const cards = STATE.flashcards;
  const { index, flipped } = STATE.fc;
  if (!cards.length) return;

  const card = cards[index];
  $("fcFront").textContent = card.front;
  $("fcBack").textContent  = card.back;
  $("fcCounter").textContent = `Card ${index + 1} / ${cards.length}`;
  $("fcProgressFill").style.width = `${((index + 1) / cards.length) * 100}%`;

  $("fcCard").classList.toggle("flipped", flipped);
  $("fcPrev").disabled = index === 0;
  $("fcNext").disabled = index === cards.length - 1;
}

$("fcFlip").addEventListener("click", () => {
  STATE.fc.flipped = !STATE.fc.flipped;
  $("fcCard").classList.toggle("flipped", STATE.fc.flipped);
});
$("fcCard").addEventListener("click", () => {
  STATE.fc.flipped = !STATE.fc.flipped;
  $("fcCard").classList.toggle("flipped", STATE.fc.flipped);
});
$("fcPrev").addEventListener("click", () => {
  if (STATE.fc.index > 0) { STATE.fc.index--; STATE.fc.flipped = false; updateFlashcard(); }
});
$("fcNext").addEventListener("click", () => {
  if (STATE.fc.index < STATE.flashcards.length - 1) { STATE.fc.index++; STATE.fc.flipped = false; updateFlashcard(); }
});

/* ── Quiz ───────────────────────────────────────────────────── */
function renderQuiz() {
  const container = $("quizContainer");
  $("quizResults").classList.add("hidden");
  $("quizResults").innerHTML = "";

  if (!STATE.quiz.length) {
    container.innerHTML = '<p style="color:var(--muted);margin-bottom:16px">No quiz generated from this content.</p>';
    $("quizActions").classList.add("hidden");
    return;
  }

  container.innerHTML = STATE.quiz.map((q, i) => {
    const badgeClass = `qb-${q.q_type}`;
    const opts = Object.entries(q.labeled).map(([k, v]) => `
      <label class="q-option">
        <input type="radio" name="q${i}" value="${k}" />
        <strong>${k}.</strong> ${esc(v)}
      </label>`).join("");
    return `
      <div class="q-block">
        <div class="q-badge ${badgeClass}">${q.q_type}</div>
        <div class="q-text">Q${i+1}. ${esc(q.q)}</div>
        <div class="q-options">${opts}</div>
        ${q.hint ? `<small style="color:var(--muted);display:block;margin-top:8px">💡 ${esc(q.hint)}</small>` : ""}
      </div>`;
  }).join("");

  $("quizActions").classList.remove("hidden");
}

/* Regenerate quiz */
$("btnRegenQuiz").addEventListener("click", async () => {
  if (!STATE.rawText) return;
  try {
    const res  = await fetch("/api/quiz", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ raw_text: STATE.rawText, difficulty: STATE.difficulty })
    });
    const data = await res.json();
    STATE.quiz = data.quiz || [];
    renderQuiz();
  } catch(err) { alert("Failed: " + err.message); }
});

/* Submit quiz */
$("btnSubmitQuiz").addEventListener("click", () => {
  const quiz  = STATE.quiz;
  let correct = 0;

  const results = quiz.map((q, i) => {
    const chosen = document.querySelector(`input[name="q${i}"]:checked`);
    const label  = chosen ? chosen.value : null;
    const isOk   = label === q.correct_label;
    if (isOk) correct++;

    // Update per-topic stats
    if (!STATE.qStats[q.subject]) STATE.qStats[q.subject] = { correct: 0, total: 0 };
    STATE.qStats[q.subject].total++;
    if (isOk) STATE.qStats[q.subject].correct++;

    return { isOk, q, chosenLabel: label };
  });

  const pct = quiz.length ? Math.round((correct / quiz.length) * 100) : 0;
  STATE.streak = pct >= 70 ? STATE.streak + 1 : 0;
  STATE.quizHistory.push({ time: now(), score: pct, correct, total: quiz.length });

  // Render results
  const resEl = $("quizResults");
  resEl.classList.remove("hidden");
  resEl.innerHTML = `
    <div class="card score-display">
      <div class="score-number">${pct}%</div>
      <div class="score-sub">${correct}/${quiz.length} correct</div>
    </div>
    ${results.map(({isOk, q, chosenLabel}, i) => `
      <div class="result-item ${isOk ? "correct" : "wrong"}">
        <strong>Q${i+1}:</strong>
        ${isOk ? "✅ Correct" : `❌ Wrong — Correct answer: <strong>${q.correct_label}</strong>`}<br />
        <span style="opacity:.8">${esc(q.exp)}</span>
      </div>`).join("")}`;

  updateDashboard();
});

/* ── Chat ────────────────────────────────────────────────────── */
$("btnAsk").addEventListener("click", askChat);
$("chatInput").addEventListener("keydown", e => { if (e.key === "Enter") askChat(); });

async function askChat() {
  const question = $("chatInput").value.trim();
  if (!question) return;

  appendChatMsg("user", question);
  $("chatInput").value = "";

  try {
    const res  = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        question,
        raw_text:      STATE.rawText,
        history:       STATE.chatHistory,
        openai_api_key: $("openaiKey").value.trim(),
        groq_api_key:   $("groqKey").value.trim(),
      })
    });
    const data = await res.json();
    const answer = data.answer || "No answer returned.";
    appendChatMsg("bot", answer);
    STATE.chatHistory.push([question, answer]);
  } catch(err) {
    appendChatMsg("bot", "Error: " + err.message);
  }
}

function appendChatMsg(role, text) {
  const history = $("chatHistory");
  history.querySelector(".chat-empty")?.remove();

  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.innerHTML = `
    <div class="chat-label">${role === "user" ? "You" : "AI"}</div>
    <div class="chat-bubble">${esc(text)}</div>`;
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
}

/* ── Dashboard ──────────────────────────────────────────────── */
function updateDashboard() {
  const h = STATE.quizHistory;
  const total   = h.length;
  const avg     = total ? Math.round(h.reduce((s,x)=>s+x.score,0)/total) : 0;
  const best    = total ? Math.max(...h.map(x=>x.score)) : 0;

  $("metAttempts").textContent = total;
  $("metAvg").textContent      = avg + "%";
  $("metBest").textContent     = best + "%";
  $("metStreak").textContent   = STATE.streak;

  renderWeakAreas();
  renderRecent();
  renderScoreChart();
}

function renderWeakAreas() {
  const el = $("weakList");
  const list = Object.entries(STATE.qStats)
    .map(([t,s]) => [t, s.total ? Math.round((s.correct/s.total)*100) : 0])
    .sort((a,b) => a[1]-b[1])
    .slice(0,6);

  if (!list.length) { el.innerHTML = '<p style="color:var(--muted);font-size:.86rem">No stats yet.</p>'; return; }
  el.innerHTML = list.map(([t,p]) =>
    `<div class="weak-item"><span class="weak-name">${esc(t)}</span><span class="weak-pct">${p}%</span></div>`
  ).join("");
}

function renderRecent() {
  const el = $("recentList");
  const list = [...STATE.quizHistory].reverse().slice(0,5);
  if (!list.length) { el.innerHTML = '<p style="color:var(--muted);font-size:.86rem">No attempts yet.</p>'; return; }
  el.innerHTML = list.map(h =>
    `<div class="recent-item">
      <span>⏱</span>
      <span>${h.time}<br /><strong>${h.score}%</strong> (${h.correct}/${h.total})</span>
    </div>`).join("");
}

function renderScoreChart() {
  const canvas = $("scoreChart");
  const labels = STATE.quizHistory.map((_,i) => `#${i+1}`);
  const data   = STATE.quizHistory.map(h => h.score);

  if (STATE.scoreChart) STATE.scoreChart.destroy();

  STATE.scoreChart = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Score %",
        data,
        borderColor: "#a78bfa",
        backgroundColor: "rgba(167,139,250,0.12)",
        pointBackgroundColor: "#34d399",
        pointRadius: 5,
        tension: 0.4,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid:{ color:"rgba(255,255,255,0.05)" }, ticks:{ color:"#6b7280" } },
        y: { min:0, max:100, grid:{ color:"rgba(255,255,255,0.05)" }, ticks:{ color:"#6b7280" } },
      }
    }
  });
}

/* ── Download ────────────────────────────────────────────────── */
$("btnDownload").addEventListener("click", async () => {
  try {
    const res  = await fetch("/api/export", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ summary: STATE.summary, notes: STATE.notes, defs: STATE.defs })
    });
    const data = await res.json();
    const blob = new Blob([data.text], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "ai_tutor_notes.txt";
    a.click();
    URL.revokeObjectURL(url);
  } catch(err) { alert("Download failed: " + err.message); }
});

/* ── Utilities ───────────────────────────────────────────────── */
function esc(str) {
  return String(str ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function now() {
  return new Date().toLocaleTimeString("en-US", { hour:"2-digit", minute:"2-digit", second:"2-digit" });
}
