const API = "/api";
let tickets = [];
let currentStatus = "";
let currentCategory = "";
let selectedId = null;

const statusTabs = document.getElementById("statusTabs");
const categoryFilter = document.getElementById("categoryFilter");
const ticketRows = document.getElementById("ticketRows");
const emptyState = document.getElementById("emptyState");
const ticketCount = document.getElementById("ticketCount");
const casePanelInner = document.getElementById("casePanelInner");
const ollamaDot = document.getElementById("ollamaDot");
const ollamaLabel = document.getElementById("ollamaLabel");

const CATEGORIES = [
  "Loan Status Query", "Document Upload Issue", "Disbursement Delay",
  "Interest Rate Query", "Account Access", "Partner Onboarding",
  "KYC Issue", "Payment Reconciliation", "Technical / Portal Bug",
  "Complaint - Escalation",
];

function init() {
  CATEGORIES.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    categoryFilter.appendChild(opt);
  });

  statusTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    currentStatus = btn.dataset.status;
    loadTickets();
  });

  categoryFilter.addEventListener("change", () => {
    currentCategory = categoryFilter.value;
    loadTickets();
  });

  loadTickets();
  checkHealth();
  setInterval(checkHealth, 15000);
}

async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    ollamaDot.className = "dot " + (data.ollama ? "up" : "down");
    ollamaLabel.textContent = data.ollama ? `${data.model} ready` : "Ollama offline";
  } catch {
    ollamaDot.className = "dot down";
    ollamaLabel.textContent = "Backend unreachable";
  }
}

async function loadTickets() {
  const params = new URLSearchParams();
  if (currentStatus) params.set("status", currentStatus);
  if (currentCategory) params.set("category", currentCategory);
  const res = await fetch(`${API}/tickets?${params}`);
  tickets = await res.json();
  renderTable();
}

function renderTable() {
  ticketRows.innerHTML = "";
  ticketCount.textContent = tickets.length;
  emptyState.hidden = tickets.length > 0;

  tickets.forEach((t) => {
    const tr = document.createElement("tr");
    if (t.id === selectedId) tr.classList.add("selected");
    tr.innerHTML = `
      <td class="case-id">${t.id}</td>
      <td class="subject-cell">${escapeHtml(t.subject)}</td>
      <td>${escapeHtml(t.sender)}</td>
      <td><span class="stamp ${t.priority}">${t.priority}</span></td>
      <td><span class="badge-status ${t.status}">${t.status.replace("_", " ")}</span></td>
      <td class="received-cell">${formatDate(t.timestamp)}</td>
    `;
    tr.addEventListener("click", () => openCase(t.id));
    ticketRows.appendChild(tr);
  });
}

function formatDate(iso) {
  const d = new Date(iso);
  return (
    d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
  );
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

async function openCase(id) {
  selectedId = id;
  renderTable();
  const res = await fetch(`${API}/tickets/${id}`);
  const ticket = await res.json();
  renderCase(ticket);
}

function renderCase(ticket) {
  casePanelInner.innerHTML = `
    <p class="case-heading">${ticket.id} &middot; ${ticket.sender_type}</p>
    <h2 class="case-subject">${escapeHtml(ticket.subject)}</h2>
    <div class="case-meta">
      <span>${escapeHtml(ticket.sender)}</span>
      <span>${formatDate(ticket.timestamp)}</span>
      <span class="stamp ${ticket.priority}">${ticket.priority}</span>
    </div>
    <div class="case-body">${escapeHtml(ticket.body)}</div>

    <div class="case-actions">
      <select class="status-select" id="statusSelect">
        ${["open", "in_progress", "closed"]
          .map((s) => `<option value="${s}" ${s === ticket.status ? "selected" : ""}>${s.replace("_", " ")}</option>`)
          .join("")}
      </select>
      <button class="primary" id="saveStatusBtn">Update status</button>
    </div>

    <div class="case-actions">
      <button class="primary" id="classifyBtn">Run AI triage</button>
    </div>

    <div id="classifyResult"></div>
  `;

  document.getElementById("saveStatusBtn").addEventListener("click", async () => {
    const newStatus = document.getElementById("statusSelect").value;
    await fetch(`${API}/tickets/${ticket.id}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: newStatus }),
    });
    await loadTickets();
  });

  document.getElementById("classifyBtn").addEventListener("click", () => runClassify(ticket.id));
}

async function runClassify(id) {
  const resultDiv = document.getElementById("classifyResult");
  const btn = document.getElementById("classifyBtn");
  btn.disabled = true;
  resultDiv.innerHTML = `<p class="spinner-text">Running the local model &mdash; this can take up to a minute on CPU&hellip;</p>`;

  try {
    const res = await fetch(`${API}/tickets/${id}/classify`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      resultDiv.innerHTML = `<div class="error-box">${escapeHtml(err.detail || "Classification failed.")}</div>`;
      return;
    }
    const data = await res.json();
    renderClassification(data);
  } catch {
    resultDiv.innerHTML = `<div class="error-box">Could not reach the backend.</div>`;
  } finally {
    btn.disabled = false;
  }
}

function renderClassification(data) {
  const resultDiv = document.getElementById("classifyResult");
  resultDiv.innerHTML = `
    <div class="result-block">
      <div class="result-row">
        <span class="result-label">Predicted category</span>
        <strong>${escapeHtml(data.category || "\u2014")}</strong>
      </div>
      <div class="result-row">
        <span class="result-label">Predicted priority</span>
        <span class="stamp ${data.priority || "low"}">${data.priority || "\u2014"}</span>
      </div>
      <div class="result-row">
        <span class="result-label">Sentiment</span>
        <span>${escapeHtml(data.sentiment || "\u2014")}</span>
      </div>
      <p class="result-label">Suggested response</p>
      <div class="response-box">${escapeHtml(data.suggested_response || "\u2014")}</div>
      <button class="copy-btn" id="copyBtn">Copy response</button>
      ${
        data.similar_ticket_ids && data.similar_ticket_ids.length
          ? `<p class="similar-list">Referenced similar cases: ${data.similar_ticket_ids.join(", ")}</p>`
          : ""
      }
    </div>
  `;
  document.getElementById("copyBtn").addEventListener("click", () => {
    navigator.clipboard.writeText(data.suggested_response || "");
  });
}

init();
