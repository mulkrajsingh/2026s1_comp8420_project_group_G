// Research-assistant UI: one tab-scoped session, PDF or text input, cancellable requests.
// Session ids persist in sessionStorage so a refresh restores the transcript.
import DOMPurify from 'dompurify';
import { marked } from 'marked';
import './style.css';
import {
  analyzePdf,
  cancelRequest,
  chat,
  completeSession,
  createSession,
  getSession,
  health,
  models,
  newRequestId,
} from './api.js';

marked.setOptions({ breaks: true, gfm: true });

document.documentElement.setAttribute('data-theme', localStorage.getItem('theme') || 'light');
const SESSION_STORAGE_KEY = 'comp8420ActiveSessionId';

const EMPTY = `<div class="empty" id="empty">
  <h1>What are you researching?</h1>
  <p>Ask about a topic, or attach a paper to analyse.</p>
</div>`;

document.getElementById('app').innerHTML = `
  <aside class="sidebar">
    <div class="brand">Research Assistant</div>
    <button class="new-chat" id="newChat">+ New analysis</button>
    <div class="controls">
      <div class="control-heading">Experiment configuration</div>
      <label for="model">Generation model</label>
      <select id="model">
        <option value="">Loading local models...</option>
      </select>
      <label for="retrievalStrategy">Retrieval ranker</label>
      <select id="retrievalStrategy">
        <option value="hybrid_rrf">Hybrid RRF (BM25 + dense)</option>
        <option value="tfidf">TF-IDF (lexical baseline)</option>
      </select>
      <label for="embeddingModel" id="embeddingModelLabel">Dense embedding (BM25 + RRF hybrid)</label>
      <select id="embeddingModel">
        <option value="allenai/specter2_base">SPECTER2</option>
        <option value="allenai/specter">SPECTER</option>
        <option value="all-MiniLM-L6-v2">MiniLM L6</option>
      </select>
      <label for="retrievalTopK">RAG candidate depth</label>
      <select id="retrievalTopK">
        <option value="5">Top 5</option>
        <option value="10">Top 10</option>
      </select>
      <label for="style">Response style</label>
      <select id="style">
        <option value="auto">Automatic</option>
        <option value="concise">Concise</option>
        <option value="technical">Technical</option>
        <option value="beginner">Beginner</option>
        <option value="reviewer">Reviewer</option>
      </select>
      <label for="promptStrategy">Prompt strategy</label>
      <select id="promptStrategy">
        <option value="zero_shot">Zero-shot</option>
        <option value="few_shot">Few-shot (where defined)</option>
      </select>
      <label class="check"><input id="noRelated" type="checkbox" /> PDF without related papers</label>
      <label class="check"><input id="includePeerReview" type="checkbox" /> Include peer review (adds one LLM call)</label>
      <p class="control-note" id="retrievalControlNote">Production default is hybrid RRF (BM25 + SPECTER2 reciprocal rank fusion). TF-IDF is a lexical baseline for comparison. Local <code>qwen3:8b</code> runs typically take ~15s (chat), ~60–90s (topic RAG), and ~3–5 min (PDF summary).</p>
    </div>
    <button class="theme-toggle" id="themeToggle"></button>
    <div class="hint">
      Local research-paper analysis.<br>
      Per-chat session logs live under <code>integration/data/sessions/&lt;session-id&gt;</code>.
    </div>
  </aside>
  <main class="main">
    <div class="thread" id="thread">${EMPTY}</div>
    <div class="composer">
      <div class="inputbar">
        <div class="attachments" id="attachments" hidden></div>
        <div class="inputrow">
          <button class="icon-btn" id="attach" title="Attach a PDF" aria-label="Attach a PDF">+</button>
          <input type="file" id="file" accept="application/pdf" hidden />
          <textarea id="input" rows="1" placeholder="Ask anything"></textarea>
          <button class="send" id="send" title="Send" aria-label="Send">↑</button>
        </div>
      </div>
      <div class="foot" id="backendStatus" role="status" aria-live="polite">
        Checking the local backend...
      </div>
    </div>
  </main>
`;

const thread = document.getElementById('thread');
const input = document.getElementById('input');
const fileInput = document.getElementById('file');
const attachments = document.getElementById('attachments');
const themeToggle = document.getElementById('themeToggle');
const modelInput = document.getElementById('model');
const retrievalStrategyInput = document.getElementById('retrievalStrategy');
const embeddingModelInput = document.getElementById('embeddingModel');
const embeddingModelLabel = document.getElementById('embeddingModelLabel');
const retrievalTopKInput = document.getElementById('retrievalTopK');
const styleInput = document.getElementById('style');
const promptStrategyInput = document.getElementById('promptStrategy');
const noRelatedInput = document.getElementById('noRelated');
const includePeerReviewInput = document.getElementById('includePeerReview');
const backendStatus = document.getElementById('backendStatus');
const sendButton = document.getElementById('send');
const attachButton = document.getElementById('attach');
const newChatButton = document.getElementById('newChat');
let isBusy = false;
let activeSessionId = null;
let pendingFile = null;
let activeAbortController = null;
let activeRequestId = null;
let activeThinkingBubble = null;
let requestStartedAt = null;
let requestTimerInterval = null;

function renderThemeLabel() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  themeToggle.textContent = dark ? 'Light mode' : 'Dark mode';
}
renderThemeLabel();
themeToggle.onclick = () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  renderThemeLabel();
};

function esc(s) {
  return (s ?? '').toString().replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function markdown(value, { inline = false } = {}) {
  const source = (value ?? '').toString();
  const rendered = inline ? marked.parseInline(source) : marked.parse(source);
  const template = document.createElement('template');
  template.innerHTML = DOMPurify.sanitize(rendered);
  template.content.querySelectorAll('a').forEach((link) => {
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  });
  return template.innerHTML;
}

function addMessage(role, html) {
  const empty = document.getElementById('empty');
  if (empty) empty.remove();
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.innerHTML = `<div class="bubble">${html}</div>`;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}

function selectedModel() {
  return modelInput.value || 'qwen3:8b';
}

function syncEmbeddingControl() {
  // TF-IDF ranking ignores dense embeddings; grey out the embedding picker.
  const tfidf = retrievalStrategyInput.value === 'tfidf';
  embeddingModelInput.disabled = tfidf;
  embeddingModelLabel.style.opacity = tfidf ? '0.55' : '1';
}

function options() {
  return {
    llmModel: selectedModel(),
    style: styleInput.value,
    promptStrategy: promptStrategyInput.value,
    retrievalMode: 'offline',
    retrievalStrategy: retrievalStrategyInput.value,
    retrievalEmbeddingModel: embeddingModelInput.value,
    retrievalTopK: Number(retrievalTopKInput.value),
    noRelatedPapers: noRelatedInput.checked,
    includePeerReview: includePeerReviewInput.checked,
    sessionId: activeSessionId,
  };
}

function list(items, renderItem, className = '') {
  if (!Array.isArray(items) || items.length === 0) return '';
  return `<ul class="${className}">${items.map(renderItem).join('')}</ul>`;
}

function resultHtml(r) {
  const metadata = r.metadata || {};
  const authors = Array.isArray(metadata.authors) ? metadata.authors.join(', ') : metadata.authors;
  const title = metadata.title || r.input_ref || 'Analysis result';
  const recs = list(r.recommended_papers, (p) =>
    `<li><b>${esc(p.title)}</b> <span class="score">${esc(p.score)}</span>` +
    `${p.why ? `<br><span class="muted">${markdown(p.why, { inline: true })}</span>` : ''}</li>`);
  const cites = list(r.apa_citations, (c) => `<li>${esc(c)}</li>`, 'cites');
  const findings = list(r.key_findings, (x) => `<li>${markdown(x, { inline: true })}</li>`);
  const gaps = list(r.research_gaps, (x) => `<li>${markdown(x, { inline: true })}</li>`);
  const evidence = list(r.evidence, (item) =>
    `<li><span class="evidence-id">${esc(item.paper_id || item.source || 'source')}</span>` +
    ` ${esc(item.text || '')}` +
    `${item.score != null ? ` <span class="score">${esc(item.score)}</span>` : ''}</li>`,
  'evidence');
  const sourceFlags = r.flags && r.flags.provider_sources
    ? `<div class="source-flags">${Object.entries(r.flags.provider_sources)
      .map(([key, value]) => `<span>${esc(key)}: ${esc(value)}</span>`).join('')}</div>`
    : '';
  const analysis = r.paper_analysis || {};
  const keyphrases = list(analysis.keyphrases, (item) =>
    `<li><b>${esc(item.text)}</b>` +
    `${item.section ? ` <span class="muted">(${esc(item.section)})</span>` : ''}` +
    `${item.score != null ? ` <span class="score">${esc(item.score)}</span>` : ''}</li>`);
  const entityMentions = Array.isArray(analysis.entity_mentions)
    ? analysis.entity_mentions.slice(0, 40)
    : [];
  const entities = list(entityMentions, (item) =>
    `<li><span class="evidence-id">${esc(item.type)}</span> ${esc(item.text)}` +
    ` <span class="muted">${esc(item.source || '')}</span></li>`);
  const structural = list(analysis.structural_checks, (item) =>
    `<li><b>${esc(item.severity)}</b>: ${esc(item.message)}` +
    `${item.evidence ? `<br><span class="muted">${esc(item.evidence)}</span>` : ''}</li>`);
  const extractive = analysis.extractive_summary && analysis.extractive_summary.text
    ? `<div class="section"><h3>Extractive summary</h3>` +
      `<p class="prewrap">${esc(analysis.extractive_summary.text)}</p>` +
      `<p class="muted">Deterministic TextRank; sentences remain traceable to the PDF.</p></div>`
    : '';
  const pos = analysis.pos
    ? `<div class="section"><h3>POS and syntax</h3>` +
      `<p>${esc(analysis.pos.token_count || 0)} tokens; ` +
      `${esc((analysis.pos.noun_chunks || []).length)} displayed noun chunks.</p></div>`
    : '';

  return `
    <div class="result-title">
      <h2>${esc(title)}</h2>
      ${authors ? `<p>${esc(authors)}</p>` : ''}
      ${metadata.arxiv_id ? `<p>arXiv: ${esc(metadata.arxiv_id)}</p>` : ''}
      ${sourceFlags}
    </div>
    <div class="section"><h3>Summary</h3><div class="markdown">${markdown(r.summary || 'No summary returned.')}</div></div>
    ${extractive}
    ${pos}
    ${keyphrases ? `<div class="section"><h3>Keyphrases</h3>${keyphrases}</div>` : ''}
    ${entities ? `<div class="section"><h3>Scientific entities</h3>${entities}</div>` : ''}
    ${structural ? `<div class="section"><h3>Structural checks</h3>${structural}</div>` : ''}
    ${findings ? `<div class="section"><h3>Key findings</h3>${findings}</div>` : ''}
    ${gaps ? `<div class="section"><h3>Research gaps</h3>${gaps}</div>` : ''}
    ${recs ? `<div class="section"><h3>Recommended papers</h3>${recs}</div>` : ''}
    ${cites ? `<div class="section"><h3>APA citations</h3>${cites}</div>` : ''}
    ${evidence ? `<div class="section"><h3>Evidence</h3>${evidence}</div>` : ''}
    ${r.peer_review ? `<div class="section"><h3>Peer review</h3><div class="markdown">${markdown(r.peer_review)}</div></div>` : ''}
  `;
}

function answerHtml(response) {
  return `<div class="markdown">${markdown(response.answer || 'No answer returned.')}</div>`;
}

function paperUrl(paper) {
  const direct = (paper.url || '').trim();
  if (direct) return direct;
  const pid = (paper.paper_id || '').trim();
  if (pid) return `https://arxiv.org/abs/${pid}`;
  return '';
}

function ragMessageHtml(response) {
  const papers = Array.isArray(response.recommended_papers)
    ? response.recommended_papers
    : [];
  const citations = Array.isArray(response.apa_citations)
    ? response.apa_citations
    : [];
  const sourceItems = papers.length
    ? papers.map((paper, index) => {
      const url = paperUrl(paper);
      const label = paper.title || paper.paper_id || `Source ${index + 1}`;
      const title = url
        ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`
        : esc(label);
      const citation = paper.apa_citation || citations[index] || '';
      return `<li>
        <div class="rag-source-title">${title}</div>
        ${citation ? `<div class="rag-source-citation">${esc(citation)}</div>` : ''}
      </li>`;
    })
    : citations.map((citation) =>
      `<li><div class="rag-source-citation">${esc(citation)}</div></li>`);
  const sources = sourceItems.length
    ? `<div class="section rag-sources-section">
        <h3>Sources</h3>
        <ol class="rag-sources">${sourceItems.join('')}</ol>
      </div>`
    : '';
  return `${answerHtml(response)}${sources}`;
}

function recommendationHtml(response) {
  const papers = response.recommended_papers || [];
  const cards = papers.map((paper, index) => {
    const url = paperUrl(paper);
    const title = url
      ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(paper.title)}</a>`
      : esc(paper.title);
    const authors = Array.isArray(paper.authors)
      ? paper.authors.join(', ')
      : (paper.authors || '');
    const meta = [
      authors ? esc(authors) : '',
      paper.year ? esc(paper.year) : '',
      paper.score != null ? `score ${esc(paper.score)}` : '',
    ].filter(Boolean).join(' · ');
    return `<li class="paper-card">
      <div class="paper-card-title"><span class="paper-rank">${index + 1}.</span> ${title}</div>
      ${meta ? `<div class="paper-card-meta">${meta}</div>` : ''}
      <div class="paper-card-summary markdown">${markdown(paper.summary || 'No summary available.')}</div>
    </li>`;
  }).join('');
  return `
    <div class="markdown">${markdown(response.answer || 'Recommended papers:')}</div>
    <ol class="paper-cards">${cards}</ol>
  `;
}

function chatResponseHtml(response) {
  // Backend sets kind; RAG turns may still carry recommended_papers without that flag.
  if (response.kind === 'paper_recommendations') return recommendationHtml(response);
  if ((response.recommended_papers || []).length || (response.apa_citations || []).length) {
    return ragMessageHtml(response);
  }
  return answerHtml(response);
}

function renderTranscript(messages) {
  thread.innerHTML = EMPTY;
  for (const message of messages || []) {
    if (message.role === 'user' && message.kind === 'text') {
      addMessage('user', esc(message.content));
    } else if (message.role === 'user' && message.kind === 'pdf_attachment') {
      addMessage('user', `Attached PDF: ${esc(message.content)}`);
    } else if (message.role === 'assistant' && message.kind === 'analysis_result') {
      addMessage('assistant', resultHtml(message.content || {}));
    } else if (message.role === 'assistant' && message.kind === 'text') {
      addMessage('assistant', answerHtml({ answer: message.content }));
    } else if (message.role === 'assistant' && message.kind === 'paper_recommendations') {
      addMessage('assistant', recommendationHtml(message.content || {}));
    } else if (message.role === 'assistant' && message.kind === 'rag_message') {
      addMessage('assistant', ragMessageHtml(message.content || {}));
    }
  }
}

function formatDuration(ms) {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  if (totalSec < 60) return `${totalSec}s`;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return sec > 0 ? `${min}m ${sec}s` : `${min}m`;
}

function elapsedMs() {
  return requestStartedAt ? Date.now() - requestStartedAt : 0;
}

function startRequestTimer() {
  requestStartedAt = Date.now();
  clearInterval(requestTimerInterval);
  requestTimerInterval = setInterval(() => {
    if (!isBusy) return;
    updateWorkingElapsed();
  }, 1000);
}

function stopRequestTimer() {
  clearInterval(requestTimerInterval);
  requestTimerInterval = null;
}

function thinkingMarkup(label = 'Thinking') {
  return `<span class="thinking-indicator"><span class="thinking-label">${esc(label)}</span><span class="thinking-dots" aria-hidden="true"><span>.</span><span>.</span><span>.</span></span></span>`;
}

function workingBubbleMarkup(label = 'Thinking', ms = elapsedMs()) {
  return `<div class="thinking-block">` +
    `<span class="request-timing working">Working on ${formatDuration(ms)}</span>` +
    `<span class="thinking-sep" aria-hidden="true">·</span>` +
    thinkingMarkup(label) +
    `</div>`;
}

function workedTimingMarkup(ms, state = 'ok') {
  const label = state === 'stopped'
    ? 'Stopped after'
    : state === 'failed'
      ? 'Failed after'
      : 'Worked on';
  return `<div class="request-timing ${state}">${label} ${formatDuration(ms)}</div>`;
}

function updateWorkingElapsed() {
  const timing = activeThinkingBubble?.querySelector('.request-timing.working');
  if (timing) {
    timing.textContent = `Working on ${formatDuration(elapsedMs())}`;
  }
}

function clearFooterStatus() {
  backendStatus.textContent = '';
  backendStatus.classList.remove('ok', 'failed', 'stopped', 'working');
}

function setStatus(message, state = '') {
  backendStatus.textContent = message;
  backendStatus.classList.remove('working', 'stopped');
  backendStatus.classList.toggle('ok', state === 'ok');
  backendStatus.classList.toggle('working', state === 'working');
  backendStatus.classList.toggle('failed', state === 'failed');
}

function updateSendState() {
  if (isBusy) {
    sendButton.disabled = false;
    sendButton.classList.add('stop');
    sendButton.title = 'Stop';
    sendButton.setAttribute('aria-label', 'Stop');
    sendButton.textContent = '■';
    return;
  }
  sendButton.classList.remove('stop');
  sendButton.title = 'Send';
  sendButton.setAttribute('aria-label', 'Send');
  sendButton.textContent = '↑';
  const hasContent = Boolean(input.value.trim()) || Boolean(pendingFile);
  const disabled = !hasContent;
  sendButton.disabled = disabled;
  sendButton.setAttribute('aria-disabled', String(disabled));
}

function isCancelledError(error) {
  if (!error) return false;
  if (error.name === 'AbortError') return true;
  const message = String(error.message || '');
  return message.includes('backend 499') || /cancelled/i.test(message);
}

async function stopActiveRequest() {
  if (!isBusy) return;
  const requestId = activeRequestId;
  const controller = activeAbortController;
  if (requestId) {
    try {
      await cancelRequest(requestId);
    } catch {
      // The backend may already have finished.
    }
  }
  if (controller) controller.abort();
}

function renderAttachment() {
  if (!pendingFile) {
    attachments.hidden = true;
    attachments.innerHTML = '';
    updateSendState();
    return;
  }
  attachments.hidden = false;
  attachments.innerHTML = `
    <div class="chip" title="${esc(pendingFile.name)}">
      <span class="chip-icon" aria-hidden="true">PDF</span>
      <span class="chip-name">${esc(pendingFile.name)}</span>
      <button class="chip-remove" id="chipRemove" type="button" title="Remove attachment" aria-label="Remove attachment">×</button>
    </div>`;
  const removeButton = document.getElementById('chipRemove');
  removeButton.onclick = () => {
    if (isBusy) return;
    clearAttachment();
    input.focus();
  };
  updateSendState();
}

function clearAttachment() {
  pendingFile = null;
  fileInput.value = '';
  renderAttachment();
}

function setBusy(busy) {
  isBusy = busy;
  attachButton.disabled = busy;
  newChatButton.disabled = busy;
  attachButton.setAttribute('aria-disabled', String(busy));
  newChatButton.setAttribute('aria-disabled', String(busy));
  const chipRemove = document.getElementById('chipRemove');
  if (chipRemove) chipRemove.disabled = busy;
  [
    modelInput,
    retrievalStrategyInput,
    embeddingModelInput,
    retrievalTopKInput,
    styleInput,
    promptStrategyInput,
    noRelatedInput,
    includePeerReviewInput,
  ].forEach((control) => { control.disabled = busy; });
  updateSendState();
}

function populateModels(catalog) {
  const entries = Array.isArray(catalog.models) ? catalog.models : [];
  const available = entries.filter((model) => model.available);
  if (available.length === 0) {
    modelInput.innerHTML = '<option value="">No project models installed</option>';
    throw new Error('Install qwen3:8b and qwen3-research-lora:latest in Ollama');
  }
  modelInput.innerHTML = entries.map((model) => {
    const suffix = model.available ? '' : ' (not installed)';
    return `<option value="${esc(model.id)}" ${model.available ? '' : 'disabled'}>` +
      `${esc(model.label)}${suffix}</option>`;
  }).join('');
  const preferred = catalog.active_model || catalog.default_model;
  const preferredAvailable = available.some((model) => model.id === preferred);
  modelInput.value = preferredAvailable ? preferred : available[0].id;
}

const backendError =
  `<span class="err">Could not complete the request. Check the local API, ` +
  `selected backend, and model settings.</span>`;

async function runRequest(callFactory, thinkingLabel, render) {
  // Send button doubles as stop: abort fetch and POST /api/requests/{id}/cancel.
  if (isBusy) return;
  activeAbortController = new AbortController();
  activeRequestId = newRequestId();
  setBusy(true);
  startRequestTimer();
  clearFooterStatus();
  const thinking = addMessage('assistant', workingBubbleMarkup(thinkingLabel));
  activeThinkingBubble = thinking.querySelector('.bubble');
  try {
    const r = await callFactory({
      signal: activeAbortController.signal,
      requestId: activeRequestId,
    });
    const duration = elapsedMs();
    activeThinkingBubble.innerHTML =
      workedTimingMarkup(duration, 'ok') + render(r);
  } catch (e) {
    const duration = elapsedMs();
    if (isCancelledError(e)) {
      activeThinkingBubble.innerHTML =
        workedTimingMarkup(duration, 'stopped') + '<span class="muted">Stopped.</span>';
    } else {
      activeThinkingBubble.innerHTML =
        workedTimingMarkup(duration, 'failed') +
        `${backendError}<div class="error-detail">${esc(e.message)}</div>`;
    }
  } finally {
    stopRequestTimer();
    requestStartedAt = null;
    activeAbortController = null;
    activeRequestId = null;
    activeThinkingBubble = null;
    setBusy(false);
  }
  thread.scrollTop = thread.scrollHeight;
}

function handleSend() {
  if (isBusy || !activeSessionId) return;
  const text = input.value.trim();
  if (pendingFile) {
    const file = pendingFile;
    const note = text
      ? `<div class="attach-note">${esc(text)}</div>`
      : '';
    addMessage('user', `${note}<div class="attach-line"><span class="chip-icon" aria-hidden="true">PDF</span> ${esc(file.name)}</div>`);
    input.value = '';
    input.style.height = 'auto';
    clearAttachment();
    runRequest(
      ({ signal, requestId }) => analyzePdf(file, { ...options(), signal, requestId }),
      'Parsing and analysing the paper',
      resultHtml,
    );
    return;
  }
  if (!text) return;
  addMessage('user', esc(text));
  input.value = '';
  input.style.height = 'auto';
  updateSendState();
  runRequest(
    ({ signal, requestId }) => chat(text, { ...options(), signal, requestId }),
    'Thinking',
    chatResponseHtml,
  );
}

sendButton.onclick = () => {
  if (isBusy) {
    stopActiveRequest();
    return;
  }
  handleSend();
};
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';
  updateSendState();
});

attachButton.onclick = () => {
  if (!isBusy) fileInput.click();
};
fileInput.onchange = () => {
  const f = fileInput.files[0];
  if (!f) return;
  if (isBusy) {
    fileInput.value = '';
    return;
  }
  pendingFile = f;
  renderAttachment();
  input.focus();
};

newChatButton.onclick = async () => {
  if (isBusy) return;
  setBusy(true);
  setStatus('Starting a new analysis...', 'working');
  try {
    if (activeSessionId) await completeSession(activeSessionId);
    const created = await createSession();
    activeSessionId = created.session_id;
    sessionStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
    thread.innerHTML = EMPTY;
    input.value = '';
    input.style.height = 'auto';
    clearAttachment();
    setStatus('New analysis ready', 'ok');
    input.focus();
  } catch (e) {
    setStatus(`Could not start a new analysis: ${e.message}`, 'failed');
  } finally {
    setBusy(false);
  }
};

async function initializeSession() {
  // Reuse the last active server session when the tab reloads mid-conversation.
  const stored = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (stored) {
    try {
      const existing = await getSession(stored);
      if (existing.state === 'active') {
        activeSessionId = existing.session_id;
        renderTranscript(existing.transcript);
        return;
      }
    } catch {
      sessionStorage.removeItem(SESSION_STORAGE_KEY);
    }
  }
  const created = await createSession();
  activeSessionId = created.session_id;
  sessionStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
  renderTranscript([]);
}

retrievalStrategyInput.onchange = syncEmbeddingControl;
syncEmbeddingControl();

async function bootstrap() {
  setBusy(true);
  setStatus('Restoring analysis session...', 'working');
  try {
    await initializeSession();
    const [result, modelCatalog] = await Promise.all([health(), models()]);
    populateModels(modelCatalog);
    setStatus(
      `Backend ready (${Object.keys(result.provider_sources || {}).length} providers, ` +
      `${modelInput.options.length} models)`,
      'ok',
    );
  } catch (e) {
    setStatus(`Backend unavailable: ${e.message}`, 'failed');
  } finally {
    setBusy(false);
  }
}

bootstrap();
