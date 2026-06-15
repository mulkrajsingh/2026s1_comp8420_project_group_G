// Calls to the canonical integration API (proxied by Vite in development).

function buildHeaders(isForm, requestId) {
  const headers = {};
  if (!isForm) {
    headers['Content-Type'] = 'application/json';
  }
  if (requestId) {
    headers['X-Request-Id'] = requestId;
  }
  return headers;
}

function formatBackendDetail(detail) {
  if (detail == null) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') return item;
        const loc = Array.isArray(item.loc) ? item.loc.join('.') : '';
        const msg = item.msg || JSON.stringify(item);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join('; ');
  }
  if (typeof detail === 'object') return JSON.stringify(detail);
  return String(detail);
}

async function post(path, body, isForm = false, { signal, requestId } = {}) {
  const opts = {
    method: 'POST',
    headers: buildHeaders(isForm, requestId),
    signal,
  };
  opts.body = isForm ? body : JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const raw = await res.text();
    let detail = raw;
    try {
      detail = formatBackendDetail(JSON.parse(raw).detail ?? raw);
    } catch {
      // Keep non-JSON backend errors readable.
    }
    throw new Error(`backend ${res.status}: ${detail}`);
  }
  return res.json();
}

async function get(path, { signal, requestId } = {}) {
  const headers = requestId ? { 'X-Request-Id': requestId } : undefined;
  const res = await fetch(path, { signal, headers });
  if (!res.ok) {
    const raw = await res.text();
    let detail = raw;
    try {
      detail = formatBackendDetail(JSON.parse(raw).detail ?? raw);
    } catch {
      // Keep non-JSON backend errors readable.
    }
    throw new Error(`backend ${res.status}: ${detail}`);
  }
  return res.json();
}

export function newRequestId() {
  return crypto.randomUUID();
}

export function cancelRequest(requestId) {
  return post(`/api/requests/${encodeURIComponent(requestId)}/cancel`, {});
}

export function createSession() {
  return post('/api/sessions', {});
}

export function getSession(sessionId) {
  return get(`/api/sessions/${encodeURIComponent(sessionId)}`);
}

export function completeSession(sessionId) {
  return post(`/api/sessions/${encodeURIComponent(sessionId)}/complete`, {});
}

export function models() {
  return get('/api/models');
}

export function searchTopic(query, options = {}) {
  return post('/api/search-topic', {
    query,
    style: options.style || 'auto',
    retrieval_mode: options.retrievalMode || 'offline',
    retrieval_embedding_model: options.retrievalEmbeddingModel || 'allenai/specter2_base',
    retrieval_strategy: options.retrievalStrategy || 'hybrid_rrf',
    retrieval_top_k: options.retrievalTopK || 5,
    llm_model: options.llmModel || 'qwen3:8b',
    prompt_strategy: options.promptStrategy || 'zero_shot',
  }, false, {
    signal: options.signal,
    requestId: options.requestId,
  });
}

export function chat(question, options = {}) {
  return post('/api/chat', {
    question,
    style: options.style || 'auto',
    retrieval_mode: options.retrievalMode || 'offline',
    retrieval_embedding_model: options.retrievalEmbeddingModel || 'allenai/specter2_base',
    retrieval_strategy: options.retrievalStrategy || 'hybrid_rrf',
    retrieval_top_k: options.retrievalTopK || 5,
    llm_model: options.llmModel || 'qwen3:8b',
    session_id: options.sessionId || null,
    prompt_strategy: options.promptStrategy || 'zero_shot',
  }, false, {
    signal: options.signal,
    requestId: options.requestId,
  });
}

export function analyzePdf(file, options = {}) {
  const fd = new FormData();
  fd.append('file', file);
  const params = new URLSearchParams({
    style: options.style || 'auto',
    retrieval_mode: options.retrievalMode || 'offline',
    retrieval_embedding_model: options.retrievalEmbeddingModel || 'allenai/specter2_base',
    retrieval_strategy: options.retrievalStrategy || 'hybrid_rrf',
    retrieval_top_k: String(options.retrievalTopK || 5),
    no_related_papers: String(Boolean(options.noRelatedPapers)),
    include_peer_review: String(Boolean(options.includePeerReview)),
    llm_model: options.llmModel || 'qwen3:8b',
  });
  params.set('prompt_strategy', options.promptStrategy || 'zero_shot');
  if (options.sessionId) params.set('session_id', options.sessionId);
  return post(`/api/analyze-pdf?${params}`, fd, true, {
    signal: options.signal,
    requestId: options.requestId,
  });
}

export async function health() {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error(`backend ${res.status}`);
  return res.json();
}
