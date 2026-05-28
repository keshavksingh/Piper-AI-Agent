// ── Piper AI Agent — WebSocket Client with JWT Auth ─────────────

const loginContainer = document.getElementById('login-container');
const chatContainer = document.getElementById('chat-container');
const loginBtn = document.getElementById('login-btn');
const loginEmail = document.getElementById('login-email');
const loginPassword = document.getElementById('login-password');
const loginError = document.getElementById('login-error');
const logoutLink = document.getElementById('logout-link');
const userDisplayName = document.getElementById('user-display-name');

const chatbox = document.getElementById('chatbox');
const chatBody = document.getElementById('chat-body');
const welcomeGreeting = document.getElementById('welcome-greeting');
const userinput = document.getElementById('userinput');
const sendbtn = document.getElementById('sendbtn');
const suggestions = document.getElementById('suggestions');

let ws = null;
let sessionId = null;
let authToken = localStorage.getItem('auth_token');
let currentUser = JSON.parse(localStorage.getItem('current_user') || 'null');

let currentAssistantDiv = null;
let assistantText = '';

// ── Processing Lock State ────────────────────────────────────────
let isProcessing = false;
let reasoningPanelDiv = null;
let reasoningStepCount = 0;
let processingTimeout = null;

// ── Login / Logout ──────────────────────────────────────────────

async function doLogin() {
    const email = loginEmail.value.trim();
    const password = loginPassword.value;

    if (!email || !password) {
        showLoginError('Please enter email and password.');
        return;
    }

    loginBtn.disabled = true;
    loginBtn.textContent = 'Signing in...';
    hideLoginError();

    try {
        const resp = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        const data = await resp.json();

        if (!resp.ok) {
            showLoginError(data.error || 'Login failed.');
            return;
        }

        // Store token and user info
        authToken = data.token;
        currentUser = data.user;
        localStorage.setItem('auth_token', authToken);
        localStorage.setItem('current_user', JSON.stringify(currentUser));

        showChat();
        connect();

    } catch (err) {
        showLoginError('Network error. Please try again.');
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = 'Sign In';
    }
}

function doLogout() {
    authToken = null;
    currentUser = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('current_user');
    localStorage.removeItem('customer_id');

    // Clear processing state before closing WS to prevent onclose handler
    // from calling unlockInput/focus on elements about to be hidden
    isProcessing = false;
    reasoningPanelDiv = null;
    reasoningStepCount = 0;
    if (processingTimeout) {
        clearTimeout(processingTimeout);
        processingTimeout = null;
    }

    if (ws) {
        ws.close();
        ws = null;
    }
    sessionId = null;

    showLogin();
}

function showLogin() {
    loginContainer.style.display = 'flex';
    chatContainer.style.display = 'none';
    loginEmail.value = '';
    loginPassword.value = '';
    hideLoginError();
}

function showChat() {
    loginContainer.style.display = 'none';
    chatContainer.style.display = 'flex';

    if (currentUser && currentUser.display_name) {
        userDisplayName.textContent = currentUser.display_name;
        const firstName = currentUser.display_name.split(' ')[0];
        welcomeGreeting.textContent = "What's next, " + firstName + "?";
    }

    chatbox.innerHTML = '';
    suggestions.innerHTML = '';
    suggestions.style.display = 'none';
    chatBody.classList.add('welcome-active');

    // Reset assistant streaming state (prevents stale ref to detached DOM after re-login)
    currentAssistantDiv = null;
    assistantText = '';

    // Reset processing state (handles re-login while prior session was mid-processing)
    isProcessing = false;
    reasoningPanelDiv = null;
    reasoningStepCount = 0;
    if (processingTimeout) {
        clearTimeout(processingTimeout);
        processingTimeout = null;
    }
    userinput.disabled = false;
    sendbtn.disabled = false;
    userinput.classList.remove('input-disabled');
    sendbtn.classList.remove('btn-disabled');
}

function enterConversationMode() {
    chatBody.classList.remove('welcome-active');
}

function showLoginError(msg) {
    loginError.textContent = msg;
    loginError.style.display = 'block';
}

function hideLoginError() {
    loginError.style.display = 'none';
}

// ── WebSocket Connection ─────────────────────────────────────────

function connect() {
    if (!authToken) {
        showLogin();
        return;
    }

    ws = new WebSocket(`ws://${location.host}/ws/chat`);

    ws.onopen = () => {
        console.log('WebSocket connected');
        ws.send(JSON.stringify({
            type: 'session_start',
            token: authToken,
        }));
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected. Reconnecting...');
        if (isProcessing) {
            collapseReasoningPanel();
            unlockInput();
        }
        if (authToken) {
            setTimeout(connect, 3000);
        }
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

// ── Processing Lock Functions ────────────────────────────────────

function lockInput() {
    isProcessing = true;
    userinput.disabled = true;
    sendbtn.disabled = true;
    userinput.classList.add('input-disabled');
    sendbtn.classList.add('btn-disabled');

    document.querySelectorAll('.suggestion-btn').forEach(btn => {
        btn.classList.add('suggestion-disabled');
    });

    // Safety timeout — auto-unlock after 120s to prevent permanent lock
    if (processingTimeout) clearTimeout(processingTimeout);
    processingTimeout = setTimeout(() => {
        console.warn('Processing safety timeout reached (120s)');
        collapseReasoningPanel();
        unlockInput();
    }, 120000);
}

function unlockInput() {
    isProcessing = false;
    userinput.disabled = false;
    sendbtn.disabled = false;
    userinput.classList.remove('input-disabled');
    sendbtn.classList.remove('btn-disabled');

    document.querySelectorAll('.suggestion-btn').forEach(btn => {
        btn.classList.remove('suggestion-disabled');
    });

    if (processingTimeout) {
        clearTimeout(processingTimeout);
        processingTimeout = null;
    }

    // Only refocus if the chat UI is visible (not during logout/auth-error)
    if (chatContainer.style.display !== 'none') {
        userinput.focus();
    }
}

// ── Reasoning Panel ──────────────────────────────────────────────

function createReasoningPanel() {
    reasoningPanelDiv = document.createElement('div');
    reasoningPanelDiv.className = 'reasoning-panel';
    reasoningStepCount = 0;

    const header = document.createElement('div');
    header.className = 'reasoning-header';
    header.onclick = toggleReasoningPanel;
    header.innerHTML =
        '<span class="reasoning-toggle">&#9660;</span>' +
        '<span class="reasoning-title">Reasoning</span>' +
        '<span class="reasoning-count"></span>' +
        '<span class="reasoning-pulse">&#8226;</span>';

    const body = document.createElement('div');
    body.className = 'reasoning-body';

    reasoningPanelDiv.appendChild(header);
    reasoningPanelDiv.appendChild(body);
    chatbox.appendChild(reasoningPanelDiv);
    scrollToBottom();
}

function addReasoningStep(type, content) {
    if (!reasoningPanelDiv) {
        createReasoningPanel();
    }

    reasoningStepCount++;

    const body = reasoningPanelDiv.querySelector('.reasoning-body');
    const step = document.createElement('div');
    step.className = 'reasoning-step';

    const badge = document.createElement('span');
    badge.className = 'reasoning-badge badge-' + type;
    badge.textContent = type;

    const text = document.createElement('span');
    text.className = 'reasoning-text';
    text.innerHTML = content;

    step.appendChild(badge);
    step.appendChild(text);
    body.appendChild(step);

    const countEl = reasoningPanelDiv.querySelector('.reasoning-count');
    countEl.textContent = reasoningStepCount + ' step' + (reasoningStepCount !== 1 ? 's' : '');

    const titleEl = reasoningPanelDiv.querySelector('.reasoning-title');
    const titles = {
        thinking: 'Thinking',
        planning: 'Planning',
        agent: 'Agent working',
        reflection: 'Reflecting',
        tool: 'Using tools',
        learning: 'Learning',
        guardrail: 'Safety check',
    };
    titleEl.textContent = titles[type] || 'Reasoning';

    scrollToBottom();
}

function collapseReasoningPanel() {
    if (!reasoningPanelDiv) return;

    if (reasoningStepCount === 0) {
        reasoningPanelDiv.remove();
        reasoningPanelDiv = null;
        return;
    }

    reasoningPanelDiv.classList.add('reasoning-complete');
    const body = reasoningPanelDiv.querySelector('.reasoning-body');
    body.classList.add('reasoning-collapsed');

    const titleEl = reasoningPanelDiv.querySelector('.reasoning-title');
    titleEl.textContent = 'Reasoned through ' + reasoningStepCount + ' step' + (reasoningStepCount !== 1 ? 's' : '');

    const toggle = reasoningPanelDiv.querySelector('.reasoning-toggle');
    toggle.innerHTML = '&#9654;';

    reasoningPanelDiv = null;
    reasoningStepCount = 0;
}

function toggleReasoningPanel() {
    const panel = this.parentElement;
    if (!panel.classList.contains('reasoning-complete')) return;

    const body = panel.querySelector('.reasoning-body');
    const toggle = panel.querySelector('.reasoning-toggle');

    body.classList.toggle('reasoning-collapsed');
    toggle.innerHTML = body.classList.contains('reasoning-collapsed') ? '&#9654;' : '&#9660;';
}

// ── Message Handler ──────────────────────────────────────────────

function handleMessage(msg) {
    switch (msg.type) {
        case 'session_ready':
            sessionId = msg.session_id;
            console.log('Session ready:', sessionId);
            break;

        case 'recommendations':
            showRecommendations(msg.suggestions || JSON.parse(msg.payload || '[]'));
            break;

        case 'processing_started':
            lockInput();
            break;

        case 'token':
            streamToken(msg.payload);
            break;

        case 'agent_thinking':
            showThinking(msg.payload);
            break;

        case 'clarification':
            collapseReasoningPanel();
            showClarification(msg.payload);
            break;

        case 'response_complete':
            completeResponse(msg.payload);
            collapseReasoningPanel();
            unlockInput();
            break;

        case 'auth_error':
            if (isProcessing) {
                collapseReasoningPanel();
                unlockInput();
            }
            doLogout();
            showLoginError(msg.message || 'Session expired. Please sign in again.');
            break;

        case 'reflection_evaluating':
        case 'reflection_refining':
            showReflectionThinking(msg.payload);
            break;

        case 'reflection_critique':
            showReflectionCritique(msg.payload);
            break;

        case 'reflexion_learning':
            showReflexionLearning(msg.payload);
            break;

        case 'tool_validation_error':
            showToolValidationError(msg.payload);
            break;

        case 'agent_planning':
            showPlanningIndicator(msg.payload);
            break;

        case 'agent_started':
            showAgentStarted(msg.payload);
            break;

        case 'agent_complete':
            showAgentComplete(msg.payload);
            break;

        case 'guardrail_blocked':
            showGuardrailBlocked(msg.payload);
            collapseReasoningPanel();
            unlockInput();
            break;

        case 'guardrail_sanitized':
            showGuardrailSanitized(msg.payload);
            break;

        case 'error':
            showError(msg.message || msg.payload);
            collapseReasoningPanel();
            unlockInput();
            break;

        default:
            console.log('Unknown message type:', msg.type);
    }
}

// ── UI Helpers ───────────────────────────────────────────────────

function addUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user-message';
    const content = document.createElement('div');
    content.className = 'msg-content';
    content.textContent = text;
    div.appendChild(content);
    chatbox.appendChild(div);
    scrollToBottom();
}

function startAssistantMessage() {
    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant-message';
    const content = document.createElement('div');
    content.className = 'msg-content';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.title = 'Copy to clipboard';
    copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    copyBtn.onclick = function() { copyMessage(this); };
    wrapper.appendChild(content);
    wrapper.appendChild(copyBtn);
    chatbox.appendChild(wrapper);
    currentAssistantDiv = content;
    assistantText = '';
}

function copyMessage(btn) {
    const msgContent = btn.parentElement.querySelector('.msg-content');
    if (!msgContent) return;
    const text = msgContent.innerText;
    navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
        }, 2000);
    });
}

function streamToken(token) {
    if (!currentAssistantDiv) {
        startAssistantMessage();
    }
    assistantText += token;
    currentAssistantDiv.innerHTML = renderMarkdown(assistantText);
    scrollToBottom();
}

function showThinking(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { step: payload }; }
    addReasoningStep('thinking', escapeHtml(data.step || 'Thinking...'));
}

function showClarification(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { return; }

    const div = document.createElement('div');
    div.className = 'message clarification-message';
    const content = document.createElement('div');
    content.className = 'msg-content';

    let html = escapeHtml(data.message) + '<div class="clarification-options">';

    if (data.options) {
        data.options.forEach(opt => {
            html += '<button class="clarification-btn" onclick="sendClarification(\'' + escapeHtml(opt.value) + '\')">' + escapeHtml(opt.label) + '</button>';
        });
    }

    if (data.allow_freetext) {
        html += '<div class="clarification-freetext">' +
            '<input type="text" id="clarification-input" placeholder="Or type your answer...">' +
            '<button onclick="sendClarificationFreetext()">Send</button>' +
            '</div>';
    }

    html += '</div>';
    content.innerHTML = html;
    div.appendChild(content);
    chatbox.appendChild(div);
    scrollToBottom();
}

function completeResponse(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = {}; }

    if (data.recommendations && data.recommendations.length > 0) {
        showRecommendations(data.recommendations);
    }

    if (data.response && data.response.confidence) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'response-meta';
        let metaHtml = '';
        if (data.response.sources && data.response.sources.length > 0) {
            metaHtml += 'Sources: ' + data.response.sources.join(', ') + ' | ';
        }
        if (data.response.tools_used && data.response.tools_used.length > 0) {
            metaHtml += 'Tools: ' + data.response.tools_used.join(', ') + ' | ';
        }
        metaHtml += 'Confidence: ' + Math.round(data.response.confidence * 100) + '%';
        metaDiv.textContent = metaHtml;
        chatbox.appendChild(metaDiv);
    }

    currentAssistantDiv = null;
    assistantText = '';
    scrollToBottom();
}

function showRecommendations(recs) {
    if (!recs || recs.length === 0) return;

    suggestions.innerHTML = '';
    suggestions.style.display = 'flex';

    recs.forEach(rec => {
        const btn = document.createElement('button');
        btn.className = 'suggestion-btn';
        if (isProcessing) btn.classList.add('suggestion-disabled');
        btn.textContent = rec;
        btn.onclick = () => sendMessage(rec);
        suggestions.appendChild(btn);
    });
}

function showError(msg) {
    let text;
    try { text = JSON.parse(msg).message || msg; } catch { text = msg; }

    // Transition out of welcome mode so the error is visible in the chatbox
    if (chatBody.classList.contains('welcome-active')) {
        enterConversationMode();
    }

    const div = document.createElement('div');
    div.className = 'message error-message';
    const content = document.createElement('div');
    content.className = 'msg-content';
    content.textContent = 'Error: ' + text;
    div.appendChild(content);
    chatbox.appendChild(div);
    scrollToBottom();
}

function scrollToBottom() {
    chatBody.scrollTop = chatBody.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderMarkdown(text) {
    // Escape HTML first to prevent XSS, then layer markdown transforms on top
    let html = escapeHtml(text);

    // Bold: **text**
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic: *text* (safe to run after bold — no remaining ** sequences)
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Inline code: `code`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return html;
}

function showReflectionThinking(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { step: payload }; }
    addReasoningStep('reflection', escapeHtml(data.step || 'Reflecting...'));
}

function showReflectionCritique(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { return; }

    const score = data.score || 0;
    let scoreClass = 'score-low';
    if (score >= 0.8) scoreClass = 'score-good';
    else if (score >= 0.6) scoreClass = 'score-ok';

    const pct = Math.round(score * 100);
    let html = '<span class="reflection-score ' + scoreClass + '">' + pct + '%</span> Quality check';
    if (data.issues && data.issues.length > 0 && score < 0.75) {
        html += ' — Refining...';
    }
    addReasoningStep('reflection', html);
}

function showReflexionLearning(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { message: payload }; }
    addReasoningStep('learning', escapeHtml(data.message || 'Learning from this interaction...'));
}

function showToolValidationError(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { step: payload }; }
    addReasoningStep('tool', escapeHtml(data.step || data.message || 'Tool validation issue'));
}

function showPlanningIndicator(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { steps: [] }; }

    let html = 'Planning approach...';
    if (data.steps && data.steps.length > 0) {
        html += '<div class="plan-steps">';
        data.steps.forEach((step, i) => {
            html += '<div class="plan-step"><span class="step-num">' + (i + 1) + '</span>' + escapeHtml(step.goal || '');
            if (step.suggested_tool) {
                html += ' <span class="step-tool">(' + escapeHtml(step.suggested_tool) + ')</span>';
            }
            html += '</div>';
        });
        html += '</div>';
    }
    if (data.multi_agent) {
        html += '<div style="margin-top:4px;font-size:12px;color:#34d399;">Multi-agent mode activated</div>';
    }
    addReasoningStep('planning', html);
}

function showAgentStarted(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { agent_type: 'specialist' }; }
    addReasoningStep('agent', escapeHtml(data.description || data.agent_type || 'Specialist') + ' agent working...');
}

function showAgentComplete(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { agent_type: 'specialist' }; }

    let toolsInfo = '';
    if (data.tools_used && data.tools_used.length > 0) {
        toolsInfo = ' (used: ' + data.tools_used.join(', ') + ')';
    }
    addReasoningStep('agent', '<span class="agent-check">&#10003;</span> ' + escapeHtml(data.agent_type || 'Specialist') + ' complete' + escapeHtml(toolsInfo));
}

function showGuardrailBlocked(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { reason: 'Input blocked by safety filter' }; }

    const div = document.createElement('div');
    div.className = 'message guardrail-blocked-message';
    const content = document.createElement('div');
    content.className = 'msg-content';
    content.innerHTML = '<span class="guardrail-icon">&#9888;</span> ' + escapeHtml(data.reason || 'Your message was blocked by our safety filter. Please rephrase your query.');
    div.appendChild(content);
    chatbox.appendChild(div);
    currentAssistantDiv = null;
    assistantText = '';
    scrollToBottom();
}

function showGuardrailSanitized(payload) {
    let data;
    try { data = JSON.parse(payload); } catch { data = { redacted_types: [] }; }

    const types = (data.redacted_types || []).join(', ');
    addReasoningStep('guardrail', 'PII redacted from response: ' + escapeHtml(types));
}

// ── Send Functions ───────────────────────────────────────────────

function sendMessage(text) {
    if (isProcessing) return;
    if (!text || !text.trim()) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!sessionId) return;

    // Switch from welcome to conversation mode on first message
    if (chatBody.classList.contains('welcome-active')) {
        enterConversationMode();
    }

    lockInput();

    addUserMessage(text);
    suggestions.style.display = 'none';
    userinput.value = '';

    ws.send(JSON.stringify({
        type: 'user_message',
        text: text,
        session_id: sessionId,
    }));
}

function sendClarification(value) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(JSON.stringify({
        type: 'clarification_response',
        session_id: sessionId,
        selected_option: value,
    }));

    const btns = document.querySelectorAll('.clarification-options');
    btns.forEach(b => b.remove());
}

function sendClarificationFreetext() {
    const input = document.getElementById('clarification-input');
    if (!input || !input.value.trim()) return;

    ws.send(JSON.stringify({
        type: 'clarification_response',
        session_id: sessionId,
        selected_option: '',
        freetext: input.value.trim(),
    }));

    const btns = document.querySelectorAll('.clarification-options');
    btns.forEach(b => b.remove());
}

// ── Event Listeners ──────────────────────────────────────────────

loginBtn.onclick = doLogin;
logoutLink.onclick = doLogout;

loginPassword.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') doLogin();
});

loginEmail.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') loginPassword.focus();
});

sendbtn.onclick = () => sendMessage(userinput.value.trim());

userinput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage(userinput.value.trim());
});

// ── Initialize ───────────────────────────────────────────────────

if (authToken && currentUser) {
    showChat();
    connect();
} else {
    showLogin();
}
