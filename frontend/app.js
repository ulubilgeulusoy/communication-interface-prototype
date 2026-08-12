const settingsForm = document.querySelector("#settings-form");
const messageForm = document.querySelector("#message-form");
const llmForm = document.querySelector("#llm-form");
const userInput = document.querySelector("#user-id");
const receiverInput = document.querySelector("#receiver-id");
const sessionInput = document.querySelector("#session-id");
const contentInput = document.querySelector("#message-content");
const llmContentInput = document.querySelector("#llm-content");
const statusLabel = document.querySelector("#connection-status");
const conditionLabel = document.querySelector("#condition-label");
const userMessagesEl = document.querySelector("#user-messages");
const llmMessagesEl = document.querySelector("#llm-messages");
const userSelectionStatusEl = document.querySelector("#user-selection-status");
const llmSelectionStatusEl = document.querySelector("#llm-selection-status");
const llmStatusEl = document.querySelector("#llm-status");
const askLlmButton = document.querySelector("#ask-llm-button");
const contextMenuEl = document.querySelector("#message-menu");
const contextMenuBackdropEl = document.querySelector("#message-menu-backdrop");

let socket = null;
let currentCondition = null;
let messageCounter = 0;
let userThread = [];
let llmThread = [];
let selectedUserMessageIds = new Set();
let selectedLlmMessageIds = new Set();
let contextThread = null;

hideContextMenu();
userMessagesEl.dataset.empty = "No user messages yet.";
llmMessagesEl.dataset.empty = "No LLM messages yet.";
syncEmptyState(userMessagesEl, userThread);
syncEmptyState(llmMessagesEl, llmThread);
syncSelectionStatus();

function buildClientMessageId(prefix) {
  messageCounter += 1;
  return `${prefix}-${Date.now()}-${messageCounter}`;
}

function syncEmptyState(container, messages) {
  container.classList.toggle("is-empty", messages.length === 0);
}

function getSelectedIds(thread) {
  return thread === "user" ? selectedUserMessageIds : selectedLlmMessageIds;
}

function getMessages(thread) {
  return thread === "user" ? userThread : llmThread;
}

function getContainer(thread) {
  return thread === "user" ? userMessagesEl : llmMessagesEl;
}

function getThreadLabel(thread) {
  return thread === "user" ? "user" : "LLM";
}

function getSelectionStatusLabel(thread, count) {
  if (count === 0) {
    return thread === "user" ? "No user messages selected" : "No LLM messages selected";
  }
  if (count === 1) {
    return `1 ${getThreadLabel(thread)} message selected`;
  }
  return `${count} ${getThreadLabel(thread)} messages selected`;
}

function syncSelectionStatus() {
  userSelectionStatusEl.textContent = getSelectionStatusLabel("user", selectedUserMessageIds.size);
  llmSelectionStatusEl.textContent = getSelectionStatusLabel("llm", selectedLlmMessageIds.size);
}

function setLlmStatus(text) {
  llmStatusEl.textContent = text;
}

function updateSelectionStyles(thread) {
  const selectedIds = getSelectedIds(thread);
  const container = getContainer(thread);
  const cards = container.querySelectorAll(".message");
  for (const card of cards) {
    card.classList.toggle("is-selected", selectedIds.has(card.dataset.id));
  }
}

function clearSelection(thread) {
  const selectedIds = getSelectedIds(thread);
  selectedIds.clear();
  updateSelectionStyles(thread);
  syncSelectionStatus();
}

function clearAllSelections() {
  clearSelection("user");
  clearSelection("llm");
}

function normalizeMessageText(text) {
  return String(text || "").trim();
}

function joinMessagesForForward(messages) {
  return messages
    .map((message) => normalizeMessageText(message.body))
    .filter(Boolean)
    .join("\n\n");
}

function createMetaText(message) {
  if (message.thread === "user") {
    return `${message.sender} -> ${message.receiver} | sent ${message.timestamp}`;
  }

  if (message.role === "user") {
    return `${message.sender} -> LLM | ${message.timestamp}`;
  }

  return `LLM (${message.model}) -> ${message.receiver} | ${message.timestamp}`;
}

function createDeliveryText(message) {
  if (message.thread === "user") {
    if (message.appearance === "incoming") {
      return message.receivedTimestamp ? `Received ${message.receivedTimestamp}` : "Received";
    }

    return message.deliveredTimestamp ? `Delivered ${message.deliveredTimestamp}` : "Sending";
  }

  if (message.role === "user") {
    return "Queued for the local model";
  }

  return `Answered in session ${message.sessionId}`;
}

function appendMessageToThread(message) {
  const threadMessages = getMessages(message.thread);
  threadMessages.push(message);

  const container = getContainer(message.thread);
  const row = document.createElement("article");
  row.className = `message ${message.appearance}`;
  row.dataset.id = message.id;
  row.dataset.thread = message.thread;

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = createMetaText(message);

  const body = document.createElement("p");
  body.textContent = message.body;

  const delivery = document.createElement("div");
  delivery.className = "delivery";
  delivery.textContent = createDeliveryText(message);

  row.append(meta, body, delivery);
  container.append(row);
  container.scrollTop = container.scrollHeight;
  syncEmptyState(container, threadMessages);
  updateSelectionStyles(message.thread);
}

function resetThread(thread) {
  if (thread === "user") {
    userThread = [];
    userMessagesEl.innerHTML = "";
    syncEmptyState(userMessagesEl, userThread);
  } else {
    llmThread = [];
    llmMessagesEl.innerHTML = "";
    syncEmptyState(llmMessagesEl, llmThread);
  }
}

function appendUserThreadMessage(payload, direction) {
  appendMessageToThread({
    id: String(payload.id ?? buildClientMessageId("user")),
    thread: "user",
    appearance: direction,
    sender: payload.sender,
    receiver: payload.receiver,
    body: normalizeMessageText(payload.content),
    timestamp: payload.sent_timestamp,
    deliveredTimestamp: payload.delivered_timestamp ?? null,
    receivedTimestamp: direction === "incoming" ? new Date().toISOString() : null,
  });
}

function appendHistoricalUserMessage(message) {
  const isOutgoing = message.sender === userInput.value;
  appendMessageToThread({
    id: String(message.id),
    thread: "user",
    appearance: isOutgoing ? "outgoing" : "incoming",
    sender: message.sender,
    receiver: message.receiver,
    body: normalizeMessageText(message.content),
    timestamp: message.sent_timestamp,
    deliveredTimestamp: message.delivered_timestamp ?? null,
    receivedTimestamp: isOutgoing ? null : message.delivered_timestamp ?? message.sent_timestamp,
  });
}

function appendLlmPromptMessage(text) {
  appendMessageToThread({
    id: buildClientMessageId("llm-user"),
    thread: "llm",
    appearance: "llm-user",
    role: "user",
    sender: userInput.value,
    receiver: "llm",
    body: normalizeMessageText(text),
    timestamp: new Date().toISOString(),
    sessionId: sessionInput.value.trim(),
    model: "prompt",
  });
}

function appendLlmReplyMessage(payload) {
  appendMessageToThread({
    id: buildClientMessageId("llm-assistant"),
    thread: "llm",
    appearance: "llm-assistant",
    role: "assistant",
    sender: "llm",
    receiver: payload.user_id,
    body: normalizeMessageText(payload.output_text),
    timestamp: payload.timestamp,
    sessionId: payload.session_id,
    model: payload.model,
  });
}

function appendHistoricalLlmInteraction(interaction) {
  appendMessageToThread({
    id: `llm-user-history-${interaction.id}`,
    thread: "llm",
    appearance: "llm-user",
    role: "user",
    sender: interaction.user_id,
    receiver: "llm",
    body: normalizeMessageText(interaction.input_text),
    timestamp: interaction.timestamp,
    sessionId: interaction.session_id,
    model: "prompt",
  });

  appendMessageToThread({
    id: `llm-assistant-history-${interaction.id}`,
    thread: "llm",
    appearance: "llm-assistant",
    role: "assistant",
    sender: "llm",
    receiver: interaction.user_id,
    body: normalizeMessageText(interaction.output_text),
    timestamp: interaction.timestamp,
    sessionId: interaction.session_id,
    model: interaction.model,
  });
}

function getCurrentUserId() {
  return userInput.value.trim();
}

function toggleMessageSelection(thread, messageId, { additive = true } = {}) {
  const selectedIds = getSelectedIds(thread);
  if (!additive) {
    selectedIds.clear();
  }

  if (selectedIds.has(messageId)) {
    selectedIds.delete(messageId);
  } else {
    selectedIds.add(messageId);
  }

  updateSelectionStyles(thread);
  syncSelectionStatus();
}

function ensureRightClickedSelection(thread, messageId) {
  const selectedIds = getSelectedIds(thread);
  const otherThread = thread === "user" ? "llm" : "user";

  if (!selectedIds.has(messageId)) {
    clearSelection(otherThread);
    selectedIds.clear();
    selectedIds.add(messageId);
    updateSelectionStyles(thread);
    syncSelectionStatus();
    return;
  }

  clearSelection(otherThread);
}

function getSelectedMessages(thread) {
  const selectedIds = getSelectedIds(thread);
  const messages = getMessages(thread);
  return messages.filter((message) => selectedIds.has(message.id));
}

function hideContextMenu() {
  contextMenuEl.hidden = true;
  contextMenuEl.style.display = "none";
  contextMenuEl.style.pointerEvents = "none";
  contextMenuBackdropEl.hidden = true;
  contextMenuBackdropEl.style.display = "none";
  contextThread = null;
}

function showContextMenu(event, thread) {
  contextThread = thread;
  const sendUserButton = contextMenuEl.querySelector('[data-action="send-user"]');
  const sendLlmButton = contextMenuEl.querySelector('[data-action="send-llm"]');

  sendUserButton.disabled = thread === "user";
  sendLlmButton.disabled = thread === "llm";

  contextMenuBackdropEl.hidden = false;
  contextMenuBackdropEl.style.display = "block";
  contextMenuEl.hidden = false;
  contextMenuEl.style.display = "grid";
  contextMenuEl.style.pointerEvents = "auto";
  contextMenuEl.style.left = `${event.clientX}px`;
  contextMenuEl.style.top = `${event.clientY}px`;
}

function sendMessageToUser(content) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    statusLabel.textContent = "Connect before sending";
    return false;
  }

  const sessionId = sessionInput.value.trim();
  const cleanContent = normalizeMessageText(content);
  if (!cleanContent || !sessionId) {
    return false;
  }

  socket.send(
    JSON.stringify({
      receiver: receiverInput.value,
      content: cleanContent,
      session_id: sessionId,
      experimental_condition: currentCondition,
    }),
  );

  return true;
}

async function sendMessageToLlm(content) {
  const messageText = normalizeMessageText(content);
  const sessionId = sessionInput.value.trim();
  const userId = userInput.value;

  if (!messageText || !sessionId || !userId) {
    setLlmStatus("Enter a session ID and a prompt first");
    return false;
  }

  askLlmButton.disabled = true;
  setLlmStatus("Waiting for local Ollama...");
  appendLlmPromptMessage(messageText);

  try {
    const response = await fetch("/api/llm/message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
        user_id: userId,
        message_text: messageText,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "LLM request failed");
    }

    appendLlmReplyMessage(data);
    setLlmStatus(`LLM replied using ${data.model}`);
    return true;
  } catch (error) {
    appendMessageToThread({
      id: buildClientMessageId("llm-error"),
      thread: "llm",
      appearance: "llm-assistant",
      role: "assistant",
      sender: "system",
      receiver: userId,
      body: error.message || "Unable to reach local Ollama",
      timestamp: new Date().toISOString(),
      sessionId,
      model: "error",
    });
    setLlmStatus(error.message || "Unable to reach local Ollama");
    return false;
  } finally {
    askLlmButton.disabled = false;
  }
}

async function forwardSelection(targetThread) {
  if (!contextThread) {
    return;
  }

  const selectedMessages = getSelectedMessages(contextThread);
  const forwardedText = joinMessagesForForward(selectedMessages);
  if (!forwardedText) {
    hideContextMenu();
    return;
  }

  if (targetThread === "user") {
    const sent = sendMessageToUser(forwardedText);
    if (sent) {
      setLlmStatus("Forwarded selected messages to the user chat");
    }
  } else {
    llmContentInput.value = forwardedText;
    await sendMessageToLlm(forwardedText);
    llmContentInput.value = "";
  }

  clearAllSelections();
  hideContextMenu();
}

async function loadCondition(sessionId) {
  const response = await fetch(`/api/condition/${encodeURIComponent(sessionId)}`);
  const data = await response.json();
  currentCondition = data.experimental_condition;
  conditionLabel.textContent = `Condition: ${currentCondition}`;
  sessionInput.value = data.session_id;
}

async function loadSessionHistory(sessionId) {
  const response = await fetch(`/api/session/${encodeURIComponent(sessionId)}/history`);
  const data = await response.json();
  const currentUserId = getCurrentUserId();

  resetThread("user");
  resetThread("llm");
  clearAllSelections();
  setLlmStatus("");

  for (const message of data.messages) {
    appendHistoricalUserMessage(message);
  }

  for (const interaction of data.llm_interactions) {
    if (interaction.user_id !== currentUserId) {
      continue;
    }
    appendHistoricalLlmInteraction(interaction);
  }

  sessionInput.value = data.session_id;
}

async function connect(userId, sessionId) {
  if (socket) {
    socket.close();
  }

  await loadCondition(sessionId);
  await loadSessionHistory(sessionInput.value.trim());

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/ws/${encodeURIComponent(userId)}`);

  socket.addEventListener("open", () => {
    statusLabel.textContent = `Connected as ${userId}`;
  });

  socket.addEventListener("close", () => {
    statusLabel.textContent = "Disconnected";
  });

  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "message") {
      appendUserThreadMessage(payload, "incoming");
    }
    if (payload.type === "ack") {
      appendUserThreadMessage(payload, "outgoing");
    }
  });
}

function handleMessageClick(event, thread) {
  const card = event.target.closest(".message");
  if (!card) {
    return;
  }

  const additive = event.metaKey || event.ctrlKey || event.shiftKey;
  const otherThread = thread === "user" ? "llm" : "user";
  if (!additive) {
    clearSelection(otherThread);
  }
  toggleMessageSelection(thread, card.dataset.id, { additive });
}

function handleMessageContextMenu(event, thread) {
  const card = event.target.closest(".message");
  if (!card) {
    hideContextMenu();
    return;
  }

  event.preventDefault();
  ensureRightClickedSelection(thread, card.dataset.id);
  showContextMenu(event, thread);
}

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await connect(userInput.value, sessionInput.value.trim());
});

messageForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const content = normalizeMessageText(contentInput.value);
  if (!content) {
    return;
  }

  if (sendMessageToUser(content)) {
    contentInput.value = "";
  }
});

llmForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const content = normalizeMessageText(llmContentInput.value);
  if (!content) {
    return;
  }

  const sent = await sendMessageToLlm(content);
  if (sent) {
    llmContentInput.value = "";
  }
});

userMessagesEl.addEventListener("click", (event) => {
  handleMessageClick(event, "user");
});

llmMessagesEl.addEventListener("click", (event) => {
  handleMessageClick(event, "llm");
});

userMessagesEl.addEventListener("contextmenu", (event) => {
  handleMessageContextMenu(event, "user");
});

llmMessagesEl.addEventListener("contextmenu", (event) => {
  handleMessageContextMenu(event, "llm");
});

contextMenuEl.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    hideContextMenu();
    return;
  }

  const { action } = button.dataset;
  if (action === "clear-selection") {
    clearAllSelections();
    hideContextMenu();
    return;
  }

  if (action === "send-user") {
    await forwardSelection("user");
    return;
  }

  if (action === "send-llm") {
    await forwardSelection("llm");
  }
});

contextMenuBackdropEl.addEventListener("pointerdown", () => {
  hideContextMenu();
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".message")) {
    clearAllSelections();
  }

  if (!event.target.closest(".context-menu")) {
    hideContextMenu();
  }
});

document.addEventListener(
  "pointerdown",
  (event) => {
    if (event.button !== 0) {
      return;
    }

    if (!event.target.closest(".context-menu") && !event.target.closest(".context-menu-backdrop")) {
      hideContextMenu();
    }
  },
  true,
);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    clearAllSelections();
    hideContextMenu();
  }
});

window.addEventListener("blur", () => {
  hideContextMenu();
});

userInput.addEventListener("change", () => {
  receiverInput.value = userInput.value === "user_a" ? "user_b" : "user_a";
});
