const settingsForm = document.querySelector("#settings-form");
const messageForm = document.querySelector("#message-form");
const userInput = document.querySelector("#user-id");
const receiverInput = document.querySelector("#receiver-id");
const sessionInput = document.querySelector("#session-id");
const contentInput = document.querySelector("#message-content");
const statusLabel = document.querySelector("#connection-status");
const conditionLabel = document.querySelector("#condition-label");
const messagesEl = document.querySelector("#messages");
const llmPromptInput = document.querySelector("#llm-prompt");
const llmResponseInput = document.querySelector("#llm-response");
const askLlmButton = document.querySelector("#ask-llm-button");
const sendLlmButton = document.querySelector("#send-llm-button");
const llmStatusEl = document.querySelector("#llm-status");

let socket = null;
let currentCondition = null;
let lastLlmResponse = "";

function appendMessage(message, direction) {
  const row = document.createElement("article");
  row.className = `message ${direction}`;

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = `${message.sender} -> ${message.receiver} | sent ${message.sent_timestamp}`;

  const body = document.createElement("p");
  body.textContent = message.content;

  const delivery = document.createElement("div");
  delivery.className = "delivery";
  delivery.textContent = message.delivered_timestamp
    ? `Delivered ${message.delivered_timestamp}`
    : "Not delivered yet";

  row.append(meta, body, delivery);
  messagesEl.append(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendLlmMessage(message) {
  const row = document.createElement("article");
  row.className = "message llm";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = `LLM (${message.model}) for ${message.user_id} | ${message.timestamp}`;

  const body = document.createElement("p");
  body.textContent = message.output_text;

  const note = document.createElement("div");
  note.className = "delivery";
  note.textContent = `Prompted from session ${message.session_id}`;

  row.append(meta, body, note);
  messagesEl.append(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLlmStatus(text) {
  llmStatusEl.textContent = text;
}

function setLastLlmResponse(text) {
  lastLlmResponse = text;
  llmResponseInput.value = text;
  sendLlmButton.disabled = !text;
}

function sendMessageToUser(content) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    statusLabel.textContent = "Connect before sending";
    return false;
  }

  const sessionId = sessionInput.value.trim();
  if (!content || !sessionId) {
    return false;
  }

  socket.send(
    JSON.stringify({
      receiver: receiverInput.value,
      content,
      session_id: sessionId,
      experimental_condition: currentCondition,
    }),
  );

  return true;
}

async function loadCondition(sessionId) {
  const response = await fetch(`/api/condition/${encodeURIComponent(sessionId)}`);
  const data = await response.json();
  currentCondition = data.experimental_condition;
  conditionLabel.textContent = `Condition: ${currentCondition}`;
}

async function connect(userId, sessionId) {
  if (socket) {
    socket.close();
  }

  await loadCondition(sessionId);

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
      appendMessage(payload, "incoming");
    }
    if (payload.type === "ack") {
      appendMessage(payload, "outgoing");
    }
  });
}

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await connect(userInput.value, sessionInput.value.trim());
});

messageForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const content = contentInput.value.trim();
  if (!content) {
    return;
  }

  if (sendMessageToUser(content)) {
    contentInput.value = "";
  }
});

askLlmButton.addEventListener("click", async () => {
  const messageText = llmPromptInput.value.trim();
  const sessionId = sessionInput.value.trim();
  const userId = userInput.value;

  if (!messageText || !sessionId || !userId) {
    setLlmStatus("Enter a session ID and a prompt first");
    return;
  }

  askLlmButton.disabled = true;
  setLlmStatus("Asking local Ollama...");

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

    appendLlmMessage(data);
    setLastLlmResponse(data.output_text);
    setLlmStatus(`LLM replied using ${data.model}`);
  } catch (error) {
    setLastLlmResponse("");
    setLlmStatus(error.message || "Unable to reach local Ollama");
  } finally {
    askLlmButton.disabled = false;
  }
});

sendLlmButton.addEventListener("click", () => {
  if (!lastLlmResponse) {
    setLlmStatus("Ask the LLM for a draft first");
    return;
  }

  if (sendMessageToUser(lastLlmResponse)) {
    setLlmStatus(`Sent latest LLM message to ${receiverInput.value}`);
  }
});

userInput.addEventListener("change", () => {
  receiverInput.value = userInput.value === "user_a" ? "user_b" : "user_a";
});
