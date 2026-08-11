const settingsForm = document.querySelector("#settings-form");
const messageForm = document.querySelector("#message-form");
const userInput = document.querySelector("#user-id");
const receiverInput = document.querySelector("#receiver-id");
const sessionInput = document.querySelector("#session-id");
const contentInput = document.querySelector("#message-content");
const statusLabel = document.querySelector("#connection-status");
const conditionLabel = document.querySelector("#condition-label");
const messagesEl = document.querySelector("#messages");

let socket = null;
let currentCondition = null;

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

  if (!socket || socket.readyState !== WebSocket.OPEN) {
    statusLabel.textContent = "Connect before sending";
    return;
  }

  const content = contentInput.value.trim();
  const sessionId = sessionInput.value.trim();
  if (!content || !sessionId) {
    return;
  }

  socket.send(
    JSON.stringify({
      receiver: receiverInput.value,
      content,
      session_id: sessionId,
      experimental_condition: currentCondition,
    }),
  );
  contentInput.value = "";
});

userInput.addEventListener("change", () => {
  receiverInput.value = userInput.value === "user_a" ? "user_b" : "user_a";
});
