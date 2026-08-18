const form = document.getElementById("chat-form");
const input = document.getElementById("query-input");
const submitButton = form.querySelector("button");
const chatLog = document.getElementById("chat-log");
const errorText = document.getElementById("error-text");

// Identifies this browser session's conversation to the orchestrator's
// checkpointer, so follow-up questions can see prior turns without the
// client having to resend the whole transcript on every request.
const threadId = crypto.randomUUID();

function renderError(message) {
  errorText.textContent = message;
  errorText.classList.remove("hidden");
}

function clearError() {
  errorText.classList.add("hidden");
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendUserTurn(query) {
  const bubble = document.createElement("div");
  bubble.className = "bubble user";
  bubble.textContent = query;
  chatLog.appendChild(bubble);
  scrollToBottom();
}

function appendAssistantTurn() {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";
  bubble.innerHTML = `
    <div class="loading retrieving">
      <span class="spinner"></span>Retrieving relevant information&hellip;
    </div>
    <div class="sources hidden">
      <p class="sources-label">Retrieved snippets</p>
      <ul class="sources-list"></ul>
    </div>
    <div class="loading generating hidden">
      <span class="spinner"></span>Generating answer&hellip;
    </div>
    <p class="answer-text"></p>
    <p class="note hidden">Note: answer may be incomplete relative to the knowledge base.</p>
  `;
  chatLog.appendChild(bubble);
  scrollToBottom();
  return bubble;
}

function hideLoadingIndicators(bubble) {
  bubble.querySelector(".loading.retrieving").classList.add("hidden");
  bubble.querySelector(".loading.generating").classList.add("hidden");
}

function renderChunksInto(bubble, chunks) {
  bubble.querySelector(".loading.retrieving").classList.add("hidden");

  if (chunks.length > 0) {
    const list = bubble.querySelector(".sources-list");
    list.textContent = "";
    for (const chunk of chunks) {
      const li = document.createElement("li");
      li.textContent = chunk;
      list.appendChild(li);
    }
    bubble.querySelector(".sources").classList.remove("hidden");
  }

  bubble.querySelector(".loading.generating").classList.remove("hidden");
  scrollToBottom();
}

function renderReportInto(bubble, report) {
  hideLoadingIndicators(bubble);
  bubble.querySelector(".answer-text").textContent = report.answer;
  bubble.querySelector(".note").classList.toggle("hidden", report.grounded);
  scrollToBottom();
}

// Parses the SSE stream from a fetch() response body, since EventSource
// doesn't support POST requests with a body.
async function consumeEventStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      onEvent(parseSseEvent(rawEvent));
    }
  }
}

function parseSseEvent(rawEvent) {
  let event = "message";
  let data = "";
  for (const line of rawEvent.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data = line.slice(5).trim();
  }
  return { event, data: data ? JSON.parse(data) : {} };
}

form.addEventListener("submit", async (submitEvent) => {
  submitEvent.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  clearError();
  input.value = "";
  submitButton.disabled = true;
  appendUserTurn(query);
  const assistantBubble = appendAssistantTurn();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, thread_id: threadId }),
    });

    if (!response.ok) {
      hideLoadingIndicators(assistantBubble);
      renderError(`Request failed (${response.status})`);
      return;
    }

    await consumeEventStream(response, ({ event, data }) => {
      if (event === "chunks_retrieved") renderChunksInto(assistantBubble, data.chunks);
      if (event === "report_generated") renderReportInto(assistantBubble, data);
      if (event === "error") {
        hideLoadingIndicators(assistantBubble);
        renderError(data.message);
      }
    });
  } catch (err) {
    hideLoadingIndicators(assistantBubble);
    renderError(`Network error: ${err.message}`);
  } finally {
    hideLoadingIndicators(assistantBubble);
    submitButton.disabled = false;
    input.focus();
  }
});
