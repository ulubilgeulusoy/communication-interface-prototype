# Tailscale Communication Interface Prototype

Two-user communication web app for prototyping real-time interaction over a Tailscale tailnet. It uses a FastAPI backend, WebSocket messaging plus multipart upload endpoints, a plain HTML/JavaScript frontend, per-session SQLite storage, and local Ollama-backed LLM replies with optional image/file attachments.

## Features

- Two browser clients can connect as `user_a` and `user_b`.
- User chat messages are delivered in real time and support optional file/image attachments.
- Each session is stored in its own SQLite file and can be reopened by using the same session ID.
- User chat and LLM chat are shown in separate side-by-side panes.
- Both `User Chat` and `LLM Chat` support selecting, previewing, and removing attachments before send.
- Messages can be multi-selected and forwarded between the user chat and LLM chat.
- Message timing includes sent, delivered, and client-side received status in the UI.
- A separate backend LLM service can query local Ollama at `http://127.0.0.1:11434`.
- LLM chat uses `llama3.2-vision:11b` for structured image findings, then always uses `gpt-oss:20b` for the user-facing response.
- PDF and text-like attachments are text-extracted by the backend and included in the LLM prompt context.
- Image findings are combined with the original request for the same RAG/vector store and embedding workflow used by text requests.
- LLM interactions log selected models, raw vision findings, retrieved sources, final response, and input attachment metadata.
- Frontend, backend, and experimental condition assignment are kept in separate modules.

## Project Structure

```text
app/
  main.py          FastAPI app, WebSocket routing, API endpoints
  db.py            Per-session SQLite setup and message logging helpers
  experiment.py    Experimental condition assignment logic
  llm_service.py   Local Ollama integration and response handling
frontend/
  index.html       Browser UI
  app.js           WebSocket client behavior
  styles.css       UI styling
uploads/           Runtime attachment storage (created locally, gitignored)
requirements.txt
README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install and run Ollama locally, then pull the required models:

```powershell
ollama pull gpt-oss:20b
ollama pull llama3.2-vision:11b
ollama pull nomic-embed-text
ollama serve
```

If `ollama serve` prints `bind: address already in use`, that usually means the Ollama daemon is already running on `127.0.0.1:11434`. In that case, do not start a second copy. Verify it with:

```powershell
curl http://127.0.0.1:11434/api/tags
ss -ltnp | grep 11434
```

`ollama ps` can still be empty in this state. That command only shows models currently loaded for inference, not whether the Ollama server itself is running.

## Run In Visual Studio Code

Use this checklist if you are starting the app from the VS Code terminal. It assumes the virtual environment and dependencies are already set up.

1. Open this folder in VS Code:

```text
/home/parc/communication-interface-prototype
```

2. Open a new terminal in VS Code and make sure you are in the repo root:

```powershell
pwd
```

In Ubuntu or another Linux shell, this usually prints:

```text
/home/parc/communication-interface-prototype
```

In PowerShell, you can also run:

```powershell
Get-Location
```

3. Make sure the virtual environment is already loaded.

```powershell
which python
```

In Ubuntu or another Linux shell, a loaded venv usually looks like:

```text
/home/parc/communication-interface-prototype/.comm_interface_env/bin/python
```

In PowerShell, use:

```powershell
Get-Command python
```

That usually shows a Python path inside the project virtual environment.

4. Check whether Ollama is already running:

```powershell
curl http://127.0.0.1:11434/api/tags
```

If you get JSON back, Ollama is up.

5. Check whether the required Ollama models are available:

```powershell
ollama list
```

You should see `gpt-oss:20b`, `llama3.2-vision:11b`, and `nomic-embed-text` in the model list.

6. If `curl` in step 4 fails, start Ollama in a separate VS Code terminal:

```powershell
ollama serve
```

If `ollama serve` returns `bind: address already in use`, treat that as "Ollama is already running" and move on to the next step.

7. Start the FastAPI app from the repo root:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In either PowerShell or Ubuntu/Linux terminal, a healthy startup usually includes lines like:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

8. Check that the app is responding locally:

```powershell
curl http://127.0.0.1:8000/api/messages
```

9. Open the app in your browser:

```text
http://127.0.0.1:8000
```

10. Open two browser windows, connect one as `user_a` and the other as `user_b`, and use the same session ID in both windows.

11. Test normal chat by sending a message between the two users.

12. Test user-chat attachments by attaching a file or image, sending it, and reloading the session.

13. Test the LLM path by entering a prompt in the `LLM Chat` pane and clicking `Send to LLM`.

14. Test image analysis in `LLM Chat` by attaching an image and asking a direct vision prompt such as `What do you see in this image?`

15. If you want to confirm the app is listening on port `8000`:

```powershell
ss -ltnp | grep 8000
```

16. If you want to confirm Ollama is listening on port `11434`:

```powershell
ss -ltnp | grep 11434
```

`ollama ps` only shows active model processes. An empty table there does not mean the Ollama server is down.

17. If you want to share the app over Tailscale, open another VS Code terminal and check that Tailscale is connected:

```powershell
tailscale status
```

18. Serve the app over Tailscale:

```powershell
tailscale serve 8000
```

19. Confirm what Tailscale is serving:

```powershell
tailscale serve status
```

20. Open the local URL above, or open the HTTPS tailnet URL printed by `tailscale serve`.

## Session Behavior

Each session ID maps to its own SQLite file under `sessions/`. If you reconnect later with the same session ID, the app reloads both the user chat history and the LLM chat history and continues from there, including attachment metadata for stored messages and LLM prompts.

## Ask The Local LLM

To query the local model, type into the `LLM Chat` pane and click `Send to LLM`. You can also attach files or images in the LLM pane.

- Backend target: `http://127.0.0.1:11434`
- Default text model: `gpt-oss:20b`
- Vision findings model: `llama3.2-vision:11b`
- Final user-facing model for every request: `gpt-oss:20b`
- Endpoint: `POST /api/llm/message`
- Attachment endpoint: `POST /api/llm/message-upload`

Example request body:

```json
{
  "session_id": "session-001",
  "user_id": "user_a",
  "message_text": "Summarize the last idea in one sentence."
}
```

If Ollama is down or the model is unavailable, the app returns a `503` response and the frontend shows the error instead of crashing.

### Attachment Analysis Behavior

- Image attachments are analyzed by `llama3.2-vision:11b`, which returns structured visual findings only.
- The original question, structured visual findings, retrieved knowledge, and conversation history are sent to `gpt-oss:20b`, which produces the only user-facing response.
- PDF attachments are read by the backend with `pypdf`; extracted text is then added to the prompt context for the LLM.
- Text-like files such as `.txt`, `.md`, `.csv`, `.json`, `.py`, and `.log` are read by the backend and added to the prompt context.
- Other binary file types are stored and displayed in the UI, but are not deeply analyzed yet.
- If the vision stage fails, the backend logs the failure and still asks `gpt-oss:20b` to respond using the original question and any available non-image context.

### Attachment Disclaimer

Attachment support is still experimental and should be tested more thoroughly end to end. Analysis quality may vary by file type, extracted-text quality, Ollama version, and local model behavior. The current implementation is useful for prototyping, but it should not yet be treated as a finalized attachment-analysis pipeline.

Attachments are also not yet part of a dedicated knowledge base or indexing pipeline. A future design decision is still needed on whether uploaded attachments should remain session-local only, be added to the existing retrieval/indexing flow, or be stored in a separate attachment-specific knowledge base.

## Index The Knowledge Base

The RAG indexer uses a separate Ollama embedding model from the chat model.

- Text chat/generation model: `gpt-oss:20b`
- Vision/image model: `llama3.2-vision:11b`
- Embedding model: `nomic-embed-text`
- The UI also includes a `Re-index Knowledge Base` button in the `LLM Chat` panel for manual refreshes after you add or change files.

At the moment, uploaded attachments are not automatically indexed into the RAG knowledge base.

Index the local `knowledge_base/` files into `vector_store/` with:

```powershell
python -m app.rag.index_knowledge
```

If the embedding model has not been pulled yet, run:

```powershell
ollama pull nomic-embed-text
```

## Serve Over Tailscale

With the FastAPI server running locally on port `8000`, expose it inside your Tailscale tailnet from a second terminal:

```powershell
tailscale serve 8000
```

Tailscale will print a tailnet URL similar to:

```text
https://your-device.your-tailnet.ts.net/
```

Share that exact URL with another device that is logged into the same Tailscale tailnet. The Tailscale URL proxies to:

```text
http://127.0.0.1:8000
```

For a persistent background serve:

```powershell
tailscale serve --bg 8000
```

To stop serving:

```powershell
tailscale serve reset
```

This setup is intended for tailnet-only prototyping. It does not make the app public on the internet unless you separately configure Tailscale Funnel.

## Inspect Logged Data

The app creates one SQLite file per session under `sessions/`. Human messages for a session are available through:

```text
http://127.0.0.1:8000/api/messages?session_id=session-001
http://127.0.0.1:8000/api/session/session-001/history
```

Each session database also contains an `llm_interactions` table for model request/response logging.

Uploaded attachment files are stored locally under `uploads/` and are ignored by git.

## Experimental Logic

`app/experiment.py` assigns a deterministic condition from the session ID. This keeps both users in the same session on the same condition and makes later experimental changes isolated from transport and UI code.
