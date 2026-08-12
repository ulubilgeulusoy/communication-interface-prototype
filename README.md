# Tailscale Communication Interface Prototype

Minimal two-user communication web app for prototyping real-time interaction over a Tailscale tailnet. It uses a FastAPI backend, WebSocket messaging, a plain HTML/JavaScript frontend, SQLite message logging, and optional local Ollama-backed LLM replies.

## Features

- Two browser clients can connect as `user_a` and `user_b`.
- Messages are sent over WebSockets in real time.
- SQLite logs sender, receiver, content, sent timestamp, delivered timestamp, session ID, and experimental condition.
- A separate backend LLM service can query local Ollama at `http://localhost:11434`.
- LLM interactions are logged in SQLite with timestamp, session, user, model, prompt, and response.
- Frontend, backend, and experimental condition assignment are kept in separate modules.

## Project Structure

```text
app/
  main.py          FastAPI app, WebSocket routing, API endpoints
  db.py            SQLite setup and message logging helpers
  experiment.py    Experimental condition assignment logic
  llm_service.py   Local Ollama integration and response handling
frontend/
  index.html       Browser UI
  app.js           WebSocket client behavior
  styles.css       UI styling
requirements.txt
README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install and run Ollama locally, then pull the default model:

```powershell
ollama pull gpt-oss:20b
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

5. Check whether `gpt-oss:20b` is available:

```powershell
ollama list
```

You should see `gpt-oss:20b` in the model list.

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

12. Test the LLM path by entering a prompt and clicking `Ask LLM`.

13. If you want to confirm the app is listening on port `8000`:

```powershell
ss -ltnp | grep 8000
```

14. If you want to confirm Ollama is listening on port `11434`:

```powershell
ss -ltnp | grep 11434
```

`ollama ps` only shows active model processes. An empty table there does not mean the Ollama server is down.

15. If you want to share the app over Tailscale, open another VS Code terminal and check that Tailscale is connected:

```powershell
tailscale status
```

16. Serve the app over Tailscale:

```powershell
tailscale serve 8000
```

17. Confirm what Tailscale is serving:

```powershell
tailscale serve status
```

18. Open the local URL above, or open the HTTPS tailnet URL printed by `tailscale serve`.

## Ask The Local LLM

The normal WebSocket chat between `user_a` and `user_b` is unchanged. To query the local model instead, type into the same message box and click `Ask LLM`.

- Backend target: `http://localhost:11434`
- Default model: `gpt-oss:20b`
- Endpoint: `POST /api/llm/message`

Example request body:

```json
{
  "session_id": "session-001",
  "user_id": "user_a",
  "message_text": "Summarize the last idea in one sentence."
}
```

If Ollama is down or the model is unavailable, the app returns a `503` response and the frontend shows the error instead of crashing.

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

The app creates `messages.sqlite3` in the repository root. Human messages are available through:

```text
http://127.0.0.1:8000/api/messages
http://127.0.0.1:8000/api/messages?session_id=session-001
```

The same SQLite database also contains an `llm_interactions` table for model request/response logging.

## Experimental Logic

`app/experiment.py` assigns a deterministic condition from the session ID. This keeps both users in the same session on the same condition and makes later experimental changes isolated from transport and UI code.
