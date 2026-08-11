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
ollama pull qwen3:8b
ollama serve
```

## Run

```powershell
uvicorn app.main:app --reload
```

Open two browser windows at:

```text
http://127.0.0.1:8000
```

In one window, connect as `User A`. In the other, connect as `User B`. Use the same session ID in both windows.

## Application Startup Checklist

1. Go to the project folder:

```powershell
cd /home/parc/communication-interface-prototype
```

2. Check whether Ollama is already running:

```powershell
curl http://127.0.0.1:11434/api/tags
```

If you get JSON back, Ollama is up.

3. Check whether `qwen3:8b` is available:

```powershell
ollama list
```

4. If Ollama is not running yet, start it:

```powershell
ollama serve
```

5. Activate the virtual environment:

```powershell
source .comm_interface_env/bin/activate
```

6. Confirm the virtual environment is active:

```powershell
which python
```

7. If needed, verify the required Python packages are installed:

```powershell
pip show fastapi httpx uvicorn
```

If any are missing, install them with:

```powershell
pip install -r requirements.txt
```

8. Start the FastAPI app:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

9. Check that the app is responding locally:

```powershell
curl http://127.0.0.1:8000/api/messages
```

10. Check that Tailscale is connected if you want to share the app:

```powershell
tailscale status
```

11. Serve the app over Tailscale:

```powershell
tailscale serve 8000
```

12. Confirm what Tailscale is serving:

```powershell
tailscale serve status
```

13. Open the app:

```text
http://127.0.0.1:8000
```

Or open the HTTPS tailnet URL printed by `tailscale serve`.

14. Test normal chat:

Open two browser windows, connect one as `user_a` and the other as `user_b`, and send a message.

15. Test the LLM path:

Type a prompt in the message box and click `Ask LLM`.

16. If you want to confirm the app is listening on port `8000`:

```powershell
ss -ltnp | grep 8000
```

17. If you want to confirm Ollama is listening on port `11434`:

```powershell
ss -ltnp | grep 11434
```

## Ask The Local LLM

The normal WebSocket chat between `user_a` and `user_b` is unchanged. To query the local model instead, type into the same message box and click `Ask LLM`.

- Backend target: `http://localhost:11434`
- Default model: `qwen3:8b`
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
