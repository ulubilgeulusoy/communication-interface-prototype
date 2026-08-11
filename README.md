# Tailscale Communication Interface Prototype

Minimal two-user communication web app for prototyping real-time interaction over a Tailscale tailnet. It uses a FastAPI backend, WebSocket messaging, a plain HTML/JavaScript frontend, and SQLite message logging.

## Features

- Two browser clients can connect as `user_a` and `user_b`.
- Messages are sent over WebSockets in real time.
- SQLite logs sender, receiver, content, sent timestamp, delivered timestamp, session ID, and experimental condition.
- Frontend, backend, and experimental condition assignment are kept in separate modules.

## Project Structure

```text
app/
  main.py          FastAPI app, WebSocket routing, API endpoints
  db.py            SQLite setup and message logging helpers
  experiment.py    Experimental condition assignment logic
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

## Run

```powershell
uvicorn app.main:app --reload
```

Open two browser windows at:

```text
http://127.0.0.1:8000
```

In one window, connect as `User A`. In the other, connect as `User B`. Use the same session ID in both windows.

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

## Inspect Logged Messages

The app creates `messages.sqlite3` in the repository root. You can also query messages through:

```text
http://127.0.0.1:8000/api/messages
http://127.0.0.1:8000/api/messages?session_id=session-001
```

## Experimental Logic

`app/experiment.py` assigns a deterministic condition from the session ID. This keeps both users in the same session on the same condition and makes later experimental changes isolated from transport and UI code.
