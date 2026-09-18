# Prerequisites — Before You Start

Install/set up everything below **before** the event starts so setup issues don't eat into the 4-hour window.

## 1. Install

| Tool | Version | Check with |
|---|---|---|
| Python | 3.11 or newer | `python3 --version` |
| pip | latest | `pip --version` |
| Git | any recent | `git --version` |
| Docker Desktop (or Docker Engine) | any recent | `docker --version` |
| A code editor | VS Code recommended | — |
| curl or Postman | for hitting the API manually | `curl --version` |

## 2. Python environment

From the repo root, once it exists:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Expected core packages in `requirements.txt`:

```
fastapi
uvicorn[standard]
pydantic>=2
google-genai
pulp
python-dotenv
httpx        # for tests hitting the running server
```

## 3. Gemini API key

We're using **`gemini-3.5-flash-lite`** via Google's Gen AI SDK.

1. Get a key from Google AI Studio: https://aistudio.google.com/apikey
2. Each teammate should get their **own** key for local dev (don't share one key over chat — avoid rate-limit collisions and secret sprawl). The team lead sets the key that's used in the actual deployed submission.
3. Create a `.env` file in the repo root (this file is gitignored — never commit it):

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

4. Confirm `.env` is listed in `.gitignore` before your first commit.

## 4. Git access

- Make sure you're added as a collaborator on the repo once it's created (after question reveal — it stays **private** during the event).
- Set up SSH or a personal access token for GitHub if you haven't already, so you're not fumbling with auth mid-event:
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "you@example.com"
  ```

## 5. Docker

- Confirm Docker actually runs before the event: `docker run hello-world`
- We'll containerize the FastAPI app at the end — no need to write the Dockerfile yourself in advance, but make sure Docker itself isn't broken on your machine now.

## 6. Deployment account

Pick one now so nobody's creating an account mid-event:

- Render (https://render.com) — free tier, straightforward for FastAPI + Docker
- or Railway (https://railway.app)
- or Fly.io (https://fly.io)

Whoever's doing deployment should have an account ready and know how to connect a GitHub repo to it.

## 7. Sanity check before the event

Run through this once so you know your machine is ready:

```bash
python3 --version        # 3.11+
docker run hello-world    # should print the hello-world message
git --version
```

If all three work and you have a Gemini API key saved somewhere safe (not in Slack/Discord history), you're ready to start the moment the question drops.

## 8. What you do NOT need to install

- Node.js / npm — no frontend in this challenge, it's an API-only service.
- A database — the challenge is stateless, one request in, one response out.
- Any GPU / local model tooling — we're calling the Gemini API, not running anything locally.