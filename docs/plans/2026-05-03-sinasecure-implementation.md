# SinaSecure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Chrome extension + local FastAPI backend that detects deepfakes in TikTok videos every 2 seconds and shows an in-page warning overlay.

**Architecture:** Manifest V3 Chrome extension injects a content script on `tiktok.com/*/video/*`. The content script captures a frame from the `<video>` element every 2 seconds via canvas, POSTs the JPEG to a local FastAPI server. The server runs `transformers.pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")` and returns `{label, score}`. On the first response with `label == "Fake"` and `score > 0.7`, the extension injects a red warning overlay on top of the video and stops polling.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, transformers, torch, Pillow, python-multipart. Chrome Manifest V3 (vanilla JS — no bundler).

**Reference:** See `docs/plans/2026-05-03-sinasecure-design.md` for the approved design.

---

## Pragmatic note for this 2-hour sprint

This plan applies strict TDD to the **backend** (where it's natural) and uses **manual smoke tests** for the Chrome extension parts (where setting up jsdom/puppeteer would burn the whole time budget). Each task ends in a commit. If you blow the budget, ship what you have — the design doc lists future work.

---

## Task 1: Project skeleton

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/main.py`
- Create: `extension/manifest.json`
- Create: `extension/content.js`
- Create: `extension/overlay.css`
- Create: `.gitignore`

**Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
.DS_Store
node_modules/
*.log
```

**Step 2: Create `backend/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
transformers==4.46.0
torch==2.5.0
pillow==11.0.0
python-multipart==0.0.12
```

**Step 3: Create empty placeholders for the other files**

Create `backend/main.py`, `extension/manifest.json`, `extension/content.js`, `extension/overlay.css` as empty files. We'll fill them in subsequent tasks.

**Step 4: Commit**

```bash
git add .gitignore backend/ extension/
git commit -m "scaffold backend and extension directories"
```

---

## Task 2: Set up Python venv and install deps

**Files:** none (env setup only).

**Step 1: Create venv and install**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Verify imports**

```bash
python -c "from transformers import pipeline; from fastapi import FastAPI; print('ok')"
```

Expected: `ok`. If torch fails to install on Apple Silicon, use `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

**Step 3: No commit** (env is git-ignored).

---

## Task 3: Write failing test for `/detect` endpoint shape

**Files:**
- Create: `backend/test_main.py`

**Step 1: Write the test**

```python
# backend/test_main.py
from fastapi.testclient import TestClient
from PIL import Image
import io


def make_jpeg_bytes():
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_detect_returns_label_and_score():
    from main import app
    client = TestClient(app)
    resp = client.post(
        "/detect",
        files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "label" in body
    assert "score" in body
    assert isinstance(body["score"], float)
    assert 0.0 <= body["score"] <= 1.0
```

**Step 2: Run the test, verify it fails**

```bash
cd backend
source .venv/bin/activate
pytest test_main.py -v
```

Expected: FAIL — `main` module has no `app` (file is empty).

**Step 3: No commit yet** (we commit after green).

---

## Task 4: Implement minimal `/detect` to pass the test

**Files:**
- Modify: `backend/main.py`

**Step 1: Implement**

```python
# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load the model once at startup. First import is slow (downloads weights).
detector = pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-v2-Model",
)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    results = detector(img)
    # pipeline returns a list of {label, score} sorted by score desc
    top = results[0]
    return {"label": top["label"], "score": float(top["score"])}
```

**Step 2: Run the test**

```bash
pytest test_main.py -v
```

Expected: PASS. (First run downloads model weights — be patient, ~2 min.)

**Step 3: Commit**

```bash
git add backend/main.py backend/test_main.py
git commit -m "add /detect endpoint backed by HF deepfake pipeline"
```

---

## Task 5: Smoke-test the backend with real images

**Files:** none (manual verification).

**Step 1: Start the server**

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Wait for `Application startup complete.` (model loads on first request — first call may be slow if reload re-imports).

**Step 2: Get sample images**

In a second terminal:

```bash
mkdir -p /tmp/sina-samples
# Real face from Unsplash
curl -L -o /tmp/sina-samples/real.jpg "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=400"
# StyleGAN-generated fake face
curl -L -o /tmp/sina-samples/fake.jpg "https://thispersondoesnotexist.com/"
```

**Step 3: Test both**

```bash
curl -s -X POST http://localhost:8000/detect -F "file=@/tmp/sina-samples/real.jpg" | python -m json.tool
curl -s -X POST http://localhost:8000/detect -F "file=@/tmp/sina-samples/fake.jpg" | python -m json.tool
```

Expected: both return `{"label": "...", "score": ...}`. The labels should differ between the two images. If they don't, note it but proceed — model accuracy is a known caveat (see design doc). The pipeline is working as long as you get valid JSON.

**Step 4: No commit** (manual test only).

---

## Task 6: Chrome extension manifest

**Files:**
- Modify: `extension/manifest.json`

**Step 1: Write the manifest**

```json
{
  "manifest_version": 3,
  "name": "SinaSecure",
  "version": "0.1.0",
  "description": "Detects deepfakes in TikTok videos.",
  "permissions": ["activeTab", "scripting"],
  "host_permissions": [
    "https://www.tiktok.com/*",
    "http://localhost:8000/*"
  ],
  "content_scripts": [
    {
      "matches": ["https://www.tiktok.com/*/video/*"],
      "js": ["content.js"],
      "css": ["overlay.css"],
      "run_at": "document_idle"
    }
  ]
}
```

**Step 2: Load the unpacked extension**

1. Open `chrome://extensions/`.
2. Toggle **Developer mode** on (top right).
3. Click **Load unpacked** → select the `extension/` folder.
4. Confirm "SinaSecure" appears with no errors.

**Step 3: Commit**

```bash
git add extension/manifest.json
git commit -m "add Chrome extension manifest"
```

---

## Task 7: Content script — wait for the video element

**Files:**
- Modify: `extension/content.js`

**Step 1: Implement**

```js
// extension/content.js
console.log("[SinaSecure] content script loaded");

function waitFor(predicate, { intervalMs = 250, timeoutMs = 10000 } = {}) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const tick = () => {
      const result = predicate();
      if (result) return resolve(result);
      if (Date.now() - start > timeoutMs) return reject(new Error("timeout"));
      setTimeout(tick, intervalMs);
    };
    tick();
  });
}

async function init() {
  try {
    const video = await waitFor(() => document.querySelector("video"));
    console.log("[SinaSecure] video found:", video);
  } catch (e) {
    console.warn("[SinaSecure] no video on page:", e.message);
  }
}

init();
```

**Step 2: Smoke test**

1. Reload the extension at `chrome://extensions/` (click the refresh icon on the SinaSecure card).
2. Open `https://www.tiktok.com/@gingermedz/video/7521054617959025933`.
3. Open DevTools → Console.
4. Confirm you see: `[SinaSecure] content script loaded` and `[SinaSecure] video found: <video ...>`.

**Step 3: Commit**

```bash
git add extension/content.js
git commit -m "wait for video element in content script"
```

---

## Task 8: Frame capture

**Files:**
- Modify: `extension/content.js`

**Step 1: Add `grabFrame` and call it once for verification**

Append to `content.js`:

```js
function grabFrame(video) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return new Promise(r => c.toBlob(r, "image/jpeg", 0.85));
}
```

And modify `init()` to capture once after the video is found:

```js
async function init() {
  try {
    const video = await waitFor(() => document.querySelector("video"));
    console.log("[SinaSecure] video found:", video);
    const blob = await grabFrame(video);
    console.log("[SinaSecure] captured blob:", blob.size, "bytes");
  } catch (e) {
    console.warn("[SinaSecure] error:", e.message);
  }
}
```

**Step 2: Smoke test**

1. Reload extension, reload the TikTok page.
2. Make sure the video is playing (click if needed).
3. Confirm console logs: `[SinaSecure] captured blob: <some number> bytes` with size > 1000.

If size is 0 or you see a SecurityError, the canvas got tainted — fall back to `chrome.tabCapture`. (Pre-flight check confirmed this isn't expected to happen for TikTok.)

**Step 3: Commit**

```bash
git add extension/content.js
git commit -m "capture single frame from video to JPEG blob"
```

---

## Task 9: POST frame to backend

**Files:**
- Modify: `extension/content.js`

**Step 1: Add `postFrame` and verify roundtrip**

Append:

```js
async function postFrame(blob) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  const resp = await fetch("http://localhost:8000/detect", {
    method: "POST",
    body: fd,
  });
  if (!resp.ok) throw new Error(`backend ${resp.status}`);
  return resp.json();
}
```

Modify `init()` to post the captured frame:

```js
async function init() {
  try {
    const video = await waitFor(() => document.querySelector("video"));
    const blob = await grabFrame(video);
    const result = await postFrame(blob);
    console.log("[SinaSecure] verdict:", result);
  } catch (e) {
    console.warn("[SinaSecure] error:", e.message);
  }
}
```

**Step 2: Smoke test**

1. Confirm the FastAPI server is still running (`uvicorn main:app --reload --port 8000`).
2. Reload extension, reload the TikTok page, make the video play.
3. Console should log `[SinaSecure] verdict: {label: "...", score: 0.xx}`.

If you get a CORS error in console, double-check the backend's CORSMiddleware allows `*` (Task 4). If you get `Failed to fetch`, the backend isn't running or isn't on port 8000.

**Step 3: Commit**

```bash
git add extension/content.js
git commit -m "POST captured frames to detection backend"
```

---

## Task 10: 2-second polling loop with flag-once behavior

**Files:**
- Modify: `extension/content.js`

**Step 1: Replace the one-shot `init` with the polling loop**

Replace the body of `init()` (everything inside the `try` block) with:

```js
const video = await waitFor(() => document.querySelector("video"));
console.log("[SinaSecure] watching video");

let flagged = false;
setInterval(async () => {
  if (flagged || video.paused || video.ended) return;
  try {
    const blob = await grabFrame(video);
    const { label, score } = await postFrame(blob);
    console.log("[SinaSecure]", label, score.toFixed(2));
    if (label === "Fake" && score > 0.7) {
      flagged = true;
      injectOverlay(video, score);
    }
  } catch (e) {
    console.warn("[SinaSecure] tick failed:", e.message);
  }
}, 2000);
```

Add a stub `injectOverlay` at the bottom (real impl in next task):

```js
function injectOverlay(video, score) {
  console.log("[SinaSecure] FLAGGED at", score);
}
```

**Note:** Verify the exact label string the model returns (it might be `"Fake"`, `"fake"`, `"Deepfake"`, etc. — check the verdicts logged in Task 9). Adjust the comparison accordingly.

**Step 2: Smoke test**

1. Reload extension, reload page, play the video.
2. Console should log a verdict roughly every 2 seconds.
3. Pause the video — logging should stop.

**Step 3: Commit**

```bash
git add extension/content.js
git commit -m "poll video every 2s and stop on first deepfake flag"
```

---

## Task 11: Warning overlay

**Files:**
- Modify: `extension/overlay.css`
- Modify: `extension/content.js`

**Step 1: Write the CSS**

```css
/* extension/overlay.css */
.sinasecure-warning {
  position: fixed;
  top: 16px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 2147483647;
  background: #d92626;
  color: #fff;
  padding: 12px 20px;
  border-radius: 8px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 16px;
  font-weight: 600;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  gap: 12px;
}

.sinasecure-warning button {
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.6);
  color: #fff;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 14px;
  cursor: pointer;
}

.sinasecure-warning button:hover {
  background: rgba(255, 255, 255, 0.15);
}
```

**Step 2: Replace the stub `injectOverlay` in `content.js`**

```js
function injectOverlay(video, score) {
  if (document.querySelector(".sinasecure-warning")) return;
  const banner = document.createElement("div");
  banner.className = "sinasecure-warning";
  banner.innerHTML = `
    <span>⚠ Deepfake detected (${Math.round(score * 100)}% confidence)</span>
    <button type="button">×</button>
  `;
  banner.querySelector("button").addEventListener("click", () => banner.remove());
  document.body.appendChild(banner);
}
```

**Step 3: Smoke test**

1. Reload extension, reload page, play video.
2. To force the overlay during testing, temporarily change the threshold check in `content.js` to `if (true)` and reload. Confirm the banner appears with the dismiss button working.
3. Revert the threshold back to `label === "Fake" && score > 0.7`.

**Step 4: Commit**

```bash
git add extension/content.js extension/overlay.css
git commit -m "inject deepfake warning overlay on first positive verdict"
```

---

## Task 12: End-to-end demo run

**Files:** none (validation only).

**Step 1: Full restart**

1. Backend: `uvicorn main:app --reload --port 8000` (wait for startup).
2. Extension: reload at `chrome://extensions/`.
3. Open `https://www.tiktok.com/@gingermedz/video/7521054617959025933`.

**Step 2: Validate**

- DevTools console shows `[SinaSecure] watching video` and a verdict every ~2s.
- If the model classifies the video as Fake at any tick with score > 0.7, the red overlay appears and polling stops.
- Dismiss button removes the overlay.
- Pausing the video stops the polling logs.

**Step 3: Test on a different video**

Try at least one more video URL. Note that classification accuracy is variable — see design doc caveat.

**Step 4: If working, tag a demo commit**

```bash
git commit --allow-empty -m "demo ready"
```

---

## Out of scope (do not do)

- Face detection / cropping pre-pass.
- Multi-frame averaging.
- Live confidence badge.
- FYP feed support (would need MutationObserver).
- Extension popup UI.
- Production hardening (auth, rate limiting, error UI).

These are listed in the design doc's "Future work" section.
