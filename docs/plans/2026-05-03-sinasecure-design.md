# SinaSecure — TikTok Deepfake Detector (Design)

**Date:** 2026-05-03
**Context:** 2-hour hackathon sprint.

## Goal

A Chrome extension that, while a user watches a TikTok video page (e.g.
`https://www.tiktok.com/@gingermedz/video/7521054617959025933`), samples
frames from the video, sends them to a local deepfake-detection model, and
shows an in-page warning overlay if the model classifies a frame as a
deepfake.

## Non-goals

- Working on the For-You feed / scrolling feed (only single-video pages).
- Face detection / cropping (out of scope for 2 hours).
- Multi-frame aggregation / averaging.
- User accounts, history, or telemetry.
- Cross-browser support (Chrome only).

## Architecture

```
[TikTok video tab]                       [Local backend]
content.js ──captures frame every 2s──>  POST /detect
                                         FastAPI + transformers.pipeline
                                         model: prithivMLmods/
                                                Deep-Fake-Detector-v2-Model
           <── { label, score } ────     prediction
overlay shown if label == "Fake" && score > 0.7
```

Two processes:

1. **Chrome extension** — Manifest V3, content script injected on TikTok
   video pages. Captures frames via canvas, posts to backend, injects
   overlay on positive verdict.
2. **FastAPI backend** — local `uvicorn` on `http://localhost:8000`.
   Loads the HF pipeline once at startup, exposes `POST /detect`.

Communication is plain `fetch()` from the extension to localhost. CORS
middleware on the backend allows `chrome-extension://*` origin.

## Components

### Backend (`backend/`)

- `main.py` — FastAPI app. Single route `POST /detect` accepting a
  multipart image upload, returning `{ "label": "Real"|"Fake", "score": float }`.
- Pipeline is loaded **once at module load** via
  `pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")`.
- CORS middleware permitting extension origins.
- `requirements.txt`: `fastapi`, `uvicorn`, `transformers`, `torch`,
  `pillow`, `python-multipart`.

### Extension (`extension/`)

- `manifest.json` — Manifest V3.
  - `permissions`: `activeTab`, `scripting`.
  - `host_permissions`: `https://www.tiktok.com/*`,
    `http://localhost:8000/*`.
  - `content_scripts.matches`: `https://www.tiktok.com/*/video/*`.
- `content.js` — finds the single `<video>` element on the page, runs the
  2-second sampling loop, posts frames to the backend, injects the overlay
  on a "Fake" verdict.
- `overlay.css` — styling for the warning banner.
- `popup.html` (optional, time-permitting) — toggle + backend status.

## Frame capture & detection loop

Single `<video>` element per page. No `MutationObserver` needed.

```js
async function init() {
  const video = await waitFor(() => document.querySelector("video"));
  let flagged = false;
  setInterval(async () => {
    if (flagged || video.paused || video.ended) return;
    const blob = await grabFrame(video);
    const { label, score } = await postFrame(blob);
    if (label === "Fake" && score > 0.7) {
      flagged = true;
      injectOverlay(video, score);
    }
  }, 2000);
}

function grabFrame(video) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return new Promise(r => c.toBlob(r, "image/jpeg", 0.85));
}
```

**Canvas tainting check passed** (verified 2026-05-03 in console on the
sample TikTok video page). `drawImage` from a TikTok `<video>` does not
taint the canvas, so `toBlob` works.

## Overlay

On flag, inject a positioned `<div>` over the video container:

```
⚠ Deepfake detected (87% confidence)   [×]
```

Red background, white text, `position: absolute`, dismissible via `×`
button. Polling stops after the first flag — no re-trigger.

## Error handling (minimal)

- Backend down / fetch fails → `console.warn` and skip this tick.
- Model score < 0.7 → ignore, keep polling.
- Video element missing → `waitFor()` polls every 250ms up to 10s.
- Backend cold start → pipeline preloaded at FastAPI startup so the first
  request isn't penalized.

No retry logic, no backoff, no queueing.

## Testing the model in isolation

Before wiring the extension, verify the backend with `curl` against known
inputs:

- **Known real:** photo of self or any Unsplash face photo.
- **Known fake:** an image from `thispersondoesnotexist.com` (StyleGAN).

```bash
curl -X POST http://localhost:8000/detect -F "file=@real_face.jpg"
curl -X POST http://localhost:8000/detect -F "file=@fake_face.jpg"
```

Confirm both classes are detected before integrating.

**Caveat:** the model is trained on face crops; we feed it full TikTok
frames (UI overlays, captions, etc.). Expect noisier results. For the
demo, pick a TikTok video with a centered face.

## Time budget

| Time      | Task                                                            |
| --------- | --------------------------------------------------------------- |
| 0:00–0:20 | FastAPI scaffold, load pipeline, test `/detect` with `curl`     |
| 0:20–0:50 | Extension scaffold, frame capture + POST, log result to console |
| 0:50–1:10 | Overlay (CSS + inject logic)                                    |
| 1:10–1:30 | 2s loop + threshold + flag-once behavior                        |
| 1:30–1:50 | End-to-end test on real TikTok page, fix breakage               |
| 1:50–2:00 | Buffer / demo polish (icon, README screenshot)                  |

## Future work (out of scope for this sprint)

- Face detection (MTCNN / `facenet-pytorch`) before classification.
- Multi-frame averaging instead of first-flag-wins.
- Live confidence badge that updates every 2s.
- Support for the FYP scrolling feed (would need `MutationObserver`).
- Move inference to Hugging Face Inference API to avoid running a local
  backend.
