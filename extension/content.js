console.log("[SinaSecure] content script loaded");

function waitFor(predicate, opts = {}) {
  const intervalMs = opts.intervalMs || 250;
  const timeoutMs = opts.timeoutMs || 10000;
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const timer = setInterval(() => {
      let result;
      try {
        result = predicate();
      } catch (err) {
        clearInterval(timer);
        reject(err);
        return;
      }
      if (result) {
        clearInterval(timer);
        resolve(result);
        return;
      }
      if (Date.now() - start >= timeoutMs) {
        clearInterval(timer);
        reject(new Error("waitFor: timeout"));
      }
    }, intervalMs);
  });
}

function grabFrame(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85);
  });
}

async function postFrame(blob) {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  const response = await fetch("http://localhost:8001/detect", {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error("postFrame: HTTP " + response.status);
  }
  return response.json();
}

function injectOverlay(score) {
  if (document.querySelector(".sinasecure-warning")) {
    return;
  }
  const div = document.createElement("div");
  div.className = "sinasecure-warning";
  const pct = Math.round(score * 100);
  const label = document.createElement("span");
  label.textContent = "\u26A0 Deepfake detected (" + pct + "% confidence)";
  const button = document.createElement("button");
  button.textContent = "\u00D7";
  button.addEventListener("click", () => {
    div.remove();
  });
  div.appendChild(label);
  div.appendChild(button);
  document.body.appendChild(div);
}

async function init() {
  const video = await waitFor(() => document.querySelector("video"));
  console.log("[SinaSecure] watching video");
  let flagged = false;
  setInterval(async () => {
    if (flagged || video.paused || video.ended) {
      return;
    }
    try {
      const blob = await grabFrame(video);
      const result = await postFrame(blob);
      const { label, score } = result;
      console.log("[SinaSecure] " + label + " " + score.toFixed(2));
      if (label === "Fake" && score > 0.7) {
        flagged = true;
        injectOverlay(score);
      }
    } catch (err) {
      console.warn("[SinaSecure] tick failed", err);
    }
  }, 2000);
}

init();
