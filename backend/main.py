from pathlib import Path
import io

import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "best_model-v3.pt"


def _load_model() -> torch.nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2),
    )
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
    model.eval()
    return model


model = _load_model()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_LABELS = ["Real", "Fake"]

DEBUG_DIR = Path(__file__).resolve().parent / "debug_frames"
DEBUG_DIR.mkdir(exist_ok=True)
MAX_DEBUG_FRAMES = 3
_debug_count = 0


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global _debug_count
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    input_tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = int(torch.argmax(probs).item())
    label = _LABELS[pred]
    score = float(probs[pred].item())

    if _debug_count < MAX_DEBUG_FRAMES:
        _debug_count += 1
        fname = f"frame_{_debug_count:02d}_{label}_{score:.3f}.jpg"
        img.save(DEBUG_DIR / fname, "JPEG", quality=90)
        print(f"[debug] saved {fname} ({img.size[0]}x{img.size[1]})")

    return {"label": label, "score": score}
