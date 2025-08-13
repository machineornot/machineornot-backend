
import io, math, os, re, tempfile, subprocess, json
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
LIGHT_MODE = os.getenv("LIGHT_MODE", "0") == "1"


app = FastAPI(title="AI Usage Detector", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
@app.get("/debug")
def debug():
    return {"light_mode": LIGHT_MODE}
def health():
    return {"ok": True}

@app.post("/analyze")
async def analyze(
    input: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None)
) -> Dict[str, Any]:
    # lazy imports so the app boots fast and healthcheck passes
    if LIGHT_MODE:
        from detectors.text_detector_light import analyze_text_or_url  # lightweight
    else:
        from detectors.text_detector import analyze_text_or_url        # heavy

    from detectors.image_detector import analyze_image
    from detectors.video_detector import analyze_video

    result = {"id": "job", "overall_confidence": 0.0, "modalities": {}, "explanations": []}

    if input:
        text_res = await analyze_text_or_url(input)
        result["modalities"]["text"] = text_res

    if file and getattr(file, "content_type", None):
        if file.content_type.startswith("image/"):
            img_res = await analyze_image(file)
            result["modalities"]["image"] = img_res
        elif file.content_type.startswith("video/"):
            vid_res = await analyze_video(file)
            result["modalities"]["video"] = vid_res

    if result["modalities"]:
        result["overall_confidence"] = max(v["score"] for v in result["modalities"].values())

    result["explanations"].append(
        "Scores are probabilistic. Light grammar/style tools are hard to distinguish definitively."
    )
    return result
