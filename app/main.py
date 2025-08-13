from fastapi.responses import JSONResponse
import traceback
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

from fastapi.responses import JSONResponse  # add this near your other imports
import traceback  # add this near your other imports

@app.post("/analyze")
async def analyze(
    input: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None)
) -> Dict[str, Any]:
    try:
        # Decide which text detector to use (LIGHT_MODE or full)
        if LIGHT_MODE:
            from detectors.text_detector_light import analyze_text_or_url  # lightweight
        else:
            from detectors.text_detector import analyze_text_or_url        # heavy

        result = {"id": "job", "overall_confidence": 0.0, "modalities": {}, "explanations": []}

        # TEXT path (only import and run text if we actually have input)
        if input:
            text_res = await analyze_text_or_url(input)
            result["modalities"]["text"] = text_res

        # FILE path (import only what we need)
        if file and getattr(file, "content_type", None):
            ctype = file.content_type or ""
            if ctype.startswith("image/"):
                from detectors.image_detector import analyze_image
                img_res = await analyze_image(file)
                result["modalities"]["image"] = img_res
            elif ctype.startswith("video/"):
                from detectors.video_detector import analyze_video
                vid_res = await analyze_video(file)
                result["modalities"]["video"] = vid_res

        if result["modalities"]:
            result["overall_confidence"] = max(v["score"] for v in result["modalities"].values())

        result["explanations"].append(
            "Scores are probabilistic. Light grammar/style tools are hard to distinguish definitively."
        )
        return result

    except Exception as e:
        # Return the error details to the client so we can see what's wrong
        tb = traceback.format_exc()
        # Print to logs too
        print(tb)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": tb}
        )
