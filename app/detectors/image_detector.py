
from typing import Dict, Any
from PIL import Image, ImageChops
import numpy as np
import io

async def analyze_image(file) -> Dict[str, Any]:
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    ela_buf = io.BytesIO()
    img.save(ela_buf, 'JPEG', quality=95)
    ela_img = Image.open(io.BytesIO(ela_buf.getvalue()))
    diff = ImageChops.difference(img, ela_img)
    resid = np.array(diff).astype(np.float32)
    ela_score = float(resid.mean() / 255.0)
    arr = np.array(img).astype(np.float32) / 255.0
    lap = abs(arr[1:-1,1:-1,:] - 0.25*(arr[:-2,1:-1,:]+arr[2:,1:-1,:]+arr[1:-1,:-2,:]+arr[1:-1,2:,:]))
    texture = float(lap.mean())
    score = min(1.0, max(0.0, 0.4*ela_score + 0.6*(0.08 - texture) * 6))
    evidence = [f"ELA≈{ela_score:.3f}", f"texture≈{texture:.3f}"]
    return {"score": max(0.0, score), "likely_usage": (["ai_image"] if score>0.65 else []), "evidence": evidence, "limitations": ["Heuristic; add CNN detector for robustness"]}
