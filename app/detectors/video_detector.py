
from typing import Dict, Any
import tempfile, subprocess, os
from .image_detector import analyze_image

async def analyze_video(file) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        path = tmp.name
    frames_dir = path + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(["ffmpeg","-i",path,"-vf","fps=1","-qscale:v","3",f"{frames_dir}/f%03d.jpg","-hide_banner","-loglevel","error"], check=False)
    frames = sorted([os.path.join(frames_dir,f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    pick = frames[::max(1,len(frames)//5)] or frames[:5]
    img_scores = []
    for f in pick:
        with open(f,"rb") as fh:
            class U:
                content_type = "image/jpeg"
                async def read(self_inner): return fh.read()
            res = await analyze_image(U())
            img_scores.append(res["score"])
    vid_score = sum(img_scores)/len(img_scores) if img_scores else 0.3
    usage = ["ai_frames"] if vid_score>0.7 else []
    return {"score": vid_score, "likely_usage": usage, "evidence": [f"frame_scores={ [round(s,2) for s in img_scores] }"], "limitations": ["No ASR in MVP; add Whisper + TTS cues"]}
