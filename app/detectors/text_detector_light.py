# app/detectors/text_detector_light.py
import re
from typing import Dict, Any

def _burstiness(text: str) -> float:
    sents = re.split(r'[.!?]', text)
    lengths = [len(s.split()) for s in sents if len(s.split())>0]
    if len(lengths) < 3:
        return 0.0
    mean = sum(lengths)/len(lengths)
    var = sum((x-mean)**2 for x in lengths)/len(lengths)
    return var**0.5

async def analyze_text_or_url(input_or_url: str) -> Dict[str, Any]:
    # Accept both raw text and URLs (very light handling)
    text = input_or_url.strip()

    # If it's a URL, return a low-confidence placeholder (LIGHT_MODE skips fetching)
    if re.match(r'^https?://', text, re.I):
        return {
            "score": 0.5,
            "likely_usage": [],
            "evidence": ["LIGHT_MODE active: URL fetch & heavy models disabled"],
            "limitations": ["Enable full mode to analyze remote text"]
        }

    if len(text) < 200:
        return {
            "score": 0.3,
            "likely_usage": [],
            "evidence": ["Short text reduces certainty (LIGHT_MODE)"],
            "limitations": ["Need >200 words for reliability"]
        }

    bur = _burstiness(text)

    # Heuristic scoring in LIGHT_MODE
    score = 0.55
    likely = []
    if bur < 6:
        score = max(score, 0.65)
        likely.append("likely_generation")

    evidence = [f"burstiness≈{bur:.1f}", "LIGHT_MODE"]
    return {"score": score, "likely_usage": likely, "evidence": evidence, "limitations": []}
