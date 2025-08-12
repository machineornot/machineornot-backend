
import re, math, asyncio
from typing import Dict, Any, List
from bs4 import BeautifulSoup
import httpx
import language_tool_python
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tool = language_tool_python.LanguageTool('en-US')
tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2.eval()
if torch.cuda.is_available():
    gpt2.cuda()

async def fetch_url_text(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script","style","noscript","header","footer","nav","aside"]):
            t.decompose()
        text = ' '.join(soup.get_text(" ").split())
        return text[:20000]

def get_text_perplexity(text: str) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        enc = {k:v.cuda() for k,v in enc.items()}
    with torch.no_grad():
        loss = gpt2(**enc, labels=enc["input_ids"]).loss
    return math.exp(loss.item())

def burstiness(text: str) -> float:
    sents = re.split(r'[.!?]', text)
    lengths = [len(s.split()) for s in sents if len(s.split())>0]
    if len(lengths) < 3:
        return 0.0
    mean = sum(lengths)/len(lengths)
    var = sum((x-mean)**2 for x in lengths)/len(lengths)
    return var**0.5

def classify_usage(ppx: float, bur: float, errs_before: int, errs_after: int, token_len: int) -> List[str]:
    labels = []
    if errs_before - errs_after >= 5 and abs(bur) < 10 and 40 < ppx < 70:
        labels.append("grammar_fix")
    if token_len > 200 and bur < 6 and ppx < 45:
        labels.append("likely_generation")
    return labels

async def analyze_text_or_url(input_or_url: str) -> Dict[str, Any]:
    text = input_or_url
    if re.match(r'^https?://', input_or_url.strip(), re.I):
        text = await fetch_url_text(input_or_url)
    text = text.strip()
    if len(text) < 200:
        return {"score": 0.3, "likely_usage": [], "evidence": ["short text reduces certainty"], "limitations": ["Need >200 words for reliability"]}
    raw_errs = len(tool.check(text))
    noised = re.sub(r'\b(the)\b', 'teh', text, flags=re.I)
    errs_after = len(tool.check(noised))
    ppx = get_text_perplexity(text)
    bur = burstiness(text)
    token_len = len(tok(text)["input_ids"])
    usage = classify_usage(ppx, bur, errs_after, raw_errs, token_len)
    score = 0.5
    if "likely_generation" in usage: score = max(score, 0.75)
    if "grammar_fix" in usage: score = max(score, 0.6)
    if ppx < 45 and bur < 8: score = max(score, 0.65)
    evidence = [f"perplexity≈{ppx:.1f}", f"burstiness≈{bur:.1f}", f"grammar_issues≈{raw_errs}"]
    return {"score": score, "likely_usage": usage, "evidence": evidence, "limitations": []}
