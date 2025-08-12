
# MachineOrNot Backend (MVP)
FastAPI service exposing `/analyze` for text, image, and video. Probabilistic results with evidence.
Endpoints:
- GET /health
- POST /analyze (multipart: input=text/url, file=image|video)
Deploy: Docker (any VPS) or Render/Railway (connect GitHub, port 8000).
