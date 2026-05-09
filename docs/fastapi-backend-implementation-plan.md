# FastAPI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal FastAPI backend in `HateSpeech-BE` that serves Indonesian toxic speech model inference to the React/Vite frontend.

**Architecture:** Keep the backend small but separated by responsibility: settings, schemas, inference service, API routes, and app bootstrap. The app loads ONNX CPU artifacts at startup from backend-local `HateSpeech-BE/artifacts` and exposes non-versioned endpoints for health, model metadata, and single-text prediction.

**Tech Stack:** FastAPI, Uvicorn, Pydantic, ONNX Runtime CPU, Transformers tokenizer, NumPy, pytest, Docker.

---

## API Contract

Base URL for local frontend development:

```text
http://localhost:8000
```

### `GET /health`

Returns backend and model readiness.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /model-info`

Returns model metadata loaded from the artifact bundle.

```json
{
  "model_key": "indobertweet_production",
  "model_name": "indolem/indobertweet-base-uncased",
  "labels": ["non_toxic", "toxic"],
  "threshold": 0.49,
  "max_length": 128
}
```

### `POST /predict`

Request:

```json
{
  "text": "contoh teks komentar"
}
```

Validation:

- `text` is required.
- `text` must be a string.
- `text` must contain at least one non-whitespace character.
- `text` must be at most `5000` characters.

Response:

```json
{
  "label": "toxic",
  "is_toxic": true,
  "scores": {
    "non_toxic": 0.08,
    "toxic": 0.92
  },
  "threshold": 0.49
}
```

Errors:

- Use FastAPI's standard `422` validation response for invalid request bodies.
- Return `503` only if the model is unavailable after app startup.

Frontend usage:

```ts
const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text }),
});
```

Frontend environment:

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## File Structure

Create these backend files inside `HateSpeech-BE`:

```text
HateSpeech-BE/
+-- app/
|   +-- __init__.py
|   +-- main.py
|   +-- api.py
|   +-- inference.py
|   +-- schemas.py
|   +-- settings.py
+-- tests/
|   +-- test_api.py
|   +-- test_inference.py
+-- .dockerignore
+-- .gitignore
+-- Dockerfile
+-- README.md
+-- requirements.txt
```

Artifact location:

```text
HateSpeech-BE/artifacts/
+-- label_mapping.json
+-- metrics.json
+-- threshold.json
+-- toxic_speech_model_onnx_cpu/
    +-- config.json
    +-- model.onnx
    +-- tokenizer.json
    +-- tokenizer_config.json
```

`HateSpeech-BE/artifacts/` must be ignored by git because the ONNX model is large.

---

## Tasks

### Task 1: Backend Skeleton And Dependencies

**Files:**

- Create: `HateSpeech-BE/requirements.txt`
- Create: `HateSpeech-BE/app/__init__.py`
- Create: `HateSpeech-BE/.gitignore`

- [ ] Add dependencies:

```text
fastapi
uvicorn[standard]
pydantic
numpy
onnxruntime
transformers
pytest
httpx
```

- [ ] Add backend git ignores:

```gitignore
__pycache__/
.pytest_cache/
.venv/
artifacts/
```

- [ ] Verify dependency installation:

```bash
cd HateSpeech-BE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: dependencies install without errors.

### Task 2: Settings And Schemas

**Files:**

- Create: `HateSpeech-BE/app/settings.py`
- Create: `HateSpeech-BE/app/schemas.py`
- Test: `HateSpeech-BE/tests/test_api.py`

- [ ] Define settings:

```python
from functools import lru_cache
from pathlib import Path
import os


class Settings:
    app_name = "Indonesia Toxic Speech Detector API"
    artifact_dir = Path(os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))
    cors_origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] Define Pydantic schemas:

```python
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_key: str
    model_name: str
    labels: list[str]
    threshold: float
    max_length: int


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned


class PredictionScores(BaseModel):
    non_toxic: float
    toxic: float


class PredictResponse(BaseModel):
    label: str
    is_toxic: bool
    scores: PredictionScores
    threshold: float
```

- [ ] Add validation tests for blank and overlong text.

Expected: direct Pydantic validation rejects invalid input before route logic runs.

### Task 3: ONNX Inference Service

**Files:**

- Create: `HateSpeech-BE/app/inference.py`
- Test: `HateSpeech-BE/tests/test_inference.py`

- [ ] Implement an `InferenceService` that:
  - Loads `metrics.json`, `threshold.json`, and `label_mapping.json`.
  - Reads `max_length` from `metrics.json`.
  - Loads tokenizer from `artifacts/toxic_speech_model_onnx_cpu`.
  - Loads `model.onnx` with `CPUExecutionProvider`.
  - Applies stable softmax.
  - Returns `non_toxic` and `toxic` probabilities.
  - Uses threshold `0.49` from `threshold.json` to set `label` and `is_toxic`.

- [ ] Use this softmax behavior:

```python
def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)
```

- [ ] Add tests for threshold decisions:

```python
def test_threshold_decision_toxic():
    result = service._format_prediction(non_toxic_score=0.4, toxic_score=0.6)
    assert result.label == "toxic"
    assert result.is_toxic is True


def test_threshold_decision_non_toxic():
    result = service._format_prediction(non_toxic_score=0.8, toxic_score=0.2)
    assert result.label == "non_toxic"
    assert result.is_toxic is False
```

- [ ] Add a real-artifact smoke test marked/skippable when `HateSpeech-BE/artifacts` is missing.

Expected: CI can run mocked/unit tests without requiring the 443 MB model, while local developers can verify real inference.

### Task 4: FastAPI App And Routes

**Files:**

- Create: `HateSpeech-BE/app/api.py`
- Create: `HateSpeech-BE/app/main.py`
- Test: `HateSpeech-BE/tests/test_api.py`

- [ ] Create the FastAPI app with CORS:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.inference import InferenceService
from app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.inference_service = InferenceService(settings.artifact_dir)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
```

- [ ] Implement routes:

```python
from fastapi import APIRouter, Request

from app.schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=hasattr(request.app.state, "inference_service"))


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    return request.app.state.inference_service.model_info()


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    return request.app.state.inference_service.predict(payload.text)
```

- [ ] Add API tests using dependency/state injection with a fake inference service:

```python
class FakeInferenceService:
    def model_info(self):
        return {
            "model_key": "indobertweet_production",
            "model_name": "indolem/indobertweet-base-uncased",
            "labels": ["non_toxic", "toxic"],
            "threshold": 0.49,
            "max_length": 128,
        }

    def predict(self, text: str):
        return {
            "label": "toxic",
            "is_toxic": True,
            "scores": {"non_toxic": 0.1, "toxic": 0.9},
            "threshold": 0.49,
        }
```

- [ ] Verify endpoints:

```bash
cd HateSpeech-BE
pytest -q
```

Expected: all API and inference unit tests pass.

### Task 5: Docker And Local Run Documentation

**Files:**

- Create: `HateSpeech-BE/Dockerfile`
- Create: `HateSpeech-BE/.dockerignore`
- Modify: `HateSpeech-BE/README.md`

- [ ] Add Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] Add `.dockerignore`:

```text
.venv
__pycache__
.pytest_cache
tests
artifacts
```

- [ ] Document local setup:

```bash
cd HateSpeech-BE
mkdir -p artifacts
# copy or extract the inference artifact bundle into HateSpeech-BE/artifacts
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- [ ] Document Docker run:

```bash
cd HateSpeech-BE
docker build -t hate-speech-be .
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  -e CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173" \
  hate-speech-be
```

- [ ] Document frontend contract examples for `/health`, `/model-info`, and `/predict`.

Expected: backend can be started locally and from Docker with mounted artifacts.

### Task 6: Final Verification

**Files:**

- No new files.

- [ ] Run tests:

```bash
cd HateSpeech-BE
pytest -q
```

Expected: all tests pass.

- [ ] Run local server:

```bash
cd HateSpeech-BE
uvicorn app.main:app --reload
```

Expected: server starts at `http://127.0.0.1:8000`.

- [ ] Verify health:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok","model_loaded":true}
```

- [ ] Verify prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"contoh teks kasar"}'
```

Expected: response includes `label`, `is_toxic`, `scores.non_toxic`, `scores.toxic`, and `threshold`.

- [ ] Verify OpenAPI:

```text
http://127.0.0.1:8000/docs
```

Expected: docs show non-versioned `/health`, `/model-info`, and `/predict` endpoints only.

---

## Assumptions

- No API versioning is used; endpoints are root-level.
- No authentication or API key is required.
- Only single-text prediction is in scope.
- Batch prediction, history, moderation workflows, and feedback capture are out of scope.
- The backend owns `HateSpeech-BE/artifacts` locally, but model files remain ignored by git.
- The app loads the model at startup and fails fast if required artifacts are missing or invalid.
- Default CORS allows Vite dev origins and can be overridden through `CORS_ORIGINS`.
