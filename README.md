# Indonesia Toxic Speech Detector

Notebook-first training and export workflow for an Indonesian toxic speech classifier based on `indolem/indobertweet-base-uncased`.

The project trains a binary classifier for Indonesian text and exports production inference artifacts for CPU deployment with ONNX Runtime. The ONNX artifact is kept in FP32 because a quantized ONNX candidate produced unacceptable probability drift.

## Repository Structure

```text
.
+-- dataset/
|   +-- indonesian_toxicspeech.csv
+-- notebooks/
    +-- indobert_production_training.ipynb
```

## Dataset

The training notebook expects a CSV with exactly these columns:

| Column | Type | Description |
| --- | --- | --- |
| `text` | string | Indonesian text to classify. |
| `is_toxic` | integer | Binary label: `0` for non-toxic, `1` for toxic. |

The included dataset is stored at:

```text
dataset/indonesian_toxicspeech.csv
```

## Model Workflow

The notebook performs the full production workflow:

1. Load and validate the dataset.
2. Create stratified train, validation, and test splits.
3. Fine-tune IndoBERTweet for binary sequence classification.
4. Tune the toxic-class decision threshold on the validation split.
5. Report final test metrics once using the selected threshold.
6. Export Hugging Face and FP32 ONNX CPU inference artifacts.
7. Run ONNX Runtime CPU smoke tests and probability-drift checks.

## Quick Start

Open the notebook:

```text
notebooks/indobert_production_training.ipynb
```

Run the cells from top to bottom in Colab or a Python environment with GPU support for training. The setup cell installs the notebook-specific dependencies:

```python
%pip install -q --upgrade --upgrade-strategy only-if-needed transformers datasets accelerate onnx onnxscript onnxruntime scikit-learn
```

The notebook can train on CPU, but GPU is strongly recommended.

## Output Artifacts

After a successful notebook run, artifacts are written under `artifacts/`:

```text
artifacts/
+-- toxic_speech_model_pt/
+-- toxic_speech_model_onnx_cpu/
|   +-- model.onnx
|   +-- tokenizer.json
|   +-- tokenizer/config files
+-- label_mapping.json
+-- threshold.json
+-- metrics.json
```

The notebook also creates:

```text
toxic_speech_inference_artifacts.zip
```

Generated model artifacts are not part of the committed repository.

## CPU Inference Contract

The exported ONNX model is intended for CPU inference:

```python
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

artifact_dir = "artifacts"
model_dir = f"{artifact_dir}/toxic_speech_model_onnx_cpu"

with open(f"{artifact_dir}/threshold.json", "r", encoding="utf-8") as handle:
    threshold = float(json.load(handle)["threshold"])

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
session = ort.InferenceSession(
    f"{model_dir}/model.onnx",
    providers=["CPUExecutionProvider"],
)
input_names = [item.name for item in session.get_inputs()]

def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)

def predict_toxic(text, max_length):
    encoded = tokenizer(
        str(text).strip(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    feed = {name: encoded[name] for name in input_names if name in encoded}
    logits = session.run(None, feed)[0]
    toxic_score = float(softmax(logits)[0, 1])
    return {
        "label": "toxic" if toxic_score >= threshold else "non_toxic",
        "score": toxic_score,
        "threshold": threshold,
    }
```

Use the `max_length` value saved in `artifacts/metrics.json`.

## Notes on Quantization

ONNX quantization is intentionally not part of the current production path. A quantized candidate showed a maximum probability difference of `0.7465`, which is too large for reliable classification. The deployment artifact is therefore FP32 ONNX running on ONNX Runtime CPU.

## Limitations

- This repository is a training and artifact-export project, not a packaged web service.
- Toxic speech classifiers can produce false positives and false negatives, especially on slang, sarcasm, code-switching, and domain-specific language.
- Review model outputs before using them for moderation or enforcement decisions.

## License

No license file is currently included in this repository. Add one before distributing or reusing the project outside its current owner context.
