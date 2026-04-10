"""
Flask web app for MMEC Earnings Call Volatility Prediction.
Run: python app.py
"""
import os, yaml, torch, numpy as np, pandas as pd
from flask import Flask, render_template, request, jsonify
from src.model import build_model

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Globals loaded once at startup
# ---------------------------------------------------------------------------
CONFIG_PATH = "configs/default.yaml"
PROCESSED_DIR = "data/processed"
CHECKPOINTS_DIR = "checkpoints"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

DEVICE = "cpu"
MANIFEST = pd.read_csv(os.path.join(PROCESSED_DIR, "manifest.csv"))
LABELS = pd.read_csv(CFG["data"]["labels_path"])

MODEL_TYPES = ["early_fusion", "cross_modal", "text_only", "audio_only"]
MODELS = {}

for mtype in MODEL_TYPES:
    ckpt = os.path.join(CHECKPOINTS_DIR, f"{mtype}_seed0.pt")
    if os.path.exists(ckpt):
        try:
            model = build_model(CFG, mtype).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
            model.eval()
            MODELS[mtype] = model
            print(f"  ✓ Loaded {mtype}")
        except Exception as e:
            print(f"  ✗ Failed to load {mtype}: {e}")

print(f"\nReady — {len(MODELS)} models loaded, {len(MANIFEST)} calls available.\n")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(pt_path, model_type="early_fusion"):
    if model_type not in MODELS:
        return None, f"Model '{model_type}' not loaded"

    model = MODELS[model_type]
    data = torch.load(pt_path, weights_only=True)

    n = min(data["n_utterances"], CFG["data"]["max_utterances"])
    acoustic = data["acoustic"][:n].unsqueeze(0)
    text_emb = data["text_emb"][:n].unsqueeze(0)
    wav_emb = data["wav_emb"][:n].unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)
    mask[0, :n] = False

    with torch.amp.autocast(device_type=DEVICE, enabled=False):
        preds = model(acoustic, text_emb, wav_emb, mask)

    if preds.dim() == 1 and preds.shape[0] > 1:
        # Multi-task: [vol_1d, vol_3d, vol_7d]
        targets = CFG["label"].get("multi_task_targets", ["abnormal_vol_1d", "abnormal_vol_3d", "abnormal_vol_7d"])
        results = {}
        for i, t in enumerate(targets):
            label = t.replace("abnormal_vol_", "").replace("d", "-day")
            results[label] = float(preds[i])
        return results, None
    else:
        val = float(preds.squeeze())
        return {"3-day": val}, None


def get_actual_labels(call_id):
    """Get actual labels from labels.csv for comparison."""
    row = LABELS[LABELS["call_id"] == call_id]
    if row.empty:
        return {}
    row = row.iloc[0]
    result = {}
    for window in ["1", "3", "7"]:
        col = f"abnormal_vol_{window}d"
        if col in row and pd.notna(row[col]):
            result[f"{window}-day"] = float(row[col])
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    calls = []
    for _, row in MANIFEST.iterrows():
        calls.append({
            "call_id": row["call_id"],
            "ticker": row["ticker"],
            "date": row["call_date"],
            "n_utts": int(row["n_utterances"]),
        })
    return render_template("index.html",
                           calls=calls,
                           model_types=list(MODELS.keys()))


@app.route("/predict", methods=["POST"])
def predict():
    call_id = request.form.get("call_id", "")
    model_type = request.form.get("model_type", "early_fusion")

    pt_path = os.path.join(PROCESSED_DIR, f"{call_id}.pt")
    if not os.path.exists(pt_path):
        return jsonify({"error": f"File not found: {call_id}.pt"}), 404

    # Get call metadata
    call_row = MANIFEST[MANIFEST["call_id"] == call_id]
    meta = {}
    if not call_row.empty:
        r = call_row.iloc[0]
        meta = {"ticker": r["ticker"], "date": r["call_date"],
                "n_utterances": int(r["n_utterances"])}

    # Run all models for comparison
    all_predictions = {}
    for mtype in MODELS:
        preds, err = run_inference(pt_path, mtype)
        if preds:
            all_predictions[mtype] = preds

    # Get actual labels
    actuals = get_actual_labels(call_id)

    return jsonify({
        "call_id": call_id,
        "meta": meta,
        "selected_model": model_type,
        "predictions": all_predictions,
        "actuals": actuals,
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pt"):
        return jsonify({"error": "Only .pt files are supported"}), 400

    tmp_path = os.path.join("/tmp", file.filename)
    file.save(tmp_path)

    model_type = request.form.get("model_type", "early_fusion")
    all_predictions = {}
    for mtype in MODELS:
        preds, err = run_inference(tmp_path, mtype)
        if preds:
            all_predictions[mtype] = preds

    os.remove(tmp_path)

    return jsonify({
        "call_id": file.filename.replace(".pt", ""),
        "meta": {"source": "uploaded"},
        "selected_model": model_type,
        "predictions": all_predictions,
        "actuals": {},
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
