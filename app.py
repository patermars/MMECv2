"""
Flask web app for MMEC Earnings Call Volatility Prediction.
Run: python app.py
"""
import os, uuid, yaml, torch, numpy as np, pandas as pd, librosa, tempfile
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify
from src.model import build_model
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model

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

# Load inference models once
print("Loading models...")
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

# Load text and audio encoders once
print("Loading encoders...")
TEXT_TOKENIZER = AutoTokenizer.from_pretrained(CFG["text"]["model_name"])
TEXT_MODEL = AutoModel.from_pretrained(CFG["text"]["model_name"]).to(DEVICE).eval()
WAV_PROCESSOR = Wav2Vec2Processor.from_pretrained(CFG["wav2vec"]["model_name"])
WAV_MODEL = Wav2Vec2Model.from_pretrained(CFG["wav2vec"]["model_name"]).to(DEVICE).eval()
print("  ✓ Encoders loaded")

# Max segments to process for uploaded audio (30s each → ~15 min of audio)
MAX_UPLOAD_SEGS = 30

# Async job store: job_id -> {"status": "pending"|"done"|"error", "result": ...}
JOBS: dict = {}
_executor = ThreadPoolExecutor(max_workers=2)

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

    preds = preds.squeeze(0) if preds.dim() == 2 else preds
    
    if preds.dim() == 1 and preds.shape[0] > 1:
        # Multi-task: [vol_1d, vol_3d, vol_7d]
        targets = CFG["label"].get("multi_task_targets", ["abnormal_vol_1d", "abnormal_vol_3d", "abnormal_vol_7d"])
        results = {}
        for i, t in enumerate(targets):
            label = t.replace("abnormal_vol_", "").replace("d", "-day")
            results[label] = float(preds[i])
        return results, None
    else:
        return {"3-day": float(preds.item())}, None


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
    return render_template("index.html",
                           model_types=list(MODELS.keys()))



def extract_acoustic_features(audio, sr=16000):
    """Extract basic acoustic features from audio."""
    features = []
    pitch = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
    pitch_mean = np.nanmean(pitch) if not np.all(np.isnan(pitch)) else 0
    pitch_std = np.nanstd(pitch) if not np.all(np.isnan(pitch)) else 0
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    
    feat_vec = [
        pitch_mean, pitch_std,
        *mfccs.mean(axis=1), *mfccs.std(axis=1),
        spectral_centroid.mean(), spectral_rolloff.mean(), zcr.mean(), rms.mean()
    ]
    return np.array(feat_vec[:29], dtype=np.float32)


@torch.no_grad()
def process_audio_file(audio_path, company_name, transcript_text=None):
    """Process raw audio file and extract features (fast version)."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Split audio into segments (30s each)
    segment_len = 30 * sr
    segments = [audio[i:i+segment_len] for i in range(0, len(audio), segment_len) if len(audio[i:i+segment_len]) > sr]
    
    if len(segments) == 0:
        return None, "Audio too short"
    
    # Sample uniformly up to MAX_UPLOAD_SEGS BEFORE any encoding (key speedup)
    if len(segments) > MAX_UPLOAD_SEGS:
        indices = np.linspace(0, len(segments)-1, MAX_UPLOAD_SEGS, dtype=int)
        segments = [segments[i] for i in indices]
    else:
        segments = segments[:MAX_UPLOAD_SEGS]
    
    # Extract acoustic features (fast)
    acoustic_feats = []
    for seg in segments:
        feat = extract_acoustic_features(seg, sr)
        acoustic_feats.append(feat)
    acoustic_t = torch.tensor(np.array(acoustic_feats), dtype=torch.float32)
    
    # Normalize
    mean = acoustic_t.mean(dim=0, keepdim=True)
    std = acoustic_t.std(dim=0, keepdim=True).clamp(min=1e-6)
    acoustic_t = (acoustic_t - mean) / std
    
    # Process text
    if transcript_text:
        # Split transcript into chunks matching segment count
        words = transcript_text.split()
        chunk_size = max(len(words) // len(segments), 10)
        texts = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        texts = texts[:len(segments)]  # Match segment count
        # Pad if needed
        while len(texts) < len(segments):
            texts.append(texts[-1] if texts else "earnings call discussion")
    else:
        # Use generic placeholder (fast)
        placeholder_text = "Earnings call discussion about quarterly financial results and business performance."
        texts = [placeholder_text] * len(segments)
    
    # Encode text (batch processing)
    tokens = TEXT_TOKENIZER(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(DEVICE)
    text_embs = TEXT_MODEL(**tokens).last_hidden_state[:, 0, :].cpu()
    
    # Encode audio with wav2vec2 (batch processing for speed)
    wav_embs = []
    batch_size = 4
    for i in range(0, len(segments), batch_size):
        batch_segs = segments[i:i+batch_size]
        # Process batch
        batch_inputs = []
        for seg in batch_segs:
            inputs = WAV_PROCESSOR(seg, sampling_rate=sr, return_tensors="pt", padding=True)
            batch_inputs.append(inputs.input_values.squeeze(0))
        
        # Pad to same length
        max_len = max(inp.shape[0] for inp in batch_inputs)
        padded = torch.zeros(len(batch_inputs), max_len)
        for j, inp in enumerate(batch_inputs):
            padded[j, :inp.shape[0]] = inp
        
        padded = padded.to(DEVICE)
        batch_out = WAV_MODEL(padded).last_hidden_state.mean(dim=1).cpu()
        wav_embs.append(batch_out)
    
    wav_embs = torch.cat(wav_embs, dim=0)
    
    return {
        "acoustic": acoustic_t,
        "text_emb": text_embs,
        "wav_emb": wav_embs,
        "n_utterances": len(segments),
    }, None


def _process_upload_job(job_id, tmp_path, company_name, transcript_text, model_type):
    """Background worker: process audio and run inference, store result in JOBS."""
    try:
        data, err = process_audio_file(tmp_path, company_name, transcript_text)
        if err:
            JOBS[job_id] = {"status": "error", "error": err}
            return

        all_predictions = {}
        n = data["n_utterances"]
        acoustic = data["acoustic"].unsqueeze(0)
        text_emb = data["text_emb"].unsqueeze(0)
        wav_emb  = data["wav_emb"].unsqueeze(0)
        mask = torch.zeros(1, n, dtype=torch.bool)

        for mtype, model in MODELS.items():
            with torch.no_grad(), torch.amp.autocast(device_type=DEVICE, enabled=False):
                preds = model(acoustic, text_emb, wav_emb, mask)
            preds = preds.squeeze(0) if preds.dim() == 2 else preds
            if preds.dim() == 1 and preds.shape[0] > 1:
                targets = CFG["label"].get("multi_task_targets", ["abnormal_vol_1d", "abnormal_vol_3d", "abnormal_vol_7d"])
                all_predictions[mtype] = {
                    t.replace("abnormal_vol_", "").replace("d", "-day"): float(preds[i])
                    for i, t in enumerate(targets)
                }
            else:
                all_predictions[mtype] = {"3-day": float(preds.item())}

        JOBS[job_id] = {
            "status": "done",
            "result": {
                "call_id": f"{company_name}_upload",
                "meta": {"source": "uploaded", "company": company_name,
                         "segments": n, "has_transcript": transcript_text is not None},
                "selected_model": model_type,
                "predictions": all_predictions,
                "actuals": {},
            }
        }
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    company_name = request.form.get("company_name", "UNKNOWN")
    model_type = request.form.get("model_type", "early_fusion")
    transcript_file = request.files.get("transcript")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    transcript_text = transcript_file.read().decode("utf-8") if transcript_file else None

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending"}
    _executor.submit(_process_upload_job, job_id, tmp_path, company_name, transcript_text, model_type)

    return jsonify({"job_id": job_id, "status": "pending"}), 202


@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    if job["status"] == "pending":
        return jsonify({"status": "pending"}), 202
    if job["status"] == "error":
        return jsonify({"status": "error", "error": job["error"]}), 500
    # done — return result and clean up
    result = job.pop("result")
    JOBS.pop(job_id, None)
    return jsonify({"status": "done", **result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
