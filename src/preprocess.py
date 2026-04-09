import os, csv, re, torch, numpy as np, librosa, soundfile as sf
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import argparse, yaml


def load_features_csv(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            vals = []
            for v in row:
                v = v.strip()
                if v == "--undefined--" or v == "":
                    vals.append(0.0)
                else:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(0.0)
            if len(vals) == len(header):
                rows.append(vals)
    if not rows:
        return None
    return np.array(rows, dtype=np.float32)


def load_text(txt_path):
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def get_mp3_files(call_dir):
    mp3s = sorted([f for f in os.listdir(call_dir) if f.endswith(".mp3")])
    return [os.path.join(call_dir, f) for f in mp3s]


@torch.no_grad()
def encode_text_batch(texts, tokenizer, model, device, max_len=512):
    embeddings = []
    bs = 32
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        tokens = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=max_len, padding=True).to(device)
        out = model(**tokens)
        cls = out.last_hidden_state[:, 0, :]
        embeddings.append(cls.cpu())
    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def encode_audio_batch(mp3_paths, processor, model, device, sr=16000, max_sec=30):
    embeddings = []
    for p in mp3_paths:
        try:
            audio, orig_sr = librosa.load(p, sr=sr, mono=True, duration=max_sec)
        except Exception:
            embeddings.append(torch.zeros(768))
            continue
        if len(audio) < 400:
            embeddings.append(torch.zeros(768))
            continue
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        embeddings.append(emb)
    return torch.stack(embeddings)


def parse_call_id(dirname):
    parts = dirname.split("_", 1)
    if len(parts) != 2:
        return None, None, None
    date_str, ticker = parts
    try:
        call_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except (IndexError, ValueError):
        return None, None, None
    return f"{ticker}_{date_str}", ticker, call_date


def process_single_call(call_dir, text_tok, text_model, wav_proc, wav_model,
                        device, max_utts=200):
    dirname = os.path.basename(call_dir)
    call_id, ticker, call_date = parse_call_id(dirname)
    if call_id is None:
        return None

    feat_path = os.path.join(call_dir, "features.csv")
    txt_path = os.path.join(call_dir, "text.txt")
    if not os.path.exists(feat_path) or not os.path.exists(txt_path):
        return None

    acoustic = load_features_csv(feat_path)
    if acoustic is None:
        return None
    texts = load_text(txt_path)
    mp3s = get_mp3_files(call_dir)

    n = min(len(acoustic), len(texts), max_utts)
    if n < 3:
        return None

    acoustic = acoustic[:n]
    texts = texts[:n]
    mp3s = mp3s[:min(len(mp3s), n)]

    acoustic_t = torch.tensor(acoustic, dtype=torch.float32)

    mean = acoustic_t.mean(dim=0, keepdim=True)
    std = acoustic_t.std(dim=0, keepdim=True).clamp(min=1e-6)
    acoustic_t = (acoustic_t - mean) / std

    text_embs = encode_text_batch(texts, text_tok, text_model, device)

    if len(mp3s) >= n:
        wav_embs = encode_audio_batch(mp3s[:n], wav_proc, wav_model, device)
    else:
        wav_embs = torch.zeros(n, 768)

    return {
        "call_id": call_id,
        "ticker": ticker,
        "call_date": call_date,
        "acoustic": acoustic_t,
        "text_emb": text_embs[:n],
        "wav_emb": wav_embs[:n],
        "n_utterances": n,
        "texts": texts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_calls", type=int, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing .pt files")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or cfg["data"]["dataset_dir"]
    output_dir = args.output_dir or cfg["data"]["processed_dir"]
    os.makedirs(output_dir, exist_ok=True)
    max_utts = cfg["data"]["max_utterances"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading FinBERT...")
    text_tok = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
    text_model = AutoModel.from_pretrained(cfg["text"]["model_name"]).to(device).eval()

    print("Loading wav2vec2...")
    wav_proc = Wav2Vec2Processor.from_pretrained(cfg["wav2vec"]["model_name"])
    wav_model = Wav2Vec2Model.from_pretrained(cfg["wav2vec"]["model_name"]).to(device).eval()

    call_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ])
    if args.max_calls:
        call_dirs = call_dirs[:args.max_calls]

    print(f"Processing {len(call_dirs)} calls...")
    
    processed_set = set()
    manifest = []
    
    manifest_path = os.path.join(output_dir, "manifest.csv")
    if args.resume:
        existing_pts = {f[:-3] for f in os.listdir(output_dir) if f.endswith(".pt")}
        for call_dir in call_dirs:
            dirname = os.path.basename(call_dir)
            call_id, _, _ = parse_call_id(dirname)
            if call_id and call_id in existing_pts:
                processed_set.add(dirname)
        print(f"Resuming: {len(processed_set)} calls already processed")
        
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                reader = csv.DictReader(f)
                manifest = list(reader)
    
    for call_dir in tqdm(call_dirs, desc="Preprocessing"):
        dirname = os.path.basename(call_dir)
        if dirname in processed_set:
            continue
        
        result = process_single_call(call_dir, text_tok, text_model,
                                     wav_proc, wav_model, device, max_utts)
        if result is None:
            continue

        out_path = os.path.join(output_dir, f"{result['call_id']}.pt")
        torch.save({
            "acoustic": result["acoustic"],
            "text_emb": result["text_emb"],
            "wav_emb": result["wav_emb"],
            "n_utterances": result["n_utterances"],
        }, out_path)

        manifest.append({
            "call_id": result["call_id"],
            "ticker": result["ticker"],
            "call_date": result["call_date"],
            "n_utterances": result["n_utterances"],
            "path": out_path,
        })

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["call_id", "ticker", "call_date", "n_utterances", "path"])
        w.writeheader()
        w.writerows(manifest)

    print(f"Done. {len(manifest)} calls processed → {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
