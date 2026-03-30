import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path

from .maec_text_cleaner import segment_maec_call
from .audio_preprocessor import preprocess_audio
from .diarization import build_speaker_segments_from_diarization, forced_align_within_speaker_segments
from .feature_extractor import extract_all_utterances_with_tracking
from .text_encoder import encode_batch
from .linguistic_features import extract_lm_features


MAEC_SPLITS = {
    "train": ("2007-01-01", "2015-12-31"),
    "val":   ("2016-01-01", "2017-12-31"),
    "test":  ("2018-01-01", "2020-06-30"),
}

OUR_SPLITS = {
    "train": ("2007-01-01", "2016-12-31"),
    "val":   ("2017-01-01", "2018-12-31"),
    "test":  ("2019-01-01", "2020-06-30"),
}


def build_call_record(
    call_meta:        dict,
    audio_path:       str,
    aligned_records:  list,
    audio_features:   np.ndarray,
    text_embeddings:  np.ndarray,
    ling_features:    np.ndarray,
    labels:           dict,
    drop_stats:       dict,
) -> dict:
    audio, sr = sf.read(audio_path)
    call_duration = len(audio) / sr

    maec_utts_with_sections = segment_maec_call(call_meta["utterances"])
    section_map = {u["utterance_index"]: u.get("section", "unknown")
                   for u in maec_utts_with_sections}

    utterances = []
    for i, rec in enumerate(aligned_records):
        if i >= len(audio_features):
            break

        utt = {
            "call_id":          call_meta["call_id"],
            "utterance_id":     f"{call_meta['call_id']}_{i:04d}",
            "utterance_index":  rec.get("utterance_index", i),
            "text":             rec["text"],
            "speaker_role":     rec["speaker_role"],
            "speaker_raw":      rec["speaker_raw"],
            "high_confidence_speaker": rec.get("high_confidence", False),
            "section":          section_map.get(rec.get("utterance_index", i), "unknown"),
            "audio_start":      rec["start"],
            "audio_end":        rec["end"],
            "duration_s":       rec["duration_s"],
            "audio_speaker_id": rec.get("audio_speaker", "UNKNOWN"),
            "text_embedding":   text_embeddings[i],
            "audio_features":   audio_features[i],
            "ling_features":    ling_features[i],
            "timeline_fraction": rec["start"] / call_duration if call_duration > 0 else 0,
            "word_timestamps":  rec.get("words", []),
        }
        utterances.append(utt)

    return {
        "call_id":            call_meta["call_id"],
        "ticker":             call_meta["ticker"],
        "call_date":          call_meta["call_date"],
        "call_duration_s":    call_duration,
        "utterances":         utterances,
        "n_utterances":       len(utterances),
        "maec_lld_summary":   call_meta.get("maec_lld_summary", {}),
        "structured":         labels.get("structured", {}),
        "labels":             labels,
        "drop_stats":         drop_stats,
    }


def assemble_full_dataset(calls_with_audio: list[dict],
                           labels_df: pd.DataFrame,
                           hf_token: str = None,
                           audio_norm_dir: str = "data/processed/audio_normalized",
                           device: str = "cuda") -> list[dict]:
    records = []
    attrition = {
        "started":       len(calls_with_audio),
        "no_audio":      0,
        "low_snr":       0,
        "align_failed":  0,
        "no_ceo":        0,
        "no_label":      0,
        "final":         0,
    }

    for call in calls_with_audio:
        if not call.get("audio_path"):
            attrition["no_audio"] += 1
            continue

        norm_path = str(Path(audio_norm_dir) / f"{call['call_id']}.wav")
        audio_meta = preprocess_audio(call["audio_path"], norm_path)
        if audio_meta["quality_flag"] == "low":
            attrition["low_snr"] += 1
            continue

        diar_segs = build_speaker_segments_from_diarization(
            audio_meta["output_path"], hf_token=hf_token, device=device
        )
        aligned = forced_align_within_speaker_segments(
            audio_meta["output_path"], diar_segs, call["utterances"], device=device
        )

        if not aligned:
            attrition["align_failed"] += 1
            continue

        call["utterances"] = segment_maec_call(call["utterances"])

        audio, sr = sf.read(audio_meta["output_path"])
        feats, drop_stats = extract_all_utterances_with_tracking(aligned, audio, sr)

        if not any(r["speaker_role"] == "CEO" and r.get("high_confidence")
                   for r in feats):
            attrition["no_ceo"] += 1
            continue

        label_row = labels_df[labels_df["call_id"] == call["call_id"]]
        if label_row.empty or pd.isna(label_row["abnormal_vol_3d"].values[0]):
            attrition["no_label"] += 1
            continue

        audio_feats = np.stack([r["audio_features"] for r in feats])
        text_embs   = encode_batch([r["text"] for r in feats], device=device)
        ling_feats  = np.stack([np.array(list(extract_lm_features(r["text"]).values()))
                                 for r in feats])

        record = build_call_record(
            call_meta=call,
            audio_path=audio_meta["output_path"],
            aligned_records=feats,
            audio_features=audio_feats,
            text_embeddings=text_embs,
            ling_features=ling_feats,
            labels=label_row.iloc[0].to_dict(),
            drop_stats=drop_stats,
        )
        records.append(record)

    attrition["final"] = len(records)
    print("Attrition summary:", attrition)
    return records


def create_temporal_splits(labels_df: pd.DataFrame,
                            split_config: dict = None,
                            output_dir: str = "data/splits") -> dict:
    if split_config is None:
        split_config = MAEC_SPLITS

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    splits = {}

    for split_name, (start_date, end_date) in split_config.items():
        mask = (labels_df["call_date"] >= start_date) & (labels_df["call_date"] <= end_date)
        split_df = labels_df[mask]
        split_df.to_csv(f"{output_dir}/{split_name}.csv", index=False)
        splits[split_name] = split_df
        print(f"{split_name}: {len(split_df)} calls ({start_date} to {end_date})")

    return splits
