from pyannote.audio import Pipeline
import torch
import whisperx
import numpy as np
import librosa
from collections import Counter


def build_speaker_segments_from_diarization(
    audio_path: str,
    hf_token: str,
    device: str = "cuda"
) -> list[dict]:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    pipeline.to(torch.device(device))

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker":  speaker,
            "start":    turn.start,
            "end":      turn.end,
            "duration": turn.end - turn.start,
        })

    return segments


def forced_align_within_speaker_segments(
    audio_path: str,
    diar_segments: list[dict],
    maec_utterances: list[dict],
    device: str = "cuda"
) -> list[dict]:
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    align_model, metadata = whisperx.load_align_model(
        language_code="en", device=device
    )

    aligned_records = []

    for seg in diar_segments:
        seg_start  = seg["start"]
        seg_end    = seg["end"]
        seg_audio  = audio[int(seg_start * sr): int(seg_end * sr)]

        overlapping_maec = [
            u for u in maec_utterances
            if _utterance_likely_in_segment(u, seg_start, seg_end, len(maec_utterances))
        ]

        if not overlapping_maec:
            continue

        maec_text_segments = [{"text": u["text"]} for u in overlapping_maec]

        try:
            aligned = whisperx.align(
                maec_text_segments,
                align_model,
                metadata,
                seg_audio,
                device=device,
                return_char_alignments=False
            )
        except Exception as e:
            print(f"Alignment failed for segment {seg_start:.1f}-{seg_end:.1f}: {e}")
            continue

        for wx_seg, maec_utt in zip(aligned.get("segments", []), overlapping_maec):
            abs_start = seg_start + wx_seg.get("start", 0)
            abs_end   = seg_start + wx_seg.get("end", seg_end - seg_start)

            aligned_records.append({
                "text":         maec_utt["text"],
                "speaker_role": maec_utt["speaker_role"],
                "speaker_raw":  maec_utt["speaker_raw"],
                "high_confidence": maec_utt.get("high_confidence", False),
                "audio_speaker": seg["speaker"],
                "start":        abs_start,
                "end":          abs_end,
                "duration_s":   abs_end - abs_start,
                "words":        wx_seg.get("words", []),
                "utterance_index": maec_utt.get("utterance_index", 0),
            })

    return aligned_records


def _utterance_likely_in_segment(utt: dict, seg_start: float, seg_end: float,
                                  total_utts: int) -> bool:
    return True


def map_audio_speakers_to_maec_roles(
    aligned_records: list[dict]
) -> dict:
    audio_to_role = {}
    role_votes = {}

    for rec in aligned_records:
        spk  = rec["audio_speaker"]
        role = rec["speaker_role"]
        if spk not in role_votes:
            role_votes[spk] = Counter()
        role_votes[spk][role] += 1

    for spk, votes in role_votes.items():
        audio_to_role[spk] = votes.most_common(1)[0][0]

    return audio_to_role
