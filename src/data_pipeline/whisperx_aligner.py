import whisperx
import torch


def forced_align_maec_text_to_audio(
    audio_path: str,
    maec_utterances: list[dict],
    device: str = "cuda"
) -> list[dict]:
    audio = whisperx.load_audio(audio_path)

    align_model, metadata = whisperx.load_align_model(
        language_code="en", device=device
    )

    segments_for_alignment = [
        {"text": utt["text"]} for utt in maec_utterances
    ]

    aligned = whisperx.align(
        segments_for_alignment,
        align_model,
        metadata,
        audio,
        device=device,
        return_char_alignments=False
    )

    enriched = []
    for i, (wx_seg, maec_utt) in enumerate(
        zip(aligned.get("segments", []), maec_utterances)
    ):
        enriched.append({
            "utterance_index": maec_utt["utterance_index"],
            "text":            maec_utt["text"],
            "speaker_role":    maec_utt["speaker_role"],
            "speaker_raw":     maec_utt["speaker_raw"],
            "high_confidence": maec_utt.get("high_confidence", False),
            "start":           wx_seg.get("start", 0.0),
            "end":             wx_seg.get("end", 0.0),
            "duration_s":      wx_seg.get("end", 0.0) - wx_seg.get("start", 0.0),
            "words":           wx_seg.get("words", []),
            "alignment_source": "forced",
        })

    return enriched
