"""
ACL19 dataset parser (GeminiLn/EarningsCall_Dataset, Qin & Yang ACL 2019).

Actual on-disk structure (verified from unzip output):
  {acl19_root}/
    {CompanyName}_{YYYYMMDD}/
      CEO/
        {SpeakerName}_{para}_{sent}.mp3   # sentence-level audio
      TextSequence.txt                     # one CEO sentence per line

Line N in TextSequence.txt corresponds exactly to the N-th MP3 when
sorted by (paragraph, sentence) index. Speaker is always the most-spoken
executive — CEO-only by construction, no diarization needed.
"""

import re
from pathlib import Path


def _parse_folder_name(folder_name: str) -> tuple[str, str]:
    """Return (company_name, call_date) from e.g. 'Amazon.com Inc._20170202'."""
    match = re.search(r'_(\d{8})$', folder_name)
    if not match:
        return folder_name, ""
    date_str = match.group(1)
    company = folder_name[:match.start()].strip()
    call_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return company, call_date


def _sort_key(mp3_path: Path) -> tuple[int, int]:
    """Sort MP3s by (paragraph, sentence) from filename SpeakerName_P_S.mp3."""
    parts = mp3_path.stem.rsplit("_", 2)
    try:
        return int(parts[-2]), int(parts[-1])
    except (ValueError, IndexError):
        return 0, 0


def _ticker_from_company(company: str, ticker_map: dict) -> str:
    return ticker_map.get(company, re.sub(r'[^A-Z0-9]', '', company.upper())[:6])


def parse_acl19_call(call_dir: Path, ticker_map: dict = None) -> dict | None:
    """
    Parse one ACL19 call folder into a structured call record.
    Returns None if TextSequence.txt or CEO/ audio folder is missing/empty.
    """
    ticker_map = ticker_map or {}
    company, call_date = _parse_folder_name(call_dir.name)
    if not call_date:
        return None

    txt_path  = call_dir / "TextSequence.txt"
    audio_dir = call_dir / "CEO"

    if not txt_path.exists() or not audio_dir.exists():
        return None

    with open(txt_path, encoding="utf-8", errors="replace") as f:
        sentences = [l.strip() for l in f if l.strip()]

    mp3_files = sorted(audio_dir.glob("*.mp3"), key=_sort_key)

    if not sentences or not mp3_files:
        return None

    n = min(len(sentences), len(mp3_files))
    if len(sentences) != len(mp3_files):
        print(f"  [warn] {call_dir.name}: {len(sentences)} sentences vs "
              f"{len(mp3_files)} mp3s — using first {n}")

    speaker_raw = mp3_files[0].stem.rsplit("_", 2)[0] if mp3_files else "UNKNOWN"
    ticker  = _ticker_from_company(company, ticker_map)
    call_id = f"{ticker}_{call_date.replace('-', '')}"

    utterances = []
    for i in range(n):
        text = sentences[i]
        utterances.append({
            "utterance_index": i,
            "text":            text,
            "speaker_raw":     speaker_raw,
            "speaker_role":    "CEO",
            "high_confidence": True,
            "audio_path":      str(mp3_files[i]),
            "word_count":      len(text.split()),
            "char_count":      len(text),
        })

    return {
        "call_id":      call_id,
        "ticker":       ticker,
        "company":      company,
        "call_date":    call_date,
        "acl19_dir":    str(call_dir),
        "speaker":      speaker_raw,
        "utterances":   utterances,
        "n_utterances": n,
    }


def ingest_acl19_dataset(acl19_root: str, ticker_map: dict = None) -> list[dict]:
    """
    Walk the ACL19 root and parse every call folder.

    ticker_map: optional {company_name: ticker} dict.
    Without it, tickers are derived from company names (imperfect for
    multi-word names like 'Amazon.com Inc.' -> 'AMAZONCOMINC').
    A curated map for S&P 500 2017 companies is strongly recommended.
    """
    root = Path(acl19_root)
    if not root.exists():
        raise FileNotFoundError(
            f"ACL19 root not found: {acl19_root}\n"
            "Download from: https://drive.google.com/drive/folders/"
            "1BKCANORbcmUJKkOkBOghw6uNHPqS_az1\n"
            "Then: zip -s0 ACL19_Release.zip --out ACL19_Release_All.zip "
            "&& unzip -q ACL19_Release_All.zip"
        )

    folders = sorted(d for d in root.iterdir() if d.is_dir())
    print(f"[acl19] Found {len(folders)} folders in {acl19_root}")

    calls = []
    for i, call_dir in enumerate(folders, 1):
        record = parse_acl19_call(call_dir, ticker_map)
        if record is None:
            print(f"  [{i}/{len(folders)}] SKIP {call_dir.name}")
            continue
        calls.append(record)
        print(f"  [{i}/{len(folders)}] OK   {call_dir.name} "
              f"-> {record['ticker']} {record['call_date']} "
              f"({record['n_utterances']} utterances)")

    print(f"\n[acl19] Ingested {len(calls)}/{len(folders)} calls")
    return calls


def validate_acl19_parse(calls: list[dict]) -> dict:
    import numpy as np
    n_utts = [c["n_utterances"] for c in calls]
    total_words = sum(u["word_count"] for c in calls for u in c["utterances"])
    return {
        "total_calls":        len(calls),
        "total_utterances":   sum(n_utts),
        "avg_utterances":     float(np.mean(n_utts)) if n_utts else 0,
        "min_utterances":     int(np.min(n_utts)) if n_utts else 0,
        "max_utterances":     int(np.max(n_utts)) if n_utts else 0,
        "avg_words_per_call": total_words / max(len(calls), 1),
        "date_range":         (min(c["call_date"] for c in calls),
                               max(c["call_date"] for c in calls)) if calls else ("", ""),
    }
