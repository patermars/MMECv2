import requests
import yt_dlp
from pathlib import Path


def build_ir_url_candidates(ticker: str, call_date: str) -> list[str]:
    year  = call_date[:4]
    month = call_date[5:7]

    candidates = [
        f"https://investor.{ticker.lower()}.com/events",
        f"https://ir.{ticker.lower()}.com/earnings",
        f"https://investors.{ticker.lower()}.com/results",
    ]
    return candidates


def scrape_call_audio(ticker: str, call_date: str, output_dir: str) -> str | None:
    output_path = Path(output_dir) / f"{ticker}_{call_date.replace('-', '')}.wav"

    if output_path.exists():
        return str(output_path)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path.with_suffix("")),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    for url in build_ir_url_candidates(ticker, call_date):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if output_path.exists():
                return str(output_path)
        except Exception:
            continue

    return None


def batch_download_maec_audio(calls: list[dict], output_dir: str) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {"success": [], "failed": []}

    for call in calls:
        try:
            path = scrape_call_audio(call["ticker"], call["call_date"], output_dir)
        except Exception:
            path = None
        if path:
            call["audio_path"] = path
            results["success"].append(call["call_id"])
        else:
            call["audio_path"] = None
            results["failed"].append(call["call_id"])

    print(f"Audio download: {len(results['success'])} success, "
          f"{len(results['failed'])} failed")
    return results
