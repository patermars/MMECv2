import subprocess
import sys
import json
from pathlib import Path

_WORKER_TIMEOUT = 30  # seconds per URL attempt


def build_ir_url_candidates(ticker: str, call_date: str) -> list[str]:
    t = ticker.lower()
    return [
        f"https://investor.{t}.com/events",
        f"https://ir.{t}.com/earnings",
        f"https://investors.{t}.com/results",
    ]


def _download_worker(url: str, outtmpl: str) -> bool:
    """Run in a subprocess: returns True if file was created."""
    import yt_dlp
    from pathlib import Path

    output_path = Path(outtmpl + ".wav")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "nocheckcertificate": True,
        "socket_timeout": 15,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception:
        pass
    print(json.dumps({"exists": output_path.exists()}))


def scrape_call_audio(ticker: str, call_date: str, output_dir: str) -> str | None:
    output_path = Path(output_dir) / f"{ticker}_{call_date.replace('-', '')}.wav"
    if output_path.exists():
        return str(output_path)

    outtmpl = str(output_path.with_suffix(""))

    for url in build_ir_url_candidates(ticker, call_date):
        code = (
            f"import sys; sys.path.insert(0, {repr(str(Path(__file__).parent.parent.parent))})\n"
            f"from src.data_pipeline.maec_audio_scraper import _download_worker\n"
            f"_download_worker({repr(url)}, {repr(outtmpl)})\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                timeout=_WORKER_TIMEOUT,
                capture_output=True,
                text=True,
            )
            last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            if last_line:
                result = json.loads(last_line)
                if result.get("exists"):
                    return str(output_path)
        except (subprocess.TimeoutExpired, Exception):
            continue

    return None


def batch_download_maec_audio(calls: list[dict], output_dir: str) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {"success": [], "failed": []}

    for call in calls:
        path = scrape_call_audio(call["ticker"], call["call_date"], output_dir)
        if path:
            call["audio_path"] = path
            results["success"].append(call["call_id"])
        else:
            call["audio_path"] = None
            results["failed"].append(call["call_id"])

    print(f"Audio download: {len(results['success'])} success, "
          f"{len(results['failed'])} failed")
    return results
