import re


def clean_maec_text(text: str) -> str:
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Q\d \d{4} Earnings Call Transcript', '', text)
    text = re.sub(r'-\n([a-z])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def segment_maec_call(utterances: list[dict]) -> list[dict]:
    n = len(utterances)
    qa_started = False

    for i, utt in enumerate(utterances):
        text_lower = utt["text"].lower()
        role       = utt["speaker_role"]
        frac       = i / max(n - 1, 1)

        if role == "OPERATOR" and any(kw in text_lower for kw in [
            "question", "open", "first caller", "first question"
        ]):
            qa_started = True
            utt["section"] = "operator"
            continue

        if not qa_started and any(kw in text_lower for kw in [
            "open the floor", "first question", "question-and-answer",
            "question and answer", "q&a", "operator instructions"
        ]):
            qa_started = True

        if role == "OPERATOR":
            utt["section"] = "operator"
        elif role == "ANALYST" or (qa_started and role not in ["CEO", "CFO", "MANAGEMENT"]):
            utt["section"] = "qa_analyst"
            qa_started = True
        elif qa_started and role in ["CEO", "CFO", "MANAGEMENT"]:
            utt["section"] = "qa_management"
        elif any(kw in text_lower for kw in [
            "guidance", "outlook", "expect", "anticipate", "project",
            "looking ahead", "next quarter", "full year"
        ]):
            utt["section"] = "guidance"
        elif frac < 0.15 and role in ["CEO", "CFO", "MANAGEMENT"]:
            utt["section"] = "opening_remarks"
        else:
            utt["section"] = "financial_review"

    return utterances
