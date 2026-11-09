"""Download and convert the public Apache SpamAssassin mail corpus to CSV.

Run from the repository root:
    python dataset/download_spamassassin.py

The generated CSV is stored locally at dataset/spamassassin_emails.csv.
It is intentionally excluded from Git by .gitignore because source message
copyright remains with the original senders; review corpus terms before
redistributing the derived text publicly.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import tarfile
import urllib.request
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = DATASET_DIR / "spamassassin_emails.csv"

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = {
    "20030228_easy_ham.tar.bz2": "ham",
    "20030228_hard_ham.tar.bz2": "ham",
    "20030228_easy_ham_2.tar.bz2": "ham",
    "20030228_spam.tar.bz2": "spam",
    "20050311_spam_2.tar.bz2": "spam",
}


class HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML email bodies."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def get_message_text(raw_message: bytes) -> str:
    message = BytesParser(policy=policy.default).parsebytes(raw_message)
    subject = str(message.get("Subject", "") or "").strip()

    plain_parts: list[str] = []
    html_parts: list[str] = []

    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.is_multipart():
            continue
        content_type = part.get_content_type()
        if content_type not in {"text/plain", "text/html"}:
            continue
        try:
            content = part.get_content()
        except (LookupError, UnicodeError, AttributeError):
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            content = payload.decode(charset, errors="replace")

        if not isinstance(content, str):
            continue
        if content_type == "text/plain":
            plain_parts.append(content)
        else:
            html_parts.append(content)

    if plain_parts:
        body = " ".join(plain_parts)
    elif html_parts:
        extractor = HTMLTextExtractor()
        for html in html_parts:
            extractor.feed(html)
        body = " ".join(extractor.parts)
    else:
        body = ""

    combined = clean_text(f"{subject}. {body}" if subject else body)
    return combined


def download_archive(filename: str) -> bytes:
    url = BASE_URL + filename
    print(f"Downloading {filename} ...")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AI-Spam-Email-Detector/1.0 (educational project)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    for archive_name, label in ARCHIVES.items():
        try:
            archive_bytes = download_archive(archive_name)
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:bz2") as archive:
                for member in archive.getmembers():
                    if not member.isfile() or member.size <= 0 or member.size > 2_000_000:
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        continue
                    try:
                        text = get_message_text(stream.read())
                    except Exception as exc:  # keep processing the remaining corpus
                        print(f"Skipping unreadable message {member.name}: {exc}")
                        continue

                    if len(text) < 20:
                        continue
                    dedupe_key = re.sub(r"\s+", " ", text).casefold()
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    rows.append((label, text))
        except Exception as exc:
            print(f"Could not process {archive_name}: {exc}", file=sys.stderr)
            return 1

    if not rows:
        print("No messages were extracted. Check your internet connection.")
        return 1

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label", "text"])
        writer.writerows(rows)

    spam_count = sum(label == "spam" for label, _ in rows)
    ham_count = sum(label == "ham" for label, _ in rows)
    print(f"Created: {OUTPUT_FILE}")
    print(f"Messages: {len(rows)} | Spam: {spam_count} | Ham: {ham_count}")
    print("Next step: run python train_model.py to retrain the classifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
