"""Minimal Flask web UI for classifying URLs with the trained model.

Run:
    python -m web.app --host 127.0.0.1 --port 5000
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from flask import Flask, jsonify, render_template, request  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishing_detector import PhishingDetector  # noqa: E402

MAX_URL_LENGTH = 2048

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
_detector: PhishingDetector | None = None


def get_detector() -> PhishingDetector:
    global _detector
    if _detector is None:
        _detector = PhishingDetector()
    return _detector


@app.route("/")
def index():
    return render_template("index.html", metrics=get_detector().metrics)


import re
from urllib.parse import urlparse

_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)
_SUSPICIOUS_TLDS = (".tk", ".xyz", ".top", ".ml", ".ga", ".cf", ".gq")
_SHORTENERS = ("bit.ly", "goo.gl", "t.co", "tinyurl.com", "is.gd", "cli.gs", "short.cm")
_DANGEROUS_EXTS = (".exe", ".php", ".asp", ".aspx", ".jsp", ".py")
_PHISHING_WORDS = ("secure", "account", "verify", "login", "update", "signin", "banking", "confirm")
_TOP_LEGIT_BRANDS = (
    "google.com", "github.com", "microsoft.com", "apple.com", "amazon.com",
    "stackoverflow.com", "wikipedia.org", "netflix.com", "paypal.com",
    "youtube.com", "linkedin.com", "facebook.com", "twitter.com", "reddit.com",
)


def inspect_url(url: str) -> dict:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    hostname = (parsed.netloc or "").split(":")[0].lower()
    path = (parsed.path or "").lower()
    url_lower = url.lower()

    has_ip = bool(_IPV4_RE.match(hostname))
    punycode = "xn--" in hostname
    suspicious_tld = next((t for t in _SUSPICIOUS_TLDS if hostname.endswith(t)), None)
    detected_hints = [w for w in _PHISHING_WORDS if w in url_lower]
    dangerous_ext = next((e for e in _DANGEROUS_EXTS if path.endswith(e)), None)
    shortener = next((s for s in _SHORTENERS if s in hostname), None)
    subdomain_count = max(0, hostname.count(".") - 1) if "." in hostname else 0
    is_trusted = any(
        hostname == b or hostname.endswith("." + b) for b in _TOP_LEGIT_BRANDS
    )

    scheme = parsed.scheme if parsed.scheme else ("https" if url.startswith("https") else "http")

    return {
        "protocol": scheme,
        "hostname": hostname,
        "path": parsed.path or "/",
        "has_ip": has_ip,
        "punycode": punycode,
        "subdomain_count": subdomain_count,
        "suspicious_tld": suspicious_tld,
        "phish_hints": detected_hints,
        "dangerous_ext": dangerous_ext,
        "shortener": shortener,
        "is_trusted_brand": is_trusted,
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    raw = data.get("url")
    if not isinstance(raw, str):
        return jsonify({"error": "'url' must be a string"}), 400
    url = raw.strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    if len(url) > MAX_URL_LENGTH:
        return jsonify({"error": f"URL exceeds maximum length of {MAX_URL_LENGTH} characters"}), 400
    try:
        result = get_detector().predict(url)
        payload = result.as_dict()
        payload["inspection"] = inspect_url(url)
        return jsonify(payload)
    except Exception as e:  # pragma: no cover
        app.logger.exception("prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # Warm up the model so the first request isn't slow.
    get_detector()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
