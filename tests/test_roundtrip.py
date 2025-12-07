import base64

from atc.decoder import decode
from atc.encoder import encode

CASES = [
    "I am in it, okay?  YES!",
    "Hello, world!  This is Adaptive Text Compression.",
    "Edge cases:   multiple   spaces, punctuation!!! and CAPS.",
    "A  B   C    D",
    "Start punctuation?! End...",
    "No punctuation just   spaces   and CAPS",
]


def test_roundtrip_cases():
    for t in CASES:
        pkg = encode(t)
        out = decode(pkg)
        assert out == t


def test_lengths_align():
    t = "Hello,   world!!!  OK?"
    pkg = encode(t)
    carriers = pkg["carriers"]
    style_b64 = pkg["style_b64"]
    style_bytes = base64.b64decode(style_b64)
    assert len(carriers) == len(style_bytes)
