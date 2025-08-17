import base64
from atc.encoder import encode
from atc.decoder import decode

def test_atc_roundtrip():
    cases = [
        "I am in it, okay?  YES!",
        "Hello, world!  This is Adaptive Text Compression.",
        "Edge cases:   multiple   spaces, punctuation!!! and CAPS.",
        "A  B   C    D",
        "Start punctuation?! End...",
    ]
    for t in cases:
        pkg = encode(t)
        out = decode(pkg)
        assert out == t
    # length align
    pkg = encode("Hello,   world!!!  OK?")
    assert len(pkg["carriers"]) == len(base64.b64decode(pkg["style_b64"]))
