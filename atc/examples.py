"""
Demo for the ATC prototype.
"""

from atc.decoder import decode
from atc.encoder import encode


def run_demo():
    texts = [
        "I am in it, okay?  YES!",
        "Hello, world!  This is Adaptive Text Compression.",
        "Edge cases:   multiple   spaces, punctuation!!! and CAPS.",
    ]
    for t in texts:
        pkg = encode(t)
        roundtrip = decode(pkg)
        ok = t == roundtrip
        print("=" * 60)
        print("IN :", repr(t))
        print("PKG:", {"carriers": pkg["carriers"], "style_b64_len": len(pkg["style_b64"])})
        print("OUT:", repr(roundtrip))
        print("OK :", ok)


if __name__ == "__main__":
    run_demo()
