import argparse
import base64
import json
import sys
from typing import Dict

from .utils import CODE2PUNCT, parse_style_byte, unpack_style_bytes

ZERO_WIDTH = "\u200b"


def decode(pkg: Dict[str, str]) -> str:
    carriers = pkg["carriers"]
    style_bytes = unpack_style_bytes(base64.b64decode(pkg["style_b64"]))
    if len(carriers) != len(style_bytes):
        raise ValueError("Length mismatch: carriers vs style bytes")

    out = []
    for ch, b in zip(carriers, style_bytes):
        spaces_before, punct_code, cap_flag = parse_style_byte(b)
        out.append(" " * spaces_before)
        if ch != ZERO_WIDTH:
            out.append(ch.upper() if cap_flag else ch)
        punct = CODE2PUNCT.get(punct_code)
        if punct is not None:
            out.append(punct)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description="ATC decoder")
    ap.add_argument("--in", dest="infile", type=str, default="-", help="JSON input path or '-'")
    args = ap.parse_args()
    pkg = json.load(
        sys.stdin if args.infile in ("-", None) else open(args.infile, "r", encoding="utf-8")
    )
    print(decode(pkg))


if __name__ == "__main__":
    main()
