import argparse
import json
import sys

from .decoder import decode as _decode
from .encoder import encode as _encode


def encode_main(argv=None):
    ap = argparse.ArgumentParser(description="ATC encoder (carriers + style bytes)")
    ap.add_argument("--text", type=str, help="Input text to encode")
    ap.add_argument("--infile", type=str, help="Read text from file (mutually exclusive)")
    ap.add_argument("--out", type=str, default="-", help="JSON output (or '-')")
    args = ap.parse_args(argv)
    if (args.text is None) == (args.infile is None):
        print("Provide exactly one of --text or --infile", file=sys.stderr)
        sys.exit(1)
    text = args.text if args.text is not None else open(args.infile, "r", encoding="utf-8").read()
    pkg = _encode(text)
    out_json = json.dumps(pkg, ensure_ascii=False, indent=2)
    if args.out in ("-", None):
        print(out_json)
    else:
        open(args.out, "w", encoding="utf-8").write(out_json)


def decode_main(argv=None):
    ap = argparse.ArgumentParser(description="ATC decoder (JSON package to text)")
    ap.add_argument("--in", dest="infile", type=str, default="-", help="JSON input (or '-')")
    args = ap.parse_args(argv)
    pkg = json.load(
        sys.stdin if args.infile in ("-", None) else open(args.infile, "r", encoding="utf-8")
    )
    print(_decode(pkg))
