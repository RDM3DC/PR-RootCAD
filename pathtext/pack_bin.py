import argparse
import json
from pathlib import Path

from .binfmt import pack_binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    args = ap.parse_args()
    obj = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    blob = pack_binary(obj)
    Path(args.out).write_bytes(blob)
    print(f"Wrote {args.out} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
