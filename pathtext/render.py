import argparse
import json
from pathlib import Path

from .pathtext import to_svg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    args = ap.parse_args()
    obj = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    to_svg(obj, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
