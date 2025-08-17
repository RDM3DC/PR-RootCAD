
import argparse
from pathlib import Path
from .binfmt import unpack_to_svg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    args = ap.parse_args()
    blob = Path(args.inp).read_bytes()
    out = unpack_to_svg(blob, args.out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
