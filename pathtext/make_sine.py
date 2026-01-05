import argparse
import json
import math
from pathlib import Path


def sine_path(width=800, height=300, cycles=2.5, amp=40, n=200):
    y0 = height / 2
    anchors = []
    for i in range(n):
        x = i * (width - 40) / (n - 1) + 20
        t = i / (n - 1)
        y = y0 + amp * math.sin(2 * math.pi * cycles * t)
        anchors.append([x, y])
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--cycles", type=float, default=2.5)
    ap.add_argument("--amp", type=float, default=40)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    anchors = sine_path(args.width, args.height, args.cycles, args.amp, args.n)
    obj = {"paths": [{"id": "p0", "anchors": anchors}]}
    Path(args.out).write_text(json.dumps(obj), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
