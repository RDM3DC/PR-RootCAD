import argparse, base64, json, sys
from typing import Dict, List
from collections import deque
from .utils import make_style_byte, pack_style_bytes, PUNCT2CODE

PUNCT_SET = set([".", ",", "!", "?", ";", ":"])
ZERO_WIDTH = "\u200b"  # zero-width carrier

def encode(text: str) -> Dict[str, str]:
    carriers: List[str] = []
    style_bytes: List[int] = []
    spaces_queue = deque()
    last_carrier_idx = None
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch == " ":
            if not spaces_queue or spaces_queue[-1] == 3:
                spaces_queue.append(1)
            else:
                spaces_queue[-1] += 1
            i += 1
            continue

        if ch in PUNCT_SET:
            run = []
            while i < n and text[i] in PUNCT_SET:
                run.append(text[i])
                i += 1
            if last_carrier_idx is None:
                for p in run:
                    carriers.append(ZERO_WIDTH)
                    style_bytes.append(make_style_byte(0, PUNCT2CODE[p], 0))
            else:
                first = True
                for p in run:
                    if first:
                        b = style_bytes[last_carrier_idx]
                        b = (b & 0b11100011) | ((PUNCT2CODE[p] & 0b111) << 2)
                        style_bytes[last_carrier_idx] = b
                        first = False
                    else:
                        carriers.append(ZERO_WIDTH)
                        style_bytes.append(make_style_byte(0, PUNCT2CODE[p], 0))
            continue

        base = ch.lower()
        cap_flag = 1 if ("A" <= ch <= "Z") else 0

        # flush extra space chunks via zero-width carriers
        while len(spaces_queue) > 1:
            sb = spaces_queue.popleft()
            carriers.append(ZERO_WIDTH)
            style_bytes.append(make_style_byte(sb, 0, 0))

        spaces_before = spaces_queue.popleft() if spaces_queue else 0

        b = make_style_byte(spaces_before, 0, cap_flag)
        carriers.append(base)
        style_bytes.append(b)
        last_carrier_idx = len(style_bytes) - 1
        i += 1

    carriers_str = "".join(carriers)
    style_b64 = base64.b64encode(pack_style_bytes(style_bytes)).decode("ascii")
    return {"carriers": carriers_str, "style_b64": style_b64}

def main():
    ap = argparse.ArgumentParser(description="ATC encoder")
    ap.add_argument("--text", type=str, help="Input text")
    ap.add_argument("--infile", type=str, help="Read text from file")
    ap.add_argument("--out", type=str, default="-", help="Write JSON to path or '-'")
    args = ap.parse_args()
    if (args.text is None) == (args.infile is None):
        print("Provide exactly one of --text or --infile", file=sys.stderr); sys.exit(1)
    text = args.text if args.text is not None else open(args.infile, "r", encoding="utf-8").read()
    pkg = encode(text)
    out_json = json.dumps(pkg, ensure_ascii=False, indent=2)
    if args.out == "-" or args.out is None:
        print(out_json)
    else:
        open(args.out, "w", encoding="utf-8").write(out_json)

if __name__ == "__main__":
    main()
