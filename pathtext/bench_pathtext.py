"""
Benchmark ATC-PATH: sweep path simplification epsilon and report size/bpc.
Usage:
  python -m pathtext.bench_pathtext --text bench/samples/paragraph.txt --out_csv pathtext/bench_pathtext.csv
"""

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Cmd failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--eps_list", default="0.5,1,2,4,6,8")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    text_path = (repo / args.text) if not args.text.startswith("/") else Path(args.text)
    text = text_path.read_text(encoding="utf-8")
    n_chars = len(text)

    anchors_json = repo / "pathtext" / "bench_anchors.json"
    # generate anchors (sine)
    run(["python", "-m", "pathtext.make_sine", str(anchors_json)], cwd=repo)

    # encode base container
    src_json = repo / "pathtext" / "bench_src.atcp.json"
    run(
        [
            "python",
            "-m",
            "pathtext.encode",
            "--text",
            str(text_path),
            "--anchors",
            str(anchors_json),
            "--out",
            str(src_json),
        ],
        cwd=repo,
    )

    rows = []
    for eps in [float(x) for x in args.eps_list.split(",")]:
        # compress paths with eps
        cmp_json = repo / "pathtext" / f"bench_eps{eps}.atcp.json"
        run(
            [
                "python",
                "-m",
                "pathtext.compress_paths",
                str(src_json),
                str(cmp_json),
                "--eps",
                str(eps),
            ],
            cwd=repo,
        )
        # binary pack
        bin_path = repo / "pathtext" / f"bench_eps{eps}.atcp2"
        run(["python", "-m", "pathtext.pack_bin", str(cmp_json), str(bin_path)], cwd=repo)
        size = bin_path.stat().st_size
        bpc = 8 * size / n_chars
        # get counts from json
        obj = json.loads(cmp_json.read_text(encoding="utf-8"))
        for p in obj.get("paths", []):
            comp = p.get("anchors_cmc", {})
            rows.append(
                {
                    "eps": eps,
                    "n_chars": n_chars,
                    "orig_points": comp.get("orig_points"),
                    "new_points": comp.get("new_points"),
                    "max_err_px": comp.get("max_err_px"),
                    "bytes": size,
                    "bpc_bits": bpc,
                }
            )
        # bin also
    # write CSV
    import csv

    with open(repo / args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "eps",
                "n_chars",
                "orig_points",
                "new_points",
                "max_err_px",
                "bytes",
                "bpc_bits",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {repo / args.out_csv}")


if __name__ == "__main__":
    main()
