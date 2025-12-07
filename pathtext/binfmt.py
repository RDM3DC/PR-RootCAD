import base64
import struct
from pathlib import Path
from typing import Dict

from atc.codec_ac import unpack as atc_unpack

MAGIC = b"ATCP2\x01"


def _zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def _zigzag_decode(u: int) -> int:
    return (u >> 1) ^ (-(u & 1))


def _put_varint(u: int, out: bytearray):
    while True:
        b = u & 0x7F
        u >>= 7
        if u:
            out.append(0x80 | b)
        else:
            out.append(b)
            break


def _get_varint(data: bytes, off: int):
    shift = 0
    val = 0
    while True:
        b = data[off]
        off += 1
        val |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return val, off
        shift += 7


def pack_binary(container: Dict) -> bytes:
    assert container.get("format") == "ATC-PATH-v1"
    text_ac = container["text_ac"]
    n = int(text_ac["n"])
    ext = text_ac.get("ext", "").encode("utf-8")
    atc_data = base64.b64decode(text_ac["data_b64"])
    paths = container.get("paths", [])
    layout = container.get("layout", [])

    chunks = [MAGIC]
    chunks.append(struct.pack("<I H I H H", n, len(ext), len(atc_data), len(paths), len(layout)))
    chunks.append(ext)
    chunks.append(atc_data)

    # Paths: pack with varint deltas (flags=1)
    for p in paths:
        if "anchors_cmc" in p:
            anchors = p["anchors_cmc"].get("anchors", [])
            eps = float(p.get("max_err", 0.0))
        else:
            anchors = p.get("anchors", [])
            eps = float(p.get("max_err", 0.0))
        # quantize
        xi = [int(round(pt[0])) for pt in anchors]
        yi = [int(round(pt[1])) for pt in anchors]
        count = len(xi)
        flags = 1  # varint deltas
        chunks.append(struct.pack("<H f I", flags, eps, count))
        if count == 0:
            continue
        # write first absolute
        buf = bytearray()
        buf += struct.pack("<ii", xi[0], yi[0])
        px, py = xi[0], yi[0]
        for i in range(1, count):
            dx = xi[i] - px
            dy = yi[i] - py
            _put_varint(_zigzag_encode(dx), buf)
            _put_varint(_zigzag_encode(dy), buf)
            px, py = xi[i], yi[i]
        chunks.append(bytes(buf))

    # Layout: index paths by order
    id2idx = {p.get("id", f"p{i}"): i for i, p in enumerate(paths)}
    for lay in layout:
        idx = id2idx[lay["path"]]
        a, b = lay["range"]
        spacing = float(lay.get("spacing_px", 14))
        offset = float(lay.get("offset_px", 0))
        chunks.append(struct.pack("<H I I f f", idx, a, b, spacing, offset))

    return b"".join(chunks)


def unpack_to_svg(
    blob: bytes, out_svg: str, width: int = 800, height: int = 300, font_family: str = "sans-serif"
) -> str:
    if not blob.startswith(MAGIC):
        raise ValueError("Not ATCP2")
    off = len(MAGIC)
    n, ext_len, atc_size, num_paths, num_layout = struct.unpack_from("<I H I H H", blob, off)
    off += struct.calcsize("<I H I H H")
    ext = blob[off : off + ext_len]
    off += ext_len
    atc_data = blob[off : off + atc_size]
    off += atc_size
    text_obj = {
        "format": "ATC-AC2-v2",
        "n": int(n),
        "ext": ext.decode("utf-8"),
        "data_b64": base64.b64encode(atc_data).decode("ascii"),
    }
    text = atc_unpack(text_obj)

    # paths
    paths_pts = []
    for _ in range(num_paths):
        flags, eps, count = struct.unpack_from("<H f I", blob, off)
        off += struct.calcsize("<H f I")
        if count == 0:
            paths_pts.append([])
            continue
        x0, y0 = struct.unpack_from("<ii", blob, off)
        off += 8
        pts = [(x0, y0)]
        if (flags & 1) == 0:
            # legacy (not used here)
            for _ in range(count - 1):
                dx, dy = struct.unpack_from("<hh", blob, off)
                off += 4
                x0 += dx
                y0 += dy
                pts.append((x0, y0))
        else:
            # varint+zigzag pairs
            for _ in range(count - 1):
                dx_u, off = _get_varint(blob, off)
                dy_u, off = _get_varint(blob, off)
                dx = _zigzag_decode(dx_u)
                dy = _zigzag_decode(dy_u)
                x0 += dx
                y0 += dy
                pts.append((x0, y0))
        paths_pts.append(pts)

    # layout
    layouts = []
    for _ in range(num_layout):
        idx, a, b, spacing, offset = struct.unpack_from("<H I I f f", blob, off)
        off += struct.calcsize("<H I I f f")
        layouts.append((idx, a, b, spacing, offset))

    # svg
    def path_d(pts):
        if not pts:
            return ""
        d = [f"M {pts[0][0]},{pts[0][1]}"]
        for x, y in pts[1:]:
            d.append(f"L {x},{y}")
        return " ".join(d)

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append("<defs>")
    for i, pts in enumerate(paths_pts):
        parts.append(
            f'<path id="p{i}" d="{path_d(pts)}" fill="none" stroke="#ccc" stroke-width="1"/>'
        )
    parts.append("</defs>")
    for i, pts in enumerate(paths_pts):
        parts.append(f'<path d="{path_d(pts)}" fill="none" stroke="#e0e0e0" stroke-width="1"/>')
    for idx, a, b, spacing, offset in layouts:
        frag = (text[a:b]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f'<text font-family="{font_family}" font-size="{spacing}px" letter-spacing="0.5px">'
        )
        parts.append(f'  <textPath href="#p{idx}" startOffset="{offset}">{frag}</textPath>')
        parts.append("</text>")
    parts.append("</svg>")
    Path(out_svg).write_text("\n".join(parts), encoding="utf-8")
    return out_svg
