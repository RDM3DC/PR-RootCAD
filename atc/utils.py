from typing import List

PUNCT2CODE = {None: 0, ".": 1, ",": 2, "!": 3, "?": 4, ";": 5, ":": 6}
CODE2PUNCT = {v: k for k, v in PUNCT2CODE.items()}

def make_style_byte(spaces_before: int, punct_after_code: int, capitalize_self: int) -> int:
    if not (0 <= spaces_before <= 3):
        raise ValueError("spaces_before must be 0..3")
    if not (0 <= punct_after_code <= 7):
        raise ValueError("punct_after_code must be 0..7")
    if not (capitalize_self in (0, 1)):
        raise ValueError("capitalize_self must be 0 or 1")
    b = (spaces_before & 0b11) | ((punct_after_code & 0b111) << 2) | ((capitalize_self & 0b1) << 5)
    return b

def parse_style_byte(b: int):
    spaces_before = b & 0b11
    punct_after_code = (b >> 2) & 0b111
    capitalize_self = (b >> 5) & 0b1
    return spaces_before, punct_after_code, capitalize_self

def pack_style_bytes(bytes_list: List[int]) -> bytes:
    return bytes(bytes_list)

def unpack_style_bytes(blob: bytes) -> List[int]:
    return list(blob)
