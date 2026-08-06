#!/usr/bin/env python3
"""Deterministically decimate a binary_little_endian PLY by vertex record."""

from __future__ import annotations

import argparse
from pathlib import Path


TYPE_SIZES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


def read_header(path: Path) -> tuple[bytes, int, int, int]:
    with path.open("rb") as f:
        data = f.read(65536)

    marker = b"end_header\n"
    marker_index = data.find(marker)
    if marker_index < 0:
        raise ValueError("PLY header missing end_header")

    header_end = marker_index + len(marker)
    header_text = data[:header_end].decode("ascii")
    lines = header_text.splitlines()

    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("Only binary_little_endian 1.0 PLY files are supported")

    vertex_count = None
    in_vertex = False
    record_size = 0

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and parts[0] == "property":
            if parts[1] == "list":
                raise ValueError("List properties in vertex records are not supported")
            record_size += TYPE_SIZES[parts[1]]

    if vertex_count is None:
        raise ValueError("PLY header missing element vertex")
    if record_size <= 0:
        raise ValueError("PLY vertex record size could not be determined")

    return data[:header_end], header_end, vertex_count, record_size


def rewrite_vertex_count(header: bytes, new_count: int) -> bytes:
    lines = header.decode("ascii").splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.startswith("element vertex "):
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            lines[idx] = f"element vertex {new_count}{newline}"
            return "".join(lines).encode("ascii")
    raise ValueError("PLY header missing element vertex")


def copy_even_sample(input_path: Path, output_path: Path, target_count: int, chunk_records: int) -> int:
    header, header_end, vertex_count, record_size = read_header(input_path)
    output_count = min(target_count, vertex_count)
    if output_count <= 0:
        raise ValueError("target_count must be positive")

    expected_size = header_end + vertex_count * record_size
    actual_size = input_path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(f"PLY payload is truncated: expected at least {expected_size}, got {actual_size}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_header = rewrite_vertex_count(header, output_count)

    selected = 0
    next_selected = 0
    with input_path.open("rb") as src, output_path.open("wb") as dst:
        src.seek(header_end)
        dst.write(out_header)

        idx = 0
        while idx < vertex_count and selected < output_count:
            records_left = vertex_count - idx
            batch_count = min(chunk_records, records_left)
            batch = src.read(batch_count * record_size)
            if len(batch) != batch_count * record_size:
                raise ValueError("Unexpected EOF while reading vertex records")

            batch_start = idx
            batch_end = idx + batch_count
            while selected < output_count and next_selected < batch_end:
                if next_selected >= batch_start:
                    offset = (next_selected - batch_start) * record_size
                    dst.write(batch[offset : offset + record_size])
                    selected += 1
                    if output_count == 1:
                        next_selected = vertex_count
                    else:
                        next_selected = round(selected * (vertex_count - 1) / (output_count - 1))
                else:
                    break

            idx = batch_end

    return output_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-count", required=True, type=int)
    parser.add_argument("--chunk-records", type=int, default=200_000)
    args = parser.parse_args()

    count = copy_even_sample(args.input, args.output, args.target_count, args.chunk_records)
    print(f"Wrote {count} vertices to {args.output}")


if __name__ == "__main__":
    main()
