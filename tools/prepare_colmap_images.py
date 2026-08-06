#!/usr/bin/env python3
"""
Prepare a deterministic local COLMAP image set from the the source photos.

For smoke tests, downscale images while preserving EXIF bytes so COLMAP/exiftool
can still read GPS and camera metadata. For production, use --max-size 0 to copy
the originals unchanged.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


IMAGE_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def list_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix in IMAGE_EXTS
    )


def select_images(images: list[Path], mode: str, max_count: int | None) -> list[Path]:
    if max_count is None or max_count >= len(images):
        return images
    if max_count <= 0:
        raise ValueError("--max-count must be positive")
    if mode == "first":
        return images[:max_count]
    if mode == "even":
        if max_count == 1:
            return [images[0]]
        last = len(images) - 1
        indexes = sorted({round(i * last / (max_count - 1)) for i in range(max_count)})
        return [images[i] for i in indexes]
    raise ValueError(f"Unsupported selection mode: {mode}")


def make_unique_name(image: Path, used: set[str]) -> str:
    name = image.name
    if name not in used:
        used.add(name)
        return name
    stem = image.stem
    suffix = image.suffix.lower()
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


def resize_or_copy(src: Path, dst: Path, max_size: int) -> tuple[int, int]:
    if max_size <= 0:
        shutil.copy2(src, dst)
        with Image.open(dst) as im:
            return im.size

    with Image.open(src) as im:
        exif = im.info.get("exif")
        width, height = im.size
        scale = min(1.0, max_size / max(width, height))
        if scale < 1.0:
            new_size = (round(width * scale), round(height * scale))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        else:
            new_size = (width, height)

        save_kwargs = {"quality": 95, "subsampling": 0}
        if exif:
            save_kwargs["exif"] = exif
        im.save(dst, "JPEG", **save_kwargs)
        return new_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(r"/path/to/source/images"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/example_site/local_colmap/images"),
    )
    parser.add_argument("--max-count", type=int, default=300)
    parser.add_argument("--selection", choices=("even", "first"), default="even")
    parser.add_argument(
        "--max-size",
        type=int,
        default=4096,
        help="Longest output image side in pixels. Use 0 to copy originals.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {output}. Use --force to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    images = list_images(source)
    selected = select_images(images, args.selection, args.max_count)
    used: set[str] = set()
    rows = []
    for index, image in enumerate(selected, start=1):
        name = make_unique_name(image, used)
        dst = output / name
        width, height = resize_or_copy(image, dst, args.max_size)
        rows.append(
            {
                "index": index,
                "image_name": name,
                "source_path": str(image),
                "width": width,
                "height": height,
            }
        )

    manifest = output.parent / "image_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image_name", "source_path", "width", "height"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Source images found: {len(images)}")
    print(f"Images prepared: {len(selected)}")
    print(f"Output images: {output}")
    print(f"Manifest: {manifest}")
    print(f"Max size: {'original' if args.max_size <= 0 else args.max_size}")


if __name__ == "__main__":
    main()
