#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".webp", ".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4"}


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int
    size: int
    dhash: int
    ahash: int
    thumb: bytes

    @property
    def quality_key(self) -> Tuple[int, int]:
        return (self.width * self.height, self.size)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def average_hash(image: Image.Image, hash_size: int = 16) -> int:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= avg)
    return value


def difference_hash(image: Image.Image, hash_size: int = 16) -> int:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for y in range(hash_size):
        row = y * (hash_size + 1)
        for x in range(hash_size):
            value = (value << 1) | int(pixels[row + x] > pixels[row + x + 1])
    return value


def thumbnail_bytes(image: Image.Image, size: int = 96) -> bytes:
    thumb = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return bytes(thumb.getdata())


def mean_abs_diff(left: bytes, right: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def load_image_info(path: Path) -> ImageInfo:
    with Image.open(path) as image:
        width, height = image.size
        return ImageInfo(
            path=path,
            width=width,
            height=height,
            size=path.stat().st_size,
            dhash=difference_hash(image),
            ahash=average_hash(image),
            thumb=thumbnail_bytes(image),
        )


def are_visual_duplicates(left: ImageInfo, right: ImageInfo) -> bool:
    if hamming(left.dhash, right.dhash) > 10:
        return False
    if hamming(left.ahash, right.ahash) > 12:
        return False
    return mean_abs_diff(left.thumb, right.thumb) <= 3.0


def collect_files(root: Path) -> Tuple[List[Path], List[Path]]:
    images: List[Path] = []
    videos: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            images.append(path)
        elif suffix in VIDEO_SUFFIXES:
            videos.append(path)
    return sorted(images), sorted(videos)


def extract_aweme_id(path: Path) -> str:
    for part in (path.parent.name, path.stem):
        match = re.search(r"(\d{16,20})", part)
        if match:
            return match.group(1)
    return "unknown"


def extract_date(path: Path) -> str:
    for part in (path.parent.name, path.stem):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", part)
        if match:
            return match.group(1)
    return "unknown-date"


def extract_asset_index(path: Path) -> str:
    stem = path.stem
    match = re.search(r"_(live_\d+|\d+)$", stem)
    if match:
        return match.group(1)
    return "main"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def group_images(images: List[Path]) -> Tuple[List[List[ImageInfo]], List[Path]]:
    groups: List[List[ImageInfo]] = []
    representatives: List[ImageInfo] = []
    unreadable: List[Path] = []

    for index, path in enumerate(images, start=1):
        try:
            info = load_image_info(path)
        except Exception:
            unreadable.append(path)
            continue

        matched = False
        for group_index, rep in enumerate(representatives):
            if are_visual_duplicates(info, rep):
                groups[group_index].append(info)
                if info.quality_key > rep.quality_key:
                    representatives[group_index] = info
                matched = True
                break

        if not matched:
            groups.append([info])
            representatives.append(info)

        if index % 250 == 0:
            print(f"Scanned images: {index}/{len(images)}")

    return groups, unreadable


def organize(source: Path, output: Path) -> Dict[str, int]:
    if output.exists():
        shutil.rmtree(output)
    image_output = output / "图片_去重"
    video_output = output / "MP4"
    report_output = output / "reports"
    image_output.mkdir(parents=True)
    video_output.mkdir(parents=True)
    report_output.mkdir(parents=True)

    images, videos = collect_files(source)
    groups, unreadable = group_images(images)

    duplicate_records = []
    kept_images = []
    for out_index, group in enumerate(groups, start=1):
        best = max(group, key=lambda item: item.quality_key)
        kept_images.append(best.path)
        aweme_id = extract_aweme_id(best.path)
        date = extract_date(best.path)
        asset_index = extract_asset_index(best.path)
        dst = image_output / (
            f"{date}_img_{out_index:04d}_{aweme_id}_{asset_index}{best.path.suffix.lower()}"
        )
        link_or_copy(best.path, dst)

        if len(group) > 1:
            duplicate_records.append(
                {
                    "kept": str(best.path),
                    "duplicates": [str(item.path) for item in group if item.path != best.path],
                    "group_size": len(group),
                    "kept_width": best.width,
                    "kept_height": best.height,
                    "kept_size": best.size,
                }
            )

    for out_index, video in enumerate(videos, start=1):
        aweme_id = extract_aweme_id(video)
        date = extract_date(video)
        dst = video_output / f"{date}_mp4_{out_index:04d}_{aweme_id}{video.suffix.lower()}"
        link_or_copy(video, dst)

    report = {
        "source": str(source),
        "output": str(output),
        "original_images": len(images),
        "kept_images": len(kept_images),
        "visual_duplicate_images_removed": len(images) - len(kept_images),
        "duplicate_groups": len(duplicate_records),
        "unreadable_images": [str(path) for path in unreadable],
        "mp4_files": len(videos),
    }
    (report_output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_output / "visual_duplicates.json").write_text(
        json.dumps(duplicate_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = organize(args.source.resolve(), args.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
