#!/usr/bin/env python3
"""Generate binary masks from segmentation label images using mask_report.json.

For each segmask PNG in the input folder, pixels whose label id is in the
requested set are written as black (0); everything else is white (255).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_labels(labels_arg):
    try:
        return {int(x.strip()) for x in labels_arg.split(",") if x.strip() != ""}
    except ValueError:
        raise SystemExit(f"Invalid --labels value: {labels_arg!r} (expected comma-separated integers)")


def load_all_classes(mask_report_path):
    with open(mask_report_path, "r") as f:
        report = json.load(f)
    all_classes = report.get("all_classes")
    if not all_classes:
        raise SystemExit(f"'all_classes' not found in {mask_report_path}")
    return all_classes


def resolve_png_path(input_dir, report_name):
    """mask_report.json entries may be missing the '.jpg' segment that's
    actually present in the on-disk filename (e.g. 'keyframe_00147.png' vs
    'keyframe_00147.jpg.png'), so try a few reasonable variants."""
    candidates = [
        input_dir / report_name,
        input_dir / report_name.replace(".png", ".jpg.png"),
    ]
    stem = Path(report_name).stem
    candidates.append(input_dir / f"{stem}.jpg.png")

    for c in candidates:
        if c.exists():
            return c

    matches = list(input_dir.glob(f"{stem}*.png"))
    if len(matches) == 1:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-dir", required=True, type=Path,
                         help="Folder containing segmask PNG files")
    parser.add_argument("-m", "--mask-report", default=Path("mask_report.json"), type=Path,
                         help="Path to the mask report JSON, relative to --input-dir "
                              "(default: mask_report.json)")
    parser.add_argument("-o", "--output-dir", required=True, type=Path,
                         help="Folder to write binary masks to")
    parser.add_argument("-l", "--labels", required=True,
                         help="Comma-separated label ids to write as black (0); "
                              "all other pixels become white (255), e.g. '7,8,9'")
    parser.add_argument("-r", "--reverse", action="store_true",
                         help="Reverse the output: selected labels become white (255) "
                              "and everything else black (0)")
    args = parser.parse_args()

    include_ids = parse_labels(args.labels)

    # An absolute --mask-report wins; otherwise it is resolved inside --input-dir.
    mask_report_path = args.input_dir / args.mask_report

    all_classes = load_all_classes(mask_report_path)
    id_to_name = {v: k for k, v in all_classes.items()}
    unknown = sorted(i for i in include_ids if i not in id_to_name)
    if unknown:
        print(f"Warning: label id(s) {unknown} not found in mask_report.json all_classes", file=sys.stderr)

    included_names = [id_to_name.get(i, f"<unknown:{i}>") for i in sorted(include_ids)]
    print(f"Including labels: {included_names} (ids={sorted(include_ids)})")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(mask_report_path, "r") as f:
        report = json.load(f)
    mask_info = report.get("mask_info", {})
    if not mask_info:
        raise SystemExit("'mask_info' not found or empty in mask_report.json")

    processed = 0
    skipped = 0
    for report_name in sorted(mask_info.keys()):
        png_path = resolve_png_path(args.input_dir, report_name)
        if png_path is None:
            print(f"Skipping {report_name}: matching PNG not found in {args.input_dir}", file=sys.stderr)
            skipped += 1
            continue

        img = Image.open(png_path)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[..., 0]

        include_ids_arr = np.array(sorted(include_ids), dtype=arr.dtype)
        selected = np.isin(arr, include_ids_arr)
        if not args.reverse:
            selected = ~selected
        binary = selected.astype(np.uint8) * 255

        out_path = args.output_dir / png_path.name
        Image.fromarray(binary, mode="L").save(out_path)
        processed += 1

    print(f"Done. Wrote {processed} binary masks to {args.output_dir} ({skipped} skipped).")


if __name__ == "__main__":
    main()
